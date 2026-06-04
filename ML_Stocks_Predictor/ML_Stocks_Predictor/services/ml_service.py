

import os

import sys

import json

import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, ParameterGrid

from typing import Tuple, Dict, List, Optional

import pickle

from config import Config

from utils.logger import setup_logger

from models.lstm_model import LSTMModel

from models.gru_model import GRUModel

from models.transformer_model import TransformerModel

from models.ensemble_model import EnsembleModel

logger = setup_logger(__name__)

class MLService:

    def __init__(self, device: str = None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

        self.models = {}

        self.scaler = None

        self.best_model_name = None

    def prepare_sequences(self, data: np.ndarray, sequence_length: int,

                         target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:

        x_data, y_data = [], []

        for i in range(sequence_length, len(data)):

            x_data.append(data[i - sequence_length:i, :])

            y_data.append(data[i, target_col_idx])

        return np.array(x_data), np.array(y_data)

    def train_model(self, model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,

                   x_val: torch.Tensor, y_val: torch.Tensor, 

                   lr: float = 0.001, num_epochs: int = 150,

                   weight_decay: float = 1e-4) -> Tuple[float, dict]:

        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode='min',

            factor=Config.LR_SCHEDULER_FACTOR,

            patience=Config.LR_SCHEDULER_PATIENCE,

            min_lr=Config.MIN_LEARNING_RATE

        )

        train_dataset = TensorDataset(x_train, y_train)

        train_loader = DataLoader(

            train_dataset,

            batch_size=Config.BATCH_SIZE,

            shuffle=True,  

            drop_last=False

        )

        best_val_loss = np.inf

        best_model_state = None

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):

            model.train()

            epoch_train_loss = 0.0

            num_train_samples = 0

            for batch_x, batch_y in train_loader:

                batch_x = batch_x.to(self.device)

                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                outputs = model(batch_x)

                loss = criterion(outputs, batch_y)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_train_loss += loss.item() * len(batch_x)

                num_train_samples += len(batch_x)

            train_loss = epoch_train_loss / num_train_samples

            model.eval()

            with torch.no_grad():

                val_outputs = model(x_val.to(self.device))

                val_loss = criterion(val_outputs, y_val.to(self.device)).item()

            current_lr = optimizer.param_groups[0]['lr']

            scheduler.step(val_loss)

            new_lr = optimizer.param_groups[0]['lr']

            if new_lr < current_lr:

                msg = f"  >> LR reduced: {current_lr:.2e} -> {new_lr:.2e}"

                logger.info(msg)

                print(msg, file=sys.stdout, flush=True)

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:

                msg = f"  Early stopping at epoch {epoch} (best val_loss: {best_val_loss:.6f})"

                logger.info(msg)

                print(msg, file=sys.stdout, flush=True)

                break

            if epoch % 5 == 0 or patience_counter == 0:

                msg = (f'  Epoch [{epoch:>3}/{num_epochs}]  '

                       f'train={train_loss:.6f}  val={val_loss:.6f}  '

                       f'lr={new_lr:.2e}  patience={patience_counter}/{Config.EARLY_STOPPING_PATIENCE}')

                logger.info(msg)

                print(msg, file=sys.stdout, flush=True)

        return best_val_loss, best_model_state

    def train_all_models(self, x_train: torch.Tensor, y_train: torch.Tensor,

                        x_val: torch.Tensor, y_val: torch.Tensor,

                        input_dim: int) -> Dict[str, Dict]:

        results = {}

        model_configs = {

            'LSTM': {

                'class': LSTMModel,

                'params': {'input_dim': input_dim, 'hidden_dim': 128, 'num_layers': 2, 

                          'output_dim': 1, 'dropout': 0.3}

            },

            'GRU': {

                'class': GRUModel,

                'params': {'input_dim': input_dim, 'hidden_dim': 128, 'num_layers': 2,

                          'output_dim': 1, 'dropout': 0.3}

            },

            'Transformer': {

                'class': TransformerModel,

                'params': {'input_dim': input_dim, 'd_model': 128, 'nhead': 8,

                          'num_layers': 2, 'output_dim': 1, 'dropout': 0.3}

            }

        }

        for model_name, config in model_configs.items():

            logger.info(f"\n{'='*50}")

            logger.info(f"  TRAINING {model_name}...")

            logger.info(f"{'='*50}")

            model = config['class'](**config['params'])

            val_loss, best_state = self.train_model(

                model, x_train, y_train, x_val, y_val,

                lr=Config.LEARNING_RATE,

                num_epochs=Config.NUM_EPOCHS,

                weight_decay=Config.WEIGHT_DECAY

            )

            model.load_state_dict(best_state)

            model.eval()

            results[model_name] = {

                'model': model,

                'val_loss': val_loss,

                'params': config['params']

            }

            self.models[model_name] = model

            logger.info(f"  {model_name} DONE — best val loss: {val_loss:.6f}\n")

        best_model_name = min(results.keys(), key=lambda k: results[k]['val_loss'])

        self.best_model_name = best_model_name

        logger.info(f"\nBest model: {best_model_name}")

        return results

    def create_ensemble(self, models_dict: Dict[str, Dict]) -> EnsembleModel:

        models = [models_dict[name]['model'] for name in ['LSTM', 'GRU', 'Transformer']]

        val_losses = [models_dict[name]['val_loss'] for name in ['LSTM', 'GRU', 'Transformer']]

        ensemble = EnsembleModel(models, device=self.device)

        ensemble.update_weights(val_losses, temperature=2.0, min_weight=0.15, max_weight=0.50)

        weights_dict = ensemble.get_weights()

        logger.info(f"Ensemble weights: {weights_dict}")

        logger.info(f"Validation losses - LSTM: {val_losses[0]:.6f}, GRU: {val_losses[1]:.6f}, Transformer: {val_losses[2]:.6f}")

        return ensemble

    def predict(self, model: nn.Module, x: torch.Tensor) -> np.ndarray:

        model.eval()

        with torch.no_grad():

            predictions = model(x.to(self.device))

        return predictions.cpu().numpy()

    def inverse_transform_predictions(self, predictions: np.ndarray, scaler, 

                                     feature_columns: List[str]) -> np.ndarray:

        close_idx = feature_columns.index('Close') if 'Close' in feature_columns else -1

        dummy = np.zeros((len(predictions), len(feature_columns)))

        dummy[:, close_idx] = predictions.flatten()

        rescaled = scaler.inverse_transform(dummy)

        return rescaled[:, close_idx]

    def save_models(self, models_dict: Dict[str, Dict], ensemble: EnsembleModel = None):

        os.makedirs(Config.MODELS_DIR, exist_ok=True)

        for model_name, model_info in models_dict.items():

            model_path = os.path.join(Config.MODELS_DIR, f'{model_name.lower()}_model.pth')

            torch.save(model_info['model'].state_dict(), model_path)

            params_path = os.path.join(Config.MODELS_DIR, f'{model_name.lower()}_params.json')

            with open(params_path, 'w') as f:

                json.dump(model_info['params'], f)

            logger.info(f"Saved {model_name} model to {model_path}")

        best_info = {

            'best_model': self.best_model_name,

            'models': {name: {'val_loss': info['val_loss']} 

                      for name, info in models_dict.items()}

        }

        with open(Config.BEST_PARAMS_PATH, 'w') as f:

            json.dump(best_info, f, indent=2)

        logger.info(f"Saved model info to {Config.BEST_PARAMS_PATH}")

    def load_models(self, input_dim: int) -> Dict[str, nn.Module]:

        models = {}

        if not os.path.exists(Config.BEST_PARAMS_PATH):

            logger.warning("No saved models found")

            return models

        with open(Config.BEST_PARAMS_PATH, 'r') as f:

            best_info = json.load(f)

        self.best_model_name = best_info.get('best_model')

        model_configs = {

            'LSTM': LSTMModel,

            'GRU': GRUModel,

            'Transformer': TransformerModel

        }

        for model_name, model_class in model_configs.items():

            model_path = os.path.join(Config.MODELS_DIR, f'{model_name.lower()}_model.pth')

            params_path = os.path.join(Config.MODELS_DIR, f'{model_name.lower()}_params.json')

            if os.path.exists(model_path) and os.path.exists(params_path):

                with open(params_path, 'r') as f:

                    params = json.load(f)

                params['input_dim'] = input_dim

                model = model_class(**params)

                model.load_state_dict(torch.load(model_path, map_location=self.device))

                model.to(self.device)

                model.eval()

                models[model_name] = model

                logger.info(f"Loaded {model_name} model")

        self.models = models

        return models

    def calculate_metrics(self, predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:

        mse = np.mean((predictions - actual) ** 2)

        rmse = np.sqrt(mse)

        mae = np.mean(np.abs(predictions - actual))

        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        return {

            'MSE': mse,

            'RMSE': rmse,

            'MAE': mae,

            'MAPE': mape

        }

