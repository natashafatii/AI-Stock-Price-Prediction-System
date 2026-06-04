

import torch

import torch.nn as nn

import numpy as np

from typing import List, Dict, Optional

class EnsembleModel:
    """
    Ensemble Model that combines predictions from multiple individual base models 
    (e.g., LSTM, GRU, Transformer). By averaging their outputs (either equally or via learned weights),
    the ensemble reduces individual model variance and overfitting, leading to more robust 
    and accurate overall stock price predictions.
    """

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None, device='cpu'):

        self.models = models

        self.device = device

        if weights is None:

            self.weights = [1.0 / len(models)] * len(models)

        else:

            total = sum(weights)

            self.weights = [w / total for w in weights]

        for model in self.models:

            model.to(device)

            model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:

        predictions = []

        with torch.no_grad():

            for model in self.models:

                pred = model(x.to(self.device))

                predictions.append(pred)

        ensemble_pred = torch.zeros_like(predictions[0])

        for pred, weight in zip(predictions, self.weights):

            ensemble_pred += pred * weight

        return ensemble_pred

    def predict_with_confidence(self, x: torch.Tensor) -> tuple:
        """
        Generates predictions from all individual models and calculates the mean and standard deviation.
        The standard deviation acts as a measure of 'confidence' or uncertainty:
        - Low std = high confidence (all models agree on the prediction).
        - High std = low confidence (models disagree on the prediction).
        """

        predictions = []

        with torch.no_grad():

            for model in self.models:

                pred = model(x.to(self.device))

                predictions.append(pred.cpu().numpy())

        predictions_array = np.array(predictions)

        mean_pred = np.mean(predictions_array, axis=0)

        std_pred = np.std(predictions_array, axis=0)

        weighted_pred = np.zeros_like(predictions_array[0])

        for pred, weight in zip(predictions_array, self.weights):

            weighted_pred += pred * weight

        return weighted_pred, std_pred, predictions_array

    def update_weights(self, performance_scores: List[float], temperature: float = 2.0, 
                      min_weight: float = 0.15, max_weight: float = 0.50):
        """
        Dynamically updates the weight/importance of each model in the ensemble based on
        its recent performance scores. Better performing models get higher weights.
        The softmax function with temperature controls how aggressively weights are shifted.
        """

        scores = np.array(performance_scores)

        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        exp_scores = np.exp(-scores_normalized / temperature)

        raw_weights = exp_scores / exp_scores.sum()

        constrained_weights = np.clip(raw_weights, min_weight, max_weight)

        self.weights = (constrained_weights / constrained_weights.sum()).tolist()

    def get_model_names(self) -> List[str]:

        names = []

        for model in self.models:

            if hasattr(model, 'get_model_name'):

                names.append(model.get_model_name())

            else:

                names.append(model.__class__.__name__)

        return names

    def get_weights(self) -> Dict[str, float]:

        names = self.get_model_names()

        return dict(zip(names, self.weights))

    def get_model_name(self):

        return "Ensemble"

