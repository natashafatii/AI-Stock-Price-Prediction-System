import os
import sys
import json
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, jsonify
from config import Config
from utils.logger import setup_logger, _bootstrap_root_logger
from utils.validators import validate_ticker, validate_date, ValidationError
from utils.indicators import get_feature_columns
from utils.visualizations import create_enhanced_prediction_chart, create_enhanced_backtesting_chart
from services.data_service import DataService
from services.ml_service import MLService

Config.init_app()

logger = setup_logger('app')

# Initialize the Flask web application
app = Flask(__name__)

app.config.from_object(Config)

_bootstrap_root_logger()

_wz_logger = logging.getLogger('werkzeug')

_wz_logger.setLevel(logging.INFO)

if not any(isinstance(h, logging.StreamHandler) for h in _wz_logger.handlers):

    _wz_handler = logging.StreamHandler(sys.stdout)

    _wz_handler.setLevel(logging.INFO)

    _wz_logger.addHandler(_wz_handler)

_wz_logger.propagate = False

data_service = DataService()

ml_service = MLService()

logger.info(f"[OK] {Config.APP_NAME} v{Config.VERSION} started successfully")

@app.route('/')

def home():
    """
    Renders the main homepage (index.html) of the application.
    This is the default route when a user visits the root URL.
    """

    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    """
    Handles the prediction request submitted from the homepage.
    This function performs the following critical steps:
    1. Validates the input stock ticker and target date.
    2. Fetches historical stock data via the DataService.
    3. Preprocesses data and calculates technical indicators.
    4. Prepares sequence data and normalizes it for the neural networks.
    5. Trains an ensemble of Machine Learning models via the MLService.
    6. Generates stock price predictions and calculates performance metrics (RMSE, MAE).
    7. Creates visualization charts (prediction plot and backtesting plot).
    8. Returns the prediction results to the user via the prediction_result.html template.
    """

    try:

        ticker_input = request.form.get('ticker', '').strip()

        date_input = request.form.get('date', '').strip()

        logger.info(f"Prediction request: ticker={ticker_input}, date={date_input}")

        try:

            ticker = validate_ticker(ticker_input)

        except ValidationError as e:

            logger.warning(f"Ticker validation failed: {e}")

            return f"<h2>Validation Error</h2><p>{str(e)}</p><a href='/'>Go Back</a>", 400

        is_valid, date_obj, error_msg = validate_date(date_input)

        if not is_valid:

            logger.warning(f"Date validation failed: {error_msg}")

            return f"<h2>Validation Error</h2><p>{error_msg}</p><a href='/'>Go Back</a>", 400

        end_date = date_input

        logger.info(f"Fetching data for {ticker}")

        df = data_service.fetch_stock_data(ticker, Config.START_DATE, end_date)

        if df is None or df.empty:

            error_html = f"""
            <h2>Data Error</h2>
            <p>No data found for ticker '{ticker}'. This could be due to:</p>
            <ul>
                <li>Invalid ticker symbol</li>
                <li>API rate limiting (try again in a few minutes)</li>
                <li>No data available for the specified date range</li>
            </ul>
            <a href='/'>Try Again</a>
            """

            return error_html, 404

        logger.info(f"Fetched {len(df)} days of data")

        if len(df) < Config.MIN_DATA_POINTS:

            return f"""
            <h2>Insufficient Data</h2>
            <p>Found only {len(df)} days of data for '{ticker}'. 
            Need at least {Config.MIN_DATA_POINTS} days for reliable predictions.</p>
            <a href='/'>Go Back</a>
            """, 400

        logger.info("Adding technical indicators...")

        df_processed = data_service.preprocess_data(df, add_indicators=True)

        if df_processed.empty or len(df_processed) < Config.SEQUENCE_LENGTH + 20:

            return f"""
            <h2>Insufficient Data</h2>
            <p>After processing, only {len(df_processed)} days remain. 
            Need at least {Config.SEQUENCE_LENGTH + 20} days.</p>
            <a href='/'>Go Back</a>
            """, 400

        logger.info(f"After preprocessing: {len(df_processed)} days, {df_processed.shape[1]} features")

        feature_cols = get_feature_columns()

        available_cols = [col for col in feature_cols if col in df_processed.columns]

        if 'Close' not in available_cols:

            available_cols.insert(0, 'Close')

        close_col_idx = available_cols.index('Close')

        raw_data = df_processed[available_cols].values

        n_total = len(raw_data) - Config.SEQUENCE_LENGTH

        n_test = int(n_total * Config.TEST_SIZE)

        n_val = int(n_total * Config.VALIDATION_SIZE)

        n_train = n_total - n_test - n_val

        if n_train < 20 or n_val < 5 or n_test < 5:

            return """
            <h2>Insufficient Data</h2>
            <p>Not enough data for train/val/test split after sequencing.</p>
            <a href='/'>Go Back</a>
            """, 400

        train_end = Config.SEQUENCE_LENGTH + n_train

        val_end = train_end + n_val

        raw_train = raw_data[:train_end]          

        raw_val = raw_data[train_end - Config.SEQUENCE_LENGTH:val_end]  

        raw_test = raw_data[val_end - Config.SEQUENCE_LENGTH:]         

        logger.info("Fitting scaler on training data only (no leakage)...")

        scaler = MinMaxScaler(feature_range=(-1, 1))

        scaler.fit(raw_train)

        data_service.scaler = scaler

        data_service.save_scaler()

        scaled_train = scaler.transform(raw_train)

        scaled_val = scaler.transform(raw_val)

        scaled_test = scaler.transform(raw_test)

        logger.info("Preparing sequences...")

        x_train, y_train = ml_service.prepare_sequences(

            scaled_train, Config.SEQUENCE_LENGTH, target_col_idx=close_col_idx

        )

        x_val, y_val = ml_service.prepare_sequences(

            scaled_val, Config.SEQUENCE_LENGTH, target_col_idx=close_col_idx

        )

        x_test, y_test = ml_service.prepare_sequences(

            scaled_test, Config.SEQUENCE_LENGTH, target_col_idx=close_col_idx

        )

        if len(x_train) == 0:

            return "<h2>Error</h2><p>Unable to create prediction sequences.</p><a href='/'>Go Back</a>", 400

        x_train = torch.tensor(x_train, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        x_val = torch.tensor(x_val, dtype=torch.float32)

        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        x_test = torch.tensor(x_test, dtype=torch.float32)

        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        input_dim = x_train.shape[2]

        logger.info(f"Data prepared: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}, features={input_dim}")

        logger.info("Training fresh models for this stock...")

        models_dict = ml_service.train_all_models(

            x_train, y_train, x_val, y_val, input_dim

        )

        ensemble = ml_service.create_ensemble(models_dict)

        use_ensemble = True

        logger.info("Making predictions...")

        predictions_scaled, std_scaled, individual_preds = ensemble.predict_with_confidence(x_test)

        predictions = ml_service.inverse_transform_predictions(

            predictions_scaled, scaler, available_cols

        )

        actual = df_processed['Close'].values[-len(predictions):]

        metrics = ml_service.calculate_metrics(predictions, actual)

        logger.info(f"Metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

        logger.info("Generating enhanced visualizations...")

        dates = df_processed.index[-len(predictions):]

        plot_path = os.path.join(Config.STATIC_FOLDER, 'prediction_plot.png')

        create_enhanced_prediction_chart(

            ticker=ticker,

            dates=dates,

            actual=actual,

            predictions=predictions,

            std_rescaled=ml_service.inverse_transform_predictions(std_scaled, scaler, available_cols) if std_scaled is not None else None,

            df_processed=df_processed,

            save_path=plot_path,

            dpi=120

        )

        signals = np.diff(predictions) > 0

        signals = np.insert(signals, 0, False)

        daily_returns = np.diff(actual) / actual[:-1]

        daily_returns = np.insert(daily_returns, 0, 0)

        strategy_returns = signals[:-1] * daily_returns[1:]

        cumulative_strategy = np.cumsum(strategy_returns)

        cumulative_stock = np.cumsum(daily_returns)

        plot_path1 = os.path.join(Config.STATIC_FOLDER, 'backtesting_plot.png')

        create_enhanced_backtesting_chart(

            ticker=ticker,

            cumulative_stock=cumulative_stock,

            cumulative_strategy=cumulative_strategy,

            actual=actual,

            predictions=predictions,

            save_path=plot_path1,

            dpi=120

        )

        latest_actual = float(np.squeeze(actual)[-1])

        latest_predicted = float(np.squeeze(predictions)[-1])

        return render_template('prediction_result.html',

                             plot_path='static/prediction_plot.png',

                             plot_path1='static/backtesting_plot.png',

                             ticker=ticker,

                             metrics=metrics,

                             model_weights=ensemble.get_weights() if use_ensemble else {},

                             latest_actual=latest_actual,

                             latest_predicted=latest_predicted)

    except Exception as e:

        logger.error(f"Prediction error: {str(e)}", exc_info=True)

        return f"""
        <h2>Error</h2>
        <p>An error occurred: {str(e)}</p>
        <p>Please try a different stock ticker or date range.</p>
        <a href='/'>Go Back</a>
        """, 500

if __name__ == '__main__':

    logger.info("Starting Flask application...")

    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000, use_reloader=False)

