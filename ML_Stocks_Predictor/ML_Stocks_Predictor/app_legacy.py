import json

import os

import time

from datetime import datetime, timedelta

from flask import Flask, render_template, request

from models.lstm_model import LSTMModel

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

import torch

import torch.nn as nn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import ParameterGrid

import requests

from requests.adapters import HTTPAdapter

from urllib3.util.retry import Retry

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

BEST_MODEL_PATH = 'best_model.pth'

BEST_PARAMS_PATH = 'best_params.json'

def setup_yfinance_session():
    """
    Configures and returns a custom requests.Session object for the yfinance library.
    This session includes standard web browser headers to prevent getting blocked by APIs,
    and implements a retry strategy to automatically handle temporary network errors or rate limits (e.g., HTTP 429).
    """

    session = requests.Session()

    session.headers.update({

        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',

        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',

        'Accept-Language': 'en-US,en;q=0.5',

        'Accept-Encoding': 'gzip, deflate',

        'Connection': 'keep-alive',

        'Upgrade-Insecure-Requests': '1'

    })

    retry_strategy = Retry(

        total=3,  

        backoff_factor=2,  

        status_forcelist=[429, 500, 502, 503, 504],  

        allowed_methods=["HEAD", "GET", "OPTIONS"]

    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("http://", adapter)

    session.mount("https://", adapter)

    return session

yf_session = setup_yfinance_session()

def fetch_stock_data_with_retry(ticker_symbol, start_date, end_date, max_retries=2):
    """
    Robust function to fetch historical stock data using multiple fallback data sources:
    1. Stooq API (via pandas_datareader) - Usually has no rate limits.
    2. Yahoo Finance API (via pandas_datareader).
    3. yfinance library directly (with built-in retries).
    4. Local dataset file ('trimed_dataset.xls') as a final fallback if APIs are unavailable.
    Returns a pandas DataFrame containing the stock's Open, High, Low, Close, and Volume data.
    """

    import pandas_datareader as pdr

    import pandas as pd

    try:

        print(f"[Method 1] Fetching {ticker_symbol} from Stooq (no rate limits)...")

        df = pdr.get_data_stooq(ticker_symbol, start=start_date, end=end_date)

        if not df.empty and len(df) > 0:

            df = df.sort_index()

            df.columns = [col.capitalize() for col in df.columns]

            print(f"[OK] Successfully fetched {len(df)} days from Stooq!")

            return df

    except Exception as e:

        print(f"  Stooq failed: {e}")

    try:

        print(f"[Method 2] Trying pandas_datareader with Yahoo...")

        df = pdr.get_data_yahoo(ticker_symbol, start=start_date, end=end_date)

        if not df.empty and len(df) > 0:

            print(f"[OK] Successfully fetched {len(df)} days from Yahoo via pandas_datareader!")

            return df

    except Exception as e:

        print(f"  pandas_datareader Yahoo failed: {e}")

    for attempt in range(max_retries):

        try:

            print(f"[Method 3] Trying yfinance (attempt {attempt + 1}/{max_retries})...")

            if attempt > 0:

                wait_time = 2 ** attempt

                print(f"  Waiting {wait_time} seconds...")

                time.sleep(wait_time)

            ticker = yf.Ticker(ticker_symbol, session=yf_session)

            df = ticker.history(start=start_date, end=end_date)

            if not df.empty and len(df) > 0:

                print(f"[OK] Successfully fetched {len(df)} days from yfinance!")

                return df

            df = yf.download(ticker_symbol, start=start_date, end=end_date, 

                           progress=False, session=yf_session)

            if not df.empty and len(df) > 0:

                print(f"[OK] Successfully downloaded {len(df)} days from yfinance!")

                return df

        except Exception as e:

            print(f"  yfinance attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:

                continue

    try:

        print(f"[Method 4] All API sources failed. Trying local trimed_dataset.xls...")

        local_dataset_path = 'trimed_dataset.xls'

        if os.path.exists(local_dataset_path):

            df_full = pd.read_excel(local_dataset_path)

            df_ticker = df_full[df_full['Ticker'].str.upper() == ticker_symbol.upper()].copy()

            if not df_ticker.empty:

                df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])

                df_ticker.set_index('Date', inplace=True)

                df_ticker.sort_index(inplace=True)

                df_filtered = df_ticker.loc[start_date:end_date]

                if not df_filtered.empty and len(df_filtered) > 0:

                    df_result = df_filtered[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                    print(f"[OK] Successfully loaded {len(df_result)} days from local trimed_dataset.xls!")

                    return df_result

                else:

                    print(f"  No data found for {ticker_symbol} in date range {start_date} to {end_date}")

            else:

                print(f"  Ticker {ticker_symbol} not found in local dataset")

        else:

            print(f"  Local dataset file not found: {local_dataset_path}")

    except Exception as e:

        print(f"  Local dataset fallback failed: {e}")

    print("[ERROR] All data sources (API + local) failed!")

    return None

def add_moving_average_features(df, window_sizes=[5, 10, 20]):
    """
    Calculates Simple Moving Averages (SMA) for the given window sizes (e.g., 5-day, 10-day, 20-day)
    based on the stock's 'Close' price and adds them as new columns to the DataFrame.
    Moving averages are important technical indicators that help smooth out price data to identify trends.
    """

    for window in window_sizes:

        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

    df.dropna(inplace=True)

    return df

def train_evaluate_model(x_train, y_train, x_test, y_test, params, device='cpu', num_epochs=100):
    """
    Initializes, trains, and evaluates an LSTM (Long Short-Term Memory) neural network model
    using the provided hyperparameters (params). It uses Mean Squared Error (MSE) loss and the Adam optimizer.
    It tracks the test loss during training and returns the best test loss achieved along with the best model's state.
    """

    model = LSTMModel(input_dim=x_train.shape[2],

                      hidden_dim=params['hidden_dim'],

                      num_layers=params['num_layers'],

                      output_dim=params['output_dim'],

                      dropout=params['dropout']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    criterion = nn.MSELoss()

    best_loss = np.inf

    best_model_state = None

    for epoch in range(1, num_epochs + 1):

        model.train()

        optimizer.zero_grad()

        outputs = model(x_train.to(device))

        loss = criterion(outputs, y_train.to(device))

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            test_outputs = model(x_test.to(device))

            test_loss = criterion(test_outputs, y_test.to(device))

        if test_loss < best_loss:

            best_loss = test_loss

            best_model_state = model.state_dict()

        if epoch % 10 == 0:

            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

    return best_loss.item(), best_model_state

@app.route('/')

def home():
    """
    Renders the homepage template ('index.html') where the user can input the stock ticker.
    """

    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    """
    Main endpoint for handling stock prediction requests via POST method.
    It performs an end-to-end pipeline:
    1. Fetches historical stock data.
    2. Preprocesses data (adds moving averages, normalizes to [-1, 1], prepares time-series sequences).
    3. Performs hyperparameter tuning using Grid Search to find the best LSTM model configuration.
    4. Evaluates the model and generates future stock price predictions.
    5. Calculates backtesting strategy returns and generates performance plots.
    """

    try:

        tickerSymbol = request.form['ticker'].upper().strip()

        recent_update_date = request.form['date']

        print(f"Using device: {device}")

        print(f"Fetching data for {tickerSymbol} from 2010-01-01 to {recent_update_date}")

        df = fetch_stock_data_with_retry(

            ticker_symbol=tickerSymbol,

            start_date='2010-01-01',

            end_date=recent_update_date,

            max_retries=3

        )

        if df is None or df.empty:

            return (f"Error: No data found for ticker '{tickerSymbol}'. This could be due to:<br>"
                   f"1. Invalid ticker symbol<br>"
                   f"2. Yahoo Finance API rate limiting (try again in a few minutes)<br>"
                   f"3. No data available for the specified date range<br><br>"
                   f"<a href='/'>Try again</a>")

        print(f"Initial data shape: {df.shape}")

        if len(df) < 30:

            return f"Error: Insufficient data for ticker '{tickerSymbol}'. Found only {len(df)} days of data. Need at least 30 days for moving averages and predictions."

        def normalize_data(data):

            if len(data) == 0:

                raise ValueError("No data available for normalization")

            scaler = MinMaxScaler(feature_range=(-1, 1))

            scaled_data = scaler.fit_transform(data)

            return scaler, scaled_data

        def prepare_sequences(data, sequence_length):

            x_data, y_data = [], []

            for i in range(sequence_length, len(data)):

                x_data.append(data[i - sequence_length:i, :-1])

                y_data.append(data[i, -1])

            return np.array(x_data), np.array(y_data)

        df = add_moving_average_features(df)

        print(f"Data shape after moving averages: {df.shape}")

        if df.empty or len(df) == 0:

            return f"Error: Insufficient data after calculating moving averages for '{tickerSymbol}'. Please try a more recent end date or a different stock with more historical data."

        sequence_length = 60

        if len(df) < sequence_length + 20:  

            return f"Error: Insufficient data for predictions. Found {len(df)} days after processing, but need at least {sequence_length + 20} days. Try using a more recent date or a stock with more historical data."

        data = df[['Close', 'MA_5', 'MA_10', 'MA_20']].values

        scaler, scaled_data = normalize_data(data)

        x, y = prepare_sequences(scaled_data, sequence_length)

        if len(x) == 0 or len(y) == 0:

            return f"Error: Unable to create prediction sequences. Need more historical data for '{tickerSymbol}'."

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        x_train = torch.tensor(x_train, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        x_test = torch.tensor(x_test, dtype=torch.float32)

        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_PARAMS_PATH):

            best_params = json.load(open(BEST_PARAMS_PATH))

            model = LSTMModel(input_dim=x_train.shape[2],  

                              hidden_dim=best_params['hidden_dim'],

                              num_layers=best_params['num_layers'],

                              output_dim=best_params['output_dim'],

                              dropout=best_params['dropout']).to(device)

            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

            model.eval()

        else:

            param_grid = {

                'hidden_dim': [64, 128],

                'num_layers': [2, 3],

                'output_dim': [1],

                'dropout': [0.2, 0.3],

                'lr': [0.001, 0.0001]

            }

            best_test_loss = np.inf

            best_params = None

            best_model_state = None

            for params in ParameterGrid(param_grid):

                print(f'Training with parameters: {params}')

                test_loss, current_best_model_state = train_evaluate_model(x_train, y_train, x_test, y_test, params,

                                                                           device=device.__str__())

                if current_best_model_state is not None and (best_model_state is None or test_loss < best_test_loss):

                    best_test_loss = test_loss

                    best_params = params

                    best_model_state = current_best_model_state

            if best_model_state is None:

                return "No valid model state found. Training did not succeed."

            print(f'\\nBest parameters: {best_params}')

            print(f'Best test loss: {best_test_loss:.6f}')

            torch.save(best_model_state, BEST_MODEL_PATH)

            with open(BEST_PARAMS_PATH, 'w') as params_file:

                json.dump(best_params, params_file)

            model = LSTMModel(input_dim=x_train.shape[2],  

                              hidden_dim=best_params['hidden_dim'],

                              num_layers=best_params['num_layers'],

                              output_dim=best_params['output_dim'],

                              dropout=best_params['dropout']).to(device)

            model.load_state_dict(best_model_state)

            model.eval()

        with torch.no_grad():

            test_outputs = model(x_test.to(device))

        max_original = df['Close'].max()

        min_original = df['Close'].min()

        test_outputs_np = test_outputs.cpu().numpy().reshape(-1)

        predictions_rescaled = test_outputs_np * (max_original - min_original) + min_original

        print("Shape of predictions_rescaled:", predictions_rescaled.shape)

        plt.figure(figsize=(12, 6))

        plt.plot(df.index[-len(predictions_rescaled):], df['Close'].values[-len(predictions_rescaled):],

                 label='Actual Prices')

        plt.plot(df.index[-len(predictions_rescaled):], predictions_rescaled, label='Predicted Prices')

        plt.title('Stock Price Prediction')

        plt.xlabel('Time')

        plt.ylabel('Price')

        plt.legend()

        plot_path = 'static/prediction_plot.png'

        plt.savefig(plot_path)

        plt.close()

        signals = np.diff(predictions_rescaled) > 0

        signals = np.insert(signals, 0, False)

        actual = df['Close'].values[-len(predictions_rescaled):]

        rmse = np.sqrt(np.mean((predictions_rescaled - actual) ** 2))

        print(f'Test RMSE: {rmse}')

        daily_returns = np.diff(actual) / actual[:-1]

        daily_returns = np.insert(daily_returns, 0, 0)

        strategy_returns = signals[:-1] * daily_returns[1:]

        cumulative_strategy_returns = np.cumsum(strategy_returns)

        cumulative_stock_returns = np.cumsum(daily_returns)

        plt.figure(figsize=(12, 6))

        plt.plot(cumulative_stock_returns, label='Stock Returns (Buy and Hold)')

        plt.plot(cumulative_strategy_returns, label='Strategy Returns')

        plt.title('Backtesting Stock Price Prediction Strategy')

        plt.xlabel('Time')

        plt.ylabel('Cumulative Returns')

        plt.legend()

        plot_path1 = 'static/backtesting_plot.png'

        plt.savefig(plot_path1)

        plt.close()

        return render_template('prediction_result.html', plot_path=plot_path, plot_path1=plot_path1)

    except Exception as e:

        error_message = f"An error occurred while processing '{tickerSymbol}': {str(e)}"

        print(f"Error: {error_message}")

        import traceback

        traceback.print_exc()

        return f"Error: {error_message}. Please try a different stock ticker or date range."

if __name__ == '__main__':

    app.run(debug=True)

