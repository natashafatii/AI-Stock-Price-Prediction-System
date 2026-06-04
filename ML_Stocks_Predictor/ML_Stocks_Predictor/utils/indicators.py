

import numpy as np

import pandas as pd

from typing import Dict

def add_moving_averages(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:

    for window in windows:

        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

    return df

def add_exponential_moving_averages(df: pd.DataFrame, windows: list = [12, 26]) -> pd.DataFrame:

    for window in windows:

        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:

    delta = df['Close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()

    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss

    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()

    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    df['MACD'] = ema_fast - ema_slow

    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df

def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:

    df['BB_Middle'] = df['Close'].rolling(window=period).mean()

    std = df['Close'].rolling(window=period).std()

    df['BB_Upper'] = df['BB_Middle'] + (std * std_dev)

    df['BB_Lower'] = df['BB_Middle'] - (std * std_dev)

    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:

    high_low = df['High'] - df['Low']

    high_close = np.abs(df['High'] - df['Close'].shift())

    low_close = np.abs(df['Low'] - df['Close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    df['ATR'] = true_range.rolling(window=period).mean()

    return df

def add_obv(df: pd.DataFrame) -> pd.DataFrame:

    obv = [0]

    for i in range(1, len(df)):

        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:

            obv.append(obv[-1] + df['Volume'].iloc[i])

        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:

            obv.append(obv[-1] - df['Volume'].iloc[i])

        else:

            obv.append(obv[-1])

    df['OBV'] = obv

    return df

def add_stochastic_oscillator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:

    low_min = df['Low'].rolling(window=period).min()

    high_max = df['High'].rolling(window=period).max()

    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)

    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    return df

def add_all_indicators(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:

    if config is None:

        config = {

            'ma_windows': [5, 10, 20],

            'ema_windows': [12, 26],

            'rsi_period': 14,

            'macd': {'fast': 12, 'slow': 26, 'signal': 9},

            'bollinger': {'period': 20, 'std_dev': 2},

            'atr_period': 14

        }

    df = df.copy()

    df = add_moving_averages(df, config.get('ma_windows', [5, 10, 20]))

    df = add_exponential_moving_averages(df, config.get('ema_windows', [12, 26]))

    df = add_rsi(df, config.get('rsi_period', 14))

    macd_config = config.get('macd', {})

    df = add_macd(df, 

                  macd_config.get('fast', 12),

                  macd_config.get('slow', 26),

                  macd_config.get('signal', 9))

    bollinger_config = config.get('bollinger', {})

    df = add_bollinger_bands(df,

                            bollinger_config.get('period', 20),

                            bollinger_config.get('std_dev', 2))

    df = add_atr(df, config.get('atr_period', 14))

    df = add_obv(df)

    df = add_stochastic_oscillator(df)

    df['Daily_Return'] = df['Close'].pct_change()

    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    df.dropna(inplace=True)

    return df

def get_feature_columns() -> list:

    return [

        'Open', 'High', 'Low', 'Close', 'Volume',

        'Daily_Return', 'Volatility',

        'MA_5', 'MA_10', 'MA_20',

        'EMA_12', 'EMA_26',

        'RSI',

        'MACD', 'MACD_Signal', 'MACD_Hist',

        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',

        'ATR',

        'OBV',

        'Stoch_K', 'Stoch_D'

    ]

