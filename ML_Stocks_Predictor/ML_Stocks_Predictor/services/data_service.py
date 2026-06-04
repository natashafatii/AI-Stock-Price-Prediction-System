

import os

import io

import time

import pandas as pd

import yfinance as yf

import requests

from requests.adapters import HTTPAdapter

from urllib3.util.retry import Retry

from typing import Optional, Tuple

import pickle

from config import Config

from utils.logger import setup_logger

from utils.indicators import add_all_indicators

from sklearn.preprocessing import MinMaxScaler

logger = setup_logger(__name__)

class DataService:

    def __init__(self):

        self.session = self._setup_session()

        self.scaler = None

    def _setup_session(self) -> requests.Session:

        session = requests.Session()

        session.headers.update({

            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',

            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',

            'Accept-Language': 'en-US,en;q=0.5',

            'Connection': 'keep-alive',

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

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:

        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")

        df = self._fetch_from_yfinance(ticker, start_date, end_date)

        if df is not None:

            return df

        df = self._fetch_from_stooq(ticker, start_date, end_date)

        if df is not None:

            return df

        df = self._fetch_from_local(ticker, start_date, end_date)

        if df is not None:

            return df

        logger.error(f"All data sources failed for {ticker}")

        return None

    def _fetch_from_stooq(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:

        try:

            logger.info(f"[Method 2] Fetching {ticker} from Stooq...")

            url = (

                f"https://stooq.com/q/d/l/?s={ticker}.US&d1="

                f"{start_date.replace('-', '')}&d2={end_date.replace('-', '')}&i=d"

            )

            response = self.session.get(url, timeout=15)

            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))

            if not df.empty and len(df) > 0 and 'Date' in df.columns:

                df['Date'] = pd.to_datetime(df['Date'])

                df.set_index('Date', inplace=True)

                df = df.sort_index()

                df.columns = [col.capitalize() for col in df.columns]

                logger.info(f"[OK] Fetched {len(df)} days from Stooq")

                return df

        except Exception as e:

            logger.warning(f"Stooq failed: {e}")

        return None

    def _fetch_from_yfinance(self, ticker: str, start_date: str, end_date: str, 

                            max_retries: int = 3) -> Optional[pd.DataFrame]:

        for attempt in range(max_retries):

            try:

                logger.info(f"[Method 1] Fetching {ticker} from yfinance (attempt {attempt + 1}/{max_retries})...")

                if attempt > 0:

                    wait_time = 2 ** attempt

                    logger.info(f"Waiting {wait_time} seconds...")

                    time.sleep(wait_time)

                ticker_obj = yf.Ticker(ticker)

                df = ticker_obj.history(start=start_date, end=end_date)

                if df is not None and not df.empty and len(df) > 0:

                    if isinstance(df.columns, pd.MultiIndex):

                        df.columns = df.columns.get_level_values(0)

                    logger.info(f"[OK] Fetched {len(df)} days from yfinance")

                    return df

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if df is not None and not df.empty and len(df) > 0:

                    if isinstance(df.columns, pd.MultiIndex):

                        df.columns = df.columns.get_level_values(0)

                    logger.info(f"[OK] Downloaded {len(df)} days from yfinance")

                    return df

            except Exception as e:

                logger.warning(f"yfinance attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:

                    continue

        return None

    def _fetch_from_local(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:

        try:

            logger.info(f"[Method 4] Fetching {ticker} from local dataset...")

            if not os.path.exists(Config.LOCAL_DATASET_PATH):

                logger.warning(f"Local dataset not found: {Config.LOCAL_DATASET_PATH}")

                return None

            df_full = None

            try:

                df_full = pd.read_excel(Config.LOCAL_DATASET_PATH, engine='openpyxl')

            except Exception as e:

                logger.warning(f"openpyxl engine failed, trying xlrd: {e}")

                try:

                    df_full = pd.read_excel(Config.LOCAL_DATASET_PATH, engine='xlrd')

                except Exception as e_xlrd:

                    logger.error(f"Both openpyxl and xlrd failed to read local dataset: {e_xlrd}")

                    return None

            if df_full is None:

                return None

            symbol_col = None

            if 'Ticker' in df_full.columns:

                symbol_col = 'Ticker'

            elif 'Symbol' in df_full.columns:

                symbol_col = 'Symbol'

            if symbol_col:

                df_ticker = df_full[df_full[symbol_col].str.upper() == ticker.upper()].copy()

            else:

                logger.warning(f"Neither 'Ticker' nor 'Symbol' column found in local dataset.")

                return None

            if not df_ticker.empty:

                if 'Date' in df_ticker.columns:

                    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])

                    df_ticker.set_index('Date', inplace=True)

                    df_ticker.sort_index(inplace=True)

                else:

                    logger.warning(f"No 'Date' column found for {ticker} in local dataset.")

                    return None

                df_filtered = df_ticker.loc[start_date:end_date]

                if not df_filtered.empty and len(df_filtered) > 0:

                    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

                    available_ohlcv = [col for col in ohlcv_cols if col in df_filtered.columns]

                    if len(available_ohlcv) == len(ohlcv_cols):

                        df_result = df_filtered[available_ohlcv].copy()

                        logger.info(f"[OK] Loaded {len(df_result)} days from local dataset")

                        return df_result

                    else:

                        logger.warning(f"Missing OHLCV columns for {ticker} in local dataset. Found: {available_ohlcv}")

                        return None

                else:

                    logger.warning(f"No data for {ticker} in date range {start_date} to {end_date} from local dataset")

            else:

                logger.warning(f"Ticker {ticker} not found in local dataset")

        except Exception as e:

            logger.error(f"Local dataset fallback failed: {e}")

        return None

    def preprocess_data(self, df: pd.DataFrame, add_indicators: bool = True) -> pd.DataFrame:

        logger.info(f"Preprocessing data: {df.shape}")

        if add_indicators:

            df = add_all_indicators(df)

            logger.info(f"After adding indicators: {df.shape}")

        return df

    def normalize_data(self, data: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, MinMaxScaler]:

        if fit or self.scaler is None:

            self.scaler = MinMaxScaler(feature_range=(-1, 1))

            scaled_data = self.scaler.fit_transform(data)

            logger.info("Fitted new scaler")

        else:

            scaled_data = self.scaler.transform(data)

            logger.info("Used existing scaler")

        scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        return scaled_df, self.scaler

    def save_scaler(self, path: str = None):

        if self.scaler is None:

            logger.warning("No scaler to save")

            return

        path = path or Config.SCALER_PATH

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:

            pickle.dump(self.scaler, f)

        logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path: str = None):

        path = path or Config.SCALER_PATH

        if not os.path.exists(path):

            logger.warning(f"Scaler file not found: {path}")

            return

        with open(path, 'rb') as f:

            self.scaler = pickle.load(f)

        logger.info(f"Loaded scaler from {path}")

