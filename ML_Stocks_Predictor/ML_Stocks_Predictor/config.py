

import os

from datetime import datetime

class Config:
    """
    Base Configuration Class for the Stock Price Prediction System.
    This class contains all the essential settings, such as file paths,
    hyperparameters for the machine learning models, and application constants.
    """

    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    DEBUG = True

    APP_NAME = 'Stock Price Prediction System'

    VERSION = '2.0.0'

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

    TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')

    MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

    BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')

    BEST_PARAMS_PATH = os.path.join(MODELS_DIR, 'best_params.json')

    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

    LOG_DIR = os.path.join(BASE_DIR, 'logs')

    LOG_FILE = os.path.join(LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d")}.log')

    LOG_LEVEL = 'INFO'

    LOCAL_DATASET_PATH = os.path.join(BASE_DIR, 'cleaned_stock_price_data.xls')

    # --- Data & Sequence Parameters ---
    START_DATE = '2023-01-01'  # The earliest date to fetch historical stock data from.

    MIN_DATA_POINTS = 150  # Minimum required days of historical data for reliable predictions.

    SEQUENCE_LENGTH = 60  # The number of past consecutive days the model looks at to predict the next day's price.

    # --- Machine Learning Split & Training Hyperparameters ---
    TEST_SIZE = 0.2  # 20% of the dataset is reserved for final model testing/evaluation.

    VALIDATION_SIZE = 0.1  # 10% of the dataset is used during training to monitor and tune performance.

    NUM_EPOCHS = 150  # Maximum number of passes the model will make over the training dataset.

    BATCH_SIZE = 32  # Number of data sequences processed together in one training step.

    EARLY_STOPPING_PATIENCE = 20  # Training will stop early if the model doesn't improve for 20 consecutive epochs to prevent overfitting.

    LEARNING_RATE = 0.001  # Controls how quickly the model updates its weights during learning.

    LR_SCHEDULER_PATIENCE = 7  

    LR_SCHEDULER_FACTOR = 0.5  

    MIN_LEARNING_RATE = 1e-6  

    WEIGHT_DECAY = 1e-4  

    # --- Technical Indicator Settings (Feature Engineering) ---
    MOVING_AVERAGE_WINDOWS = [5, 10, 20]  # Calculate 5-day, 10-day, and 20-day Simple Moving Averages to smooth price trends.

    RSI_PERIOD = 14  # The timeframe (14 days) used to calculate the Relative Strength Index (momentum indicator).

    RSI_OVERBOUGHT = 70  # Threshold indicating a stock might be overvalued (sell signal).

    RSI_OVERSOLD = 30  # Threshold indicating a stock might be undervalued (buy signal).

    MACD_FAST = 12

    MACD_SLOW = 26

    MACD_SIGNAL = 9

    BOLLINGER_PERIOD = 20

    BOLLINGER_STD = 2

    ATR_PERIOD = 14

    # --- API Request & Hardware Settings ---
    MAX_RETRIES = 3  # Maximum number of times to retry fetching data if the API fails.

    RETRY_DELAY = 2  # Seconds to wait between retry attempts.

    REQUEST_TIMEOUT = 30  # Maximum time (in seconds) to wait for an API response.

    DEVICE = 'cuda' if os.environ.get('USE_CUDA') == 'true' else 'cpu'  # Uses GPU (cuda) for faster training if available, else CPU.

    @classmethod
    def init_app(cls):
        """
        Initializes the application environment by creating necessary directories
        (like models, logs, and static folders) if they do not already exist.
        This ensures the app has the required file structure to run correctly.
        """

        os.makedirs(cls.MODELS_DIR, exist_ok=True)

        os.makedirs(cls.LOG_DIR, exist_ok=True)

        os.makedirs(cls.STATIC_FOLDER, exist_ok=True)

        print(f"[OK] {cls.APP_NAME} v{cls.VERSION} initialized")

        print(f"[OK] Device: {cls.DEVICE}")

        print(f"[OK] Data range: {cls.START_DATE} to present")

class DevelopmentConfig(Config):
    """
    Development configuration with debugging enabled.
    Used when testing and developing the app locally.
    """

    DEBUG = True

    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """
    Production configuration with debugging disabled for better security and performance.
    Used when deploying the application to a live server.
    """

    DEBUG = False

    LOG_LEVEL = 'WARNING'

    SECRET_KEY = os.environ.get('SECRET_KEY')  

config = {

    'development': DevelopmentConfig,

    'production': ProductionConfig,

    'default': DevelopmentConfig

}

def get_config(env='default'):
    """
    Returns the appropriate configuration object based on the given environment name
    (e.g., 'development', 'production'). Defaults to development.
    """

    return config.get(env, config['default'])

