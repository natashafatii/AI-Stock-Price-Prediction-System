`   # 📈 AI Stock Price Prediction System  > Ensemble Deep Learning (LSTM + GRU + Transformer) for Stock Market Forecasting  ![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)  ![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)  ![License](https://img.shields.io/badge/License-MIT-yellow.svg)  ---  ## 🚀 Overview  AI-powered stock price prediction system that uses an ensemble of three deep learning models:  - LSTM (Long Short-Term Memory)  - GRU (Gated Recurrent Unit)  - Transformer (Self-Attention Network)  The system predicts stock closing prices using historical stock data and 24 technical indicators. A Flask-based web interface allows users to enter a stock ticker and receive predictions, visualizations, performance metrics, and backtesting results.  ---  ## ✨ Features  - 🔮 Predict stock closing prices  - 🧠 Ensemble learning (LSTM + GRU + Transformer)  - 📊 24 technical indicators  - 🌐 Flask web application  - 📈 Backtesting engine  - 📉 Professional visualizations  - 💾 Multi-source stock data fetching  - 📊 RMSE, MAE, and MAPE evaluation metrics  - 📌 Confidence intervals for predictions  ---  ## 🛠 Tech Stack  | Category | Technology |  |-----------|------------|  | Language | Python |  | Web Framework | Flask |  | Deep Learning | PyTorch |  | Data Processing | Pandas, NumPy |  | Machine Learning | Scikit-Learn |  | Visualization | Matplotlib |  | Stock Data | yfinance, Stooq |  | Excel Support | openpyxl |  ---  ## 🏗 System Architecture  ```text  User Input       │       ▼  Flask Web Application       │       ▼  Data Service  (yfinance / Stooq / Local Dataset)       │       ▼  Feature Engineering  (24 Technical Indicators)       │       ▼  Data Scaling & Sequence Creation       │       ▼  LSTM + GRU + Transformer       │       ▼  Weighted Ensemble       │       ▼  Predictions + Metrics + Charts   `

📊 Technical Indicators
-----------------------

The model uses 24 features including:

### Raw Features

*   Open
    
*   High
    
*   Low
    
*   Close
    
*   Volume
    

### Trend Indicators

*   MA\_5
    
*   MA\_10
    
*   MA\_20
    
*   EMA\_12
    
*   EMA\_26
    

### Momentum Indicators

*   RSI
    
*   MACD
    
*   MACD Signal
    
*   MACD Histogram
    

### Volatility Indicators

*   Bollinger Bands
    
*   ATR
    

### Volume Indicators

*   OBV
    

### Oscillators

*   Stochastic K
    
*   Stochastic D
    

### Return Features

*   Daily Return
    
*   Volatility
    

🤖 Models Used
--------------

### LSTM

*   Learns long-term patterns in stock prices
    
*   Uses Forget, Input, and Output gates
    
*   Effective for time-series forecasting
    

### GRU

*   Simplified version of LSTM
    
*   Faster training
    
*   Fewer parameters
    
*   Lower overfitting risk
    

### Transformer

*   Uses self-attention mechanism
    
*   Captures long-range dependencies
    
*   Processes sequences in parallel
    

🧠 Ensemble Learning
--------------------

Predictions from all three models are combined using weighted averaging.

Formula:

`   Final Prediction =  (LSTM × Weight)  +  (GRU × Weight)  +  (Transformer × Weight)   `

Weights are calculated automatically based on validation performance.

📈 Data Pipeline
----------------

1.  Fetch stock data
    
2.  Generate technical indicators
    
3.  Remove missing values
    
4.  Normalize data using MinMaxScaler
    
5.  Create 60-day sequences
    
6.  Split into Train / Validation / Test
    
7.  Train all models
    
8.  Create ensemble
    
9.  Generate predictions
    
10.  Evaluate performance
    
11.  Create charts
    
12.  Display results
    

⚙ Training Configuration
------------------------

ParameterValueSequence Length60Hidden Dimension128Layers2Batch Size32Learning Rate0.001Epochs150Dropout0.3Early Stopping20Weight Decay1e-4

📊 Evaluation Metrics
---------------------

### RMSE

Root Mean Squared Error

### MAE

Mean Absolute Error

### MAPE

Mean Absolute Percentage Error

### MAPE Rating

MAPERating< 10%🟢 Excellent10–20%🟡 Good20–30%🟠 Acceptable> 30%🔴 Needs Improvement

📉 Backtesting
--------------

The project includes a simple trading strategy.

### AI Strategy

Buy when predicted price is increasing.

### Benchmark

Buy and Hold strategy.

Comparison charts are generated to evaluate performance.

📥 Installation
---------------

### Clone Repository

`   git clone https://github.com/natashafatii/AI-Stock-Price-Prediction-System.git  cd AI-Stock-Price-Prediction-System   `

### Create Virtual Environment

`   python -m venv venv   `

### Activate Environment

Windows:

`   venv\Scripts\activate   `

Linux / Mac:

`   source venv/bin/activate   `

### Install Dependencies

`   pip install -r requirements.txt   `

### Run Application

`   python main.py   `

### Open Browser

`   http://localhost:5000   `

🎮 Usage
--------

1.  Enter stock ticker (AAPL, TSLA, AMZN, etc.)
    
2.  Select end date
    
3.  Click Predict
    
4.  View:
    
    *   Predicted Prices
        
    *   RMSE
        
    *   MAE
        
    *   MAPE
        
    *   Prediction Charts
        
    *   Backtesting Results
        

🔮 Future Improvements
----------------------

*   Sentiment Analysis using News
    
*   Social Media Integration
    
*   Multi-Step Forecasting
    
*   GPU Support
    
*   Transfer Learning
    
*   Real-Time Streaming Data
    
*   Hyperparameter Optimization
    
*   Advanced Trading Strategies
    

🙏 Acknowledgments
------------------

*   Yahoo Finance
    
*   yfinance
    
*   PyTorch
    
*   Flask
    
*   Scikit-Learn
    
*   Pandas
    
*   NumPy
    
*   Matplotlib
    

📄 License
----------

This project is licensed under the MIT License.

⭐ Support
---------

If you found this project useful, consider giving it a ⭐ on GitHub.

⚠ Disclaimer
------------

This project is developed for educational and research purposes only.

Stock market predictions are inherently uncertain and should not be used as financial advice or real trading recommendations.


