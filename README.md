# 📈 AI Stock Price Prediction System

> **Ensemble Deep Learning (LSTM + GRU + Transformer) for Stock Market Forecasting**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Technical Indicators](#technical-indicators)
- [Models Used](#models-used)
- [Ensemble Learning](#ensemble-learning)
- [Data Pipeline](#data-pipeline)
- [Training Configuration](#training-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Backtesting](#backtesting)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## 🚀 Overview

AI-powered stock price prediction system that uses an ensemble of three deep learning models:

| Model | Type | Strength |
|-------|------|----------|
| **LSTM** | Long Short-Term Memory | Learns long-term patterns in stock prices |
| **GRU** | Gated Recurrent Unit | Faster training, fewer parameters |
| **Transformer** | Self-Attention Network | Captures long-range dependencies |

The system predicts stock closing prices using historical stock data and 24 technical indicators. A Flask-based web interface allows users to enter a stock ticker and receive predictions, visualizations, performance metrics, and backtesting results.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Stock Prediction** | Predicts closing price for any public stock |
| 🧠 **Ensemble Learning** | Combines 3 neural networks (LSTM + GRU + Transformer) |
| 📊 **24 Technical Indicators** | RSI, MACD, Bollinger Bands, ATR, OBV, Moving Averages |
| 🌐 **Web Interface** | Flask-based UI with interactive forms |
| 📈 **Backtesting Engine** | Compare AI strategy vs Buy & Hold |
| 📉 **Professional Visualizations** | 4-panel charts (Price, RSI, MACD, Volume) |
| 💾 **Multi-Source Data Fetching** | Yahoo Finance → Stooq → Local Dataset |
| 📊 **Performance Metrics** | RMSE, MAE, MAPE with color-coded ratings |
| 📌 **Confidence Intervals** | Uncertainty bands around predictions |

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python |
| **Web Framework** | Flask |
| **Deep Learning** | PyTorch |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Visualization** | Matplotlib |
| **Stock Data** | yfinance, Stooq |
| **Excel Support** | openpyxl |

---

## 🏗 System Architecture ```
User Input
│
▼
Flask Web Application
│
▼
Data Service
(yfinance / Stooq / Local Dataset)
│
▼
Feature Engineering
(24 Technical Indicators)
│
▼
Data Scaling & Sequence Creation
│
▼
LSTM + GRU + Transformer
│
▼
Weighted Ensemble
│
▼
Predictions + Metrics + Charts
```


---

## 📈 Data Pipeline

| Step | Description |
|------|-------------|
| 1 | Fetch stock data (yfinance → Stooq → Local) |
| 2 | Generate 24 technical indicators |
| 3 | Remove missing values (drop NaN rows) |
| 4 | Normalize data using MinMaxScaler (-1 to 1) |
| 5 | Create 60-day sliding window sequences |
| 6 | Split into Train (70%) / Validation (10%) / Test (20%) |
| 7 | Train LSTM, GRU, and Transformer models |
| 8 | Create weighted ensemble |
| 9 | Generate predictions on test data |
| 10 | Calculate RMSE, MAE, MAPE metrics |
| 11 | Generate prediction charts and backtesting charts |
| 12 | Display results on web page |

---

## ⚙ Training Configuration

| Parameter | Value |
|-----------|-------|
| Sequence Length | 60 days |
| Hidden Dimension | 128 |
| Number of Layers | 2 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Max Epochs | 150 |
| Dropout | 0.3 |
| Early Stopping Patience | 20 epochs |
| LR Scheduler Patience | 7 epochs |
| LR Scheduler Factor | 0.5 |
| Weight Decay (L2) | 1e-4 |
| Gradient Clipping | Max norm 1.0 |

---

## 📊 Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MSE** | (1/n) × Σ(pred - actual)² | Mean Squared Error |
| **RMSE** | √MSE | Root Mean Squared Error (in dollars) |
| **MAE** | (1/n) × Σ\|pred - actual\| | Mean Absolute Error (in dollars) |
| **MAPE** | (100/n) × Σ\|actual-pred\|/actual | Mean Absolute Percentage Error (%) |

### MAPE Rating Scale

| MAPE Range | Rating |
|------------|--------|
| < 10% | 🟢 Excellent |
| 10-20% | 🟡 Good |
| 20-30% | 🟠 Acceptable |
| > 30% | 🔴 Needs Improvement |

### Sample Results (Amazon - AMZN)

| Metric | Value | Rating |
|--------|-------|--------|
| MAPE | 1.64% | 🟢 Excellent |
| RMSE | $5.02 | Very Good |
| MAE | $3.69 | Very Good |

---

## 📉 Backtesting

The project includes a simple trading strategy to evaluate model performance.

### AI Strategy
- Generate signal: Buy when predicted price is increasing
- Signal = `predicted_next_day > predicted_today`
- Calculate returns: Daily return × Signal (only earn when in market)

### Benchmark
- Buy and Hold strategy (buy once at beginning, hold through entire period)

### Comparison
- Cumulative returns: AI Strategy vs Buy & Hold
- Chart shows which strategy performed better

### Limitations
- No transaction costs
- No slippage
- Assumes perfect execution
- Theoretical upper bound only

---

## 📥 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone Repository

```bash
git clone https://github.com/natashafatii/AI-Stock-Price-Prediction-System.git
cd AI-Stock-Price-Prediction-System
```

### Create Virtual Environment
```
`   python -m venv venv   `
```
### Activate Environment

Windows:
```
`   venv\Scripts\activate   `
```
Linux / Mac:
```
`   source venv/bin/activate   `
```
### Install Dependencies
```
`   pip install -r requirements.txt   `
```
### Run Application
```
`   python main.py   `
```
### Open Browser
```
`   http://localhost:5000   `
```
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


