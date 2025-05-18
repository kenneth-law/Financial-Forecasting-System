# Copyright 2025 Kenneth Law
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ASX Financial Data Model Training with Optimized LSTM

This script demonstrates how to stream data from a PostgreSQL database and train
an LSTM model for financial time series prediction optimized for RTX 3080 GPUs.
Includes a web-based UI for training and monitoring.

Long short-term memory (LSTM) is a type of recurrent neural network (RNN) aimed at 
mitigating the vanishing gradient problem commonly encountered by traditional RNNs. 
Its relative insensitivity to gap length is its advantage over other RNNs, hidden
Markov models, and other sequence learning methods.

https://huggingface.co/docs/transformers/en/index

Author: Kenneth Law
Date: 2025-03-11

DISCLAIMER:
This software is provided for educational and informational purposes only.
The author(s) are not registered investment advisors and do not provide financial advice.
This software does not guarantee accuracy of data and should not be the sole basis for any investment decision.
Users run and use this software at their own risk. The author(s) accept no liability for any loss or damage 
resulting from its use. Always consult with a qualified financial professional before making investment decisions.
"""

import os
import time
import json
import torch
import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify, send_from_directory
from joblib import dump, load
import yfinance as yf  # Added for supplementary data
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # For sentiment analysis
import requests
from bs4 import BeautifulSoup
import re

# Database connection details remain unchanged
DB_USER = "asx_user"
DB_PASSWORD = "asx200"
DB_NAME = "asx_financials"
DB_HOST = "localhost"

# Enhanced model parameters
SEQUENCE_LENGTH = 30  # Increased from 20
PREDICTION_HORIZON = 5
BATCH_SIZE = 256
LEARNING_RATE = 3e-5
NUM_EPOCHS = 40  # Increased from 20
HIDDEN_SIZE = 384  # Increased from 256
NUM_LAYERS = 3  # Increased from 2
DROPOUT = 0.3  # Increased from 0.2
MODEL_OUTPUT_DIR = "models/asx_price_predictor"

# Create model output directory if it doesn't exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Flask app for UI
app = Flask(__name__, 
            template_folder='templates_model',  
            static_folder='static_model')

# Global variables for tracking training progress
training_progress = {
    'current_epoch': 0,
    'total_epochs': NUM_EPOCHS,
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_price_mae': [],  # Added for price prediction metrics
    'best_accuracy': 0.0,
    'best_price_mae': float('inf'),  # Added for price prediction metrics
    'is_training': False,
    'ticker': '',
    'start_date': '',
    'end_date': '',
    'status': 'Ready'
}

# Memory tracking remains the same
def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e18
        reserved = torch.cuda.memory_reserved() / 1e18
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")
        return {'allocated': allocated, 'reserved': reserved}
    return {'allocated': 0, 'reserved': 0}

# Enhanced LSTM Model Definition - Now with dual outputs for classification and regression
class EnhancedFinancialLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Added bidirectional for better feature extraction
        )
        self.dropout = nn.Dropout(dropout)
        
        # Combined hidden size from bidirectional LSTM
        combined_hidden = hidden_size * 2
        
        # Classification head
        self.fc_class = nn.Sequential(
            nn.Linear(combined_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)  # Binary classification
        )
        
        # Regression head for price prediction
        self.fc_price = nn.Sequential(
            nn.Linear(combined_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Price regression
        )
        
        # Auxiliary heads for additional tasks (helps with feature learning)
        self.fc_volatility = nn.Sequential(
            nn.Linear(combined_hidden, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)  # Volatility prediction
        )
        
    def forward(self, x):
        # Optimize memory layout
        self.lstm.flatten_parameters()
        
        # x shape: [batch_size, seq_length, input_size]
        lstm_out, _ = self.lstm(x)
        
        # Take output from last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Apply dropout to prevent overfitting
        feature_vector = self.dropout(last_time_step)
        
        # Get class probabilities and predicted price
        class_output = self.fc_class(feature_vector)
        price_output = self.fc_price(feature_vector)
        volatility_output = self.fc_volatility(feature_vector)
        
        return class_output, price_output, volatility_output

# Database connection function remains unchanged
def connect_to_db():
    """Establish connection to PostgreSQL database"""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST
    )
    return conn

def numpy_to_python_types(obj):
    """Convert numpy types to standard Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [numpy_to_python_types(i) for i in obj]
    elif hasattr(obj, 'item'):  # NumPy scalars have .item() method
        return obj.item()  # Converts to Python scalar
    else:
        return obj

# Sentiment analysis function - NEW
def get_news_sentiment(ticker, lookback_days=7):
    """
    Retrieve and analyze sentiment from financial news articles
    
    Args:
        ticker (str): Stock ticker symbol
        lookback_days (int): Number of days to look back for news
        
    Returns:
        dict: Sentiment scores and article counts
    """
    try:
        # Initialize sentiment model
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Search for financial news - this would normally use a real API
        # For demonstration, return simulated data
        sentiment_scores = {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'article_count': 0
        }
        
        # In a real implementation, you would:
        # 1. Get news articles about the stock
        # 2. Process each article with FinBERT
        # 3. Aggregate sentiment scores
        
        # Simulated sentiment analysis
        simulated_article_count = np.random.randint(5, 30)
        sentiment_scores['article_count'] = simulated_article_count
        
        # Random but reasonable sentiment distribution
        pos = np.random.uniform(0.3, 0.6)
        neg = np.random.uniform(0.1, 0.4)
        neut = 1.0 - pos - neg
        
        sentiment_scores['positive'] = pos
        sentiment_scores['negative'] = neg
        sentiment_scores['neutral'] = neut
        
        return sentiment_scores
    except Exception as e:
        print(f"Error fetching news sentiment: {e}")
        # Return neutral sentiment if analysis fails
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'article_count': 0}

# Enhanced feature creation function
def create_time_series_features(df, market_df=None, ticker=None, include_sentiment=True):
    """
    Create enhanced financial time series features from price data
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_df (pandas.DataFrame, optional): Market index data for relative metrics
        ticker (str, optional): Stock ticker for sentiment analysis
        include_sentiment (bool): Whether to include sentiment features
        
    Returns:
        pandas.DataFrame: DataFrame with additional features
    """
    # Make a copy to avoid modifying the original dataframe
    df_features = df.copy()
    
    # Technical indicators - Price based
    df_features['daily_return'] = df_features['close_price'].pct_change()
    df_features['log_return'] = np.log(df_features['close_price'] / df_features['close_price'].shift(1))
    
    # Enhanced moving averages
    for period in [5, 10, 20, 50, 100]:
        df_features[f'ma{period}'] = df_features['close_price'].rolling(window=period).mean()
        df_features[f'ma{period}_ratio'] = df_features['close_price'] / df_features[f'ma{period}']
    
    # Exponential moving averages (more weight to recent prices)
    for period in [5, 10, 20, 50]:
        df_features[f'ema{period}'] = df_features['close_price'].ewm(span=period, adjust=False).mean()
        df_features[f'ema{period}_ratio'] = df_features['close_price'] / df_features[f'ema{period}']
    
    # MACD (Moving Average Convergence Divergence)
    df_features['macd'] = df_features['ema12'] - df_features['ema26'] if 'ema12' in df_features.columns and 'ema26' in df_features.columns else np.nan
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean() if 'macd' in df_features.columns else np.nan
    df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal'] if 'macd' in df_features.columns and 'macd_signal' in df_features.columns else np.nan
    
    # Enhanced volatility measures
    for period in [5, 10, 20, 50]:
        df_features[f'volatility_{period}d'] = df_features['daily_return'].rolling(window=period).std()
        # Add normalized volatility (helps with scale invariance)
        df_features[f'volatility_{period}d_norm'] = df_features[f'volatility_{period}d'] / df_features['close_price']
    
    # Calculate price momentum for multiple periods
    for period in [3, 5, 10, 20, 50]:
        df_features[f'momentum_{period}d'] = df_features['close_price'].pct_change(periods=period)
    
    # Bollinger Bands (measure of price volatility)
    for period in [20]:  # Standard period for Bollinger Bands
        middle_band = df_features['close_price'].rolling(window=period).mean()
        std_dev = df_features['close_price'].rolling(window=period).std()
        df_features[f'bollinger_upper_{period}'] = middle_band + (std_dev * 2)
        df_features[f'bollinger_lower_{period}'] = middle_band - (std_dev * 2)
        df_features[f'bollinger_width_{period}'] = (df_features[f'bollinger_upper_{period}'] - df_features[f'bollinger_lower_{period}']) / middle_band
        # Bollinger %B indicator (where price is relative to the bands)
        df_features[f'bollinger_b_{period}'] = (df_features['close_price'] - df_features[f'bollinger_lower_{period}']) / (df_features[f'bollinger_upper_{period}'] - df_features[f'bollinger_lower_{period}'])
    
    # RSI (Relative Strength Index) - momentum oscillator
    delta = df_features['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Enhanced volume features
    df_features['volume_change'] = df_features['volume'].pct_change()
    df_features['volume_change_norm'] = df_features['volume_change'] / df_features['volume'].rolling(window=20).mean()
    for period in [5, 10, 20, 50]:
        df_features[f'volume_ma{period}'] = df_features['volume'].rolling(window=period).mean()
        df_features[f'volume_ratio_{period}'] = df_features['volume'] / df_features[f'volume_ma{period}']
    
    # Volume price trend
    df_features['vpt'] = (df_features['volume'] * df_features['daily_return']).cumsum()
    
    # OBV (On Balance Volume)
    df_features['obv_change'] = np.where(df_features['close_price'] > df_features['close_price'].shift(1), 
                                         df_features['volume'], 
                                         np.where(df_features['close_price'] < df_features['close_price'].shift(1), 
                                                 -df_features['volume'], 0))
    df_features['obv'] = df_features['obv_change'].cumsum()
    
    # Price ratios and ranges
    df_features['hl_ratio'] = df_features['high_price'] / df_features['low_price']
    df_features['cl_ratio'] = df_features['close_price'] / df_features['low_price']
    df_features['ho_ratio'] = df_features['high_price'] / df_features['open_price']
    df_features['co_ratio'] = df_features['close_price'] / df_features['open_price']
    df_features['daily_range'] = (df_features['high_price'] - df_features['low_price']) / df_features['open_price']
    
    # Gap features
    df_features['gap_raw'] = df_features['open_price'] - df_features['close_price'].shift(1)
    df_features['gap'] = df_features['gap_raw'] / df_features['close_price'].shift(1)
    
    # Add market relative features if market data is provided
    if market_df is not None:
        # Merge market data based on date index
        market_returns = market_df['close_price'].pct_change().rename('market_return')
        df_features = pd.merge(df_features, market_returns, left_index=True, right_index=True, how='left')
        
        # Calculate relative strength
        df_features['rel_strength'] = df_features['daily_return'] - df_features['market_return']
        
        # Relative strength over multiple periods
        for period in [5, 10, 20]:
            df_features[f'rel_strength_{period}d'] = df_features[f'momentum_{period}d'] - df_features['market_return'].rolling(window=period).sum()
            
        # Beta calculation (measure of volatility compared to the market)
        # Need sufficient data points for regression
        if len(df_features) > 50:
            rolling_cov = df_features['daily_return'].rolling(window=50).cov(df_features['market_return'])
            rolling_var = df_features['market_return'].rolling(window=50).var()
            df_features['beta_50d'] = rolling_cov / rolling_var
    
    # Add sentiment features
    if include_sentiment and ticker is not None:
        # In real implementation, this would analyze actual news
        # For demonstration, we'll use simulated sentiment that shifts over time
        
        # Create date range index for sentiment
        date_range = pd.date_range(start=df_features.index.min(), end=df_features.index.max())
        sentiment_df = pd.DataFrame(index=date_range)
        
        # Create synthetic sentiment that changes over time with some autocorrelation
        n = len(date_range)
        
        # Create slightly autocorrelated sentiment (using AR process)
        # This mimics how sentiment tends to persist for periods of time
        ar_param = 0.92  # Autocorrelation parameter
        white_noise = np.random.normal(0, 0.15, n)
        sentiment_series = np.zeros(n)
        
        sentiment_series[0] = white_noise[0]
        for t in range(1, n):
            sentiment_series[t] = ar_param * sentiment_series[t-1] + white_noise[t]
        
        # Scale to a reasonable range for sentiment scores [0,1]
        sentiment_series = (sentiment_series - sentiment_series.min()) / (sentiment_series.max() - sentiment_series.min())
        
        # Create positive and negative sentiment
        sentiment_df['positive_sentiment'] = 0.5 + sentiment_series * 0.3
        sentiment_df['negative_sentiment'] = 0.5 - sentiment_series * 0.3
        sentiment_df['neutral_sentiment'] = 1 - (sentiment_df['positive_sentiment'] + sentiment_df['negative_sentiment'])
        
        # Add moving averages of sentiment
        for period in [3, 7, 14]:
            sentiment_df[f'positive_sentiment_ma{period}'] = sentiment_df['positive_sentiment'].rolling(window=period).mean()
            sentiment_df[f'negative_sentiment_ma{period}'] = sentiment_df['negative_sentiment'].rolling(window=period).mean()
        
        # Add sentiment momentum (change in sentiment)
        sentiment_df['sentiment_momentum'] = sentiment_df['positive_sentiment'].diff(periods=3)
        
        # Merge sentiment with main features
        df_features = df_features.join(sentiment_df, how='left')
        
        # In case dates don't exactly align, forward fill sentiment values then backfill
        sentiment_cols = [col for col in df_features.columns if 'sentiment' in col]
        df_features[sentiment_cols] = df_features[sentiment_cols].ffill().bfill()
    
    # Fill NaN values that result from calculations
    df_features = df_features.fillna(0)

    # Handle infinite values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(0)

    # Outlier handling: clip extreme values to improve model training stability
    for col in df_features.columns:
        if df_features[col].dtype in [np.float64, np.float32]:
            q1 = df_features[col].quantile(0.005)  # More conservative clip
            q3 = df_features[col].quantile(0.995)
            df_features[col] = df_features[col].clip(q1, q3)
    
    return df_features

def prepare_sequences(df, seq_length=SEQUENCE_LENGTH, pred_horizon=PREDICTION_HORIZON, ticker=None):
    """
    Convert time series data into sequences for model training with both classification and regression targets
    
    Args:
        df (pandas.DataFrame): DataFrame with price and feature data
        seq_length (int): Length of input sequence
        pred_horizon (int): How many days ahead to predict
        ticker (str, optional): Stock ticker for saving scaler
        
    Returns:
        tuple: (X, y_class, y_price) where X is input sequences, y_class is direction target, y_price is price target
    """
    # Drop columns that might cause data leakage (future information)
    # We need to be careful about forward-looking features
    exclude_columns = ['date', 'ticker']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Use RobustScaler instead of MinMaxScaler for better outlier handling
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df[feature_columns]).astype(np.float32)
    
    # Save scaler for later use with new data (using joblib)
    if ticker is not None:  # Only save if ticker is provided
        if not os.path.exists(os.path.join(MODEL_OUTPUT_DIR, 'scalers')):
            os.makedirs(os.path.join(MODEL_OUTPUT_DIR, 'scalers'))
        dump(scaler, os.path.join(MODEL_OUTPUT_DIR, 'scalers', f"{ticker}_scaler.joblib"))
        # Also save feature columns for reference
        np.save(os.path.join(MODEL_OUTPUT_DIR, 'scalers', f"{ticker}_features.npy"), feature_columns)
    
    # Create sequences
    X = []
    y_class = []  # For direction classification
    y_price = []  # For price regression
    y_vol = []    # For volatility prediction
    
    # Calculate future returns for classification
    future_returns = df['close_price'].pct_change(periods=pred_horizon).shift(-pred_horizon)
    
    # Get future prices for regression target
    future_prices = df['close_price'].shift(-pred_horizon)
    current_prices = df['close_price']
    
    # Calculate future volatility
    future_volatility = df['close_price'].rolling(window=pred_horizon).std().shift(-pred_horizon)
    
    for i in range(len(df) - seq_length - pred_horizon + 1):
        X.append(scaled_features[i:i+seq_length])
        
        # Binary classification: 1 if price goes up, 0 if it goes down
        y_class.append(1 if future_returns.iloc[i+seq_length-1] > 0 else 0)
        
        # Regression target: normalized future price (% change from current)
        current_price = current_prices.iloc[i+seq_length-1]
        future_price = future_prices.iloc[i+seq_length-1]
        price_change_pct = (future_price - current_price) / current_price
        y_price.append(price_change_pct)
        
        # Volatility target
        volatility = future_volatility.iloc[i+seq_length-1]
        normalized_vol = volatility / current_price
        y_vol.append(normalized_vol)
    
    return (np.array(X, dtype=np.float32), 
            np.array(y_class, dtype=np.int64),
            np.array(y_price, dtype=np.float32),
            np.array(y_vol, dtype=np.float32))

# Data streaming function remains largely unchanged
def stream_stock_data(ticker, start_date=None, end_date=None, batch_size=1000):
    """
    Stream stock price data from database in batches
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        batch_size (int): Number of rows to fetch per batch
        
    Yields:
        pandas.DataFrame: Batch of stock data
    """
    conn = connect_to_db()
    
    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        # Use more historical data - 4 years instead of 2
        start_date = (datetime.now() - timedelta(days=365*4)).strftime('%Y-%m-%d')
    
    # Create a server-side cursor for efficient streaming
    cursor = conn.cursor(name='stock_data_cursor')
    
    # Query data in chronological order
    query = """
    SELECT date, open_price, high_price, low_price, close_price, volume
    FROM stock_prices
    WHERE ticker = %s AND date BETWEEN %s AND %s
    ORDER BY date
    """
    
    cursor.execute(query, (ticker, start_date, end_date))
    
    # Fetch and yield batches
    columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    
    while True:
        records = cursor.fetchmany(batch_size)
        if not records:
            break
            
        df_batch = pd.DataFrame(records, columns=columns)
        df_batch['date'] = pd.to_datetime(df_batch['date'])
        df_batch = df_batch.set_index('date')
        
        yield df_batch
    
    # Clean up
    cursor.close()
    conn.close()

# The get_market_data function remains largely unchanged
def get_market_data(market_ticker="^AXJO", start_date=None, end_date=None):
    """
    Get market index data for relative comparison
    
    Args:
        market_ticker (str): Market index ticker (ASX 200 = ^AXJO)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pandas.DataFrame: Market index data
    """
    conn = connect_to_db()
    
    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        # Extend lookback period
        start_date = (datetime.now() - timedelta(days=365*4)).strftime('%Y-%m-%d')
    
    cursor = conn.cursor()
    
    # Query market data
    query = """
    SELECT date, close_price
    FROM stock_prices
    WHERE ticker = %s AND date BETWEEN %s AND %s
    ORDER BY date
    """
    
    cursor.execute(query, (market_ticker, start_date, end_date))
    records = cursor.fetchall()
    
    # Create DataFrame
    market_df = pd.DataFrame(records, columns=['date', 'close_price'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df = market_df.set_index('date')
    
    cursor.close()
    conn.close()
    
    return market_df

# Enhanced training function with multi-task learning
def train_model(ticker, start_date=None, end_date=None):
    """
    Train enhanced LSTM model with mixed precision training, multi-task learning and GPU optimizations
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        dict: Training results and metrics
    """
    global training_progress
    
    # Update training status
    training_progress['is_training'] = True
    training_progress['ticker'] = ticker
    training_progress['start_date'] = start_date if start_date else "4 years ago"
    training_progress['end_date'] = end_date if end_date else "today"
    training_progress['status'] = "Loading data..."
    training_progress['train_loss'] = []
    training_progress['val_loss'] = []
    training_progress['val_accuracy'] = []
    training_progress['val_price_mae'] = []
    training_progress['current_epoch'] = 0
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        initial_mem = print_gpu_memory()
    
    try:
        # Stream and collect stock data
        training_progress['status'] = f"Streaming data for {ticker}..."
        all_data = pd.DataFrame()
        for batch in stream_stock_data(ticker, start_date, end_date):
            all_data = pd.concat([all_data, batch])
        
        # Get market data for relative comparisons
        try:
            market_data = get_market_data("^AXJO", start_date, end_date)
        except Exception as e:
            print(f"Warning: Could not retrieve market data: {e}")
            market_data = None
        
        print(f"Collected {len(all_data)} rows of data for {ticker}.")
        training_progress['status'] = "Creating enhanced features..."
        
        # Create features with sentiment analysis
        all_data = create_time_series_features(all_data, market_data, ticker=ticker, include_sentiment=True)
        
        # Drop the first few rows with NaN values from calculations
        all_data = all_data.dropna()
        
        # Prepare sequences with multiple targets
        training_progress['status'] = "Preparing sequences with multi-task targets..."
        X, y_class, y_price, y_vol = prepare_sequences(all_data, ticker=ticker)

        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Split data - time ordered split with validation set
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        test_size = len(X) - train_size - val_size
        
        X_train, y_class_train, y_price_train, y_vol_train = (
            X[:train_size], 
            y_class[:train_size], 
            y_price[:train_size],
            y_vol[:train_size]
        )
        
        X_val, y_class_val, y_price_val, y_vol_val = (
            X[train_size:train_size+val_size], 
            y_class[train_size:train_size+val_size], 
            y_price[train_size:train_size+val_size],
            y_vol[train_size:train_size+val_size]
        )
        
        X_test, y_class_test, y_price_test, y_vol_test = (
            X[train_size+val_size:], 
            y_class[train_size+val_size:], 
            y_price[train_size+val_size:],
            y_vol[train_size+val_size:]
        )
        
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_class_train_tensor = torch.tensor(y_class_train, dtype=torch.long).to(device)
        y_price_train_tensor = torch.tensor(y_price_train, dtype=torch.float32).view(-1, 1).to(device)
        y_vol_train_tensor = torch.tensor(y_vol_train, dtype=torch.float32).view(-1, 1).to(device)
        
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_class_val_tensor = torch.tensor(y_class_val, dtype=torch.long).to(device)
        y_price_val_tensor = torch.tensor(y_price_val, dtype=torch.float32).view(-1, 1).to(device)
        y_vol_val_tensor = torch.tensor(y_vol_val, dtype=torch.float32).view(-1, 1).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_class_train_tensor, y_price_train_tensor, y_vol_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_class_val_tensor, y_price_val_tensor, y_vol_val_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            pin_memory=False,  # Already on GPU
            num_workers=0      # No need for workers as data is already loaded
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            pin_memory=False,
            num_workers=0
        )
        
        # Initialize model
        input_size = X_train.shape[2]  # Number of features
        model = EnhancedFinancialLSTM(input_size=input_size).to(device)
        
        # Loss functions
        classification_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.SmoothL1Loss()  # More robust to outliers than MSE
        
        # Optimizer with weight decay to prevent overfitting
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # For mixed precision training
        scaler = GradScaler()
        
        # Training loop
        best_accuracy = 0.0
        best_mae = float('inf')
        training_progress['status'] = "Training multi-task model..."
        
        # For early stopping
        early_stop_patience = 10
        early_stop_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            training_progress['current_epoch'] = epoch + 1
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_class_loss = 0.0
            train_price_loss = 0.0
            train_vol_loss = 0.0
            
            # Progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
            
            for inputs, class_targets, price_targets, vol_targets in train_pbar:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with autocast():
                    class_outputs, price_outputs, vol_outputs = model(inputs)
                    
                    # Multi-task loss with weighting
                    # At first, prioritize classification to get directional prediction right
                    # As training progresses, shift focus to price prediction
                    epoch_progress = min(epoch / (NUM_EPOCHS * 0.5), 1.0)  # Progress ratio (0->1) over first half
                    
                    class_weight = 1.0 - 0.3 * epoch_progress  # Starts at 1.0, decreases to 0.7
                    price_weight = 0.5 + 0.5 * epoch_progress   # Starts at 0.5, increases to 1.0
                    vol_weight = 0.3 + 0.1 * epoch_progress     # Starts at 0.3, increases to 0.4
                    
                    c_loss = classification_criterion(class_outputs, class_targets)
                    p_loss = regression_criterion(price_outputs, price_targets)
                    v_loss = regression_criterion(vol_outputs, vol_targets)
                    
                    # Combine losses
                    loss = class_weight * c_loss + price_weight * p_loss + vol_weight * v_loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Track losses
                train_loss += loss.item() * inputs.size(0)
                train_class_loss += c_loss.item() * inputs.size(0)
                train_price_loss += p_loss.item() * inputs.size(0)
                train_vol_loss += v_loss.item() * inputs.size(0)
                
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'class_loss': c_loss.item(),
                    'price_loss': p_loss.item()
                })
            
            # Calculate average training loss
            train_loss = train_loss / len(train_loader.dataset)
            train_class_loss = train_class_loss / len(train_loader.dataset)
            train_price_loss = train_price_loss / len(train_loader.dataset)
            
            training_progress['train_loss'].append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_class_loss = 0.0
            val_price_loss = 0.0
            correct = 0
            total = 0
            price_abs_errors = []
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            
            with torch.no_grad():
                for inputs, class_targets, price_targets, vol_targets in val_pbar:
                    # Mixed precision inference
                    with autocast():
                        class_outputs, price_outputs, vol_outputs = model(inputs)
                        
                        # Calculate losses
                        c_loss = classification_criterion(class_outputs, class_targets)
                        p_loss = regression_criterion(price_outputs, price_targets)
                        v_loss = regression_criterion(vol_outputs, vol_targets)
                        
                        # Combined loss (same weights as training for consistency)
                        loss = class_weight * c_loss + price_weight * p_loss + vol_weight * v_loss
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_class_loss += c_loss.item() * inputs.size(0)
                    val_price_loss += p_loss.item() * inputs.size(0)
                    
                    # Calculate classification accuracy
                    _, predicted = torch.max(class_outputs, 1)
                    total += class_targets.size(0)
                    correct += (predicted == class_targets).sum().item()
                    
                    # Calculate price prediction MAE
                    price_abs_error = torch.abs(price_outputs - price_targets)
                    price_abs_errors.extend(price_abs_error.cpu().numpy())
                    
                    val_pbar.set_postfix({
                        'loss': loss.item(),
                        'class_acc': correct/total if total > 0 else 0
                    })
            
            # Calculate validation metrics
            val_loss = val_loss / len(val_loader.dataset)
            val_class_loss = val_class_loss / len(val_loader.dataset)
            val_price_loss = val_price_loss / len(val_loader.dataset)
            accuracy = correct / total if total > 0 else 0
            price_mae = np.mean(price_abs_errors)
            
            training_progress['val_loss'].append(val_loss)
            training_progress['val_accuracy'].append(accuracy)
            training_progress['val_price_mae'].append(price_mae)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Price: {train_price_loss:.4f}), "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, "
                  f"Price MAE: {price_mae:.6f}")
            
            # Check if this is the best model
            is_best = False
            
            # We consider both accuracy and price MAE in determining the best model
            # with a weighted average of both metrics
            current_score = 0.6 * accuracy - 0.4 * price_mae  # Higher is better
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                training_progress['best_accuracy'] = best_accuracy
                is_best = True
            
            if price_mae < best_mae:
                best_mae = price_mae
                is_best = True
            
            if is_best:
                # Save the best model
                model_path = os.path.join(MODEL_OUTPUT_DIR, f"{ticker}_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'price_mae': price_mae,
                    'input_size': input_size,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'sequence_length': SEQUENCE_LENGTH,
                    'prediction_horizon': PREDICTION_HORIZON,
                }, model_path)
                
                print(f"Saved new best model with accuracy: {accuracy:.4f}, Price MAE: {price_mae:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Step the scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Optional: Save memory snapshot
            if torch.cuda.is_available() and epoch % 5 == 0:
                print_gpu_memory()
        
        # Evaluate on test set for final metrics
        model.eval()
        test_correct = 0
        test_total = 0
        test_price_errors = []
        
        # Convert test data to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_class_test_tensor = torch.tensor(y_class_test, dtype=torch.long).to(device)
        y_price_test_tensor = torch.tensor(y_price_test, dtype=torch.float32).view(-1, 1).to(device)
        
        test_dataset = TensorDataset(X_test_tensor, y_class_test_tensor, y_price_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for inputs, class_targets, price_targets in test_loader:
                # Forward pass
                class_outputs, price_outputs, _ = model(inputs)
                
                # Classification accuracy
                _, predicted = torch.max(class_outputs, 1)
                test_total += class_targets.size(0)
                test_correct += (predicted == class_targets).sum().item()
                
                # Price prediction errors
                price_error = torch.abs(price_outputs - price_targets)
                test_price_errors.extend(price_error.cpu().numpy())
        
        test_accuracy = test_correct / test_total
        test_price_mae = np.mean(test_price_errors)
        
        # Final update
        training_progress['status'] = "Training complete"
        training_progress['is_training'] = False
        
        # Return summary metrics
        return {
            'ticker': ticker,
            'num_data_points': len(all_data),
            'num_sequences': len(X),
            'final_train_accuracy': accuracy,
            'final_test_accuracy': test_accuracy,
            'best_validation_accuracy': best_accuracy,
            'test_price_mae': test_price_mae,
            'train_loss': training_progress['train_loss'],
            'val_loss': training_progress['val_loss'],
            'val_accuracy': training_progress['val_accuracy'],
            'val_price_mae': training_progress['val_price_mae']
        }
    
    except Exception as e:
        print(f"Error during training: {e}")
        training_progress['status'] = f"Error: {str(e)}"
        training_progress['is_training'] = False
        return {'error': str(e)}

# Enhanced prediction function with price forecasting
def predict(ticker, days=5):
    """
    Make prediction for a stock using the trained model, with confidence metrics and price forecast
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to look ahead for prediction
        
    Returns:
        dict: Prediction results including price forecast and confidence
    """
    try:
        # First, check if a model exists for this ticker
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"{ticker}_model.pth")
        if not os.path.exists(model_path):
            return {'error': f"No trained model found for {ticker}"}
        
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model with same architecture
        input_size = checkpoint['input_size']
        model = EnhancedFinancialLSTM(
            input_size=input_size,
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get the most recent data - extend lookback to ensure quality
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        # Get market data for relative metrics
        try:
            market_data = get_market_data("^AXJO", start_date, end_date)
        except Exception:
            market_data = None
            
        # Try to get sentiment data
        try:
            sentiment_data = get_news_sentiment(ticker)
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            sentiment_data = None
        
        # Fetch recent data
        all_data = pd.DataFrame()
        for batch in stream_stock_data(ticker, start_date, end_date):
            all_data = pd.concat([all_data, batch])
        
        if len(all_data) < SEQUENCE_LENGTH + days:
            return {'error': f"Not enough data for {ticker}. Need at least {SEQUENCE_LENGTH} days."}
        
        # Create features
        all_data = create_time_series_features(all_data, market_data, ticker)
        all_data = all_data.dropna()
        
        # Load scaler using joblib
        scaler_path = os.path.join(MODEL_OUTPUT_DIR, 'scalers', f"{ticker}_scaler.joblib")
        features_path = os.path.join(MODEL_OUTPUT_DIR, 'scalers', f"{ticker}_features.npy")
        
        if not os.path.exists(scaler_path) or not os.path.exists(features_path):
            return {'error': "Feature scaler not found. Please retrain the model."}
        
        # Load scaler and feature columns
        scaler = load(scaler_path)
        feature_columns = np.load(features_path, allow_pickle=True)
        
        # Extract features for prediction
        feature_columns_available = [col for col in feature_columns if col in all_data.columns]
        if len(feature_columns_available) < len(feature_columns):
            print(f"Warning: Missing {len(feature_columns) - len(feature_columns_available)} features")
        
        df_features = all_data[feature_columns_available]
        
        # Add missing columns with zeros if necessary
        for col in feature_columns:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Reorder columns to match training data
        df_features = df_features[feature_columns]
        
        # Scale features using loaded scaler
        scaled_features = scaler.transform(df_features).astype(np.float32)
        
        # Create sequence for the most recent data point
        sequence = scaled_features[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)
        
        # Make prediction with mixed precision
        with torch.no_grad(), autocast():
            class_output, price_output, vol_output = model(sequence_tensor)
            probabilities = torch.softmax(class_output, dim=1)
            _, prediction = torch.max(class_output, 1)
        
        # Get prediction and probability
        prediction = prediction.item()  # 0 or 1
        prob_up = probabilities[0, 1].item()  # Probability of going up
        prob_down = probabilities[0, 0].item()  # Probability of going down
        
        # Get the current price
        current_price = all_data['close_price'].iloc[-1]
        
        # Calculate predicted price change percentage and actual price
        predicted_change_pct = price_output.item()
        predicted_price = current_price * (1 + predicted_change_pct)
        
        # Calculate predicted volatility
        predicted_volatility = vol_output.item() * current_price
        
        # Recent historical volatility
        historical_volatility = all_data['close_price'].iloc[-20:].pct_change().std() * np.sqrt(252)  # Annualized
        
        # Calculate confidence score (0-100) based on:
        # 1. How far probability is from 0.5 (uncertainty)
        # 2. Consistency of recent predictions
        # 3. Model historical accuracy on similar market conditions
        probability_confidence = abs(prob_up - 0.5) * 2  # 0 if 50/50, 1 if 100% sure
        
        # Analyze recent market condition similarity (basic implementation)
        recent_volatility = all_data['volatility_10d'].iloc[-1] if 'volatility_10d' in all_data.columns else 0.01
        volatility_confidence = max(0, 1 - recent_volatility * 10)  # Lower confidence in high volatility
        
        # Calculate sentiment confidence factor if available
        sentiment_confidence = 0.5  # Neutral default
        if sentiment_data:
            if prediction == 1:  # UP prediction
                sentiment_confidence = sentiment_data['positive']
            else:  # DOWN prediction
                sentiment_confidence = sentiment_data['negative']
        
        # Combined confidence score (0-100%)
        # Weighting: 50% probability, 30% volatility, 20% sentiment
        confidence_score = (0.5 * probability_confidence + 
                           0.3 * volatility_confidence + 
                           0.2 * sentiment_confidence) * 100
                           
        # Apply sigmoid to smooth out extreme values and center around 85% max confidence
        # 1 / (1 + e^-(x-0.5)/0.15) scales from ~15% to ~85%
        confidence_score = 100 * (1 / (1 + np.exp(-(confidence_score/100 - 0.5) / 0.15)))
        
        # For safety, ensure confidence is at least 50% if probability is very high
        if abs(prob_up - 0.5) > 0.4:  # Very strong signal
            confidence_score = max(confidence_score, 75)
        
        # Return prediction results with enhanced information
        result = {
            'ticker': ticker,
            'prediction': "UP" if prediction == 1 else "DOWN",
            'probability': prob_up if prediction == 1 else prob_down,
            'confidence': round(confidence_score, 1),  # Rounded confidence score
            'prediction_horizon': days,
            'current_price': current_price,
            'predicted_price': round(predicted_price, 2),
            'predicted_change_pct': round(predicted_change_pct * 100, 2),
            'predicted_volatility': round(predicted_volatility, 4),
            'historical_volatility': round(historical_volatility, 4),
            'prediction_date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d'),
            'as_of_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': str(e)}

# Create a sample template directory and files if they don't exist
def create_ui_files():
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates_model'):
        os.makedirs('templates_model')
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static_model'):
        os.makedirs('static_model')
    
    # Create index.html if it doesn't exist
    if not os.path.exists('templates_model/index.html'):
        with open('templates_model/index.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>ASX Financial Forecasting System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --danger: #e74c3c;
            --warning: #f39c12;
            --light: #ecf0f1;
            --dark: #2c3e50;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            background-color: #f8f9fa;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary);
            color: white;
            padding: 20px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        header h1 {
            margin-left: 20px;
        }
        
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .disclaimer {
            font-size: 0.8rem;
            padding: 10px;
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            margin-bottom: 20px;
        }
        
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .card-header {
            background-color: var(--light);
            padding: 15px 20px;
            border-bottom: 1px solid #e9ecef;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .card-body {
            padding: 20px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        
        button {
            background-color: var(--secondary);
            color: white;
            border: none;
            padding: 12px 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;"
background-color: var(--secondary);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.2s;
        }
        
        button:hover {
            background-color: #2980b9;
        }
        
        button:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .chart {
            width: 100%;
            height: 400px;
        }
        
        .progress-container {
            margin-top: 20px;
        }
        
        .progress-bar {
            height: 25px;
            background-color: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-bar-fill {
            height: 100%;
            background-color: var(--secondary);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .status {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .status-icon {
            margin-right: 10px;
        }
        
        .status-icon.loading {
            color: var(--warning);
        }
        
        .status-icon.success {
            color: var(--success);
        }
        
        .status-icon.error {
            color: var(--danger);
        }
        
        .prediction-result {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            transition: all 0.3s;
        }
        
        .prediction-up {
            background-color: rgba(39, 174, 96, 0.2);
            border: 1px solid #27ae60;
        }
        
        .prediction-down {
            background-color: rgba(231, 76, 60, 0.2);
            border: 1px solid #e74c3c;
        }
        
        .prediction-arrow {
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        .prediction-arrow.up {
            color: #27ae60;
        }
        
        .prediction-arrow.down {
            color: #e74c3c;
        }
        
        .prediction-text {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .prediction-probability {
            font-size: 18px;
            margin-bottom: 10px;
        }
        
        .prediction-details {
            font-size: 14px;
            color: #7f8c8d;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
        }
        
        .tab.active {
            border-bottom: 3px solid var(--secondary);
            font-weight: bold;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container header-container">
            <h1>ASX Financial Forecasting System</h1>
        </div>
    </header>
    
    <div class="container">
        <div class="disclaimer">
            <strong>DISCLAIMER:</strong> This software is provided for educational and informational purposes only. 
            The author(s) are not registered investment advisors and do not provide financial advice.
            This software does not guarantee accuracy of data and should not be the sole basis for any investment decision.
        </div>
        
        <div class="grid">
            <div>
                <div class="card">
                    <div class="card-header">
                        <span>Train Model</span>
                    </div>
                    <div class="card-body">
                        <div class="form-group">
                            <label for="ticker">Stock Ticker:</label>
                            <input type="text" id="ticker" placeholder="Enter ticker symbol (e.g., CBA.AX)">
                        </div>
                        <div class="form-group">
                            <label for="start-date">Start Date:</label>
                            <input type="date" id="start-date">
                        </div>
                        <div class="form-group">
                            <label for="end-date">End Date:</label>
                            <input type="date" id="end-date">
                        </div>
                        <button id="train-btn" onclick="startTraining()">
                            <i class="fas fa-play"></i> Start Training
                        </button>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span>Training Status</span>
                    </div>
                    <div class="card-body">
                        <div class="status">
                            <div class="status-icon" id="status-icon">
                                <i class="fas fa-info-circle"></i>
                            </div>
                            <div id="status-text">Ready to train</div>
                        </div>
                        
                        <div class="progress-container">
                            <div class="progress-bar">
                                <div class="progress-bar-fill" id="progress-bar" style="width: 0%">0%</div>
                            </div>
                            <div id="epoch-text">Epoch: 0/20</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span>Make Prediction</span>
                    </div>
                    <div class="card-body">
                        <div class="form-group">
                            <label for="pred-ticker">Stock Ticker:</label>
                            <input type="text" id="pred-ticker" placeholder="Enter ticker symbol (e.g., CBA.AX)">
                        </div>
                        <div class="form-group">
                            <label for="prediction-days">Prediction Horizon (Days):</label>
                            <select id="prediction-days">
                                <option value="1">1 day</option>
                                <option value="3">3 days</option>
                                <option value="5" selected>5 days</option>
                                <option value="10">10 days</option>
                            </select>
                        </div>
                        <button id="predict-btn" onclick="makePrediction()">
                            <i class="fas fa-search"></i> Make Prediction
                        </button>
                        
                        <div id="prediction-result" style="display: none;" class="prediction-result">
                            <div class="prediction-arrow" id="prediction-arrow">
                                <i class="fas fa-arrow-up"></i>
                            </div>
                            <div class="prediction-text" id="prediction-text">UP</div>
                            <div class="prediction-probability" id="prediction-probability">Probability: 75%</div>
                            <div class="prediction-details" id="prediction-details">
                                Prediction for 5 days ahead (until 2025-03-16)
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div>
                <div class="card">
                    <div class="card-header">
                        <span>Training Metrics</span>
                        <div class="tabs">
                            <div class="tab active" onclick="changeTab(event, 'loss-chart')">Loss</div>
                            <div class="tab" onclick="changeTab(event, 'accuracy-chart')">Accuracy</div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="loss-chart" class="chart tab-content active"></div>
                        <div id="accuracy-chart" class="chart tab-content"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span>GPU Memory Usage</span>
                    </div>
                    <div class="card-body">
                        <div id="memory-chart" class="chart"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        // Initialize charts
        const lossChart = Plotly.newPlot('loss-chart', 
            [{
                name: 'Training Loss',
                type: 'scatter',
                mode: 'lines+markers',
                x: [],
                y: [],
                line: {color: '#3498db'}
            },
            {
                name: 'Validation Loss',
                type: 'scatter',
                mode: 'lines+markers',
                x: [],
                y: [],
                line: {color: '#e74c3c'}
            }], 
            {
                title: 'Loss Over Epochs',
                xaxis: {title: 'Epoch'},
                yaxis: {title: 'Loss'},
                margin: {l: 50, r: 50, b: 50, t: 50},
                legend: {orientation: 'h', y: 1.1}
            }
        );
        
        const accuracyChart = Plotly.newPlot('accuracy-chart', 
            [{
                name: 'Validation Accuracy',
                type: 'scatter',
                mode: 'lines+markers',
                x: [],
                y: [],
                line: {color: '#27ae60'}
            }], 
            {
                title: 'Accuracy Over Epochs',
                xaxis: {title: 'Epoch'},
                yaxis: {title: 'Accuracy', range: [0, 1]},
                margin: {l: 50, r: 50, b: 50, t: 50}
            }
        );
        
        const memoryChart = Plotly.newPlot('memory-chart', 
            [{
                name: 'Allocated',
                type: 'scatter',
                mode: 'lines',
                x: [],
                y: [],
                line: {color: '#3498db'}
            },
            {
                name: 'Reserved',
                type: 'scatter',
                mode: 'lines',
                x: [],
                y: [],
                line: {color: '#f39c12'}
            }], 
            {
                title: 'GPU Memory Usage (GB)',
                xaxis: {title: 'Time'},
                yaxis: {title: 'Memory (GB)'},
                margin: {l: 50, r: 50, b: 50, t: 50},
                legend: {orientation: 'h', y: 1.1}
            }
        );
        
        // Training status polling
        let statusInterval;
        
        function startTraining() {
            const ticker = document.getElementById('ticker').value;
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            
            if (!ticker) {
                alert('Please enter a ticker symbol');
                return;
            }
            
            document.getElementById('train-btn').disabled = true;
            document.getElementById('status-icon').innerHTML = '<i class="fas fa-spinner fa-spin status-icon loading"></i>';
            document.getElementById('status-text').textContent = 'Starting training...';
            
            // Reset charts
            Plotly.update('loss-chart', {x: [[]], y: [[]]}, {}, [0]);
            Plotly.update('loss-chart', {x: [[]], y: [[]]}, {}, [1]);
            Plotly.update('accuracy-chart', {x: [[]], y: [[]]}, {}, [0]);
            
            // Start training process
            fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ticker, start_date: startDate, end_date: endDate})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    statusInterval = setInterval(checkStatus, 1000);
                } else {
                    document.getElementById('status-icon').innerHTML = '<i class="fas fa-times-circle status-icon error"></i>';
                    document.getElementById('status-text').textContent = 'Error: ' + data.error;
                    document.getElementById('train-btn').disabled = false;
                }
            })
            .catch(error => {
                document.getElementById('status-icon').innerHTML = '<i class="fas fa-times-circle status-icon error"></i>';
                document.getElementById('status-text').textContent = 'Error: ' + error;
                document.getElementById('train-btn').disabled = false;
            });
        }
        
        function checkStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                // Update status text
                document.getElementById('status-text').textContent = data.status;
                
                // Update progress bar
                const progress = (data.current_epoch / data.total_epochs) * 100;
                document.getElementById('progress-bar').style.width = progress + '%';
                document.getElementById('progress-bar').textContent = Math.round(progress) + '%';
                document.getElementById('epoch-text').textContent = `Epoch: ${data.current_epoch}/${data.total_epochs}`;
                
                // Update charts if there's new data
                if (data.train_loss && data.train_loss.length > 0) {
                    const epochs = Array.from({length: data.train_loss.length}, (_, i) => i + 1);
                    
                    // Force full redraw of charts instead of just updating them
                    Plotly.react('loss-chart', [
                        {
                            name: 'Training Loss',
                            type: 'scatter',
                            mode: 'lines+markers',
                            x: epochs,
                            y: data.train_loss,
                            line: {color: '#3498db'}
                        },
                        {
                            name: 'Validation Loss',
                            type: 'scatter',
                            mode: 'lines+markers',
                            x: epochs,
                            y: data.val_loss || [],
                            line: {color: '#e74c3c'}
                        }
                    ], {
                        title: 'Loss Over Epochs',
                        xaxis: {title: 'Epoch'},
                        yaxis: {title: 'Loss'},
                        margin: {l: 50, r: 50, b: 50, t: 50},
                        legend: {orientation: 'h', y: 1.1}
                    });
                    
                    if (data.val_accuracy && data.val_accuracy.length > 0) {
                        Plotly.react('accuracy-chart', [
                            {
                                name: 'Validation Accuracy',
                                type: 'scatter',
                                mode: 'lines+markers',
                                x: epochs,
                                y: data.val_accuracy,
                                line: {color: '#27ae60'}
                            }
                        ], {
                            title: 'Accuracy Over Epochs',
                            xaxis: {title: 'Epoch'},
                            yaxis: {title: 'Accuracy', range: [0, 1]},
                            margin: {l: 50, r: 50, b: 50, t: 50}
                        });
                    }
                }
                
                // Check if training is complete
                if (!data.is_training && data.current_epoch > 0) {
                    clearInterval(statusInterval);
                    document.getElementById('status-icon').innerHTML = '<i class="fas fa-check-circle status-icon success"></i>';
                    document.getElementById('train-btn').disabled = false;
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
            });
        }
        
        function makePrediction() {
            const ticker = document.getElementById('pred-ticker').value;
            const days = document.getElementById('prediction-days').value;
            
            if (!ticker) {
                alert('Please enter a ticker symbol');
                return;
            }
            
            document.getElementById('predict-btn').disabled = true;
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ticker, days})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('predict-btn').disabled = false;
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Update prediction display
                const predResult = document.getElementById('prediction-result');
                predResult.style.display = 'block';
                
                if (data.prediction === 'UP') {
                    predResult.className = 'prediction-result prediction-up';
                    document.getElementById('prediction-arrow').innerHTML = '<i class="fas fa-arrow-up prediction-arrow up"></i>';
                } else {
                    predResult.className = 'prediction-result prediction-down';
                    document.getElementById('prediction-arrow').innerHTML = '<i class="fas fa-arrow-down prediction-arrow down"></i>';
                }
                
                // Enhanced prediction display with more details
                document.getElementById('prediction-text').textContent = data.prediction;
                document.getElementById('prediction-probability').textContent = 
                    `Probability: ${Math.round(data.probability * 100)}% (Confidence: ${data.confidence || 'N/A'}%)`;
                
                let detailsText = `Prediction for ${data.prediction_horizon} days ahead (until ${data.prediction_date})`;
                
                // Add price prediction if available
                if (data.predicted_price) {
                    detailsText += `<br>Current price: $${data.current_price.toFixed(2)}`;
                    detailsText += `<br>Predicted price: $${data.predicted_price.toFixed(2)} (${data.predicted_change_pct > 0 ? '+' : ''}${data.predicted_change_pct.toFixed(2)}%)`;
                }
                
                document.getElementById('prediction-details').innerHTML = detailsText;
            })
        
        function changeTab(event, tabId) {
            // Hide all tab contents
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }
            
            // Deactivate all tabs
            const tabs = document.getElementsByClassName('tab');
            for (let i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            
            // Activate the selected tab and content
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }
        
        // Set default dates
        const today = new Date();
        const twoYearsAgo = new Date();
        twoYearsAgo.setFullYear(today.getFullYear() - 2);
        
        document.getElementById('start-date').valueAsDate = twoYearsAgo;
        document.getElementById('end-date').valueAsDate = today;
    </script>
</body>
</html>""")
    
    return

# Flask routes
@app.route('/')
def index():
    """Render the home page"""
    create_ui_files()
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def api_train():
    data = request.json
    ticker = data.get('ticker', '')
    start_date = data.get('start_date', None)
    end_date = data.get('end_date', None)
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'})
    
    # Start training in a background thread
    import threading
    thread = threading.Thread(target=train_model, args=(ticker, start_date, end_date))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Training started', 'ticker': ticker})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    ticker = data.get('ticker', '')
    days = int(data.get('days', 5))
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'})
    
    result = predict(ticker, days)
    return jsonify(result)

@app.route('/api/status')
def api_status():
    return jsonify(numpy_to_python_types(training_progress))


@app.route('/train', methods=['POST'])
def train_route():
    """Start the training process"""
    data = request.json
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    if not ticker:
        return jsonify({'status': 'error', 'error': 'Ticker symbol is required'})
    
    # Start training in a separate thread
    import threading
    training_thread = threading.Thread(
        target=train_model, 
        args=(ticker, start_date, end_date)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/status')
def status_route():
    """Get the current training status"""
    return jsonify(numpy_to_python_types(training_progress))

@app.route('/predict', methods=['POST'])
def predict_route():
    """Make a prediction using the trained model"""
    data = request.json
    ticker = data.get('ticker')
    days = int(data.get('days', 5))
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'})
    
    result = predict(ticker, days)
    return jsonify(result)

@app.route('/models')
def list_models():
    """List all trained models"""
    models = []
    if os.path.exists(MODEL_OUTPUT_DIR):
        for file in os.listdir(MODEL_OUTPUT_DIR):
            if file.endswith('_model.pth'):
                ticker = file.replace('_model.pth', '')
                models.append(ticker)
    
    return jsonify({'models': models})

if __name__ == "__main__":
    # Create template files
    create_ui_files()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print_gpu_memory()
    else:
        print("No GPU detected. Training will use CPU (this may be slow).")
    
    print("Starting web server on http://127.0.0.1:5001/")
    app.run(host='0.0.0.0', port=5001, debug=True)