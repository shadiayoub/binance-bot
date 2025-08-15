#!/usr/bin/env python3
"""
Improve ML Model Accuracy
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from binance import AsyncClient
from prediction_models import PricePredictor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_extended_historical_data(symbol: str, interval: str, days: int = 30):
    """Fetch more historical data for better training"""
    client = await AsyncClient.create()
    
    try:
        # Fetch more data (30 days of 1-minute data = ~43,200 klines)
        klines = await client.get_historical_klines(
            symbol, interval, 
            str(datetime.now() - timedelta(days=days)),
            limit=5000  # Maximum allowed
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        logger.info(f"Fetched {len(df)} historical klines for {symbol} ({days} days)")
        return df
        
    finally:
        await client.close_connection()

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced features for better prediction"""
    df = df.copy()
    
    # Ensure numeric data types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['price_change_2'] = df['close'].pct_change(2)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)
    
    # Volatility features
    df['volatility_5'] = df['close'].rolling(5).std()
    df['volatility_10'] = df['close'].rolling(10).std()
    df['volatility_20'] = df['close'].rolling(20).std()
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    df['volume_std'] = df['volume'].rolling(20).std()
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # Price relative to moving averages
    df['price_vs_sma5'] = df['close'] / df['sma_5'] - 1
    df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
    df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
    df['ema_cross_9_21'] = df['ema_9'] - df['ema_21']
    df['ema_cross_21_50'] = df['ema_21'] - df['ema_50']
    
    # Bollinger Bands
    df['bb_upper'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
    df['bb_lower'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
    
    # RSI with multiple periods
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_histogram_change'] = df['macd_histogram'].diff()
    
    # Stochastic
    df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                    (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Williams %R
    df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / 
                       (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
    
    # Commodity Channel Index (CCI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Money Flow Index
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    
    mfi_ratio = positive_flow / (negative_flow + 1e-8)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
    # Time-based features
    df['hour'] = pd.to_datetime(df['open_time'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['open_time'], unit='ms').dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_london_open'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['is_ny_open'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    
    # Target variable (next period's price change)
    df['target'] = df['close'].shift(-1) / df['close'] - 1
    
    # Remove NaN values
    df = df.dropna()
    
    return df

def train_improved_model(df: pd.DataFrame, model_type: str = "random_forest") -> dict:
    """Train an improved model with advanced features"""
    
    # Create advanced features
    df_features = create_advanced_features(df)
    
    # Select features for training
    feature_columns = [
        # Price changes
        'price_change', 'price_change_2', 'price_change_5', 'price_change_10', 'price_change_20',
        
        # Volatility
        'volatility_5', 'volatility_10', 'volatility_20',
        
        # Volume
        'volume_change', 'volume_ratio', 'volume_std',
        
        # Moving averages
        'price_vs_sma5', 'price_vs_sma20', 'price_vs_sma50',
        'ema_cross_9_21', 'ema_cross_21_50',
        
        # Bollinger Bands
        'bb_position', 'bb_width',
        
        # RSI
        'rsi_7', 'rsi_14', 'rsi_21',
        
        # MACD
        'macd', 'macd_histogram', 'macd_histogram_change',
        
        # Stochastic
        'stoch_k', 'stoch_d',
        
        # ATR
        'atr_ratio',
        
        # Williams %R
        'williams_r',
        
        # CCI
        'cci',
        
        # MFI
        'mfi',
        
        # Time features
        'hour', 'is_weekend', 'is_london_open', 'is_ny_open'
    ]
    
    X = df_features[feature_columns]
    y = df_features['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # No shuffle for time series
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with optimized parameters
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Calculate directional accuracy
    train_direction_acc = accuracy_score(
        (y_train > 0), (y_pred_train > 0)
    )
    test_direction_acc = accuracy_score(
        (y_test > 0), (y_pred_test > 0)
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_rmse': cv_rmse,
        'train_direction_accuracy': train_direction_acc,
        'test_direction_accuracy': test_direction_acc,
        'feature_columns': feature_columns,
        'scaler': scaler,
        'model': model
    }
    
    return metrics

async def main():
    """Main function to improve model accuracy"""
    logger.info("🚀 Starting model improvement process...")
    
    # Fetch more historical data
    df = await fetch_extended_historical_data('BTCUSDT', '1m', days=30)
    
    if len(df) < 1000:
        logger.error("Insufficient data for training")
        return
    
    logger.info(f"📊 Training with {len(df)} data points")
    
    # Train improved models
    models = {}
    results = {}
    
    for model_type in ["random_forest", "gradient_boosting"]:
        logger.info(f"🔄 Training {model_type}...")
        
        try:
            metrics = train_improved_model(df, model_type)
            models[model_type] = metrics
            results[model_type] = metrics
            
            logger.info(f"✅ {model_type} training completed:")
            logger.info(f"   Test RMSE: {metrics['test_rmse']:.6f}")
            logger.info(f"   CV RMSE: {metrics['cv_rmse']:.6f}")
            logger.info(f"   Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"❌ {model_type} training failed: {e}")
    
    # Find best model
    best_model = None
    best_accuracy = 0
    
    for model_type, metrics in results.items():
        if metrics['test_direction_accuracy'] > best_accuracy:
            best_accuracy = metrics['test_direction_accuracy']
            best_model = model_type
    
    if best_model:
        logger.info(f"🏆 Best model: {best_model} (Accuracy: {best_accuracy:.2%})")
        
        # Save improved model
        model_data = {
            'model': results[best_model]['model'],
            'scaler': results[best_model]['scaler'],
            'feature_columns': results[best_model]['feature_columns'],
            'model_type': best_model,
            'trained_at': datetime.now(),
            'metrics': results[best_model]
        }
        
        model_path = f"models/BTCUSDT_improved_{best_model}.joblib"
        joblib.dump(model_data, model_path)
        logger.info(f"💾 Improved model saved to: {model_path}")
        
        # Compare with original
        logger.info("\n📈 Model Comparison:")
        logger.info("=" * 50)
        for model_type, metrics in results.items():
            logger.info(f"{model_type.upper()}:")
            logger.info(f"  Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")
            logger.info(f"  Test RMSE: {metrics['test_rmse']:.6f}")
            logger.info(f"  CV RMSE: {metrics['cv_rmse']:.6f}")
            logger.info("")

if __name__ == "__main__":
    asyncio.run(main())
