#!/usr/bin/env python3
"""
Train ML Model for Price Prediction
"""

import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from binance import AsyncClient
from prediction_models import PricePredictor
from enhanced_strategy import EnhancedStrategy
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_historical_data(symbol: str, interval: str, limit: int = 1000):
    """Fetch historical kline data from Binance"""
    client = await AsyncClient.create()
    
    try:
        # Fetch historical klines
        klines = await client.get_historical_klines(
            symbol, interval, 
            str(datetime.now() - timedelta(days=limit)),
            limit=limit
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
        
        logger.info(f"Fetched {len(df)} historical klines for {symbol}")
        return df
        
    finally:
        await client.close_connection()

async def train_prediction_model():
    """Train the prediction model"""
    config = Config()
    
    logger.info("Starting model training...")
    
    # Fetch historical data
    df = await fetch_historical_data(
        config.SYMBOL, 
        config.KLINE_INTERVAL, 
        limit=2000  # More data for better training
    )
    
    if len(df) < 500:
        logger.error("Insufficient historical data for training")
        return
    
    # Initialize predictor
    predictor = PricePredictor(model_type="random_forest")
    
    # Train model
    metrics = predictor.train(df)
    
    # Save model
    model_path = f"models/{config.SYMBOL}_predictor.joblib"
    import os
    os.makedirs("models", exist_ok=True)
    predictor.save_model(model_path)
    
    # Print results
    logger.info("=== Training Results ===")
    logger.info(f"Train RMSE: {metrics['train_rmse']:.4f}")
    logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
    logger.info(f"Train Direction Accuracy: {metrics['train_direction_accuracy']:.2%}")
    logger.info(f"Test Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")
    logger.info(f"Model saved to: {model_path}")
    
    # Test prediction
    test_prediction, confidence = predictor.predict(df.tail(100))
    logger.info(f"Sample prediction: {test_prediction:.4f} (confidence: {confidence:.2%})")

async def train_ensemble_model():
    """Train ensemble model with multiple algorithms"""
    config = Config()
    
    logger.info("Training ensemble model...")
    
    # Fetch data
    df = await fetch_historical_data(config.SYMBOL, config.KLINE_INTERVAL, limit=2000)
    
    # Train multiple models
    models = {}
    results = {}
    
    for model_type in ["random_forest", "gradient_boosting"]:
        logger.info(f"Training {model_type}...")
        predictor = PricePredictor(model_type=model_type)
        metrics = predictor.train(df)
        models[model_type] = predictor
        results[model_type] = metrics
        
        # Save individual model
        model_path = f"models/{config.SYMBOL}_{model_type}.joblib"
        predictor.save_model(model_path)
    
    # Print comparison
    logger.info("=== Model Comparison ===")
    for model_type, metrics in results.items():
        logger.info(f"{model_type}:")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train prediction model")
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble model")
    
    args = parser.parse_args()
    
    if args.ensemble:
        asyncio.run(train_ensemble_model())
    else:
        asyncio.run(train_prediction_model())

if __name__ == "__main__":
    main()
