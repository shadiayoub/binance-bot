#!/usr/bin/env python3
"""
Train ML Models for Multiple Symbols
"""

import asyncio
import os
import logging
from typing import List, Dict
import pandas as pd
import numpy as np

from binance import AsyncClient
from prediction_models import PricePredictor
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_historical_data(client: AsyncClient, symbol: str, days: int = 7) -> pd.DataFrame:
    """Fetch historical kline data for a symbol"""
    try:
        logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        # Binance has a limit of 1500 klines per request
        # For 1-minute intervals, we can get max 1500 minutes = 25 hours
        # So we'll fetch 7 days worth of data in chunks
        all_klines = []
        
        # Calculate how many requests we need
        total_minutes = days * 24 * 60
        chunk_size = 1500  # Max allowed by Binance
        num_chunks = (total_minutes + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            limit = min(chunk_size, total_minutes - i * chunk_size)
            
            klines = await client.futures_klines(
                symbol=symbol,
                interval='1m',
                limit=limit
            )
            
            all_klines.extend(klines)
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        logger.info(f"Fetched {len(df)} klines for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

async def train_symbol_model(symbol: str, df: pd.DataFrame, model_type: str = "gradient_boosting") -> bool:
    """Train ML model for a specific symbol"""
    try:
        logger.info(f"Training {model_type} model for {symbol}")
        
        # Create predictor
        predictor = PricePredictor()
        
        # Create features
        df_features = predictor.create_features(df)
        
        if len(df_features) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(df_features)} samples")
            return False
        
        # Train model
        metrics = predictor.train(df_features)
        
        # Save model
        model_filename = f"models/{symbol}_{model_type}.joblib"
        predictor.save_model(model_filename)
        
        logger.info(f"Model saved for {symbol}: {model_filename}")
        logger.info(f"Metrics - RMSE: {metrics['test_rmse']:.4f}, Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {e}")
        return False

async def train_all_symbols():
    """Train models for all configured symbols"""
    try:
        # Initialize client
        config = Config()
        client = await AsyncClient.create(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            testnet=config.TESTNET
        )
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Train models for each symbol
        results = {}
        
        for symbol in config.SYMBOLS:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*50}")
            
            # Fetch data
            df = await fetch_historical_data(client, symbol, days=30)
            
            if len(df) == 0:
                logger.warning(f"Skipping {symbol} - no data available")
                continue
            
            # Train model
            success = await train_symbol_model(symbol, df, "gradient_boosting")
            results[symbol] = success
            
            # Small delay between symbols
            await asyncio.sleep(1)
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*50}")
        
        successful = [s for s, success in results.items() if success]
        failed = [s for s, success in results.items() if not success]
        
        logger.info(f"Successful: {len(successful)} symbols")
        for symbol in successful:
            logger.info(f"  ✅ {symbol}")
        
        if failed:
            logger.info(f"Failed: {len(failed)} symbols")
            for symbol in failed:
                logger.info(f"  ❌ {symbol}")
        
        await client.close_connection()
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")

if __name__ == "__main__":
    asyncio.run(train_all_symbols())
