#!/usr/bin/env python3
"""
Prediction Models for Binance Futures Trading Bot
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features for prediction"""
        df = df.copy()
        
        # Ensure numeric data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any NaN values
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
        
        # Technical indicators
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
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
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
        
        self.feature_columns = feature_columns
        
        X = df[feature_columns]
        y = df['target']
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the prediction model"""
        logger.info("Training prediction model...")
        
        # Create features
        df_features = self.create_features(df)
        
        # Prepare data
        X, y = self.prepare_features(df_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Calculate directional accuracy
        train_direction_acc = accuracy_score(
            (y_train > 0), (y_pred_train > 0)
        )
        test_direction_acc = accuracy_score(
            (y_test > 0), (y_pred_test > 0)
        )
        
        self.is_trained = True
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_direction_accuracy': train_direction_acc,
            'test_direction_accuracy': test_direction_acc
        }
        
        logger.info(f"Model training completed. Test RMSE: {test_rmse:.4f}, "
                   f"Direction Accuracy: {test_direction_acc:.2%}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Make prediction for the next period"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features for the latest data
        df_features = self.create_features(df)
        
        if len(df_features) == 0:
            return 0.0, 0.0
        
        # Check if we have the required feature columns
        if not hasattr(self, 'feature_columns') or not self.feature_columns:
            logger.warning("No feature columns available, using default features")
            # Use default feature columns if not set
            self.feature_columns = [
                'price_change', 'price_change_2', 'price_change_5', 'price_change_10', 'price_change_20',
                'volatility_5', 'volatility_10', 'volatility_20',
                'volume_change', 'volume_ratio', 'volume_std',
                'price_vs_sma5', 'price_vs_sma20', 'price_vs_sma50',
                'ema_cross_9_21', 'ema_cross_21_50',
                'bb_position', 'bb_width',
                'rsi_7', 'rsi_14', 'rsi_21',
                'macd', 'macd_histogram', 'macd_histogram_change',
                'stoch_k', 'stoch_d',
                'atr_ratio',
                'williams_r', 'cci', 'mfi',
                'hour', 'is_weekend', 'is_london_open', 'is_ny_open'
            ]
        
        # Check which features are actually available
        available_features = [col for col in self.feature_columns if col in df_features.columns]
        missing_features = [col for col in self.feature_columns if col not in df_features.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Use only available features
            self.feature_columns = available_features
        
        if not available_features:
            logger.error("No features available for prediction")
            return 0.0, 0.0
        
        # Get latest features
        latest_features = df_features[available_features].iloc[-1:]
        
        # Scale features if scaler is available
        if self.scaler is not None:
            latest_scaled = self.scaler.transform(latest_features)
        else:
            latest_scaled = latest_features.values
        
        # Make prediction
        try:
            prediction = self.model.predict(latest_scaled)[0]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0, 0.0
        
        # Get prediction confidence
        if hasattr(self.model, 'estimators_'):
            # Random Forest - use individual estimators
            predictions = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'predict'):
                    pred = estimator.predict(latest_scaled)[0]
                    predictions.append(pred)
            
            if predictions:
                # Calculate confidence based on prediction variance
                pred_std = np.std(predictions)
                pred_mean = np.mean(predictions)
                if abs(pred_mean) > 1e-8:
                    confidence = max(0.1, min(1.0, 1 - pred_std / abs(pred_mean)))
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
        elif hasattr(self.model, 'estimators'):
            # Gradient Boosting - use individual estimators
            predictions = []
            for estimator in self.model.estimators:
                if hasattr(estimator, 'predict'):
                    pred = estimator.predict(latest_scaled)[0]
                    predictions.append(pred)
            
            if predictions:
                # Calculate confidence based on prediction variance
                pred_std = np.std(predictions)
                pred_mean = np.mean(predictions)
                if abs(pred_mean) > 1e-8:
                    confidence = max(0.1, min(1.0, 1 - pred_std / abs(pred_mean)))
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
        else:
            confidence = 0.5  # Default confidence
        
        return prediction, confidence
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'trained_at': datetime.now()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        # Handle both original and improved model formats
        if 'model' in model_data:
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
        else:
            # Original format
            self.model = model_data
            self.scaler = None
            self.feature_columns = []
            self.model_type = "unknown"
        
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


class SentimentPredictor:
    """Sentiment-based prediction using news and social media"""
    
    def __init__(self):
        self.sentiment_score = 0.0
        self.news_sources = []
    
    def update_sentiment(self, news_data: Dict[str, Any]):
        """Update sentiment based on news and social media"""
        # This would integrate with news APIs, Twitter, Reddit, etc.
        # For now, placeholder implementation
        pass
    
    def get_sentiment_signal(self) -> float:
        """Get sentiment-based signal (-1 to 1)"""
        return self.sentiment_score


class EnsemblePredictor:
    """Combine multiple prediction methods"""
    
    def __init__(self):
        self.predictors = {}
        self.weights = {}
    
    def add_predictor(self, name: str, predictor, weight: float = 1.0):
        """Add a predictor to the ensemble"""
        self.predictors[name] = predictor
        self.weights[name] = weight
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Make ensemble prediction"""
        predictions = []
        confidences = []
        
        for name, predictor in self.predictors.items():
            try:
                pred, conf = predictor.predict(df)
                predictions.append(pred * self.weights[name])
                confidences.append(conf)
            except Exception as e:
                logger.warning(f"Predictor {name} failed: {e}")
                continue
        
        if not predictions:
            return 0.0, 0.0
        
        # Weighted average
        ensemble_prediction = np.average(predictions, weights=confidences)
        ensemble_confidence = np.mean(confidences)
        
        return ensemble_prediction, ensemble_confidence
