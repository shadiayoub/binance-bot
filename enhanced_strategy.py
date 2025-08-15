#!/usr/bin/env python3
"""
Enhanced Trading Strategy with Prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

from prediction_models import PricePredictor, EnsemblePredictor

logger = logging.getLogger(__name__)

class EnhancedStrategy:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.predictor = None
        self.ensemble = None
        self.signal_history = []
        
        # Strategy parameters
        self.min_confidence = 0.6
        self.min_prediction_threshold = 0.001  # 0.1% minimum prediction
        self.signal_decay = 0.95  # Signal strength decay factor
        
    def initialize_predictor(self, model_type: str = "random_forest"):
        """Initialize the ML predictor"""
        self.predictor = PricePredictor(model_type)
        self.ensemble = EnsemblePredictor()
        
        # Add different predictors to ensemble
        self.ensemble.add_predictor("random_forest", PricePredictor("random_forest"), 1.0)
        self.ensemble.add_predictor("gradient_boosting", PricePredictor("gradient_boosting"), 0.8)
        
    def train_model(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Train the prediction model with historical data"""
        if self.predictor is None:
            self.initialize_predictor()
        
        return self.predictor.train(historical_data)
    
    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various technical indicators and signals"""
        signals = {}
        
        # Ensure data types are correct
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any NaN values
        df = df.dropna()
        
        if len(df) < 21:  # Need at least 21 periods for EMA21
            return {
                'ema_signal': 0.0,
                'rsi_signal': 0.0,
                'bb_signal': 0.0,
                'macd_signal': 0.0,
                'volume_signal': 0.0,
                'momentum_signal': 0.0
            }
        
        # EMA Crossover (original strategy)
        df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
        
        if len(df) >= 2:
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            
            ema_cross_up = prev['ema9'] < prev['ema21'] and curr['ema9'] > curr['ema21']
            ema_cross_down = prev['ema9'] > prev['ema21'] and curr['ema9'] < curr['ema21']
            
            signals['ema_signal'] = 1.0 if ema_cross_up else (-1.0 if ema_cross_down else 0.0)
        else:
            signals['ema_signal'] = 0.0
        
        # RSI Signals
        rsi = RSIIndicator(df['close'], window=14).rsi()
        if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
            current_rsi = float(rsi.iloc[-1])
            if current_rsi < 30:
                signals['rsi_signal'] = 1.0  # Oversold - bullish
            elif current_rsi > 70:
                signals['rsi_signal'] = -1.0  # Overbought - bearish
            else:
                signals['rsi_signal'] = 0.0
        else:
            signals['rsi_signal'] = 0.0
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        
        if len(bb_upper) > 0 and len(bb_lower) > 0:
            current_price = float(df['close'].iloc[-1])
            current_upper = float(bb_upper.iloc[-1])
            current_lower = float(bb_lower.iloc[-1])
            
            if current_price <= current_lower:
                signals['bb_signal'] = 1.0  # Price at lower band - bullish
            elif current_price >= current_upper:
                signals['bb_signal'] = -1.0  # Price at upper band - bearish
            else:
                signals['bb_signal'] = 0.0
        else:
            signals['bb_signal'] = 0.0
        
        # MACD
        macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        if len(macd_histogram) >= 2:
            prev_hist = float(macd_histogram.iloc[-2])
            curr_hist = float(macd_histogram.iloc[-1])
            
            if curr_hist > 0 and prev_hist <= 0:
                signals['macd_signal'] = 1.0  # MACD histogram turning positive
            elif curr_hist < 0 and prev_hist >= 0:
                signals['macd_signal'] = -1.0  # MACD histogram turning negative
            else:
                signals['macd_signal'] = 0.0
        else:
            signals['macd_signal'] = 0.0
        
        # Volume analysis
        volume_ma = df['volume'].rolling(20).mean()
        if len(volume_ma) > 0:
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = float(volume_ma.iloc[-1])
            
            if current_volume > avg_volume * 1.5:
                signals['volume_signal'] = 1.0  # High volume - bullish
            elif current_volume < avg_volume * 0.5:
                signals['volume_signal'] = -1.0  # Low volume - bearish
            else:
                signals['volume_signal'] = 0.0
        else:
            signals['volume_signal'] = 0.0
        
        # Price momentum
        price_change_1 = float(df['close'].pct_change(1).iloc[-1]) if len(df) > 1 else 0.0
        price_change_5 = float(df['close'].pct_change(5).iloc[-1]) if len(df) > 5 else 0.0
        
        if price_change_1 > 0.01:  # 1% increase
            signals['momentum_signal'] = 1.0
        elif price_change_1 < -0.01:  # 1% decrease
            signals['momentum_signal'] = -1.0
        else:
            signals['momentum_signal'] = 0.0
        
        return signals
    
    def get_ml_prediction(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Get ML prediction for price direction"""
        if self.predictor is None or not self.predictor.is_trained:
            return 0.0, 0.0
        
        try:
            prediction, confidence = self.predictor.predict(df)
            return prediction, confidence
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 0.0, 0.0
    
    def combine_signals(self, technical_signals: Dict[str, float], 
                       ml_prediction: float, ml_confidence: float) -> Tuple[str, float]:
        """Combine technical signals with ML prediction"""
        
        # Calculate weighted technical signal
        technical_weights = {
            'ema_signal': 0.3,
            'rsi_signal': 0.2,
            'bb_signal': 0.15,
            'macd_signal': 0.15,
            'volume_signal': 0.1,
            'momentum_signal': 0.1
        }
        
        technical_score = 0.0
        for signal_name, weight in technical_weights.items():
            if signal_name in technical_signals:
                technical_score += technical_signals[signal_name] * weight
        
        # Combine with ML prediction
        ml_weight = min(ml_confidence, 0.8)  # Cap ML weight at 0.8
        technical_weight = 1.0 - ml_weight
        
        combined_score = (technical_score * technical_weight + 
                         np.sign(ml_prediction) * ml_weight)
        
        # Determine final signal
        if combined_score > 0.3:
            signal = "BUY"
            strength = min(abs(combined_score), 1.0)
        elif combined_score < -0.3:
            signal = "SELL"
            strength = min(abs(combined_score), 1.0)
        else:
            signal = None
            strength = 0.0
        
        # Apply signal decay based on history
        if self.signal_history:
            last_signal = self.signal_history[-1]
            if last_signal['signal'] == signal:
                # Same signal - reduce strength
                strength *= self.signal_decay
        
        # Store signal in history
        self.signal_history.append({
            'signal': signal,
            'strength': strength,
            'technical_score': technical_score,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'combined_score': combined_score
        })
        
        # Keep only last 100 signals
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
        
        return signal, strength
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Generate trading signal using enhanced strategy"""
        
        # Calculate technical signals
        technical_signals = self.calculate_technical_signals(df)
        
        # Get ML prediction
        ml_prediction, ml_confidence = self.get_ml_prediction(df)
        
        # Combine signals
        signal, strength = self.combine_signals(technical_signals, ml_prediction, ml_confidence)
        
        # Additional filters
        if signal:
            # Check if signal meets minimum confidence
            if strength < 0.4:
                signal = None
                strength = 0.0
            
            # Check if ML prediction is too weak
            if abs(ml_prediction) < self.min_prediction_threshold:
                signal = None
                strength = 0.0
        
        # Prepare detailed signal info
        signal_info = {
            'signal': signal,
            'strength': strength,
            'technical_signals': technical_signals,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'timestamp': pd.Timestamp.now()
        }
        
        return signal, strength, signal_info
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of recent signals"""
        if not self.signal_history:
            return {}
        
        recent_signals = self.signal_history[-20:]  # Last 20 signals
        
        buy_signals = [s for s in recent_signals if s['signal'] == 'BUY']
        sell_signals = [s for s in recent_signals if s['signal'] == 'SELL']
        
        return {
            'total_signals': len(recent_signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_buy_strength': np.mean([s['strength'] for s in buy_signals]) if buy_signals else 0,
            'avg_sell_strength': np.mean([s['strength'] for s in sell_signals]) if sell_signals else 0,
            'avg_ml_confidence': np.mean([s['ml_confidence'] for s in recent_signals]),
            'last_signal': recent_signals[-1] if recent_signals else None
        }
