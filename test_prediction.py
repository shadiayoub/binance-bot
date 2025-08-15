#!/usr/bin/env python3
"""
Test Prediction Issue
"""

import pandas as pd
import numpy as np
from prediction_models import PricePredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prediction():
    """Test the prediction functionality"""
    
    try:
        # Load the improved model
        predictor = PricePredictor()
        predictor.load_model("models/BTCUSDT_improved_gradient_boosting.joblib")
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model type: {type(predictor.model)}")
        logger.info(f"Model has predict: {hasattr(predictor.model, 'predict')}")
        logger.info(f"Feature columns: {len(predictor.feature_columns) if predictor.feature_columns else 'None'}")
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        base_price = 50000
        price_changes = np.random.normal(0, 0.001, 100)
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Test prediction
        logger.info("Testing prediction...")
        prediction, confidence = predictor.predict(df)
        
        logger.info(f"✅ Prediction successful!")
        logger.info(f"   Prediction: {prediction:.6f}")
        logger.info(f"   Confidence: {confidence:.2%}")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
