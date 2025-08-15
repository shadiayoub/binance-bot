#!/usr/bin/env python3
"""
Test ML Model Functionality
"""

import pandas as pd
import numpy as np
from prediction_models import PricePredictor
from enhanced_strategy import EnhancedStrategy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the trained model can be loaded"""
    try:
        predictor = PricePredictor()
        predictor.load_model("models/BTCUSDT_predictor.joblib")
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def test_prediction():
    """Test model prediction with sample data"""
    try:
        # Load model
        predictor = PricePredictor()
        predictor.load_model("models/BTCUSDT_predictor.joblib")
        
        # Create sample data (similar to what the bot would use)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 50000
        price_changes = np.random.normal(0, 0.001, 100)  # Small random changes
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Make prediction
        prediction, confidence = predictor.predict(df)
        
        logger.info(f"✅ Prediction test successful!")
        logger.info(f"   Prediction: {prediction:.6f}")
        logger.info(f"   Confidence: {confidence:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {e}")
        return False

def test_enhanced_strategy():
    """Test the enhanced strategy with ML integration"""
    try:
        # Create strategy
        config = {
            'SYMBOL': 'BTCUSDT',
            'KLINE_INTERVAL': '1m'
        }
        strategy = EnhancedStrategy(config)
        
        # Initialize predictor
        strategy.initialize_predictor()
        strategy.predictor.load_model("models/BTCUSDT_predictor.joblib")
        
        # Create sample data
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
        
        # Generate signal
        signal, strength, signal_info = strategy.generate_signal(df)
        
        logger.info(f"✅ Enhanced strategy test successful!")
        logger.info(f"   Signal: {signal}")
        logger.info(f"   Strength: {strength:.2f}")
        logger.info(f"   ML Prediction: {signal_info.get('ml_prediction', 'N/A')}")
        logger.info(f"   ML Confidence: {signal_info.get('ml_confidence', 'N/A')}")
        
        if signal_info.get('technical_signals'):
            tech = signal_info['technical_signals']
            logger.info(f"   Technical Signals:")
            for name, value in tech.items():
                logger.info(f"     {name}: {value:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced strategy test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Testing ML Model Integration...")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction),
        ("Enhanced Strategy", test_enhanced_strategy)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 Test Results Summary:")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Your ML model is ready for trading!")
    else:
        logger.error("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
