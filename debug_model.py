#!/usr/bin/env python3
"""
Debug Model Structure
"""

import joblib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_model_structure():
    """Debug the model structure"""
    
    # Check improved model
    try:
        improved_model_path = "models/BTCUSDT_improved_gradient_boosting.joblib"
        model_data = joblib.load(improved_model_path)
        
        logger.info("=== Improved Model Structure ===")
        logger.info(f"Keys: {list(model_data.keys())}")
        
        if 'model' in model_data:
            model = model_data['model']
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Model has predict method: {hasattr(model, 'predict')}")
            logger.info(f"Model attributes: {dir(model)[:10]}...")  # First 10 attributes
            
            # Check if it's actually a numpy array
            if isinstance(model, np.ndarray):
                logger.error("❌ Model is a numpy array, not a sklearn model!")
                logger.info(f"Array shape: {model.shape}")
                logger.info(f"Array dtype: {model.dtype}")
            else:
                logger.info("✅ Model is not a numpy array")
                
        else:
            logger.error("❌ No 'model' key found")
            
    except Exception as e:
        logger.error(f"Error loading improved model: {e}")
    
    # Check original model
    try:
        original_model_path = "models/BTCUSDT_predictor.joblib"
        model_data = joblib.load(original_model_path)
        
        logger.info("\n=== Original Model Structure ===")
        logger.info(f"Keys: {list(model_data.keys())}")
        
        if 'model' in model_data:
            model = model_data['model']
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Model has predict method: {hasattr(model, 'predict')}")
            
            if isinstance(model, np.ndarray):
                logger.error("❌ Original model is also a numpy array!")
            else:
                logger.info("✅ Original model is not a numpy array")
        else:
            logger.error("❌ No 'model' key found")
            
    except Exception as e:
        logger.error(f"Error loading original model: {e}")

if __name__ == "__main__":
    debug_model_structure()
