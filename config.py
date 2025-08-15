#!/usr/bin/env python3
"""
Configuration file for Binance Futures Trading Bot
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration
    TESTNET = False  # Set to False ONLY after extensive testing
    API_KEY = os.environ.get("API_KEY", "")
    API_SECRET = os.environ.get("API_SECRET", "")
    
    # Trading Configuration
    SYMBOL = "BTCUSDT"
    KLINE_INTERVAL = "1m"
    LEVERAGE = 5
    RISK_PER_TRADE = 0.10  # 10% risk per trade (increased for small balance)
    STOPLOSS_PCT = 0.02    # 2% stop loss (increased for better position sizing)
    TAKE_PROFIT_PCT = 0.04 # 4% take profit
    DAILY_MAX_LOSS_PCT = 0.03  # 3% daily max loss
    
    # Database Configuration
    DB_FILE = "async_bot_state.db"
    
    # Technical Analysis
    KLINE_HISTORY = 200
    MIN_ORDER_USD_FALLBACK = 5.0
    
    # Risk Management
    MAX_OPEN_POSITIONS = 3
    MIN_BALANCE_THRESHOLD = 10.0  # Minimum balance to continue trading
    
    # Connection Settings
    WEBSOCKET_TIMEOUT = 30
    REST_TIMEOUT = 10
    MAX_RETRIES = 5
    RETRY_DELAY = 1.0
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.API_KEY or not cls.API_SECRET:
            raise ValueError("API_KEY and API_SECRET must be set in environment variables or .env file")
        
        if cls.RISK_PER_TRADE <= 0 or cls.RISK_PER_TRADE > 0.1:
            raise ValueError("RISK_PER_TRADE must be between 0 and 0.1 (10%)")
        
        if cls.DAILY_MAX_LOSS_PCT <= 0 or cls.DAILY_MAX_LOSS_PCT > 0.5:
            raise ValueError("DAILY_MAX_LOSS_PCT must be between 0 and 0.5 (50%)")
        
        return True
