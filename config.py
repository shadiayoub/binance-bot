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
    TESTNET = False  # Set to True for testing, False for production
    API_KEY = os.environ.get("API_KEY", "")
    API_SECRET = os.environ.get("API_SECRET", "")
    
    # Trading Configuration
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]  # Multiple symbols
    DEFAULT_SYMBOL = "BTCUSDT"  # Fallback symbol
    KLINE_INTERVAL = "1m"
    LEVERAGE = 5
    RISK_PER_TRADE = 0.05  # 5% risk per trade (reduced for larger minimum notional)
    STOPLOSS_PCT = 0.02    # 2% stop loss (increased for better position sizing)
    TAKE_PROFIT_PCT = 0.04 # 4% take profit
    DAILY_MAX_LOSS_PCT = 0.03  # 3% daily max loss
    
    # Trading Mode
    ENABLE_LIVE_TRADING = True  # Set to False for paper trading only
    MIN_SIGNAL_STRENGTH = 0.3   # Minimum signal strength to execute trades
    MAX_DAILY_TRADES = 10       # Maximum trades per day per symbol
    
    # Symbol-specific configurations
    SYMBOL_CONFIGS = {
        "BTCUSDT": {
            "leverage": 5,
            "risk_per_trade": 0.05,
            "stoploss_pct": 0.02,
            "take_profit_pct": 0.04,
            "min_quantity": 0.00005
        },
        "ETHUSDT": {
            "leverage": 5,
            "risk_per_trade": 0.05,
            "stoploss_pct": 0.02,
            "take_profit_pct": 0.04,
            "min_quantity": 0.001
        },
        "BNBUSDT": {
            "leverage": 5,
            "risk_per_trade": 0.05,
            "stoploss_pct": 0.02,
            "take_profit_pct": 0.04,
            "min_quantity": 0.01
        },
        "ADAUSDT": {
            "leverage": 5,
            "risk_per_trade": 0.05,
            "stoploss_pct": 0.02,
            "take_profit_pct": 0.04,
            "min_quantity": 1.0
        },
        "SOLUSDT": {
            "leverage": 5,
            "risk_per_trade": 0.05,
            "stoploss_pct": 0.02,
            "take_profit_pct": 0.04,
            "min_quantity": 0.1
        }
    }
    
    # Database Configuration
    DB_FILE = "async_bot_state.db"
    DEMO_DB_FILE = "demo_bot_state.db"
    LIVE_DB_FILE = "live_bot_state.db"

    # Get appropriate database file based on mode
    @staticmethod
    def get_db_file():
        if not Config.ENABLE_LIVE_TRADING:
            return Config.DEMO_DB_FILE
        return Config.LIVE_DB_FILE
    
    # Technical Analysis
    KLINE_HISTORY = 200
    MIN_ORDER_USD_FALLBACK = 5.0
    
    # Risk Management
    MAX_OPEN_POSITIONS = 3
    MIN_BALANCE_THRESHOLD = 50.0  # Minimum balance to continue trading (increased for 20 USDT min notional)
    
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
