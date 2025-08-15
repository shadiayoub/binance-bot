# API Reference Documentation

## 📋 Table of Contents

1. [Core Classes](#core-classes)
2. [Configuration](#configuration)
3. [Database Schema](#database-schema)
4. [ML Models](#ml-models)
5. [Trading Strategy](#trading-strategy)
6. [Risk Management](#risk-management)

## 🏗️ Core Classes

### MultiSymbolBot

Main bot class that orchestrates multi-symbol trading.

```python
class MultiSymbolBot:
    def __init__(self)
    async def initialize() -> bool
    async def run()
    async def cleanup()
```

**Key Methods:**

- `initialize()`: Sets up connections, loads models, configures symbols
- `run()`: Main bot loop with kline listeners and signal processing
- `cleanup()`: Graceful shutdown and resource cleanup

**Properties:**
- `symbols`: List of trading symbols
- `symbol_data`: Market data for each symbol
- `symbol_strategies`: Strategy instances for each symbol
- `symbol_positions`: Current positions for each symbol

### EnhancedStrategy

Combines technical analysis with ML predictions for signal generation.

```python
class EnhancedStrategy:
    def __init__(self, config: dict)
    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict[str, float]
    def get_ml_prediction(self, df: pd.DataFrame) -> Tuple[float, float]
    def combine_signals(self, technical_signals: Dict, ml_prediction: float, ml_confidence: float) -> Tuple[str, float]
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], float, Dict[str, Any]]
```

**Key Methods:**

- `calculate_technical_signals()`: Computes 34+ technical indicators
- `get_ml_prediction()`: Gets ML model prediction and confidence
- `combine_signals()`: Combines technical and ML signals with weights
- `generate_signal()`: Final signal generation with filters

### PricePredictor

Handles ML model training, prediction, and persistence.

```python
class PricePredictor:
    def __init__(self)
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]
    def train(self, df: pd.DataFrame) -> Dict[str, float]
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]
    def save_model(self, filepath: str)
    def load_model(self, filepath: str)
```

**Key Methods:**

- `create_features()`: Generates 34+ features from market data
- `train()`: Trains ML model with cross-validation
- `predict()`: Makes predictions with confidence scores
- `save_model()` / `load_model()`: Model persistence

### RiskManager

Manages position sizing, risk limits, and safety checks.

```python
class RiskManager:
    def __init__(self, config: Config, db_manager: DatabaseManager)
    def calculate_position_size(self, balance: float, risk_per_trade: float, price: float, min_quantity: float) -> float
    def check_daily_loss_limit(self, current_pnl: float) -> bool
    def check_balance_threshold(self, balance: float) -> bool
    def check_open_positions_limit(self, open_positions: int) -> bool
    def validate_trade_parameters(self, signal: str, strength: float, confidence: float) -> bool
```

**Key Methods:**

- `calculate_position_size()`: Dynamic position sizing based on risk
- `check_daily_loss_limit()`: Daily loss limit enforcement
- `validate_trade_parameters()`: Trade parameter validation

### DatabaseManager

Handles all database operations and persistence.

```python
class DatabaseManager:
    def __init__(self, db_file: str)
    def initialize_database(self)
    def store_order(self, order_data: Dict[str, Any])
    def store_trade(self, trade_data: Dict[str, Any])
    def snapshot_account(self, account_data: Dict[str, Any])
    def get_daily_pnl(self) -> float
    def get_open_orders(self) -> List[Dict[str, Any]]
    def update_bot_state(self, state_data: Dict[str, Any])
    def get_bot_state(self) -> Dict[str, Any]
```

## ⚙️ Configuration

### Config Class

Centralized configuration management.

```python
class Config:
    # API Configuration
    TESTNET: bool = False
    API_KEY: str = ""
    API_SECRET: str = ""
    
    # Trading Configuration
    SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    DEFAULT_SYMBOL: str = "BTCUSDT"
    KLINE_INTERVAL: str = "1m"
    LEVERAGE: int = 5
    RISK_PER_TRADE: float = 0.10
    STOPLOSS_PCT: float = 0.02
    TAKE_PROFIT_PCT: float = 0.04
    DAILY_MAX_LOSS_PCT: float = 0.03
    
    # Symbol-specific configurations
    SYMBOL_CONFIGS: Dict[str, Dict[str, Any]]
    
    # Database Configuration
    DB_FILE: str = "async_bot_state.db"
    
    # Technical Analysis
    KLINE_HISTORY: int = 200
    MIN_ORDER_USD_FALLBACK: float = 5.0
    
    # Risk Management
    MAX_OPEN_POSITIONS: int = 3
    MIN_BALANCE_THRESHOLD: float = 10.0
    
    # Connection Settings
    WEBSOCKET_TIMEOUT: int = 30
    REST_TIMEOUT: int = 10
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 1.0
```

### Environment Variables

```bash
# Required
API_KEY=your_binance_api_key
API_SECRET=your_binance_api_secret

# Optional
TESTNET=True
SYMBOL=BTCUSDT
LEVERAGE=5
RISK_PER_TRADE=0.10
STOPLOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
DAILY_MAX_LOSS_PCT=0.03
DB_FILE=async_bot_state.db
LOG_LEVEL=INFO
```

## 🗄️ Database Schema

### Tables

#### orders
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    status TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### trades
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL,
    pnl REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### account_snapshot
```sql
CREATE TABLE account_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_balance REAL NOT NULL,
    available_balance REAL NOT NULL,
    total_pnl REAL,
    total_margin REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### positions
```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL,
    margin REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### bot_state
```sql
CREATE TABLE bot_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🤖 ML Models

### Feature Engineering

The bot generates 34+ features for ML models:

#### Price Features
- `price_change`: 1-minute price change
- `price_change_2/5/10/20`: Multi-timeframe price changes
- `volatility_5/10/20`: Rolling volatility measures

#### Volume Features
- `volume_change`: Volume change percentage
- `volume_ratio`: Current volume vs average
- `volume_std`: Volume standard deviation

#### Technical Indicators
- `ema_cross_9_21/21_50`: EMA crossover signals
- `bb_position/bb_width`: Bollinger Bands metrics
- `rsi_7/14/21`: RSI at different periods
- `macd/macd_histogram`: MACD indicators
- `stoch_k/stoch_d`: Stochastic oscillator
- `atr_ratio`: ATR ratio
- `williams_r/cci/mfi`: Additional oscillators

#### Time Features
- `hour`: Hour of day (0-23)
- `is_weekend`: Weekend flag
- `is_london_open/is_ny_open`: Market session flags

### Model Training

```python
# Training parameters
model_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 42,
    'learning_rate': 0.1
}

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
```

### Model Performance

| Metric | Description | Target |
|--------|-------------|---------|
| **Direction Accuracy** | % of correct price direction predictions | >55% |
| **RMSE** | Root Mean Square Error | <0.002 |
| **Confidence** | Prediction confidence score | 0.1-1.0 |

## 📊 Trading Strategy

### Signal Generation Flow

```python
def generate_signal(df: pd.DataFrame) -> Tuple[Optional[str], float, Dict[str, Any]]:
    # 1. Calculate technical signals
    technical_signals = calculate_technical_signals(df)
    
    # 2. Get ML prediction
    ml_prediction, ml_confidence = get_ml_prediction(df)
    
    # 3. Combine signals
    signal, strength = combine_signals(technical_signals, ml_prediction, ml_confidence)
    
    # 4. Apply filters
    if signal and strength > 0.3 and ml_confidence > 0.5:
        return signal, strength, {
            'technical_signals': technical_signals,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence
        }
    
    return None, 0.0, {}
```

### Signal Weights

```python
technical_weights = {
    'ema_signal': 0.3,
    'rsi_signal': 0.2,
    'bb_signal': 0.15,
    'macd_signal': 0.15,
    'volume_signal': 0.1,
    'momentum_signal': 0.1
}

ml_weight = min(ml_confidence, 0.8)
technical_weight = 1.0 - ml_weight
```

### Trade Execution

```python
async def execute_trade(symbol: str, signal: str, strength: float, price: float):
    # 1. Check existing position
    if symbol_positions[symbol]:
        return
    
    # 2. Calculate position size
    position_size = risk_manager.calculate_position_size(balance, risk_per_trade, price, min_quantity)
    
    # 3. Place market order
    order = await client.futures_create_order(
        symbol=symbol,
        side=signal,
        type='MARKET',
        quantity=position_size
    )
    
    # 4. Place stop orders
    await place_stop_orders(symbol, signal, position_size, price)
```

## 🛡️ Risk Management

### Position Sizing

```python
def calculate_position_size(balance: float, risk_per_trade: float, price: float, min_quantity: float) -> float:
    # Calculate risk amount
    risk_amount = balance * risk_per_trade
    
    # Calculate position size based on stop loss
    stop_loss_distance = price * stoploss_pct
    position_size = risk_amount / stop_loss_distance
    
    # Apply minimum quantity check
    if position_size < min_quantity:
        return 0.0
    
    return position_size
```

### Risk Limits

```python
# Daily loss limit
DAILY_MAX_LOSS_PCT = 0.03  # 3% daily loss limit

# Position limits
MAX_OPEN_POSITIONS = 3

# Balance threshold
MIN_BALANCE_THRESHOLD = 10.0  # Minimum $10 balance

# Signal strength threshold
MIN_SIGNAL_STRENGTH = 0.3

# ML confidence threshold
MIN_ML_CONFIDENCE = 0.5
```

### Safety Checks

```python
def validate_trade_parameters(signal: str, strength: float, confidence: float) -> bool:
    # Check signal strength
    if strength < MIN_SIGNAL_STRENGTH:
        return False
    
    # Check ML confidence
    if confidence < MIN_ML_CONFIDENCE:
        return False
    
    # Check daily loss limit
    if not check_daily_loss_limit(current_pnl):
        return False
    
    # Check balance threshold
    if not check_balance_threshold(current_balance):
        return False
    
    return True
```

## 📝 Logging

### Log Levels

- **INFO**: Normal operation, signals, trades
- **WARNING**: Non-critical issues, retries
- **ERROR**: Critical errors, failures
- **DEBUG**: Detailed debugging information

### Log Format

```
2025-08-15 12:55:47,464 [INFO] prediction_models: Model loaded from models/BTCUSDT_gradient_boosting.joblib
2025-08-15 12:55:48,539 [INFO] __main__: Symbol BTCUSDT precision - Qty: 3, Price: 1, Min notional: 100.0
2025-08-15 12:55:48,808 [INFO] __main__: Leverage set for BTCUSDT: {'symbol': 'BTCUSDT', 'leverage': 5, 'maxNotionalValue': '480000000'}
```

### Sample Trade Log

```
[BTCUSDT] Price: 118,670.40, Signal: SELL, Strength: 0.65
[BTCUSDT] ML Prediction: -0.0009, Confidence: 72.00%
[BTCUSDT] Technical signals - EMA: -1.00, RSI: 0.00, BB: 0.00
[BTCUSDT] SELL order placed: 0.00008 @ 118670.40
[BTCUSDT] Stop orders placed - SL: 116297.99, TP: 123417.22
```

## 🔧 Error Handling

### Common Errors

| Error | Description | Solution |
|-------|-------------|----------|
| `APIError(code=-1111)` | Precision error | Check symbol precision settings |
| `APIError(code=-1130)` | Invalid limit | Reduce data fetch limit |
| `AttributeError` | Model loading issue | Check model file exists |
| `ConnectionError` | Network issue | Check internet connection |

### Retry Logic

```python
async def with_backoff(func, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
```

## 📊 Performance Monitoring

### Key Metrics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Total Return**: Overall portfolio performance

### Database Queries

```sql
-- Daily P&L
SELECT DATE(timestamp) as date, SUM(pnl) as daily_pnl 
FROM trades 
GROUP BY DATE(timestamp) 
ORDER BY date DESC;

-- Win Rate
SELECT 
    COUNT(CASE WHEN pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
FROM trades;

-- Symbol Performance
SELECT 
    symbol,
    COUNT(*) as trades,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl
FROM trades 
GROUP BY symbol;
```

---

**Last Updated**: August 15, 2025  
**Version**: 2.0.0
