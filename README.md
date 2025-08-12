# Binance Futures Trading Bot

An advanced, async-based Binance Futures trading bot with comprehensive risk management, modular architecture, and robust error handling.

## Features

- **Async Architecture**: Built with asyncio for high-performance concurrent operations
- **Risk Management**: Comprehensive risk controls including daily loss limits, position sizing, and market condition checks
- **Modular Design**: Separated concerns with dedicated modules for configuration, database, and risk management
- **Real-time Data**: WebSocket connections for live market data and user account updates
- **Technical Analysis**: EMA crossover strategy with extensible framework for other strategies
- **Database Persistence**: SQLite storage for orders, trades, and account snapshots
- **Error Handling**: Robust error handling with exponential backoff and graceful degradation
- **Logging**: Comprehensive logging to both file and console

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd binance-bot
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   export BINANCE_API_KEY="your_api_key_here"
   export BINANCE_API_SECRET="your_api_secret_here"
   ```

## Configuration

Edit `config.py` to customize trading parameters:

```python
class Config:
    TESTNET = True  # Set to False for live trading
    SYMBOL = "BTCUSDT"
    LEVERAGE = 5
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    STOPLOSS_PCT = 0.005   # 0.5% stop loss
    TAKE_PROFIT_PCT = 0.01 # 1% take profit
    DAILY_MAX_LOSS_PCT = 0.03  # 3% daily max loss
```

## Usage

### Running the Bot

```bash
# Run the improved version
python improved_bot.py

# Or run the original version
python bot.py
```

### Testing

**IMPORTANT**: Always test on Binance Testnet first:

1. Get testnet API credentials from [Binance Testnet](https://testnet.binancefuture.com/)
2. Set `TESTNET = True` in config
3. Run the bot and monitor its behavior
4. Only switch to live trading after extensive testing

## Architecture

### Core Modules

- **`config.py`**: Centralized configuration management
- **`database.py`**: Database operations and persistence
- **`risk_manager.py`**: Risk management and position sizing
- **`improved_bot.py`**: Main bot logic with enhanced features

### Key Components

1. **Configuration Management**: Centralized config with validation
2. **Database Layer**: SQLite with connection management and indexing
3. **Risk Management**: Multi-layered risk controls
4. **Market Data**: Real-time kline data via WebSocket
5. **User Data Stream**: Account and order updates
6. **Strategy Engine**: Extensible technical analysis framework

## Risk Management

The bot implements several risk management features:

- **Daily Loss Limits**: Automatic shutdown when daily loss threshold is reached
- **Position Sizing**: Risk-based position sizing with maximum limits
- **Market Condition Checks**: Volatility-based trading filters
- **Balance Thresholds**: Minimum balance requirements
- **Open Position Limits**: Maximum number of concurrent positions
- **Risk-Reward Validation**: Minimum risk-reward ratio enforcement

## Database Schema

The bot stores data in SQLite with the following tables:

- **orders**: All order records with status tracking
- **trades**: Completed trades with P&L
- **account_snapshot**: Account balance history
- **positions**: Current position tracking
- **bot_state**: Bot state persistence

## Logging

The bot provides comprehensive logging:

- **File Logging**: All logs saved to `bot.log`
- **Console Output**: Real-time status updates
- **Error Tracking**: Detailed error information with stack traces
- **Performance Metrics**: Trade execution times and success rates

## Monitoring

Monitor the bot through:

1. **Log Files**: Check `bot.log` for detailed information
2. **Database Queries**: Query the SQLite database for performance analysis
3. **Risk Summary**: Use risk manager methods to get current status

## Safety Features

- **Testnet Mode**: Default configuration for safe testing
- **Graceful Shutdown**: Proper cleanup on interruption
- **Error Recovery**: Automatic reconnection and retry logic
- **Validation**: Comprehensive parameter validation
- **Rate Limiting**: Built-in rate limiting for API calls

## Customization

### Adding New Strategies

1. Create a new strategy class in a separate module
2. Implement the signal generation method
3. Update the main bot to use your strategy

### Modifying Risk Parameters

Edit the `RiskLimits` class in `risk_manager.py`:

```python
@dataclass
class RiskLimits:
    max_daily_loss_pct: float = 0.03
    max_open_positions: int = 3
    min_balance_threshold: float = 10.0
    max_position_size_pct: float = 0.1
    max_leverage: int = 10
    min_risk_reward_ratio: float = 1.5
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Check API credentials and network connectivity
2. **Database Errors**: Ensure write permissions for the database file
3. **WebSocket Disconnections**: The bot automatically reconnects
4. **Order Failures**: Check symbol precision and minimum order sizes

### Debug Mode

Enable debug logging by modifying the logging level in `improved_bot.py`:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Disclaimer

**Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. This bot is for educational purposes only. Use at your own risk.**

- Always test thoroughly on testnet before live trading
- Start with small amounts
- Monitor the bot continuously
- Understand the risks involved in futures trading
- Never invest more than you can afford to lose

## License

This project is for educational purposes. Use responsibly and at your own risk.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the logs for error messages
2. Review the configuration settings
3. Test on testnet first
4. Open an issue with detailed information
