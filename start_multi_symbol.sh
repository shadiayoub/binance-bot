#!/bin/bash
"""
Quick Start Script for Multi-Symbol Trading Bot
"""

echo "🚀 Starting Multi-Symbol Binance Futures Trading Bot"
echo "=================================================="

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "📊 No ML models found. Training models for all symbols..."
    echo "This may take a few minutes..."
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Train models
    python train_multi_symbol_models.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Models trained successfully!"
    else
        echo "❌ Model training failed. Please check the logs."
        exit 1
    fi
else
    echo "✅ ML models found. Proceeding with bot startup..."
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please create one from env.example"
    exit 1
fi

# Create necessary directories
mkdir -p data logs models

echo "🐳 Starting Docker containers..."
docker-compose up -d

echo "📈 Bot is starting up..."
echo "📊 Monitor logs with: docker-compose logs -f binance-bot"
echo "🛑 Stop bot with: docker-compose down"
echo ""
echo "🎯 Trading symbols: BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT"
echo "🤖 ML-powered predictions for each symbol"
echo "⚡ Real-time market analysis and automated trading"
