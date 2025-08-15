# Deployment Guide

## 🚀 Quick Start Deployment

### Prerequisites

- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **Python**: 3.11 or higher
- **Docker**: 20.10+ and Docker Compose 2.0+
- **Git**: For version control
- **Binance Account**: With API keys (testnet recommended for testing)

### Step 1: Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd binance-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

**Required Configuration:**
```bash
# Binance API Keys (get from https://www.binance.com/en/my/settings/api-management)
API_KEY=your_binance_api_key_here
API_SECRET=your_binance_api_secret_here

# Use testnet for testing (recommended)
TESTNET=True
```

### Step 3: Train ML Models

```bash
# Train models for all symbols (takes 5-10 minutes)
python train_multi_symbol_models.py
```

**Expected Output:**
```
INFO:__main__: Successful: 5 symbols
INFO:__main__:   ✅ BTCUSDT
INFO:__main__:   ✅ ETHUSDT
INFO:__main__:   ✅ BNBUSDT
INFO:__main__:   ✅ ADAUSDT
INFO:__main__:   ✅ SOLUSDT
```

### Step 4: Run with Docker

```bash
# Build and start the bot
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f binance-bot
```

## 🐳 Docker Deployment

### ⚠️ Critical: Code Changes Require Docker Rebuild

**IMPORTANT**: After making any changes to Python code files, you MUST rebuild the Docker image:

```bash
# After code changes:
docker-compose build binance-bot
docker-compose up -d binance-bot

# Verify the changes are active:
docker-compose logs -f binance-bot
```

**Why this is necessary**: Docker containers run from pre-built images. Code changes are not automatically reflected without rebuilding the image.

### Production Deployment

```bash
# 1. Build production image
docker-compose build --no-cache

# 2. Start in detached mode
docker-compose up -d

# 3. Monitor logs
docker-compose logs -f binance-bot

# 4. Check container status
docker-compose ps
```

### Development Deployment

```bash
# Run with live log output
docker-compose up

# Or run specific services
docker-compose --profile monitoring up -d
```

### Docker Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart bot
docker-compose restart binance-bot

# View logs
docker-compose logs -f binance-bot

# Check status
docker-compose ps

# Rebuild after code changes
docker-compose build && docker-compose up -d

# Clean up
docker-compose down -v
docker system prune -f
```

## 🔧 Manual Deployment

### Local Development Setup

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Train models
python train_multi_symbol_models.py

# 3. Run bot
python multi_symbol_bot.py

# 4. Monitor logs
tail -f multi_symbol_bot.log
```

### Production Server Setup

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# 2. Clone repository
git clone <repository-url>
cd binance-bot

# 3. Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
nano .env

# 5. Train models
python train_multi_symbol_models.py

# 6. Run with systemd (optional)
sudo cp binance-bot.service /etc/systemd/system/
sudo systemctl enable binance-bot
sudo systemctl start binance-bot
```

## 📊 Monitoring & Logs

### Log Files

```bash
# Main bot logs
tail -f multi_symbol_bot.log

# Docker logs
docker-compose logs -f binance-bot

# Database location
ls -la data/async_bot_state.db
```

### Health Checks

```bash
# Check if bot is running
docker-compose ps

# Check container health
docker-compose exec binance-bot python -c "import sqlite3; print('Database OK')"

# Check API connectivity
docker-compose exec binance-bot python -c "from binance import AsyncClient; print('API OK')"
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats binance-futures-bot

# Check disk usage
du -sh data/ logs/ models/

# Monitor network connections
netstat -tulpn | grep python
```

## 🔒 Security Configuration

### API Key Security

```bash
# Set proper file permissions
chmod 600 .env
chmod 700 data/ logs/ models/

# Use environment variables (alternative)
export API_KEY="your_api_key"
export API_SECRET="your_api_secret"
```

### Docker Security

```dockerfile
# Run as non-root user
USER bot

# Read-only filesystem
read_only: true

# Resource limits
deploy:
  resources:
    limits:
      memory: 512M
      cpus: '0.5'
```

### Network Security

```bash
# Firewall configuration
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# VPN setup (recommended)
# Configure VPN for secure API access
```

## 🔄 Updates & Maintenance

### Code Updates

```bash
# 1. Pull latest changes
git pull origin main

# 2. Rebuild Docker image
docker-compose build --no-cache

# 3. Restart services
docker-compose down
docker-compose up -d

# 4. Verify deployment
docker-compose logs --tail=20 binance-bot
```

### Model Retraining

```bash
# 1. Retrain models
python train_multi_symbol_models.py

# 2. Restart bot
docker-compose restart binance-bot

# 3. Verify model loading
docker-compose logs --tail=10 binance-bot | grep "ML model loaded"
```

### Database Maintenance

```bash
# Backup database
cp data/async_bot_state.db data/backup_$(date +%Y%m%d_%H%M%S).db

# Check database integrity
sqlite3 data/async_bot_state.db "PRAGMA integrity_check;"

# Optimize database
sqlite3 data/async_bot_state.db "VACUUM;"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. API Connection Errors

```bash
# Check API keys
echo $API_KEY
echo $API_SECRET

# Test API connection
python -c "
from binance import AsyncClient
import asyncio

async def test_api():
    client = await AsyncClient.create(api_key='$API_KEY', api_secret='$API_SECRET', testnet=True)
    account = await client.futures_account()
    print('API connection successful')
    await client.close_connection()

asyncio.run(test_api())
"
```

#### 2. Model Loading Errors

```bash
# Check model files
ls -la models/

# Verify model integrity
python -c "
import joblib
model = joblib.load('models/BTCUSDT_gradient_boosting.joblib')
print('Model loaded successfully')
"
```

#### 3. Docker Issues

```bash
# Check Docker status
docker --version
docker-compose --version

# Clean Docker cache
docker system prune -f

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

#### 4. Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod 600 .env
chmod 755 data/ logs/ models/

# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
docker-compose up --verbose

# Check detailed logs
docker-compose logs --tail=100 binance-bot
```

## 📈 Performance Optimization

### Resource Tuning

```yaml
# docker-compose.yml
services:
  binance-bot:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
```

### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
```

### Log Rotation

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/binance-bot

# Add configuration
/path/to/binance-bot/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 bot bot
}
```

## 🔄 Backup & Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/binance-bot"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp data/async_bot_state.db $BACKUP_DIR/db_$DATE.db

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### Recovery Process

```bash
# 1. Stop bot
docker-compose down

# 2. Restore database
cp /backups/binance-bot/db_20250815_120000.db data/async_bot_state.db

# 3. Restore models
tar -xzf /backups/binance-bot/models_20250815_120000.tar.gz

# 4. Restart bot
docker-compose up -d

# 5. Verify recovery
docker-compose logs --tail=20 binance-bot
```

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] API keys configured and tested
- [ ] Environment variables set
- [ ] ML models trained successfully
- [ ] Database initialized
- [ ] Log directories created
- [ ] Docker images built

### Post-Deployment

- [ ] Bot starts without errors
- [ ] All ML models loaded
- [ ] WebSocket connections established
- [ ] Logs showing normal operation
- [ ] Database writing correctly
- [ ] Health checks passing

### Monitoring Setup

- [ ] Log monitoring configured
- [ ] Performance metrics tracked
- [ ] Error alerting setup
- [ ] Backup schedule configured
- [ ] Update procedures documented

---

**Last Updated**: August 15, 2025  
**Version**: 2.0.0
