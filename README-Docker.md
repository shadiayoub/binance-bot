# Docker Setup for Binance Futures Trading Bot

This guide covers how to run the Binance Futures Trading Bot using Docker containers.

## 🐳 Quick Start

### Prerequisites

1. **Docker** installed on your system
2. **Docker Compose** installed
3. **API Keys** configured in `.env` file

### 1. Setup Environment

```bash
# Copy environment template
cp env.example .env

# Edit with your API keys
nano .env
```

### 2. Run with Setup Script

```bash
# Make script executable
chmod +x scripts/docker-setup.sh

# Development (testnet)
./scripts/docker-setup.sh dev

# Production (live trading)
./scripts/docker-setup.sh prod

# Check status
./scripts/docker-setup.sh status

# Stop containers
./scripts/docker-setup.sh stop
```

## 📁 Docker Files Overview

### Core Files

- **`Dockerfile`** - Main container definition
- **`docker-compose.yml`** - Standard deployment
- **`docker-compose.dev.yml`** - Development environment
- **`docker-compose.prod.yml`** - Production environment
- **`.dockerignore`** - Files to exclude from build

### Scripts

- **`scripts/docker-setup.sh`** - Automated setup and management

## 🚀 Deployment Options

### Development Environment

```bash
# Start development (testnet)
docker-compose -f docker-compose.dev.yml up --build -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop
docker-compose -f docker-compose.dev.yml down
```

**Features:**
- ✅ Testnet trading only
- ✅ Source code mounted for development
- ✅ Debug logging enabled
- ✅ Interactive mode available
- ✅ No resource limits

### Production Environment

```bash
# Start production (live trading)
docker-compose -f docker-compose.prod.yml up --build -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop
docker-compose -f docker-compose.prod.yml down
```

**Features:**
- ⚠️ **LIVE TRADING** - Real money at risk
- 🔒 Security hardening (non-root user, read-only filesystem)
- 📊 Resource limits and monitoring
- 🔄 Automatic restarts
- 💾 Persistent data storage

### Standard Environment

```bash
# Start standard deployment
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔧 Configuration

### Environment Variables

Configure your bot by setting these in `.env`:

```bash
# API Configuration
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
TESTNET=True  # Set to False for live trading

# Trading Configuration
SYMBOL=BTCUSDT
LEVERAGE=5
RISK_PER_TRADE=0.01
STOPLOSS_PCT=0.005
TAKE_PROFIT_PCT=0.01
DAILY_MAX_LOSS_PCT=0.03

# Database
DB_FILE=/app/data/async_bot_state.db

# Logging
LOG_LEVEL=INFO
```

### Volume Mounts

The containers mount these directories:

- **`./data`** → `/app/data` - Database and persistent data
- **`./logs`** → `/app/logs` - Log files
- **`./backups`** → `/backups` - Database backups (production)

## 📊 Monitoring

### Basic Monitoring

```bash
# Check container status
docker-compose ps

# View recent logs
docker-compose logs --tail=50

# Monitor logs in real-time
docker-compose logs -f
```

### Advanced Monitoring (Production)

Enable monitoring stack:

```bash
# Start with monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# Access Grafana
# URL: http://localhost:3000
# Username: admin
# Password: admin (or set GRAFANA_PASSWORD in .env)

# Access Prometheus
# URL: http://localhost:9090
```

## 🔄 Management Commands

### Using Setup Script

```bash
# Show help
./scripts/docker-setup.sh help

# Development
./scripts/docker-setup.sh dev

# Production
./scripts/docker-setup.sh prod

# Test run
./scripts/docker-setup.sh test

# Check status
./scripts/docker-setup.sh status

# Stop all
./scripts/docker-setup.sh stop

# Clean everything
./scripts/docker-setup.sh clean
```

### Manual Docker Commands

```bash
# Build image
docker build -t binance-bot .

# Run container
docker run -d --name bot \
  -e API_KEY=your_key \
  -e API_SECRET=your_secret \
  -e TESTNET=True \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  binance-bot

# View logs
docker logs -f bot

# Stop container
docker stop bot
docker rm bot
```

## 🛡️ Security Features

### Production Security

- **Non-root user**: Container runs as user ID 1000
- **Read-only filesystem**: Root filesystem is read-only
- **Temporary filesystems**: `/tmp` and `/var/tmp` are tmpfs
- **Resource limits**: CPU and memory limits enforced
- **Network isolation**: Custom bridge network
- **Health checks**: Automatic health monitoring

### Data Protection

- **Persistent volumes**: Database and logs persist across restarts
- **Backup service**: Automatic database backups (production)
- **Log rotation**: Log files are automatically rotated
- **Environment isolation**: Separate networks for different environments

## 🔍 Troubleshooting

### Common Issues

#### 1. Container won't start

```bash
# Check logs
docker-compose logs

# Check environment variables
docker-compose config

# Verify .env file
cat .env
```

#### 2. Database connection issues

```bash
# Check database file permissions
ls -la data/

# Reset database (WARNING: loses data)
rm data/async_bot_state.db
docker-compose restart
```

#### 3. API connection issues

```bash
# Verify API keys
docker-compose exec binance-bot env | grep API

# Test API connection
docker-compose exec binance-bot python -c "
from config import Config
print('API_KEY:', 'SET' if Config.API_KEY else 'NOT SET')
"
```

#### 4. Resource issues

```bash
# Check resource usage
docker stats

# Increase limits in docker-compose.yml
# Under deploy.resources.limits
```

### Debug Mode

```bash
# Run in debug mode
docker-compose -f docker-compose.dev.yml run --rm binance-bot-dev python -u improved_bot.py

# Interactive shell
docker-compose -f docker-compose.dev.yml run --rm binance-bot-dev bash
```

## 📈 Performance

### Resource Requirements

- **Development**: 256MB RAM, 0.25 CPU
- **Production**: 512MB RAM, 0.5 CPU
- **With Monitoring**: 1GB RAM, 1.0 CPU

### Optimization Tips

1. **Use production compose file** for better performance
2. **Enable monitoring** for detailed metrics
3. **Regular backups** to prevent data loss
4. **Log rotation** to manage disk space
5. **Resource limits** to prevent resource exhaustion

## 🔄 Updates

### Updating the Bot

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up --build -d

# Or use setup script
./scripts/docker-setup.sh stop
./scripts/docker-setup.sh prod
```

### Updating Dependencies

```bash
# Rebuild with new requirements
docker-compose build --no-cache

# Restart services
docker-compose up -d
```

## 📋 Best Practices

1. **Always test on testnet first**
2. **Use production compose for live trading**
3. **Monitor logs regularly**
4. **Set up automated backups**
5. **Use resource limits**
6. **Keep API keys secure**
7. **Regular security updates**

## 🆘 Support

For issues:

1. Check the logs: `docker-compose logs`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `docker-compose exec bot ping google.com`
4. Check resources: `docker stats`

## ⚠️ Important Notes

- **Live trading involves real money** - test thoroughly first
- **API keys are sensitive** - never commit them to version control
- **Database contains trading data** - backup regularly
- **Monitor the bot** - don't leave it unattended
- **Understand the risks** - trading bots can lose money
