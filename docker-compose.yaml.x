version: '3.8'

services:
  binance-bot:
    build: .
    container_name: binance-futures-bot-prod
    restart: unless-stopped
    environment:
      # API Configuration
      - API_KEY=${API_KEY}
      - API_SECRET=${API_SECRET}
      - TESTNET=False
      
      # Trading Configuration
      - SYMBOL=${SYMBOL:-BTCUSDT}
      - LEVERAGE=${LEVERAGE:-5}
      - RISK_PER_TRADE=${RISK_PER_TRADE:-0.01}
      - STOPLOSS_PCT=${STOPLOSS_PCT:-0.005}
      - TAKE_PROFIT_PCT=${TAKE_PROFIT_PCT:-0.01}
      - DAILY_MAX_LOSS_PCT=${DAILY_MAX_LOSS_PCT:-0.03}
      
      # Database Configuration
      - DB_FILE=/app/data/async_bot_state.db
      
      # Logging
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    
    volumes:
      # Persist database and logs
      - bot-data:/app/data
      - bot-logs:/app/logs
    
    networks:
      - bot-network
    
    # Resource limits for production
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
    
    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import sqlite3; sqlite3.connect('/app/data/async_bot_state.db')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
    
    # Security: Run as non-root
    user: "1000:1000"
    
    # Security: Read-only root filesystem
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp

  # Database backup service
  backup:
    image: alpine:latest
    container_name: bot-backup
    restart: "no"
    volumes:
      - bot-data:/data:ro
      - ./backups:/backups
    command: |
      sh -c "
        apk add --no-cache sqlite
        cp /data/async_bot_state.db /backups/bot_backup_$$(date +%Y%m%d_%H%M%S).db
        echo 'Backup completed at $$(date)'
      "
    profiles:
      - backup

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: bot-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - bot-network
    profiles:
      - monitoring

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: bot-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - bot-network
    profiles:
      - monitoring

networks:
  bot-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  bot-data:
    driver: local
  bot-logs:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
