#!/bin/bash

# Docker setup script for Binance Futures Trading Bot
# Usage: ./scripts/docker-setup.sh [dev|prod|test]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Function to check environment file
check_env() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        cp env.example .env
        print_warning "Please edit .env file with your API keys before running the bot."
        exit 1
    fi
    
    # Check if API keys are set
    if ! grep -q "API_KEY=your_" .env && ! grep -q "API_KEY=$" .env; then
        print_success "API keys appear to be configured in .env"
    else
        print_error "Please configure your API keys in .env file"
        exit 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data logs backups monitoring/grafana/dashboards monitoring/grafana/datasources
    print_success "Directories created"
}

# Function to build and run development environment
run_dev() {
    print_status "Starting development environment..."
    docker-compose up --build -d
    print_success "Development environment started"
    print_status "View logs: docker-compose logs -f"
    print_status "Stop: docker-compose down"
}

# Function to build and run production environment
run_prod() {
    print_warning "Starting PRODUCTION environment with live trading!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting production environment..."
        # Set TESTNET to false for production
        export TESTNET=False
        docker-compose up --build -d
        print_success "Production environment started"
        print_status "View logs: docker-compose logs -f"
        print_status "Stop: docker-compose down"
    else
        print_status "Production deployment cancelled"
        exit 0
    fi
}

# Function to run tests
run_test() {
    print_status "Starting test environment..."
    docker-compose up --build -d
    print_success "Test environment started"
    print_status "Running tests..."
    sleep 10
    docker-compose logs binance-bot
    print_status "Test completed. Stopping containers..."
    docker-compose down
}

# Function to show status
show_status() {
    print_status "Container status:"
    docker-compose ps
    echo
    print_status "Recent logs:"
    docker-compose logs --tail=20
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev     - Start development environment (testnet)"
    echo "  prod    - Start production environment (live trading)"
    echo "  test    - Run tests and stop"
    echo "  status  - Show container status and logs"
    echo "  stop    - Stop all containers"
    echo "  clean   - Stop and remove all containers and volumes"
    echo "  backup  - Create database backup"
    echo "  monitor - Start with monitoring stack"
    echo "  devtools - Start development tools"
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev     # Start development environment"
    echo "  $0 prod    # Start production environment"
    echo "  $0 status  # Check status"
    echo "  $0 monitor # Start with Grafana/Prometheus"
    echo ""
    echo "Profiles available:"
    echo "  - monitoring: Grafana, Prometheus, monitoring dashboard"
    echo "  - backup: Database backup service"
    echo "  - dev-tools: Database viewer, log viewer"
    echo "  - cache: Redis cache"
    echo "  - postgres: PostgreSQL database"
    echo "  - webhooks: Webhook receiver"
}

# Function to stop containers
stop_containers() {
    print_status "Stopping containers..."
    docker-compose down 2>/dev/null || true
    print_success "Containers stopped"
}

# Function to run backup
run_backup() {
    print_status "Creating database backup..."
    docker-compose --profile backup up backup
    print_success "Backup completed"
}

# Function to run monitoring stack
run_monitoring() {
    print_status "Starting monitoring stack..."
    docker-compose --profile monitoring up -d
    print_success "Monitoring stack started"
    print_status "Grafana: http://localhost:3000 (admin/admin)"
    print_status "Prometheus: http://localhost:9090"
    print_status "Monitoring Dashboard: http://localhost:8081"
}

# Function to run development tools
run_devtools() {
    print_status "Starting development tools..."
    docker-compose --profile dev-tools up -d
    print_success "Development tools started"
    print_status "Database viewer and log viewer are running"
}

# Function to clean everything
clean_all() {
    print_warning "This will remove all containers, volumes, and data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning all containers and volumes..."
        stop_containers
        docker system prune -f
        docker volume prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main script logic
main() {
    print_status "Binance Futures Trading Bot - Docker Setup"
    echo
    
    # Check prerequisites
    check_docker
    check_env
    create_directories
    
    # Parse command
    case "${1:-help}" in
        dev)
            run_dev
            ;;
        prod)
            run_prod
            ;;
        test)
            run_test
            ;;
        status)
            show_status
            ;;
        stop)
            stop_containers
            ;;
        backup)
            run_backup
            ;;
        monitor)
            run_monitoring
            ;;
        devtools)
            run_devtools
            ;;
        clean)
            clean_all
            ;;
        help|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"
