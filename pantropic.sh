#!/bin/bash
#
# Pantropic - Intelligent Local LLM Server
# Service management script
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/.venv"
PYTHON="${VENV_PATH}/bin/python"
LOG_FILE="${SCRIPT_DIR}/pantropic.log"
PID_FILE="${SCRIPT_DIR}/pantropic.pid"
HOST="${PANTROPIC_HOST:-0.0.0.0}"
PORT="${PANTROPIC_PORT:-8090}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[Pantropic]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[Pantropic]${NC} $1"
}

print_error() {
    echo -e "${RED}[Pantropic]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[Pantropic]${NC} $1"
}

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        pgrep -f "pantropic.main" 2>/dev/null | head -1 || echo ""
    fi
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

wait_for_health() {
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    return 1
}

start_server() {
    if is_running; then
        print_warning "Pantropic is already running (PID: $(get_pid))"
        return 0
    fi

    print_status "Starting Pantropic server..."

    # Check virtual environment
    if [ ! -f "$PYTHON" ]; then
        print_error "Virtual environment not found at $VENV_PATH"
        print_error "Run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
        exit 1
    fi

    # Start server with nohup
    cd "$SCRIPT_DIR"
    nohup "$PYTHON" -m pantropic.main > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"

    print_status "Waiting for server to be ready..."

    if wait_for_health; then
        print_success "Pantropic started successfully!"
        print_success "  PID: $pid"
        print_success "  URL: http://localhost:${PORT}"
        print_success "  Log: $LOG_FILE"
    else
        print_error "Server failed to start. Check logs:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
}

stop_server() {
    if ! is_running; then
        print_warning "Pantropic is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid=$(get_pid)
    print_status "Stopping Pantropic (PID: $pid)..."

    kill "$pid" 2>/dev/null || true

    # Wait for graceful shutdown
    local attempts=0
    while [ $attempts -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        attempts=$((attempts + 1))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        print_warning "Force killing..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    print_success "Pantropic stopped"
}

restart_server() {
    print_status "Restarting Pantropic..."
    stop_server
    sleep 2
    start_server
}

status_server() {
    if is_running; then
        local pid=$(get_pid)
        print_success "Pantropic is running (PID: $pid)"

        # Get health info
        local health=$(curl -s "http://localhost:${PORT}/health" 2>/dev/null)
        if [ -n "$health" ]; then
            echo ""
            echo "Health status:"
            echo "$health" | python3 -m json.tool 2>/dev/null || echo "$health"
        fi
    else
        print_warning "Pantropic is not running"
        exit 1
    fi
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        print_error "Log file not found: $LOG_FILE"
        exit 1
    fi
}

scan_models() {
    print_status "Scanning models..."
    cd "$SCRIPT_DIR"
    "$PYTHON" -m pantropic.main --scan-only
    print_success "Models scanned successfully"
}

show_help() {
    echo "Pantropic - Intelligent Local LLM Server"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|logs|scan|help}"
    echo ""
    echo "Commands:"
    echo "  start    Start Pantropic in background"
    echo "  stop     Stop Pantropic"
    echo "  restart  Restart Pantropic"
    echo "  status   Show server status and health"
    echo "  logs     Tail the log file"
    echo "  scan     Rescan models directory"
    echo "  help     Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  PANTROPIC_HOST  Server host (default: 0.0.0.0)"
    echo "  PANTROPIC_PORT  Server port (default: 8090)"
}

# Main
case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        show_logs
        ;;
    scan)
        scan_models
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
