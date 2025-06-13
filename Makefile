# Makefile for Claude OpenAI Bridge

.PHONY: help build run stop clean logs shell test lint format

# Default target
help:
	@echo "Claude OpenAI Bridge - Docker Commands"
	@echo "======================================"
	@echo "make build       - Build Docker image"
	@echo "make run         - Run with docker-compose"
	@echo "make run-dev     - Run in development mode with hot reload"
	@echo "make run-prod    - Run in production mode with Nginx"
	@echo "make stop        - Stop all containers"
	@echo "make clean       - Stop and remove all containers, volumes"
	@echo "make logs        - View container logs"
	@echo "make shell       - Open shell in API container"
	@echo "make test        - Run tests"
	@echo "make lint        - Run linting"
	@echo "make format      - Format code with black"

# Build Docker image
build:
	docker-compose build

# Run in default mode (web-only security)
run:
	docker-compose up -d
	@echo "Claude OpenAI Bridge is running at http://localhost:8000"
	@echo "View logs with: make logs"

# Run in development mode
run-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run in production mode with Nginx
run-prod:
	docker-compose --profile production up -d
	@echo "Claude OpenAI Bridge is running at http://localhost"

# Run with Redis for distributed sessions
run-redis:
	docker-compose --profile redis up -d

# Stop all containers
stop:
	docker-compose --profile production --profile redis down

# Clean everything (containers, volumes, networks)
clean:
	docker-compose --profile production --profile redis down -v
	docker system prune -f

# View logs
logs:
	docker-compose logs -f claude-api

# Open shell in container
shell:
	docker-compose exec claude-api /bin/bash

# Run tests
test:
	docker-compose exec claude-api python -m pytest

# Lint code
lint:
	docker-compose exec claude-api flake8 .

# Format code
format:
	docker-compose exec claude-api black .

# Check API health
health:
	curl -s http://localhost:8000/health | python -m json.tool

# Quick API test
test-api:
	@echo "Testing Claude OpenAI Bridge API..."
	@curl -X POST http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Say hello"}]}'