.PHONY: help install dev backend frontend docker-up docker-down test clean docs

help:
	@echo "Options Desk - Available Commands"
	@echo "=================================="
	@echo "install       - Install all dependencies"
	@echo "dev           - Start backend and frontend for development"
	@echo "backend       - Start backend server"
	@echo "frontend      - Start frontend dev server"
	@echo "docker-up     - Start all services with Docker Compose"
	@echo "docker-down   - Stop all Docker services"
	@echo "test          - Run all tests"
	@echo "test-backend  - Run backend tests"
	@echo "test-frontend - Run frontend tests"
	@echo "lint          - Lint all code"
	@echo "format        - Format all code"
	@echo "clean         - Clean build artifacts"
	@echo "docs          - Open API documentation"

install:
	@echo "Installing backend dependencies..."
	pip install -r requirements.txt
	pip install -r backend/requirements.txt
	pip install -e .
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Installation complete!"

backend:
	@echo "Starting backend server..."
	uvicorn backend.main:app --reload --port 8000

frontend:
	@echo "Starting frontend dev server..."
	cd frontend && npm start

docker-up:
	@echo "Starting all services with Docker Compose..."
	docker-compose up --build

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

test: test-backend test-frontend

test-backend:
	@echo "Running backend tests..."
	pytest tests/ --cov=options_desk --cov-report=term-missing

test-frontend:
	@echo "Running frontend tests..."
	cd frontend && npm test -- --watchAll=false

lint:
	@echo "Linting backend..."
	flake8 backend/ src/ --count --max-line-length=100
	@echo "Linting frontend..."
	cd frontend && npm run lint || true

format:
	@echo "Formatting backend code..."
	black backend/ src/ tests/
	@echo "Formatting complete!"

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
	cd frontend && rm -rf build/ node_modules/.cache/
	@echo "Clean complete!"

docs:
	@echo "Opening API documentation..."
	open http://localhost:8000/api/v1/docs || xdg-open http://localhost:8000/api/v1/docs

deploy-backend:
	@echo "Deploying backend to Cloud Run..."
	gcloud builds submit --config=deployment/cloudbuild/backend.yaml

deploy-frontend:
	@echo "Deploying frontend to Cloud Run..."
	gcloud builds submit --config=deployment/cloudbuild/frontend.yaml
