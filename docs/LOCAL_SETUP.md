# Local Development Setup

Complete guide for setting up the Options Desk project locally.

## Prerequisites

- Python 3.9+ ([Download](https://www.python.org/downloads/))
- Node.js 18+ and npm ([Download](https://nodejs.org/))
- Git ([Download](https://git-scm.com/downloads))
- Docker Desktop (optional, for containerized development) ([Download](https://www.docker.com/products/docker-desktop))

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd option_pricer_DLC

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:80
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/api/v1/docs
```

### Option 2: Manual Setup

#### 1. Clone and Setup Python Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd option_pricer_DLC

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Install the options_desk package in development mode
pip install -e .
```

#### 2. Start the Backend

```bash
# From the project root directory
uvicorn backend.main:app --reload --port 8000

# You should see:
# INFO:     Uvicorn running on http://127.0.0.1:8000
```

Test the backend:
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

Visit the API documentation: http://localhost:8000/api/v1/docs

#### 3. Start the Frontend

Open a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Frontend will open at http://localhost:3000
```

## Project Structure Overview

```
option_pricer_DLC/
├── backend/              # FastAPI backend
│   ├── api/             # API routes
│   ├── services/        # Business logic
│   ├── schemas/         # Request/response models
│   └── main.py          # App entry point
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API client
│   │   └── types/       # TypeScript types
│   └── public/
├── src/options_desk/    # Core library
│   ├── pricing/         # Pricing models
│   ├── data/            # Data fetching
│   ├── surface/         # Vol surface
│   └── core/            # Core structures
├── deployment/          # Docker & deployment configs
└── docs/               # Documentation
```

## Testing the Application

### Test Backend Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Calculate option price
curl -X POST http://localhost:8000/api/v1/pricing/price \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100,
    "strike": 105,
    "time_to_expiry": 1.0,
    "volatility": 0.2,
    "risk_free_rate": 0.05,
    "dividend_yield": 0.0,
    "option_type": "call"
  }'

# Get option chain for AAPL
curl -X POST http://localhost:8000/api/v1/data/option-chain \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

### Test Frontend

1. Open http://localhost:3000
2. Navigate to "Option Chain"
3. Enter a symbol (e.g., "AAPL")
4. Click "Fetch Option Chain"
5. Navigate to "Vol Surface"
6. Enter a symbol and build the surface

## Development Workflow

### Backend Development

```bash
# Run with auto-reload (watches for file changes)
uvicorn backend.main:app --reload

# Run tests
pytest tests/

# Lint code
flake8 backend/ src/

# Format code
black backend/ src/
```

### Frontend Development

```bash
cd frontend

# Start dev server (auto-reloads on changes)
npm start

# Run tests
npm test

# Build for production
npm run build

# Lint code
npm run lint
```

## Common Issues and Solutions

### Backend Issues

#### Issue: ModuleNotFoundError: No module named 'options_desk'

**Solution**: Install the package in development mode:
```bash
pip install -e .
```

#### Issue: CORS errors when calling API

**Solution**: Check `backend/core/config.py` and ensure `BACKEND_CORS_ORIGINS` includes `http://localhost:3000`.

#### Issue: ImportError for scipy or numpy

**Solution**: Reinstall scientific packages:
```bash
pip install --force-reinstall numpy scipy
```

### Frontend Issues

#### Issue: Cannot connect to backend

**Solution**: Verify `REACT_APP_API_URL` in `frontend/.env.development`:
```
REACT_APP_API_URL=http://localhost:8000/api/v1
```

#### Issue: Plotly charts not rendering

**Solution**: Ensure plotly is installed:
```bash
cd frontend
npm install plotly.js react-plotly.js @types/react-plotly.js
```

### Docker Issues

#### Issue: Port already in use

**Solution**: Stop other services using ports 8000 or 80:
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

#### Issue: Docker build fails

**Solution**: Clear Docker cache and rebuild:
```bash
docker-compose down
docker system prune -a
docker-compose up --build
```

## Environment Variables

### Backend (.env)

Create a `.env` file in the project root:

```bash
# Backend configuration
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:80
DEFAULT_RISK_FREE_RATE=0.05
ENABLE_CACHE=true
CACHE_TTL_SECONDS=300
```

### Frontend

Already configured in:
- `frontend/.env.development` - For local development
- `frontend/.env.production` - For production builds

## Next Steps

- Read the [API Documentation](API.md)
- Check out the [Deployment Guide](DEPLOYMENT.md)
- Explore example notebooks in `notebooks/`
- Run the test suite
- Start building new features!

## Getting Help

- GitHub Issues: Report bugs and request features
- API Docs: http://localhost:8000/api/v1/docs
- Check logs: `docker-compose logs -f backend` or `docker-compose logs -f frontend`
