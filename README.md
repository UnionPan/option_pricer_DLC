# Options Desk

A comprehensive full-stack options pricing, risk management, and hedging platform for quantitative analysis and trading.

## Overview

Options Desk is a complete quantitative finance platform featuring:

### Core Library (Python)
- **Pricing Models**: Black-Scholes-Merton with analytical Greeks
- **Greeks Calculation**: Delta, gamma, vega, theta, and rho
- **Volatility Surface**: Implied volatility calculation and 3D surface construction
- **Risk Management**: Portfolio risk metrics and limits
- **Delta Hedging**: Automated hedging strategies with configurable rebalancing
- **Backtesting**: Historical simulation of hedging strategies

### Backend API (FastAPI)
- **RESTful API**: Modern, fast API built with FastAPI
- **Real-time Data**: Integration with yfinance for market data
- **Pricing Endpoints**: Calculate option prices and Greeks
- **Surface Building**: Construct volatility surfaces from market data
- **Interactive Docs**: Auto-generated Swagger/OpenAPI documentation

### Frontend (React + TypeScript)
- **Option Chain Viewer**: Interactive option chain tables with real-time data
- **Volatility Surface Visualization**: Interactive 3D surface plots using Plotly
- **Pricing Calculator**: Real-time option pricing and Greeks display
- **Responsive Design**: Works on desktop and mobile devices

### Deployment
- **Docker Support**: Full Docker Compose setup for local development
- **Google Cloud Run**: Production-ready deployment configurations
- **CI/CD**: Automated testing and deployment with GitHub Actions

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd option_pricer_DLC

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:80
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/api/v1/docs
```

### Using Make Commands

```bash
# Install dependencies
make install

# Start backend and frontend separately
make backend    # Terminal 1
make frontend   # Terminal 2

# Run tests
make test

# See all available commands
make help
```

## Project Structure

```
option_pricer_DLC/
├── backend/                    # FastAPI Backend
│   ├── api/v1/endpoints/      # API routes (pricing, data, surface)
│   ├── services/              # Business logic layer
│   ├── schemas/               # Pydantic models
│   └── main.py                # FastAPI app entry point
├── frontend/                   # React + TypeScript Frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API client
│   │   └── types/             # TypeScript types
│   └── public/
├── src/options_desk/          # Core Python Library
│   ├── core/                  # Core data structures
│   ├── pricing/               # Black-Scholes & Greeks
│   ├── surface/               # Volatility surface
│   ├── data/                  # Data fetching
│   └── risk/                  # Risk metrics
├── deployment/
│   ├── docker/                # Dockerfiles for services
│   └── cloudbuild/            # GCP Cloud Build configs
├── .github/workflows/         # CI/CD pipelines
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## Features in Action

### Web Interface

#### Option Chain Viewer
- Real-time option chain data via yfinance
- Separate calls and puts tables
- Strike prices, bid/ask, volume, open interest
- Implied volatility display
- Visual highlighting for ITM/OTM contracts

#### Volatility Surface Builder
- Interactive 3D surface visualization
- Configurable expiration date ranges
- Strike vs. maturity vs. IV plots
- Powered by Plotly for smooth interactions

#### Pricing Calculator
- Black-Scholes option pricing
- Full Greeks calculation (delta, gamma, vega, theta, rho)
- Real-time calculations via API

### REST API

```bash
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

# Response: {"price": 8.02, "model": "black_scholes"}
```

### Python Library

```python
from options_desk.pricing.black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega
)
from options_desk.core.option import OptionType

# Calculate option price
price = black_scholes_price(
    spot=100, strike=105, time_to_expiry=1.0,
    rate=0.05, volatility=0.2, option_type=OptionType.CALL
)

# Calculate Greeks
delta = black_scholes_delta(100, 105, 1.0, 0.05, 0.2, OptionType.CALL)
gamma = black_scholes_gamma(100, 105, 1.0, 0.05, 0.2)
vega = black_scholes_vega(100, 105, 1.0, 0.05, 0.2)

print(f"Price: ${price:.2f}")
print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}")
```

## Documentation

- **[Local Setup Guide](docs/LOCAL_SETUP.md)**: Complete setup instructions for local development
- **[API Documentation](docs/API.md)**: REST API endpoints and examples
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Docker and Google Cloud Run deployment
- **Interactive API Docs**: http://localhost:8000/api/v1/docs (when running)

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **NumPy & SciPy**: Scientific computing
- **yfinance**: Market data

### Frontend
- **React**: UI framework
- **TypeScript**: Type-safe JavaScript
- **Plotly**: Interactive 3D visualizations
- **Axios**: HTTP client

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Google Cloud Run**: Serverless deployment
- **Cloud Build**: CI/CD pipeline
- **GitHub Actions**: Automated testing

## Development

### Running Tests

```bash
# Run all tests
make test

# Backend tests only
make test-backend

# Frontend tests only
make test-frontend
```

### Code Quality

```bash
# Lint all code
make lint

# Format Python code
make format

# Clean build artifacts
make clean
```

## Deployment

### Local Development with Docker

```bash
# Start all services
docker-compose up --build

# Stop services
docker-compose down
```

### Google Cloud Run

```bash
# Deploy backend
make deploy-backend

# Deploy frontend
make deploy-frontend
```

See [Deployment Guide](docs/DEPLOYMENT.md) for complete instructions including:
- GCP project setup
- Workload Identity configuration
- Environment variables
- Cost optimization
- Monitoring and logging

## Roadmap

### Current Features
- Black-Scholes pricing and Greeks
- Real-time option chain data
- Interactive 3D volatility surfaces
- REST API with documentation
- Docker containerization
- Cloud deployment configs

### Coming Soon
- [ ] Binomial tree models for American options
- [ ] Monte Carlo simulation for path-dependent options
- [ ] Portfolio Greeks aggregation
- [ ] Delta hedging backtester
- [ ] Historical volatility analysis
- [ ] Options strategies builder
- [ ] User authentication
- [ ] Database integration
- [ ] WebSocket real-time updates

### Future Enhancements
- [ ] Deep hedging with reinforcement learning
- [ ] Latent volatility factor models
- [ ] Advanced surface interpolation (SVI, SSVI)
- [ ] Multi-asset correlations
- [ ] Risk limits and monitoring

## Screenshots

### Option Chain Viewer
View real-time option chains with calls and puts, including strikes, prices, volume, and implied volatility.

### Volatility Surface
Interactive 3D visualization of implied volatility across strikes and maturities using Plotly.

### API Documentation
Auto-generated interactive API documentation with Swagger UI.

## Use Cases

- **Portfolio Management**: Calculate portfolio Greeks for risk management
- **Derivatives Pricing**: Price custom option structures
- **Research & Education**: Learn about options pricing and volatility surfaces
- **Backtesting**: Test hedging strategies against historical data
- **Market Making**: Monitor implied volatility and identify mispricing

## Contributing

Contributions are welcome! Areas for contribution:
- Additional pricing models (Monte Carlo, binomial trees)
- More exotic option types
- Enhanced surface interpolation methods
- Performance optimizations
- UI/UX improvements
- Documentation and examples

## License

See LICENSE file for details.

## Acknowledgments

Built with:
- Options pricing formulas from Hull's "Options, Futures, and Other Derivatives"
- Volatility surface concepts from Gatheral's "The Volatility Surface"
- Market data provided by Yahoo Finance via yfinance

## Contact

- GitHub Issues: Report bugs and request features
- Email: yp1170l@nyu.edu
- Portfolio: [Your portfolio URL]

---

Made with ❤️ for the quantitative finance community
