# API Documentation

Base URL (local): `http://localhost:8000/api/v1`

## Interactive Documentation

- **Swagger UI**: `/api/v1/docs`
- **ReDoc**: `/api/v1/redoc`

## Authentication

Currently, the API does not require authentication. For production use, consider adding:
- API keys
- OAuth 2.0
- JWT tokens

## Endpoints

### Health Check

#### GET /health

Check if the service is running.

**Response**:
```json
{
  "status": "healthy"
}
```

---

## Pricing Endpoints

### Calculate Option Price

#### POST /api/v1/pricing/price

Calculate option price using Black-Scholes model.

**Request Body**:
```json
{
  "spot": 100.0,
  "strike": 105.0,
  "time_to_expiry": 1.0,
  "volatility": 0.2,
  "risk_free_rate": 0.05,
  "dividend_yield": 0.0,
  "option_type": "call"
}
```

**Response**:
```json
{
  "price": 8.02,
  "model": "black_scholes"
}
```

### Calculate Greeks

#### POST /api/v1/pricing/greeks

Calculate all Greeks for an option.

**Request Body**:
```json
{
  "spot": 100.0,
  "strike": 105.0,
  "time_to_expiry": 1.0,
  "volatility": 0.2,
  "risk_free_rate": 0.05,
  "dividend_yield": 0.0,
  "option_type": "call"
}
```

**Response**:
```json
{
  "delta": 0.4801,
  "gamma": 0.0173,
  "vega": 0.3969,
  "theta": -0.0172,
  "rho": 0.4113
}
```

---

## Data Endpoints

### Get Option Chain

#### POST /api/v1/data/option-chain

Fetch option chain data for a symbol.

**Request Body**:
```json
{
  "symbol": "AAPL",
  "expiration_date": "2024-12-20"
}
```

If `expiration_date` is omitted, returns the nearest expiration.

**Response**:
```json
{
  "symbol": "AAPL",
  "expiration_date": "2024-12-20",
  "spot_price": 175.50,
  "contracts": [
    {
      "strike": 170.0,
      "last_price": 8.50,
      "bid": 8.40,
      "ask": 8.60,
      "volume": 1234,
      "open_interest": 5678,
      "implied_volatility": 0.25,
      "option_type": "call"
    }
  ],
  "fetched_at": "2024-01-15T10:30:00"
}
```

### Get Available Expirations

#### GET /api/v1/data/expirations/{symbol}

Get list of available expiration dates.

**Path Parameters**:
- `symbol`: Ticker symbol (e.g., "AAPL")

**Response**:
```json
{
  "symbol": "AAPL",
  "expirations": [
    "2024-01-19",
    "2024-01-26",
    "2024-02-02",
    "2024-02-16"
  ]
}
```

---

## Volatility Surface Endpoints

### Build Volatility Surface

#### POST /api/v1/surface/build

Build a complete volatility surface.

**Request Body**:
```json
{
  "symbol": "AAPL",
  "spot_price": 175.50,
  "min_expiry_days": 7,
  "max_expiry_days": 180
}
```

**Response**:
```json
{
  "symbol": "AAPL",
  "spot_price": 175.50,
  "surface_points": [
    {
      "strike": 170.0,
      "expiry": 0.0822,
      "implied_vol": 0.25,
      "moneyness": 0.9686
    }
  ],
  "num_expirations": 12,
  "num_strikes": 45
}
```

### Get Volatility Smile

#### POST /api/v1/surface/smile

Get volatility smile for a specific expiration.

**Request Body**:
```json
{
  "symbol": "AAPL",
  "expiration_date": "2024-12-20"
}
```

**Response**:
```json
{
  "symbol": "AAPL",
  "expiration_date": "2024-12-20",
  "time_to_expiry": 0.95,
  "spot_price": 175.50,
  "strikes": [160.0, 165.0, 170.0, 175.0, 180.0],
  "implied_vols": [0.28, 0.26, 0.25, 0.24, 0.26]
}
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

- `200`: Success
- `404`: Resource not found (e.g., symbol has no options)
- `422`: Validation error (invalid request parameters)
- `500`: Internal server error

---

## Rate Limiting

Currently no rate limiting is enforced. For production deployment, consider:

- API Gateway with rate limiting
- Cloud Armor for DDoS protection
- Redis for distributed rate limiting

---

## Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Calculate option price
response = requests.post(
    f"{BASE_URL}/pricing/price",
    json={
        "spot": 100,
        "strike": 105,
        "time_to_expiry": 1.0,
        "volatility": 0.2,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.0,
        "option_type": "call"
    }
)
print(f"Option price: ${response.json()['price']:.2f}")

# Get option chain
response = requests.post(
    f"{BASE_URL}/data/option-chain",
    json={"symbol": "AAPL"}
)
chain = response.json()
print(f"Spot: ${chain['spot_price']:.2f}")
print(f"Contracts: {len(chain['contracts'])}")
```

---

## TypeScript Client Example

```typescript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Calculate Greeks
const response = await axios.post(`${API_BASE_URL}/pricing/greeks`, {
  spot: 100,
  strike: 105,
  time_to_expiry: 1.0,
  volatility: 0.2,
  risk_free_rate: 0.05,
  dividend_yield: 0.0,
  option_type: 'call'
});

const greeks = response.data;
console.log(`Delta: ${greeks.delta.toFixed(4)}`);
console.log(`Gamma: ${greeks.gamma.toFixed(4)}`);
```
