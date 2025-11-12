# Deployment Guide

This guide covers deploying the Options Desk application using Docker and Google Cloud Run.

## Quick Start with Docker Compose

The easiest way to run the entire stack locally:

```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:80
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/api/v1/docs
```

## Local Development

### Backend (FastAPI)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -e .

# Run the backend
uvicorn backend.main:app --reload --port 8000

# Access API docs
open http://localhost:8000/api/v1/docs
```

### Frontend (React)

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Access frontend
open http://localhost:3000
```

## Google Cloud Platform Deployment

### Prerequisites

1. GCP account with billing enabled
2. `gcloud` CLI installed and authenticated
3. Docker installed locally

### Setup

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Deploy Backend

```bash
# Submit build and deploy to Cloud Run
gcloud builds submit \
  --config=deployment/cloudbuild/backend.yaml \
  --substitutions=COMMIT_SHA=$(git rev-parse HEAD)

# Get the backend URL
gcloud run services describe options-desk-backend \
  --region=us-central1 \
  --format='value(status.url)'
```

### Deploy Frontend

```bash
# Update frontend/.env.production with backend URL
# REACT_APP_API_URL=https://your-backend-url.run.app/api/v1

# Submit build and deploy to Cloud Run
gcloud builds submit \
  --config=deployment/cloudbuild/frontend.yaml \
  --substitutions=COMMIT_SHA=$(git rev-parse HEAD)

# Get the frontend URL
gcloud run services describe options-desk-frontend \
  --region=us-central1 \
  --format='value(status.url)'
```

### App Engine with IAP (NYU Policy)

The NYU security baseline requires every App Engine app to sit behind Google authentication. The backend already enforces Identity-Aware Proxy (IAP) tokens when `ENFORCE_IAP_AUTH=true`, so deployment boils down to enabling App Engine + IAP and assigning access.

```bash
# Enable additional services
gcloud services enable appengine.googleapis.com iap.googleapis.com

# Create the App Engine application once per project
gcloud app create --region=us-central

# Deploy backend + frontend using the App Engine Cloud Build configs
COMMIT_SHA=$(git rev-parse HEAD)
gcloud builds submit --config=deployment/cloudbuild/backend-appengine.yaml --substitutions=COMMIT_SHA=$COMMIT_SHA
gcloud builds submit --config=deployment/cloudbuild/frontend-appengine.yaml --substitutions=COMMIT_SHA=$COMMIT_SHA

# Turn on IAP for the App Engine application
gcloud iap web enable --resource-type=app-engine --project $PROJECT_ID

# Grant NYU identities access (repeat per user/group/service account)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:yp1170@nyu.edu" \
  --role="roles/iap.httpsResourceAccessor"
```

Only identities with `roles/iap.httpsResourceAccessor` (or a group that contains that role) can reach the UI/API. All other requests are rejected before they hit your code, and the FastAPI middleware double-checks that every inbound request carries a valid `X-Goog-IAP-JWT-Assertion` header.

To manually exercise the backend while the policy is active:

```bash
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" \
  "https://REGION-dot-PROJECT_ID.uc.r.appspot.com/api/v1/health"
```

The middleware automatically skips health checks hitting `/health`, and the IAP audience string is derived from the metadata server. If you prefer to override it manually, set the `IAP_AUDIENCE` environment variable to `/projects/PROJECT_NUMBER/apps/PROJECT_ID`.

## GitHub Actions CI/CD

### Setup GitHub Secrets

Configure these secrets in your GitHub repository:

1. `GCP_PROJECT_ID`: Your GCP project ID
2. `GCP_WORKLOAD_IDENTITY_PROVIDER`: Workload Identity Provider for GitHub Actions
3. `GCP_SERVICE_ACCOUNT`: Service account email with Cloud Run permissions

### Workload Identity Setup

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"

# Create workload identity pool
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create workload identity provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Link service account
gcloud iam service-accounts add-iam-policy-binding \
  "github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_USERNAME/option_pricer_DLC"
```

### Automated Deployment

Once configured, deployments happen automatically:

- Push to `main` branch → Triggers deployment to Cloud Run
- Pull requests → Runs CI tests only
- Manual trigger available via GitHub Actions UI

## Environment Variables

### Backend

- `BACKEND_CORS_ORIGINS`: Allowed CORS origins (comma-separated)
- `DEFAULT_RISK_FREE_RATE`: Default risk-free rate (default: 0.05)
- `ENABLE_CACHE`: Enable response caching (default: true)
- `CACHE_TTL_SECONDS`: Cache TTL in seconds (default: 300)
- `ENFORCE_IAP_AUTH`: Set to `true` on App Engine to require Google IAP tokens
- `IAP_AUDIENCE`: Optional override for the expected IAP audience (`/projects/<number>/apps/<id>`)
- `IAP_EXEMPT_PATHS`: JSON list of path prefixes that can bypass IAP (defaults to `["/health"]`)

### Frontend

- `REACT_APP_API_URL`: Backend API URL

## Monitoring and Logs

### View Cloud Run Logs

```bash
# Backend logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=options-desk-backend" \
  --limit 50 \
  --format json

# Frontend logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=options-desk-frontend" \
  --limit 50 \
  --format json
```

### View Metrics

```bash
# Open Cloud Run metrics in browser
gcloud run services describe options-desk-backend \
  --region=us-central1 \
  --format='value(status.url)' | xargs -I {} open "https://console.cloud.google.com/run"
```

## Cost Optimization

Cloud Run pricing:

- **Backend**: ~$0.00002400 per request (2GB RAM, 2 CPU)
- **Frontend**: ~$0.00000400 per request (512MB RAM, 1 CPU)
- **Free tier**: 2 million requests/month

Estimated monthly cost for moderate usage (10K requests/day):

- Backend: ~$7/month
- Frontend: ~$1/month
- Total: ~$8/month

## Troubleshooting

### Backend won't start

```bash
# Check logs
docker logs options-desk-backend

# Verify environment variables
docker exec options-desk-backend env

# Test locally
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend can't connect to backend

1. Verify `REACT_APP_API_URL` in `.env.development` or `.env.production`
2. Check CORS settings in backend `config.py`
3. Verify backend is accessible: `curl http://localhost:8000/health`

### Cloud Run deployment fails

1. Check Cloud Build logs: `gcloud builds list --limit=10`
2. Verify service account permissions
3. Check Docker build locally: `docker build -f deployment/docker/backend/Dockerfile .`
