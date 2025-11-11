# Options Desk

Options Desk is a full-stack options analytics playground: a FastAPI backend serving pricing/Greeks endpoints, a React dashboard for visualization, and a reusable Python library that powers both.

## Highlights
- **Quant core** – Black-Scholes pricing + Greeks, implied volatility, surface builders, and risk utilities packaged under `src/options_desk`.
- **FastAPI backend** – REST endpoints for pricing, chains, and surface generation with interactive docs at `/api/v1/docs`.
- **React frontend** – Option chain explorer, pricing widgets, and 3D Plotly volatility surfaces backed by the API.
- **Container-first** – Dockerfiles for backend and frontend, docker-compose for local dev, Cloud Build + Cloud Run configs for production, and GitHub Actions for CI/CD.

## Run Locally
```bash
# Clone and enter the repo
git clone git@github.com:UnionPan/option_pricer_DLC.git
cd option_pricer_DLC

# Recommended: run everything via Docker
docker-compose up --build
```
Services then listen on:
- Frontend → `http://localhost`
- Backend → `http://localhost:8000` (docs at `/api/v1/docs`)

Prefer Make targets?
```bash
make install     # pip + npm deps
make backend     # uvicorn backend.main:app --reload
make frontend    # React dev server
make test        # backend + frontend tests
```

## Deploy to Cloud Run
1. Authenticate `gcloud`, set your project/region, and enable Cloud Build + Cloud Run.
2. Backend: `gcloud builds submit --config=deployment/cloudbuild/backend.yaml --substitutions=COMMIT_SHA=$(git rev-parse HEAD)`
3. Frontend: update `frontend/.env.production` with the backend URL, then run the matching `gcloud builds submit` using `deployment/cloudbuild/frontend.yaml`.
4. Optional: map a custom domain with `gcloud run domain-mappings create ...`

GitHub Actions (`.github/workflows/deploy-*.yml`) run the same Cloud Build scripts on pushes to `main` once Workload Identity is configured.

## Project Layout (short)
```
backend/      FastAPI app (routers, schemas, services)
frontend/     React + TypeScript UI
src/          Reusable options_desk Python package
deployment/   Dockerfiles + Cloud Build configs
docs/         Local setup, API, deployment guides
.github/      CI workflows
```

## Documentation & Links
- Local setup: `docs/LOCAL_SETUP.md`
- API reference: `docs/API.md`
- Deployment guide: `docs/DEPLOYMENT.md`
- Live API docs (when backend runs): `http://localhost:8000/api/v1/docs`

Happy hedging!
