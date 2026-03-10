# Deployment Guide — Milliman IntelliScript ML Inference Service

## Architecture Overview

```
┌──────────────┐     ┌────────────────────────────────────────────────────────────┐
│   GitHub      │     │  AWS Cloud                                                │
│   Actions     │     │                                                           │
│               │     │  ┌─────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  push to main ├────►│  │  ECR    │    │     ALB      │    │  ECS Fargate     │  │
│               │     │  │  Repo   │    │  (Internet-  │    │  ┌────────────┐  │  │
│  1. Test      │     │  │         │    │   facing)    │    │  │ Container  │  │  │
│  2. Build     │     │  │  Docker ├───►│              ├───►│  │  FastAPI   │  │  │
│  3. Push      │     │  │  Image  │    │  Port 80     │    │  │  Uvicorn   │  │  │
│  4. Deploy    │     │  │         │    │              │    │  │  Port 8080 │  │  │
│  5. Smoke     │     │  └─────────┘    └──────────────┘    │  └────────────┘  │  │
│     Test      │     │                                      │  ┌────────────┐  │  │
│               │     │  ┌──────────────────────────────┐   │  │ Container  │  │  │
└──────────────┘     │  │  CloudWatch Logs             │   │  │  (replica) │  │  │
                      │  │  /ecs/ml-inference-service   │   │  └────────────┘  │  │
                      │  └──────────────────────────────┘   └──────────────────┘  │
                      │                                                           │
                      │  Auto Scaling: 2-6 tasks, CPU target 70%                  │
                      └────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI + Uvicorn | High-performance async inference endpoint |
| Container | Docker (multi-stage) | Reproducible, minimal production image |
| Registry | Amazon ECR | Stores versioned Docker images |
| Compute | ECS Fargate | Serverless container orchestration |
| Load Balancer | Application Load Balancer | HTTP routing, health checks, HA |
| IaC | CloudFormation | Declarative infrastructure management |
| CI/CD | GitHub Actions | Automated test → build → deploy pipeline |
| Logging | CloudWatch + Loguru | Centralized structured logging |
| Auto Scaling | Application Auto Scaling | CPU-based horizontal scaling (2–6 tasks) |

---

## Prerequisites

1. **AWS Account** with permissions for ECR, ECS, ALB, IAM, VPC, CloudWatch
2. **GitHub Repository** with the following secrets configured:
   - `AWS_ROLE_ARN` — ARN of an IAM role configured for GitHub OIDC federation
3. **Local tools** (for development):
   - [UV](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
   - [Docker](https://docs.docker.com/get-docker/) — Container runtime
   - [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

---

## Local Development

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd ml-engineer-exam

# Install dependencies
uv sync --all-extras

# Run existing tests
uv run pytest -v
```

### Run the API Locally

```bash
# Start the FastAPI server
uv run uvicorn ml_engineer_exam.api.app:app --reload --port 8080

# Test health endpoint
curl http://localhost:8080/health

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "linear",
    "features": {
      "MedInc": 1.6812,
      "HouseAge": 25.0,
      "AveRooms": 4.192200557103064,
      "AveBedrms": 1.0222841225626742,
      "Population": 1392.0,
      "AveOccup": 3.877437325905293,
      "Latitude": 36.06,
      "Longitude": -119.01
    }
  }'

# Batch prediction
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ridge",
    "features": [
      {
        "MedInc": 1.6812, "HouseAge": 25.0, "AveRooms": 4.19,
        "AveBedrms": 1.02, "Population": 1392.0, "AveOccup": 3.88,
        "Latitude": 36.06, "Longitude": -119.01
      },
      {
        "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.98,
        "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.56,
        "Latitude": 37.88, "Longitude": -122.23
      }
    ]
  }'

# Interactive API docs
open http://localhost:8080/docs
```

### Run with Docker

```bash
# Build the image
docker build -t ml-inference-service .

# Run the container
docker run -p 8080:8080 ml-inference-service

# Test
curl http://localhost:8080/health
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Service info and links |
| `GET`  | `/health` | Health check — returns status and loaded models |
| `GET`  | `/models` | List available models with metadata |
| `POST` | `/predict` | Single prediction with model selection |
| `POST` | `/predict/batch` | Batch predictions (multiple inputs) |
| `GET`  | `/docs` | Interactive Swagger/OpenAPI documentation |
| `GET`  | `/redoc` | ReDoc API documentation |

### Request/Response Examples

**POST /predict**
```json
// Request
{
  "model_name": "linear",
  "features": {
    "MedInc": 1.6812,
    "HouseAge": 25.0,
    "AveRooms": 4.192,
    "AveBedrms": 1.022,
    "Population": 1392.0,
    "AveOccup": 3.877,
    "Latitude": 36.06,
    "Longitude": -119.01
  }
}

// Response
{
  "model_name": "linear",
  "prediction": 0.7191,
  "input_features": { ... }
}
```

---

## CI/CD Pipeline

### Workflow: CI (`.github/workflows/ci.yml`)

Triggered on every push and pull request:
1. **Install** dependencies with UV
2. **Run** pytest test suite
3. **Lint** with ruff (informational)

### Workflow: Deploy (`.github/workflows/deploy.yml`)

Triggered on push to `main` or manual dispatch:

```
Push to main
    │
    ▼
┌─── Test ──────────────────────┐
│  uv sync → pytest             │
└──────────────┬────────────────┘
               │ pass
               ▼
┌─── Build & Push ──────────────┐
│  docker build → ECR push      │
│  Tags: sha-<hash>, latest     │
└──────────────┬────────────────┘
               │
               ▼
┌─── Deploy ────────────────────┐
│  CloudFormation deploy        │
│  Wait for ECS stability       │
│  Smoke test /health + /predict│
└───────────────────────────────┘
```

### Setting Up GitHub OIDC for AWS

```bash
# 1. Create OIDC Identity Provider in AWS IAM
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1

# 2. Create IAM Role with trust policy for your repo
# See: https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services

# 3. Add AWS_ROLE_ARN to GitHub repository secrets
```

---

## AWS Deployment (Manual)

### First-Time Setup

```bash
# 1. Deploy infrastructure
aws cloudformation deploy \
  --template-file infrastructure/cloudformation.yaml \
  --stack-name ml-inference-service-production \
  --parameter-overrides \
    EnvironmentName=production \
    ServiceName=ml-inference-service \
  --capabilities CAPABILITY_NAMED_IAM

# 2. Get ECR repository URI
ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name ml-inference-service-production \
  --query "Stacks[0].Outputs[?OutputKey=='ECRRepositoryUri'].OutputValue" \
  --output text)

# 3. Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker build -t $ECR_URI:latest .
docker push $ECR_URI:latest

# 4. Update ECS service to pull the new image
aws ecs update-service \
  --cluster ml-inference-service-cluster \
  --service ml-inference-service \
  --force-new-deployment

# 5. Get the service URL
aws cloudformation describe-stacks \
  --stack-name ml-inference-service-production \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
  --output text
```

---

## Design Decisions & Rationale

### Why FastAPI?
- **Performance**: Built on Starlette/uvicorn, one of the fastest Python web frameworks
- **Auto-documentation**: Generates OpenAPI/Swagger docs from Pydantic models
- **Type safety**: Native Pydantic integration for request validation
- **Industry standard**: Widely adopted for ML inference services

### Why ECS Fargate?
- **Serverless containers**: No EC2 instance management
- **Consistent latency**: Always-on containers (vs. Lambda cold starts for ML models)
- **Right-sized**: 512 CPU / 1024 MB memory handles scikit-learn models efficiently
- **Auto-scaling**: CPU-based scaling from 2–6 tasks keeps costs proportional to load
- **Health checks**: ALB + container health checks ensure high availability

### Why CloudFormation?
- **AWS-native**: No third-party tooling required
- **Declarative**: Full infrastructure defined in a single YAML template
- **Drift detection**: AWS can detect manual changes to managed resources
- **Rollback**: Automatic rollback on deployment failures

### Why Multi-Stage Docker Build?
- **Smaller image**: Build dependencies excluded from production image
- **Security**: Non-root user, minimal attack surface
- **Reproducibility**: UV ensures deterministic dependency resolution

---

## Recommended Next Steps

If given more time, the following improvements would strengthen this workflow:

### Security
- [ ] Add HTTPS with ACM certificate on the ALB
- [ ] Implement API key / JWT authentication middleware
- [ ] Store models in S3 with versioning instead of bundling in Docker image
- [ ] Add WAF rules on the ALB for rate limiting / IP filtering
- [ ] Use AWS Secrets Manager for any future API keys or DB credentials

### Observability
- [ ] Add structured JSON logging with request correlation IDs
- [ ] Create CloudWatch dashboards for latency, error rate, throughput
- [ ] Set up CloudWatch Alarms (p99 latency > 500ms, error rate > 1%)
- [ ] Add X-Ray tracing for request-level profiling
- [ ] Implement Prometheus metrics endpoint (`/metrics`)

### ML Operations
- [ ] Model registry in S3 with version tracking (or MLflow)
- [ ] A/B testing support — route traffic between model versions
- [ ] Model performance monitoring — track prediction drift
- [ ] Automated retraining pipeline triggered by data drift
- [ ] Shadow mode for new model validation before promotion

### Infrastructure
- [ ] Add staging environment with separate CloudFormation stack
- [ ] Implement blue/green ECS deployments via CodeDeploy
- [ ] Add private subnets + NAT gateway for production hardening
- [ ] Set up VPC endpoints for ECR/S3/CloudWatch (reduce data transfer costs)
- [ ] Terraform alternative for multi-cloud portability

### Testing
- [ ] Integration tests that spin up Docker container and hit endpoints
- [ ] Load testing with Locust or k6 to validate scaling behavior
- [ ] Contract testing for API schema stability
- [ ] Canary deployments with automated rollback on error spike
