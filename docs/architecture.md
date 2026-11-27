# MP249 - System Architecture

## 1. End-to-End MLOps Pipeline

```mermaid
flowchart TD
    A[Training Dataset] --> B[DVC Versioning]
    B --> C[Data Quality Audit]
    C --> D[Cleaning and Preprocessing]
    D --> E[Random Forest Experiments]
    E --> F[MLflow Tracking]
    F --> G[Model Registry]
    G --> H[Champion V2]
    H --> I[Deployment Artifact]
    I --> J[FastAPI]
    J --> K[Docker]
    K --> L[Kubernetes]
    M[Web Interface] <--> J
```

## 2. Prediction Architecture

```mermaid
flowchart LR
    A[User Input] --> B[Frontend]
    B --> C[FastAPI]
    C --> D[Pydantic Validation]
    D --> E[Preprocessing Pipeline]
    E --> F[Random Forest V2]
    F --> G[Predicted Delay]
    G --> B
    F --> H[JSON Prediction Logs]
```

## 3. Kubernetes Architecture

```mermaid
flowchart TD
    A[User Browser] --> B[Port Forward 8005]
    B --> C[mp249-svc ClusterIP]
    C --> D[Pod 1]
    C --> E[Pod 2]
    C --> F[Pod 3]
    D --> G[Champion Model]
    E --> G
    F --> G
```

Each pod runs the mp249-api:1.0 image on port 8000.

Readiness and liveness probes use the /health endpoint.

## 4. Drift and Retraining Architecture

```mermaid
flowchart TD
    A[Reference Training Data] --> C[PSI Calculation]
    B[Later Dataset] --> C
    C --> D{PSI above 0.25?}
    D -->|No| E[Record Monitoring Results]
    D -->|Yes| F[Generate Drift Alert]
    F --> G[Manual Candidate Evaluation]
    G --> H[Chronological 70-30 Split]
    H --> I[Train Candidate]
    I --> J[Compare Check-set MAE]
    J --> K{Improvement at least 5 percent?}
    K -->|Yes| L[Promote Candidate]
    K -->|No| M[Retain Champion]
    L --> N[Export Model and Redeploy]
```

Monitoring does not automatically trigger model replacement.

## 5. Continuous Integration

```mermaid
flowchart LR
    A[Git Push] --> B[GitHub Actions]
    B --> C[Python 3.12]
    C --> D[Install Dependencies]
    D --> E[Run Pytest]
    E --> F{Tests Pass?}
    F -->|Yes| G[Successful CI]
    F -->|No| H[Fix Errors]
```

## 6. Implementation Components

| Component | Implementation |
|---|---|
| Configuration | params.yaml |
| Data audit | src/audit_data.py |
| Data cleaning | src/clean_data.py |
| Experiment tracking | src/train_experiments.py |
| Model registration | src/register_models.py |
| Drift detection | src/psi_monitor.py |
| Monitoring alerts | scripts/monitor_drift.py |
| Holdout prediction | src/predict_holdout.py |
| API entry point | app/main.py |
| API implementation | src/api.py |
| Frontend | frontend/ |
| Deployment artifact | models/model.joblib |
| Containerization | Dockerfile |
| Orchestration | k8s/ |
| Continuous integration | .github/workflows/ci.yml |

## 7. Verified Deployment Results

- Champion model: Version 2
- Docker image: mp249-api:1.0
- Kubernetes deployment: mp249-api
- Kubernetes service: mp249-svc
- Ready replicas: 3
- Pod restarts at final verification: 0
- Automated tests: 16 passed
- Example prediction: 13.62 minutes
- Check-set MAE improvement: 62.98 percent

## 8. Deployment Controls

Model promotion and application deployment are separate operations.

A registry alias change does not automatically update the deployed model file.

The deployed artifact must be updated and the container rebuilt before deploying a new model version.

Historical MLflow artifact paths are machine-specific. Portable model exports are provided separately.
