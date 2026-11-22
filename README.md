\# Surat City Bus Delay Prediction — End-to-End MLOps



An end-to-end Machine Learning Operations (MLOps) project for predicting bus delays in Surat, India, using machine learning, automated testing, data versioning, model tracking, containerization, Kubernetes deployment, data drift detection, and drift-triggered retraining.



\## Project Overview



The project predicts bus delays in minutes based on route characteristics, passenger demand, weather, driver experience, and temporal information.



It demonstrates the complete MLOps lifecycle, from dataset preparation and model training to production inference, monitoring, and controlled model updates.



\*\*Project ID:\*\* MP249  

\*\*Application:\*\* SURAT MOVE  

\*\*ML Algorithm:\*\* Random Forest Regressor  

\*\*Production Model:\*\* Version 2  

\*\*Docker Release:\*\* v2.1



\## Technology Stack



| Category | Technologies |

|---|---|

| Programming | Python 3.12 |

| Data Processing | Pandas, NumPy |

| Machine Learning | Scikit-learn |

| Model Tracking | MLflow |

| Data Versioning | DVC |

| Backend | FastAPI |

| Frontend | HTML, CSS, JavaScript |

| Containerization | Docker, Docker Compose |

| Orchestration | Kubernetes |

| CI | GitHub Actions |

| Automated Testing | Pytest |

| Monitoring | Python, Windows Task Scheduler |

| Version Control | Git, GitHub |



\## System Architecture



```text

&#x20;            Bus Delay Datasets

&#x20;                    |

&#x20;                    v

&#x20;           Data Preprocessing

&#x20;                    |

&#x20;                    v

&#x20;             DVC Versioning

&#x20;                    |

&#x20;                    v

&#x20;          Random Forest Training

&#x20;                    |

&#x20;                    v

&#x20;          MLflow Experiment Tracking

&#x20;                    |

&#x20;                    v

&#x20;            Model Registration

&#x20;                    |

&#x20;                    v

&#x20;          FastAPI Prediction API

&#x20;                    |

&#x20;                    v

&#x20;            Docker Container

&#x20;                    |

&#x20;                    v

&#x20;         Kubernetes Deployment

&#x20;                    |

&#x20;                    v

&#x20;           SURAT MOVE Frontend



&#x20;      Monitoring and Retraining Loop

&#x20;                    |

&#x20;                    v

&#x20;            Drift Detection

&#x20;                    |

&#x20;                    v

&#x20;          SHA256 Fingerprint Check

&#x20;                    |

&#x20;                    v

&#x20;      Candidate Model Retraining

&#x20;                    |

&#x20;                    v

&#x20;         Performance Comparison

&#x20;                    |

&#x20;                    v

&#x20;       Manual Deployment Approval

```



\## Dataset and Features



The model uses historical bus operation records.



\*\*Input features:\*\*



\- Route distance (`route\_km`)

\- Number of stops (`stops`)

\- Passenger count (`passengers`)

\- Hour of travel (`hour`)

\- Temperature (`temp\_c`)

\- Driver experience (`driver\_exp\_yr`)

\- Weather conditions (`weather`)

\- Depot zone (`depot\_zone`)

\- Day of week (`day\_of\_week`)

\- Month (`month`)



\*\*Prediction target:\*\* `delay\_min`



The API derives the day of week and month automatically from the provided date.



\## Machine Learning Model



The project uses a `RandomForestRegressor` with preprocessing for numerical and categorical variables.



\### Initial Model Performance



| Metric | Result |

|---|---:|

| MAE | 1.4222 |

| RMSE | 1.7814 |

| R² | 0.8959 |



These are baseline evaluation results.



\### Drift-Triggered Retraining Evaluation



A separate chronological holdout dataset was used to compare the previous production model against a retrained candidate.



| Metric | Previous Model | Retrained Candidate |

|---|---:|---:|

| MAE | 5.4399 | 1.8291 |

| RMSE | 6.6648 | 2.7338 |

| R² | 0.3391 | 0.8888 |



The candidate showed better performance on this holdout and was subsequently promoted through a controlled deployment process.



These metrics are from different evaluation stages and should not be interpreted as directly comparable to the initial baseline metrics.



\## MLflow Experiment Tracking



MLflow is used to:



\- Track training parameters and evaluation metrics

\- Record model artifacts

\- Compare retraining experiments

\- Maintain model versions

\- Support model rollback planning



\*\*Registered model:\*\* `Surat\_Bus\_Delay\_Predictor`



| Alias | Model Version |

|---|---|

| `champion` | 2 |

| `rollback` | 1 |



MLflow aliases identify model versions; changing an alias does not automatically redeploy Kubernetes containers.



\## Data Versioning with DVC



DVC tracks project datasets and enables reproducible data management.



Common commands:



```bash

dvc status

dvc status -c

dvc pull

dvc push

```



The configured remote uses Google Drive. Access to the remote requires appropriate credentials.



\## FastAPI Prediction Service



The application provides these endpoints:



| Endpoint | Method | Description |

|---|---|---|

| `/` | GET | API information |

| `/health` | GET | Health status |

| `/predict` | POST | Predict bus delay |

| `/app/` | GET | SURAT MOVE frontend |



\### Example Prediction Request



```json

{

&#x20; "date": "2026-08-20",

&#x20; "route\_km": 15.5,

&#x20; "stops": 12,

&#x20; "passengers": 45,

&#x20; "hour": 9,

&#x20; "temp\_c": 32,

&#x20; "driver\_exp\_yr": 20,

&#x20; "weather": "Clear",

&#x20; "depot\_zone": "Central"

}

```



\### Example Response



```json

{

&#x20; "predicted\_delay\_minutes": 13.7,

&#x20; "model": "RandomForestRegressor",

&#x20; "model\_version": "2"

}

```



The example response was verified against the Kubernetes-deployed application.



\## Local Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/DevanshLuthra/MP249-Bus-Delay.git

cd MP249-Bus-Delay

```



\### 2. Create a Python Virtual Environment



Windows PowerShell:



```powershell

py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 4. Start the API



```bash

uvicorn src.api:app --host 0.0.0.0 --port 8000

```



Open:



\- API health: `http://localhost:8000/health`

\- API documentation: `http://localhost:8000/docs`

\- Frontend: `http://localhost:8000/app/`



The trained model and required datasets must be available locally. Retrieve DVC-managed data and model artifacts where applicable before starting the service.



\## Docker Deployment



Build the container:



```bash

docker build -t surat-bus-delay:v2.1 .

```



Run the application:



```bash

docker run --rm -p 8000:8000 surat-bus-delay:v2.1

```



The API will be available on port 8000.



Docker Compose configuration is also included in the repository.



\## Kubernetes Deployment



Apply the Kubernetes manifests:



```bash

kubectl apply -f k8s/

```



Verify the deployment:



```bash

kubectl get deployments,pods,services

```



Forward the application service:



```bash

kubectl port-forward service/surat-bus-delay-service 8002:8000

```



Access the frontend:



`http://localhost:8002/app/`



\*\*Note:\*\* The Kubernetes deployment references a locally hosted Docker image registry. Other environments must configure an accessible registry and image reference before deployment.



\## Data Drift Detection



The project detects statistical changes between the reference training data and later observations.



Drift detection includes:



\- Kolmogorov–Smirnov tests for numerical features

\- Chi-square tests for categorical features

\- Per-feature drift reporting

\- Overall drift status



Run manually:



```bash

python src/detect\_drift.py

```



Outputs include:



```text

reports/drift\_report.csv

reports/drift\_summary.json

```



The monitored dataset previously showed drift in `driver\_exp\_yr`.



\## Automated Monitoring and Retraining



The monitoring workflow is implemented in:



```text

scripts/monitor\_drift.py

```



It performs the following operations:



1\. Run drift detection.

2\. Record monitoring results.

3\. Calculate a SHA256 fingerprint of the input datasets.

4\. Skip retraining when the same dataset has already been successfully evaluated.

5\. Trigger candidate retraining when drift is detected on a dataset requiring evaluation.

6\. Compare candidate performance against the existing model.

7\. Save reports and logs.

8\. Require manual approval before production deployment.



Run monitoring manually:



```bash

python scripts/monitor\_drift.py

```



\### Scheduled Monitoring



On the configured Windows development machine, Task Scheduler runs:



\*\*Task:\*\* `SuratBusDelayDriftMonitoring`  

\*\*Schedule:\*\* Daily at 9:00 PM



The task is local to the configured Windows machine and is not automatically installed by cloning the repository.



\### Monitoring Outputs



```text

reports/monitoring\_history.csv

reports/retraining\_state.json

logs/drift\_monitor.log

logs/drift\_alerts.log

reports/retraining\_report.json

```



The retraining state file is intentionally excluded from Git. New installations must initialize or establish their own evaluation state; otherwise, the first drift-triggered evaluation may run again.



\## Continuous Integration



GitHub Actions automatically executes the project's CI workflow on configured repository events.



\*\*Workflow:\*\* `.github/workflows/ci.yml`



The workflow provides automated validation of the project.



GitHub Actions history:



https://github.com/DevanshLuthra/MP249-Bus-Delay/actions



\## Automated Testing



Run the complete test suite:



```bash

python -m pytest tests -v

```



Latest verified local result:



\*\*12 tests passed, 0 failed.\*\*



Tests cover API behavior, data preprocessing, model training, inference, and unseen-category handling.



\## Deployment and Rollback Strategy



The project retains deployment revision history in Kubernetes.



| Revision | Image | Description |

|---|---|---|

| 1 | v1 | Initial model |

| 2 | v2 | Retrained model |

| 3 | v2.1 | Retrained model with corrected API version |



View rollout history:



```bash

kubectl rollout history deployment/surat-bus-delay

```



The previous revisions are available in rollout history. An actual production rollback has not yet been executed as part of final verification.



Model promotion is controlled rather than automatically performed by the monitoring script.



\## Project Highlights



\- End-to-end machine learning pipeline

\- Reproducible data versioning with DVC

\- MLflow experiment tracking and model registry

\- Production inference with FastAPI

\- User-facing SURAT MOVE frontend

\- Docker containerization

\- Kubernetes orchestration

\- GitHub Actions CI

\- Automated data drift detection

\- Drift-triggered candidate retraining

\- SHA256-based duplicate retraining prevention

\- Controlled model promotion and rollback readiness



\## Future Improvements



Potential enhancements include:



\- Live GPS and traffic data ingestion

\- Real-time passenger demand integration

\- Automated cloud deployment

\- Monitoring dashboards and alert notifications

\- Additional model-performance monitoring

\- Distributed data processing

\- Automated approval workflows with deployment safeguards



\## Repository



https://github.com/DevanshLuthra/MP249-Bus-Delay



\*\*Project:\*\* MP249 — Surat City Bus Delay Prediction



\*\*Purpose:\*\* Academic demonstration of an end-to-end MLOps lifecycle.

