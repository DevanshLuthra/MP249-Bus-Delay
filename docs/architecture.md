\# MP249 — Surat City Bus Delay Prediction



\## End-to-End MLOps Architecture



```mermaid

flowchart TD

&#x20;   A\["Bus Delay Datasets"] --> B\["Data Cleaning"]

&#x20;   B --> C\["DVC Data Versioning"]

&#x20;   C --> D\["Random Forest Model Training"]

&#x20;   D --> E\["MLflow Experiment Tracking"]

&#x20;   E --> F\["MLflow Model Registry"]

&#x20;   F --> G\["FastAPI Prediction Service"]

&#x20;   G --> H\["Docker Container"]

&#x20;   H --> I\["Kubernetes Deployment"]

&#x20;   I --> J\["SURAT MOVE Web Interface"]



&#x20;   C --> K\["Statistical Drift Detection"]

&#x20;   K --> L\["Scheduled Monitoring"]

&#x20;   L --> M{"Drift and Unevaluated Dataset?"}

&#x20;   M -->|No| N\["Log Results and Skip Retraining"]

&#x20;   M -->|Yes| O\["Candidate Model Retraining"]

&#x20;   O --> P\["Model Performance Comparison"]

&#x20;   P --> Q\["Manual Deployment Approval"]

&#x20;   Q -.->|Approved| H



&#x20;   R\["GitHub Actions CI"] -.-> G

```



\## Architecture Components



| Component | Implementation |

|---|---|

| Data preprocessing | `src/clean\_data.py` |

| Model training | `src/train\_model.py` |

| Model registry | MLflow |

| Data versioning | DVC |

| Prediction API | `src/api.py` |

| Frontend | `frontend/` |

| Containerization | Docker |

| Orchestration | Kubernetes |

| Drift detection | `src/detect\_drift.py` |

| Scheduled monitoring | `scripts/monitor\_drift.py` |

| Candidate retraining | `src/retrain\_model.py` |

| Continuous integration | GitHub Actions |



\## Production Deployment



\- \*\*Production model:\*\* RandomForestRegressor, version 2

\- \*\*Docker image:\*\* `surat-bus-delay:v2.1`

\- \*\*Kubernetes deployment:\*\* `surat-bus-delay`

\- \*\*MLflow champion alias:\*\* version 2

\- \*\*MLflow rollback alias:\*\* version 1



\## Monitoring and Retraining



The monitoring workflow detects statistical data drift and checks a SHA256 fingerprint of the datasets before triggering candidate retraining.



When a dataset has already been evaluated, retraining is skipped.



When drift is detected for a dataset that has not yet been evaluated, the system retrains a candidate and compares its performance with the existing model.



Production deployment requires manual approval. The monitoring workflow does not automatically replace the production model.

