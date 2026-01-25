# MLOps Project Walkthrough - The Flavor Alchemist

A complete MLOps pipeline using Weights & Biases (W&B). Here is how to proceed.

## 1. Setup

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Make sure you are logged into W&B:

```bash
wandb login
```

## 2. Running the MLOps Pipeline

You can run the entire lifecycle (Data Versioning → Training → Monitoring) with one command:

```bash
python mlops/run_pipeline.py
```

Or run individual steps:

- **Data Versioning**: `python mlops/scripts/data_versioning.py`
- **Training**: `python mlops/scripts/train.py --apriori_min_support 0.03`
- **Monitoring**: `python mlops/scripts/monitor.py`

## 3. Serving the Model

To start the production API:

```bash
uvicorn mlops.deployment.serve_model:app --reload
```

Test with (Linux/Mac):

```bash
curl -X POST "http://127.0.0.1:8000/recommend" -H "Content-Type: application/json" -d '{"ingredients": ["chicken", "garlic"]}'
```

Or on Windows PowerShell:

```powershell
$body = @{"ingredients" = @("chicken", "garlic")} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend" -Method Post -ContentType "application/json" -Body $body
```

## 4. Directory Structure

- **scripts/**: MLOps automation scripts (`data_versioning.py`, `train.py`, `monitor.py`).
- **deployment/**: FastAPI serving code.
- **run_pipeline.py**: Pipeline orchestrator.
