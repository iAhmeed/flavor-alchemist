from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import wandb
import pandas as pd
import os
import sys

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mining import get_recommendations

app = FastAPI(title="The Flavor Alchemist API", version="1.0.0")

PROJECT_NAME = "flavor-alchemist"
MODEL_ALIAS = "production"  # Use 'latest' or a specific tag

# Global model variable
rules = None

@app.on_event("startup")
def load_model():
    global rules
    print("Initializing W&B...")
    run = wandb.init(project=PROJECT_NAME, job_type="inference")
    
    print(f"Downloading model artifact with alias '{MODEL_ALIAS}'...")
    try:
        artifact = run.use_artifact(f'{PROJECT_NAME}/rules-model:{MODEL_ALIAS}')
        model_dir = artifact.download()
        model_path = os.path.join(model_dir, "rules_model.pkl")
        
        print(f"Loading model from {model_path}...")
        rules = pd.read_pickle(model_path)
        print(f"Model loaded. {len(rules)} rules available.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Fallback to local if available for testing? 
        # For MLOps strictness, we might want to fail or try local cache.
    finally:
        run.finish()

class RecipeRequest(BaseModel):
    ingredients: List[str]
    top_k: int = 5

class Recommendation(BaseModel):
    item: str
    confidence: float
    lift: float
    rule: str

@app.post("/recommend", response_model=List[Recommendation])
def recommend(request: RecipeRequest):
    if rules is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    recs = get_recommendations(rules, request.ingredients, top_k=request.top_k)
    return recs

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": rules is not None}
