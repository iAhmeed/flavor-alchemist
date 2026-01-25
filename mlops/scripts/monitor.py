import wandb
import pandas as pd
import random
import time
import os
import sys

# Ensure src can be imported
# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.cleaner import load_and_clean_data
from src.mining import get_recommendations

PROJECT_NAME = "flavor-alchemist"

def monitor():
    print("Initializing Monitoring...")
    run = wandb.init(project=PROJECT_NAME, job_type="monitor")
    
    # helper: load data to get sample ingredients
    # In a real scenario, this stream comes from the API logs or Kafka
    print("Loading reference data for simulation...")
    transactions, _ = load_and_clean_data()
    all_ingredients = list(set([item for sublist in transactions for item in sublist]))
    
    # Load Model
    print("Loading model from registry...")
    try:
        artifact = run.use_artifact(f"{PROJECT_NAME}/rules-model:production", type="model")
        model_dir = artifact.download()
        rules = pd.read_pickle(os.path.join(model_dir, "rules_model.pkl"))
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    # Create a Table to log predictions
    columns = ["timestamp", "input_ingredients", "recommendation", "lift", "confidence"]
    prediction_table = wandb.Table(columns=columns)
    
    print("Simulating traffic...")
    # Simulate 50 requests
    for i in range(50):
        # Randomly pick 1-3 ingredients
        k = random.randint(1, 3)
        input_ing = random.sample(all_ingredients, k)
        
        recs = get_recommendations(rules, input_ing, top_k=1)
        
        timestamp = time.time()
        
        if recs:
            top_rec = recs[0]
            prediction_table.add_data(
                timestamp, 
                str(input_ing), 
                top_rec['item'], 
                top_rec['lift'], 
                top_rec['confidence']
            )
        else:
            prediction_table.add_data(timestamp, str(input_ing), "None", 0, 0)
        
        if i % 10 == 0:
            print(f"Processed {i} requests...")

    print("Logging predictions table...")
    run.log({"production_predictions": prediction_table})
    
    run.finish()
    print("Monitoring complete.")

if __name__ == "__main__":
    monitor()
