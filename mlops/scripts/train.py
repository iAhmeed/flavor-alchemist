import argparse
import wandb
import os
import pandas as pd
import sys

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mining import mine_rules

PROJECT_NAME = "flavor-alchemist"
MODEL_ARTIFACT_TYPE = "model"

def train(args):
    run = wandb.init(project=PROJECT_NAME, job_type="train", config=args)
    config = run.config

    print("Downloading dataset artifact...")
    artifact = run.use_artifact(f'{PROJECT_NAME}/recipes-dataset:latest', type='dataset')
    data_dir = artifact.download()
    data_path = os.path.join(data_dir, "recipes.csv")
    
    print(f"Training using data at {data_path}...")
    
    rules = mine_rules(
        apriori_min_support=config.apriori_min_support,
        fp_growth_min_support=config.fp_growth_min_support,
        apriori_min_confidence=config.apriori_min_confidence,
        fp_growth_min_confidence=config.fp_growth_min_confidence,
        data_filepath=data_path
    )
    
    if rules is not None and not rules.empty:
        # Log metrics
        avg_lift = rules['lift'].mean()
        avg_conf = rules['confidence'].mean()
        num_rules = len(rules)
        
        run.log({
            "num_rules": num_rules,
            "avg_lift": avg_lift,
            "avg_confidence": avg_conf
        })
        
        print(f"Training complete. Rules found: {num_rules}")
        
        # Save model
        model_path = "rules_model.pkl"
        rules.to_pickle(model_path)
        
        # Log model artifact
        model_artifact = wandb.Artifact(
            name="rules-model",
            type=MODEL_ARTIFACT_TYPE,
            description="Trained association rules model",
            metadata=dict(config)
        )
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact, aliases=["production"])
        
    else:
        print("No rules generated. Try lowering support thresholds.")
        run.log({"num_rules": 0})
    
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apriori_min_support", type=float, default=0.03)
    parser.add_argument("--fp_growth_min_support", type=float, default=0.02)
    parser.add_argument("--apriori_min_confidence", type=float, default=0.33)
    parser.add_argument("--fp_growth_min_confidence", type=float, default=0.6)
    
    args = parser.parse_args()
    train(args)
