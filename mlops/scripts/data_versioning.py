import wandb
import os

PROJECT_NAME = "flavor-alchemist"
ARTIFACT_NAME = "recipes-dataset"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "recipes.csv")

def version_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    run = wandb.init(project=PROJECT_NAME, job_type="data-versioning")
    
    print(f"Creating artifact {ARTIFACT_NAME}...")
    artifact = wandb.Artifact(
        name=ARTIFACT_NAME, 
        type="dataset",
        description="Raw recipes dataset containing ingredients."
    )
    
    artifact.add_file(DATA_PATH)
    
    print("Logging artifact...")
    run.log_artifact(artifact)
    
    run.finish()
    print("Data versioning complete.")

if __name__ == "__main__":
    version_data()
