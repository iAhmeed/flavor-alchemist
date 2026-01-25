import os
import subprocess
import sys

def run_step(command, step_name):
    print(f"--- Starting {step_name} ---")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error in {step_name}. Exiting.")
        sys.exit(1)
    print(f"--- Finished {step_name} ---")

def main():
    # 1. Version Data
    run_step("python mlops/scripts/data_versioning.py", "Data Versioning")
    
    # 2. Train Model (Experimentation - Run multiple to show tracking)
    experiments = [
        (0.05, 0.04),
        (0.03, 0.02), # "Production" candidate
        (0.01, 0.01)
    ]
    
    for i, (apriori_supp, fp_supp) in enumerate(experiments):
        print(f"--- Experiment {i+1}/{len(experiments)}: apriori={apriori_supp}, fp_growth={fp_supp} ---")
        run_step(f"python mlops/scripts/train.py --apriori_min_support {apriori_supp} --fp_growth_min_support {fp_supp}", f"Training Run {i+1}")
    
    # 3. Simulate Monitoring (Post-deployment check)
    run_step("python mlops/scripts/monitor.py", "Production Monitoring")
    
    print("\n=== MLOps Pipeline Execution Complete ===")
    print("Check W&B Dashboard for Artifacts, Runs, and Registry.")

if __name__ == "__main__":
    main()
