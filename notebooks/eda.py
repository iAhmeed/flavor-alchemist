import pandas as pd
import matplotlib.pyplot as plt
from src.cleaner import load_and_clean_data
import os

def run_eda():
    if not os.path.exists("notebooks"):
        os.makedirs("notebooks")
        
    print("Loading data for EDA...")
    transactions, df = load_and_clean_data()
    
    if not transactions:
        print("No data found.")
        return

    # Flatten list
    all_ingredients = [item for sublist in transactions for item in sublist]
    
    # 1. Top Ingredients
    print("Generating Top Ingredients plot...")
    counts = pd.Series(all_ingredients).value_counts().head(20)
    
    plt.figure(figsize=(12, 6))
    counts.sort_values().plot(kind='barh', color='teal')
    plt.title("Top 20 Most Frequent Ingredients")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig("notebooks/top_ingredients.png")
    print("Saved notebooks/top_ingredients.png")
    
    # 2. Recipe Length Distribution
    print("Generating Recipe Length histogram...")
    lengths = [len(t) for t in transactions]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=range(1, 30), color='salmon', edgecolor='black')
    plt.title("Distribution of Number of Ingredients per Recipe")
    plt.xlabel("Number of Ingredients")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("notebooks/recipe_lengths.png")
    print("Saved notebooks/recipe_lengths.png")

if __name__ == "__main__":
    run_eda()
