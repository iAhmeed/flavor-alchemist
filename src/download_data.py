import os
import urllib.request
import requests

def download_data():
    url = "https://huggingface.co/datasets/AkashPS11/recipes_data_food.com/resolve/main/recipes.csv"
    output_dir = "data"
    output_file = os.path.join(output_dir, "recipes.csv")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading from {url}...")
    try:
        # Use requests for better handling
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_data()
