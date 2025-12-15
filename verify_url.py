import requests

import requests

candidates = [
    "https://huggingface.co/datasets/AkashPS11/recipes_data_food.com/resolve/main/recipes.csv",
    "https://huggingface.co/datasets/recipe_nlg/dataset/resolve/main/full_dataset.csv",
    "https://raw.githubusercontent.com/sameermahajan/CulinaryML/master/Data/RAW_recipes.csv"
]

for url in candidates:
    print(f"Checking {url}...")
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            print(f"SUCCESS! URL: {url}")
            print(f"Content-Length: {response.headers.get('content-length', 'unknown')} bytes")
            break
        else:
            print(f"Failed: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
