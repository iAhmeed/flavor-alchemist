from src.cleaner import load_and_clean_data

print("Loading data...")
transactions, _ = load_and_clean_data()
all_ingredients = sorted(list(set([item for sublist in transactions for item in sublist])))

print(f"Total unique ingredients: {len(all_ingredients)}")
print("Checking for 'chicken':")
print([i for i in all_ingredients if 'chicken' in i][:10])
print("Checking for 'garlic':")
print([i for i in all_ingredients if 'garlic' in i][:10])

if "chicken" in all_ingredients:
    print("'chicken' IS in the list.")
else:
    print("'chicken' IS NOT in the list.")

if "garlic" in all_ingredients:
    print("'garlic' IS in the list.")
else:
    print("'garlic' IS NOT in the list.")
