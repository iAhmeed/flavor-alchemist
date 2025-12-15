from src.cleaner import load_and_clean_data

print("Loading data...")
transactions, _ = load_and_clean_data()
all_ingredients = sorted(list(set([item for sublist in transactions for item in sublist])))

print("Testing default logic...")
default_selection = []
for default_item in ["chicken", "garlic"]:
     matches = [x for x in all_ingredients if default_item in x]
     if matches:
         # Pick the shortest match
         best_match = min(matches, key=len)
         print(f"Match for '{default_item}': '{best_match}'")
     else:
         print(f"No match for '{default_item}'")
