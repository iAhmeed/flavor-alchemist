from src.cleaner import load_and_clean_data

print("Calling load_and_clean_data...")
result = load_and_clean_data()

print(f"Result type: {type(result)}")
if isinstance(result, tuple):
    print(f"Result length: {len(result)}")
else:
    print("Result is not a tuple.")
    if hasattr(result, 'columns'):
        print(f"Columns found: {list(result.columns)}")
