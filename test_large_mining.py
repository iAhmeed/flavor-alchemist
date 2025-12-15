from src.mining import mine_rules
import time

print("Starting mining test on large dataset...")
start = time.time()
# Use high support to ensure quick termination for test
rules = mine_rules(min_support=0.05, min_confidence=0.1) 
end = time.time()

if rules is not None and not rules.empty:
    print(f"Success! Mining took {end - start:.2f} seconds.")
    print(f"Found {len(rules)} rules.")
    print(rules.head())
else:
    print("Mining returned no rules or failed.")
