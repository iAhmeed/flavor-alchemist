import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
from src.cleaner import load_and_clean_data
CACHE_PATH = "data/rules_cache.pkl"

def mine_rules(apriori_min_support=0.01, fp_growth_min_support=0.01, apriori_min_confidence=0.1, fp_growth_min_confidence=0.1, data_filepath=None):
    """
    Mines association rules from the recipe dataset.
    """
    print("Loading and cleaning data...")
    if data_filepath:
         transactions, _ = load_and_clean_data(filepath=data_filepath)
    else:
         transactions, _ = load_and_clean_data()
    
    if not transactions:
        print("No transactions found.")
        return None

    print(f"Encoding {len(transactions)} transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"Running Apriori algorithm with min_support={apriori_min_support}...")
    apriori_frequent_itemsets = apriori(df, min_support=apriori_min_support, use_colnames=True)
    print(f"Found {len(apriori_frequent_itemsets)} frequent itemsets with Apriori.")
    if apriori_frequent_itemsets.empty:
        return pd.DataFrame()
    print(f"Generating rules with min_confidence={apriori_min_confidence}...")
    apriori_rules = association_rules(apriori_frequent_itemsets, metric="confidence", min_threshold=apriori_min_confidence)
    apriori_rules = apriori_rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    print(f"Generated {len(apriori_rules)} rules with Apriori.")
    avg_lift_apriori = apriori_rules['lift'].mean() if not apriori_rules.empty else 0
    print(f"Average lift of Apriori rules: {avg_lift_apriori:.4f}")
    avg_confidence_apriori = apriori_rules['confidence'].mean() if not apriori_rules.empty else 0
    print(f"Average confidence of Apriori rules: {avg_confidence_apriori:.4f}")
    
    print(f"Running FP-Growth with min_support={fp_growth_min_support}...")
    fp_growth_frequent_itemsets = fpgrowth(df, min_support=fp_growth_min_support, use_colnames=True)
    
    print(f"Found {len(fp_growth_frequent_itemsets)} frequent itemsets.")
    if fp_growth_frequent_itemsets.empty:
        return pd.DataFrame()

    print(f"Generating rules with min_confidence={fp_growth_min_confidence}...")
    fp_growth_rules = association_rules(fp_growth_frequent_itemsets, metric="confidence", min_threshold=fp_growth_min_confidence)
    
    # Add 'lift' and sort
    fp_growth_rules = fp_growth_rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    
    print(f"Generated {len(fp_growth_rules)} rules.")
    avg_lift_fp_growth = fp_growth_rules['lift'].mean() if not fp_growth_rules.empty else 0
    print(f"Average lift of FP-Growth rules: {avg_lift_fp_growth:.4f}")
    avg_confidence_fp_growth = fp_growth_rules['confidence'].mean() if not fp_growth_rules.empty else 0
    print(f"Average confidence of FP-Growth rules: {avg_confidence_fp_growth:.4f}")
    
    return fp_growth_rules

def get_recommendations(rules, ingredients, top_k=5):
    """
    Given a set of ingredients, recommend associated items.
    """
    # Simple logic: Find rules where antecedent is a subset of ingredients
    # This matches: if {A, B} -> {C}, and user has {A, B}, suggest C.
    
    # Filter rules where antecedents overlap with input
    # Note: rigorous subset matching might be slow if we have many rules.
    # We can do exact match or partial match.
    
    # converting frozensets to sets for easier handling
    if rules is None or rules.empty:
        return []

    input_set = set([i.lower() for i in ingredients])
    
    suggestions = []
    
    relevant_rules = rules[rules['antecedents'].apply(lambda x: not x.isdisjoint(input_set))]
    
    for _, row in relevant_rules.iterrows():
        consequents = row['consequents']
        for item in consequents:
            if item not in input_set:
                suggestions.append({
                    'item': item,
                    'confidence': row['confidence'],
                    'lift': row['lift'],
                    'rule': f"{set(row['antecedents'])} -> {set(row['consequents'])}"
                })
    
    # Deduplicate and sort by lift
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s['item'] not in seen:
            seen.add(s['item'])
            unique_suggestions.append(s)
    
    unique_suggestions.sort(key=lambda x: x['lift'], reverse=True)
    return unique_suggestions[:top_k]

if __name__ == "__main__":
    rules = mine_rules(min_support=0.005) # Lower support for more interesting result
    if rules is not None and not rules.empty:
        print(rules.head())
        # Test recommendation
        test_ingredients = ["chicken", "garlic"]
        recs = get_recommendations(rules, test_ingredients)
        print(f"\nRecommendations for {test_ingredients}:")
        for r in recs:
            print(f"- {r['item']} (Lift: {r['lift']:.2f})")
