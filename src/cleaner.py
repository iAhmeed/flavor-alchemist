import pandas as pd
import ast
import re

def load_and_clean_data(filepath="data/recipes.csv"):
    """
    Loads recipes and cleans the ingredients list.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found at {filepath}")
        return None

    # Inspect the columns
    # We expect 'Ingredients' containing a string representation of a list
    # e.g., "['sugar', 'flour']"
    
    # Support for new large dataset (Food.com / RecipeNLG style)
    if 'RecipeIngredientParts' in df.columns:
        target_col = 'RecipeIngredientParts'
        # Format is likely c("item1", "item2")
        def parse_r_vector(x):
            try:
                if isinstance(x, str):
                    # Extract contents inside c(...)
                    if x.startswith('c(') and x.endswith(')'):
                        content = x[2:-1]
                        items = re.findall(r'"([^"]+)"', content)
                        return items
                    return [] # Fallback
                return x
            except Exception as e:
                return []
        
        df['parsed_ingredients'] = df[target_col].apply(parse_r_vector)
    elif 'Ingredients' in df.columns:
        # Old dataset fallback
        def parse_ingredients(x):
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                return x
            except:
                return []
        
        df['parsed_ingredients'] = df['Ingredients'].apply(parse_ingredients)
    else:
        print("Column 'RecipeIngredientParts' or 'Ingredients' not found. Available:", df.columns.tolist())
        return None, None

    # Basic cleaning
    def clean_item(item):
        # Lowercase
        item = item.lower()
        item = re.sub(r'[\d\W]+', ' ', item).strip()
        return item

    # Further processing: Expand the lists
    # Create valid transactions
    transactions = []
    # Drop NAs
    df = df.dropna(subset=['parsed_ingredients'])
    
    for row in df['parsed_ingredients']:
        if isinstance(row, list):
            clean_row = [clean_item(i) for i in row if i]
            if clean_row:
                transactions.append(list(set(clean_row))) # Unique items per recipe
    
    return transactions, df

if __name__ == "__main__":
    trans, df = load_and_clean_data()
    if trans:
        print(f"Loaded {len(trans)} recipes.")
        print("Sample:", trans[0])
