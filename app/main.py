import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from src.mining import mine_rules, get_recommendations
from src.cleaner import load_and_clean_data

st.set_page_config(page_title="The Flavor Alchemist", page_icon="🧪", layout="wide")

@st.cache_resource
def load_resources(apriori_min_supp, fp_growth_min_support, apriori_min_conf, fp_growth_min_conf):
    return mine_rules(apriori_min_supp, fp_growth_min_support, apriori_min_conf, fp_growth_min_conf)

@st.cache_data
def get_all_ingredients():
    transactions, _ = load_and_clean_data()
    all_ingredients = sorted(list(set([item for sublist in transactions for item in sublist])))
    return all_ingredients

st.title("🧪 The Flavor Alchemist")
st.markdown("""
***Select ingredients you have, and we will suggest what else goes well with them!***
""")

sidebar = st.sidebar
# Tuned for Large Dataset (800k+ recipes)
# Higher support needed to avoid OOM or slow mining on start
apriori_min_support = 0.03
apriori_min_confidence = 0.33
fp_growth_min_support = 0.02
fp_growth_min_confidence = 0.6

if sidebar.button("Retrain Model"):
    st.cache_resource.clear()

# Load ingredients for autocomplete
with st.spinner("Loading pantry..."):
    all_ingredients = get_all_ingredients()

# Load rules
with st.spinner("Brewing potions (Mining Rules)..."):
    rules = load_resources(apriori_min_support, fp_growth_min_support, apriori_min_confidence, fp_growth_min_confidence)

if rules is not None and not rules.empty:
    st.success(f"Model ready! Discovered {len(rules)} association rules.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Pantry")
        # Smart default selection
        default_selection = []
        container_ingredients = set(all_ingredients) # Fast lookup
        
        for default_item in ["chicken", "garlic"]:
             # Try exact match
             if default_item in container_ingredients:
                 default_selection.append(default_item)
             else:
                 # Try fuzzy match (find item that contains the word)
                 matches = [x for x in all_ingredients if default_item in x]
                 if matches:
                     # Pick the shortest match (heuristic for "core" ingredient)
                     best_match = min(matches, key=len)
                     default_selection.append(best_match)
        
        # Filter duplicates just in case
        default_selection = list(set(default_selection))

        # Autocomplete using multiselect
        ingredients = st.multiselect(
            "Select ingredients:",
            options=all_ingredients,
            default=default_selection
        )
        
        if st.button("🪄 Alchemize!"):
            recommendations = get_recommendations(rules, ingredients, top_k=10)
            
            if recommendations:
                st.subheader("Magic Pairings:")
                for rec in recommendations:
                    score = rec['lift']
                    st.write(f"**{rec['item'].title()}** (Lift: {score:.2f})")
                    st.progress(min(score/10, 1.0))
            else:
                st.warning("No strong associations found. Try common ingredients!")

    with col2:
        st.subheader("Flavor Network")
        if ingredients:
            relevant_rules = rules[rules['antecedents'].apply(lambda x: not x.isdisjoint(set(ingredients)))]
            
            if not relevant_rules.empty:
                G = nx.DiGraph()
                top_rules = relevant_rules.head(20)
                
                for _, row in top_rules.iterrows():
                    for ant in row['antecedents']:
                        for con in row['consequents']:
                            G.add_edge(ant, con, weight=row['lift'])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.5)
                nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                        node_size=2000, font_size=10, font_weight='bold', 
                        edge_color='gray', width=[G[u][v]['weight']/2 for u,v in G.edges()])
                st.pyplot(fig)
            else:
                st.info("No network to display.")

else:
    st.error("Could not generate rules. Try lowering Min Support.")

st.markdown("---")
st.caption("Powered by Apriori/FP-Growth Algorithm")
