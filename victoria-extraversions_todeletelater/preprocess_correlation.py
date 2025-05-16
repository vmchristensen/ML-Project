# This version only uses degree, harmonic, betweenness, eccentricity


import pandas as pd
import numpy as np
import ast
import networkx as nx
import seaborn as sns # For heatmap visualization
import matplotlib.pyplot as plt # For showing the plot

# Define the list of features we intend to use consistently
SELECTED_FEATURES = ['degree', 'harmonic', 'betweenness', 'eccentricity']

def centralities(edgelist):
    """Compute selected centrality scores for each node in the tree."""
    T = nx.from_edgelist(edgelist)
    
    # Calculate only the selected features
    dc = nx.degree_centrality(T)
    cc = nx.harmonic_centrality(T)
    bc = nx.betweenness_centrality(T)
    
    ecc_values = {}
    if nx.is_connected(T):
        try:
            ecc = nx.eccentricity(T)
            ecc_values = ecc
        except Exception as e:
            print(f"Error in eccentricity for a tree: {e}. Assigning 0s.")
            ecc_values = {node: 0.0 for node in T.nodes()}
    else:
        print("Graph is disconnected, assigning 0.0 for eccentricity.")
        ecc_values = {node: 0.0 for node in T.nodes()}

    # Return a dictionary mapping node to a tuple of selected feature values
    # Ensure the order here matches SELECTED_FEATURES if you build the DataFrame columns from it
    return {
        v: (
            dc.get(v, 0),
            cc.get(v, 0),
            bc.get(v, 0),
            ecc_values.get(v, 0)
        ) for v in T.nodes()
    }


# Load training data
train_df = pd.read_csv('train.csv')

expanded_rows = []

for idx, row in train_df.iterrows():
    edgelist_str = row['edgelist']
    if pd.isna(edgelist_str) or not edgelist_str.strip():
        # print(f"Skipping row {idx} due to empty or NaN edgelist.") # Optional: less verbose
        continue
    try:
        edgelist = ast.literal_eval(edgelist_str)
    except (ValueError, SyntaxError):
        # print(f"Skipping row {idx} due to edgelist parsing error.") # Optional
        continue

    if not edgelist:
        continue
    
    temp_T = nx.from_edgelist(edgelist)
    if not temp_T.nodes():
        continue

    root = row['root']
    cent = centralities(edgelist)
    
    if not cent:
        continue

    # Prepare data for DataFrame creation
    features_data = []
    valid_nodes_for_df = []
    for v_node in temp_T.nodes(): # Iterate over nodes in graph to ensure consistency
        if v_node in cent:
            features_data.append(cent[v_node]) # cent[v_node] is now a tuple of (dc, cc, bc, ecc)
            valid_nodes_for_df.append(v_node)
        # else: # This case should be rare if centralities processes all nodes from temp_T
            # print(f"Warning: Node {v_node} from graph not found in centralities output for sentence {row['sentence']}")

    if not features_data:
        continue

    # Create DataFrame for nodes in this sentence using the SELECTED_FEATURES list for column names
    df_sentence = pd.DataFrame(features_data, columns=SELECTED_FEATURES) # USE THE CONSISTENT LIST
    
    df_sentence['vertex_id'] = valid_nodes_for_df
    df_sentence['sentence_id'] = row['sentence']
    df_sentence['n'] = row['n']
    df_sentence['language'] = row['language']
    df_sentence['sentence_group'] = f"{row['language']}_{row['sentence']}"
    df_sentence['target'] = (df_sentence['vertex_id'] == root).astype(int)
    
    # Normalize features within the sentence using SELECTED_FEATURES
    for col in SELECTED_FEATURES: # USE THE CONSISTENT LIST
        if col in df_sentence.columns:
            mean = df_sentence[col].mean()
            std = df_sentence[col].std()
            if pd.isna(std) or std == 0:
                df_sentence[col] = 0 
            else:
                df_sentence[col] = (df_sentence[col] - mean) / std
    
    expanded_rows.append(df_sentence)

if not expanded_rows:
    print("No data was processed into expanded_rows. Exiting.")
    # exit() # Or handle as appropriate
else:
    expanded_df = pd.concat(expanded_rows, ignore_index=True)
    print(f"Shape of expanded_df: {expanded_df.shape}")
    print(expanded_df.head())

    # --- POINT TO CHECK CORRELATION MATRIX ---
    print("\n--- Feature Correlation Matrix ---")
    # Use SELECTED_FEATURES for the correlation matrix
    actual_feature_columns_for_corr = [col for col in SELECTED_FEATURES if col in expanded_df.columns]
    if not actual_feature_columns_for_corr:
        print("No feature columns (from SELECTED_FEATURES) found in expanded_df to calculate correlation.")
    else:
        correlation_matrix = expanded_df[actual_feature_columns_for_corr].corr()
        print(correlation_matrix)

        plt.figure(figsize=(8, 6)) # Adjusted size for fewer features
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Feature Correlation Matrix (Selected Features)')
        plt.show()
    # --- END OF CORRELATION CHECK ---

    # Save preprocessed data
    expanded_df.to_csv('v2_preprocessed_train_selected_features.csv', index=False)
    print("\nPreprocessed data saved to v2_preprocessed_train_selected_features.csv")