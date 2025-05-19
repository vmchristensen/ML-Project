# This version only uses degree and eccentricity

import pandas as pd
import numpy as np
import ast
import networkx as nx
import seaborn as sns # For heatmap visualization
import matplotlib.pyplot as plt # For showing the plot

# Define the list of features we intend to use consistently
SELECTED_FEATURES = ['degree', 'eccentricity'] # MODIFIED

def centralities(edgelist):
    """Compute selected centrality scores for each node in the tree."""
    T = nx.from_edgelist(edgelist)

    # Calculate only the selected features
    dc = nx.degree_centrality(T)
    # REMOVED: cc = nx.harmonic_centrality(T)
    # REMOVED: bc = nx.betweenness_centrality(T)

    ecc_values = {}
    if nx.is_connected(T):
        try:
            ecc = nx.eccentricity(T)
            ecc_values = ecc
        except Exception as e:
            print(f"Error in eccentricity for a tree: {e}. Assigning 0s.")
            ecc_values = {node: 0.0 for node in T.nodes()}
    else:
        # This case should be rare for trees from the dataset, but good to have
        print(f"Graph from edgelist {edgelist} is disconnected, assigning 0.0 for eccentricity for all nodes.")
        ecc_values = {node: 0.0 for node in T.nodes()}

    # Return a dictionary mapping node to a tuple of selected feature values
    # Ensure the order here matches SELECTED_FEATURES
    return {
        v: (
            dc.get(v, 0),
            ecc_values.get(v, 0) # MODIFIED: Only degree and eccentricity
        ) for v in T.nodes()
    }


# Load training data
train_df = pd.read_csv('train.csv')

expanded_rows = []

for idx, row in train_df.iterrows():
    edgelist_str = row['edgelist']
    if pd.isna(edgelist_str) or not edgelist_str.strip():
        continue
    try:
        edgelist = ast.literal_eval(edgelist_str)
    except (ValueError, SyntaxError):
        continue

    if not edgelist:
        continue

    temp_T = nx.from_edgelist(edgelist)
    if not temp_T.nodes(): # Check if graph has any nodes after creation
        # print(f"Skipping row {idx} as edgelist resulted in a graph with no nodes: {edgelist_str}")
        continue


    root = row['root']
    cent = centralities(edgelist)

    if not cent: # If centralities returned an empty dict (e.g., if T had no nodes)
        # print(f"Skipping row {idx} as no centralities were computed (cent dict empty).")
        continue

    # Prepare data for DataFrame creation
    features_data = []
    valid_nodes_for_df = []
    # Iterate over nodes present in the graph to ensure consistency
    for v_node in temp_T.nodes():
        if v_node in cent:
            # cent[v_node] is now a tuple of (degree, eccentricity)
            features_data.append(cent[v_node])
            valid_nodes_for_df.append(v_node)
        # else: # This should ideally not happen if centralities processes all nodes from temp_T
            # print(f"Warning: Node {v_node} from graph not found in centralities output for sentence {row['sentence']}")


    if not features_data: # If no valid nodes had features extracted
        # print(f"Skipping row {idx} as no feature data could be gathered for nodes.")
        continue

    # Create DataFrame for nodes in this sentence using the SELECTED_FEATURES list for column names
    df_sentence = pd.DataFrame(features_data, columns=SELECTED_FEATURES) # USES THE MODIFIED LIST

    df_sentence['vertex_id'] = valid_nodes_for_df
    df_sentence['sentence_id'] = row['sentence']
    df_sentence['n'] = row['n']
    df_sentence['language'] = row['language']
    df_sentence['sentence_group'] = f"{row['language']}_{row['sentence']}"
    df_sentence['target'] = (df_sentence['vertex_id'] == root).astype(int)

    # Normalize features within the sentence using SELECTED_FEATURES
    for col in SELECTED_FEATURES: # USES THE MODIFIED LIST
        if col in df_sentence.columns: # Check if column exists (it should if SELECTED_FEATURES is correct)
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
    print("\n--- Feature Correlation Matrix (Degree & Eccentricity) ---")
    # Use SELECTED_FEATURES for the correlation matrix
    actual_feature_columns_for_corr = [col for col in SELECTED_FEATURES if col in expanded_df.columns]
    if not actual_feature_columns_for_corr:
        print("No feature columns (from SELECTED_FEATURES) found in expanded_df to calculate correlation.")
    elif len(actual_feature_columns_for_corr) < 2:
        print(f"Not enough columns to compute a meaningful correlation matrix. Columns found: {actual_feature_columns_for_corr}")
    else:
        correlation_matrix = expanded_df[actual_feature_columns_for_corr].corr()
        print(correlation_matrix)

        plt.figure(figsize=(6, 4)) # Adjusted size for 2 features
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Feature Correlation Matrix (Degree & Eccentricity)')
        plt.show()
    # --- END OF CORRELATION CHECK ---

    # Save preprocessed data
    output_filename = 'v3_preprocessed_train_degree_eccentricity.csv' # MODIFIED filename
    expanded_df.to_csv(output_filename, index=False)
    print(f"\nPreprocessed data saved to {output_filename}")