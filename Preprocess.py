import pandas as pd
import numpy as np
import ast
import networkx as nx

def centralities(edgelist):
    """Compute centrality scores for each node in the tree."""
    T = nx.from_edgelist(edgelist)
    dc = nx.degree_centrality(T)
    cl = nx.closeness_centrality(T)
    bc = nx.betweenness_centrality(T)
    pc = nx.pagerank(T)
    cc = nx.harmonic_centrality(T)
    ec = nx.eigenvector_centrality(T, max_iter=10000)
    ecc = nx.eccentricity(T)
    return {v: (dc[v], cl[v], bc[v], pc[v], cc[v], ec[v], ecc[v]) for v in T}

# Load training data
train_df = pd.read_csv('train.csv')

expanded_rows = []

for idx, row in train_df.iterrows():
    # Parse edgelist and get root node
    edgelist = ast.literal_eval(row['edgelist'])
    root = row['root']
    
    # Compute centralities for all nodes
    cent = centralities(edgelist)
    nodes = list(cent.keys())
    
    # Create DataFrame for nodes in this sentence
    features = np.array([cent[v] for v in nodes])
    df_sentence = pd.DataFrame(features, columns=[
        'degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity'
    ])
    
    # Add metadata columns
    df_sentence['vertex_id'] = nodes
    df_sentence['sentence_id'] = row['sentence']
    df_sentence['n'] = row['n']
    df_sentence['language'] = row['language']
    df_sentence['sentence_group'] = f"{row['language']}_{row['sentence']}"
    
    # Set target 
    df_sentence['target'] = (df_sentence['vertex_id'] == root).astype(int)
    
    # Normalize features within the sentence
    for col in ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']:
        mean = df_sentence[col].mean()
        std = df_sentence[col].std()
        if std == 0:
            df_sentence[col] = 0
        else:
            df_sentence[col] = (df_sentence[col] - mean) / std
    
    expanded_rows.append(df_sentence)

# Combine all data
expanded_df = pd.concat(expanded_rows, ignore_index=True)

# Save preprocessed data with all columns
expanded_df.to_csv('preprocessed_train.csv', index=False)