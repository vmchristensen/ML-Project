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

# Load test data (assumes columns: id, language, n, edgelist)
test_df = pd.read_csv('test.csv')

expanded_rows = []

for idx, row in test_df.iterrows():
    edgelist = ast.literal_eval(row['edgelist'])
    
    # Compute centralities
    cent = centralities(edgelist)
    nodes = list(cent.keys())
    
    # Create DataFrame for nodes
    features = np.array([cent[v] for v in nodes])
    df_sentence = pd.DataFrame(features, columns=[
        'degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity'
    ])
    
    # Add metadata
    df_sentence['vertex_id'] = nodes
    df_sentence['sentence_id'] = row['id']
    df_sentence['n'] = row['n']
    df_sentence['language'] = row['language']
    
    # Normalize features per sentence
    for col in df_sentence.columns[:7]:
        mean = df_sentence[col].mean()
        std = df_sentence[col].std()
        df_sentence[col] = (df_sentence[col] - mean) / std if std != 0 else 0
    
    expanded_rows.append(df_sentence)

expanded_test_df = pd.concat(expanded_rows, ignore_index=True)
expanded_test_df.to_csv('preprocessed_test.csv', index=False)