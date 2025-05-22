import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import itertools

# === Load and prepare data ===
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)
#data = data.sample(10000, random_state=42)

categorical_features = ['language']
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

X = data[categorical_features + numerical_features]
y = data['target']
groups = data['sentence_group']


# Define the param grid
C_values = [0.1, 1, 10]
gamma_values = ['scale', 'auto']
param_grid = list(itertools.product(C_values, gamma_values))

best_score = 0
best_params = None

for C_val, gamma_val in param_grid:
    print(f"Testing C={C_val}, gamma={gamma_val}")
    
    model = SVC(
        kernel='linear',
        C=C_val,
        gamma=gamma_val,
        class_weight='balanced',
        probability=True,
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', model)
    ])

    group_kfold = GroupKFold(n_splits=3)
    accuracies = []

    for train_idx, test_idx in group_kfold.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_data = data.iloc[test_idx].copy()

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        test_data['proba'] = proba

        selected = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]
        acc = (selected['target'] == 1).mean()
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    print(f"→ Mean sentence-level accuracy: {mean_acc:.3f}")

    if mean_acc > best_score:
        best_score = mean_acc
        best_params = (C_val, gamma_val)

print("\nBest sentence-level accuracy:", best_score)
print("Best parameters: C =", best_params[0], ", gamma =", best_params[1])


"""
=== Best Parameters ===
Optimal Parameters: 
Optimal Features: 
Best Sentence Level Accuracy: 
"""