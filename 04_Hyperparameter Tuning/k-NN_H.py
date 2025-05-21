import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from itertools import product
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)

# Define features
categorical_features = []
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

X = data[categorical_features + numerical_features]  # stay as DataFrame
y = data['target']
groups = data['sentence_group']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Grid of hyperparameters
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan','minkowski']
}

# Generate all parameter combinations
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

# Track best
best_score = 0
best_params = {}

for params in param_combinations:
    print(f"\nTesting parameters: {params}")
    
    model = KNeighborsClassifier(**params)
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', model)
    ])
    
    group_kfold = GroupKFold(n_splits=5)
    fold_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        test_data = data.iloc[test_idx].copy()

        # Train and predict
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        test_data['proba'] = proba

        selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]
        accuracy = (selected_roots['target'] == 1).mean()
        fold_accuracies.append(accuracy)

        print(f"Fold {fold_idx + 1} Accuracy: {accuracy:.3f}")
    
    mean_accuracy = np.mean(fold_accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.3f}")

    if mean_accuracy > best_score:
        best_score = mean_accuracy
        best_params = params

# Final output
print("\n=== Best Parameters ===")
print(f"Best Accuracy: {best_score:.3f}")
print(f"Params: {best_params}")

"""
=== Best Parameters ===
Best Accuracy: 0.316
Params: {'n_neighbors': 7, 'weights': 'distance', 'metric': 'euclidean'}

"""