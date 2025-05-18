import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from itertools import product

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_train.csv")

# Define features and target
categorical_features = ['language']
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_id']

# Hyperparameter grid
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', None]
}

# Generate all parameter combinations
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

best_score = 0
best_params = {}

for params in param_combinations:
    print(f"\nTesting parameters: {params}")
    model = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42,
        **params
    )
    
    group_kfold = GroupKFold(n_splits=5)
    fold_accuracies = []
    
    for train_idx, test_idx in group_kfold.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups.iloc[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        test_data = data.iloc[test_idx].copy()
        test_data['proba'] = proba

        # Select predicted root per sentence
        selected_roots = test_data.groupby('sentence_id', group_keys=False).apply(
            lambda x: x.loc[x['proba'].idxmax()]
        )

        accuracy = (selected_roots['target'] == 1).mean()
        fold_accuracies.append(accuracy)
    
    mean_accuracy = np.mean(fold_accuracies)
    print(f"Mean Sentence-Level Accuracy: {mean_accuracy:.3f}")
    
    if mean_accuracy > best_score:
        best_score = mean_accuracy
        best_params = params

print("\n=== Best Parameters ===")
print(f"Best Sentence-Level Accuracy: {best_score:.3f}")
print(f"Optimal Parameters: {best_params}")

# === Feature Importance Calculation ===
print("\n=== Feature Importances ===")
feature_names = preprocessor.get_feature_names_out()

# Collect feature importances across folds
best_model = DecisionTreeClassifier(
    class_weight='balanced', 
    random_state=42, 
    **best_params
)

group_kfold = GroupKFold(n_splits=5)
importances = []

for train_idx, test_idx in group_kfold.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    best_model.fit(X_train, y_train)
    importances.append(best_model.feature_importances_)

# Calculate mean importance across folds
mean_importances = np.mean(importances, axis=0)

# Combine with feature names and sort
feature_importance = sorted(
    zip(feature_names, mean_importances),
    key=lambda x: x[1], 
    reverse=True
)

# Print results
for name, importance in feature_importance:
    print(f"{name}: {importance:.4f}")

"""Mean Sentence-Level Accuracy: 0.602
Optimal Parameters: {'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt'}
=== Feature Importances ===
num__betweenness: 0.8085
num__harmonic: 0.1118
num__degree: 0.0245
num__eigenvector: 0.0153
lang__language_Japanese: 0.0126
num__pagerank: 0.0116
num__closeness: 0.0094
num__eccentricity: 0.0036
lang__language_Korean: 0.0020
lang__language_Russian: 0.0008"""