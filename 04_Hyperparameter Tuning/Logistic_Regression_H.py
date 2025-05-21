import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from itertools import product
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)

# Define features
categorical_features = ['language']
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']
X = data[categorical_features + numerical_features]
y = data['target']
groups = data['sentence_group']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numerical_features)
])

# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}
param_combos = list(product(param_grid['C'], param_grid['penalty']))

# GroupKFold setup
group_kfold = GroupKFold(n_splits=5)
best_score = 0
best_params = None

# Manual grid search with sentence-level accuracy
for C_val, penalty in param_combos:
    if penalty == 'l1' and 'liblinear' not in ['liblinear', 'saga']:
        continue  # skip if solver doesn't support L1

    print(f"\nTesting: C={C_val}, penalty={penalty}")
    model = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        penalty=penalty,
        C=C_val,
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', model)
    ])

    accuracies = []
    for train_idx, test_idx in group_kfold.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_data = data.iloc[test_idx].copy()

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]

        test_data['proba'] = proba
        selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]
        acc = (selected_roots['target'] == 1).mean()
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    print(f"Mean Accuracy: {mean_acc:.3f}")
    if mean_acc > best_score:
        best_score = mean_acc
        best_params = {'C': C_val, 'penalty': penalty}

print("\n=== Best Parameters ===")
print(f"Accuracy: {best_score:.3f}")
print(f"Params: {best_params}")

# === Feature Importance (on best model) ===
print("\n=== Permutation Importance (best model) ===")
best_model = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',
    penalty=best_params['penalty'],
    C=best_params['C'],
    max_iter=1000,
    random_state=42
)

final_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', best_model)
])

final_pipeline.fit(X, y)
perm = permutation_importance(final_pipeline, X, y, n_repeats=5, random_state=42, scoring='accuracy')

feature_names = final_pipeline.named_steps['preprocess'].get_feature_names_out()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'ImportanceMean': perm.importances_mean,
    'ImportanceStd': perm.importances_std
}).sort_values(by='ImportanceMean', ascending=False)

print(importance_df)


"""
=== Best Parameters ===
Optimal Parameters: {'C': 0.1, 'penalty': 'l2', 'max_iter': 1000, 'solver': 'liblinear', 'class_weight': 'balanced'}
Optimal Features: only closeness
Best Sentence Level Accuracy: 0.291 (±0.015)

=== Permutation Importance (best model) ===
             Feature  ImportanceMean  ImportanceStd
3      num__pagerank        0.154839       0.000481
0        num__degree        0.091350       0.000509
4      num__harmonic        0.055375       0.000522
1     num__closeness        0.022999       0.000173
5   num__eigenvector        0.001909       0.000262
6  num__eccentricity       -0.007187       0.000205
2   num__betweenness       -0.034682       0.000370
"""
