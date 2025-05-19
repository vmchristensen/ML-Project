import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from joblib import dump
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)

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
groups = data['sentence_group']

# Initialize model
model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    class_weight='balanced',
    random_state=42
)

# Sentence-level accuracy evaluation
group_kfold = GroupKFold(n_splits=5)
accuracies = []

for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
    print(f"Fold {fold_idx + 1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups.iloc[test_idx]

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    test_data = data.iloc[test_idx].copy()
    test_data['proba'] = proba

    # Select predicted root per sentence
    selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]

    accuracy = (selected_roots['target'] == 1).mean()
    accuracies.append(accuracy)
    print(f"Fold {fold_idx + 1} Sentence-Level Accuracy: {accuracy:.3f}")

# Summary
print("\n=== Cross-Validation Results ===")
print(f"Sentence-Level Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")