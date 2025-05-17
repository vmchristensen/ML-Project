# Victoria – Decision Tree Classifier with sentence-level accuracy

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold

# Load the preprocessed dataset
#data = pd.read_csv("preprocessed_train.csv")
#data = pd.read_csv("v3_preprocessed_train_degree_eccentricity.csv")  # reduced version
data = pd.read_csv("v4_preprocessed_train_degree.csv")  # reduced version
# Define features and target
categorical_features = ['language']
numerical_features = ['degree']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_id']

# Best-tuned Decision Tree
model = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features='log2',
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
    selected_roots = test_data.groupby('sentence_id', group_keys=False).apply(
        lambda x: x.loc[x['proba'].idxmax()]
    )

    accuracy = (selected_roots['target'] == 1).mean()
    accuracies.append(accuracy)
    print(f"Fold {fold_idx + 1} Sentence-Level Accuracy: {accuracy:.3f}")

# Summary
print("\n=== Cross-Validation Results ===")
print(f"Sentence-Level Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")
