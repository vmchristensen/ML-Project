# Victoria – Logistic Regression with sentence-level accuracy
# fixed

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_train.csv")
#data = pd.read_csv("v4_preprocessed_train_degree.csv")  # optional reduced version

# Define features and target
categorical_features = ['language']
numerical_features = ['degree', 'harmonic', 'betweenness', 'pagerank', 'eigenvector', 'eccentricity']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_id']

# Logistic Regression model
model = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',
    penalty='l2',
    max_iter=1000,
    random_state=42
)

# Grouped CV using sentence-level accuracy
group_kfold = GroupKFold(n_splits=5)
accuracies = []

for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
    print(f"Fold {fold_idx + 1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups.iloc[test_idx]

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    # Attach predictions to test data
    test_data = data.iloc[test_idx].copy()
    test_data['proba'] = proba

    # Select one predicted root per sentence
    selected_roots = test_data.groupby('sentence_id').apply(
        lambda x: x.loc[x['proba'].idxmax()]
    )

    # Sentence-level accuracy
    accuracy = (selected_roots['target'] == 1).mean()
    accuracies.append(accuracy)
    print(f"Fold {fold_idx + 1} Sentence-Level Accuracy: {accuracy:.3f}")

# Final results
print("\n=== Cross-Validation Results ===")
print(f"Sentence-Level Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")
