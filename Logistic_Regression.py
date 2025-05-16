# Victoria

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_train.csv")
# Uncomment below if only using degree and eccentricity features:
# data = pd.read_csv("v3_preprocessed_train_degree_eccentricity.csv")

# Define features and target
categorical_features = ['language']
numerical_features = ['degree', 'harmonic', 'betweenness', 'pagerank', 'eigenvector', 'eccentricity'] # change if using other dataset

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Transform features
X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_id']  # Group by sentence ID to avoid leakage

# Initialize Logistic Regression model
model = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',  # Supports L1 and L2 penalties, works well on small datasets
    penalty='l2',
    max_iter=1000,
    random_state=42
)

# Define F1 scorer for class 1
f1_scorer = make_scorer(f1_score, pos_label=1)

# Grouped Cross-Validation (5 folds)
group_kfold = GroupKFold(n_splits=5)
cv_scores = cross_val_score(
    model, X, y,
    cv=group_kfold,
    groups=groups,
    scoring=f1_scorer
)

# Print results
print(f"F1 Scores per fold: {cv_scores}")
print(f"Mean F1 (Class 1): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

