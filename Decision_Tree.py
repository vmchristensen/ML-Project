# Victoria – Final Decision Tree Model using best hyperparameters

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_train.csv")
# data = pd.read_csv("v3_preprocessed_train_degree_eccentricity.csv")  # if using simpler set

# Define features and target
categorical_features = ['language']
numerical_features = ['degree', 'harmonic', 'betweenness', 'pagerank', 'eigenvector', 'eccentricity']  # change if needed

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Transform features
X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_id']

# Best tuned Decision Tree model
model = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features='log2',
    random_state=42
)

# Define F1 scorer for class 1
f1_scorer = make_scorer(f1_score, pos_label=1)

# Cross-validation
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

# Fit on full data (optional, e.g. for final predictions)
model.fit(X, y)
