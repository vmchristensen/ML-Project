import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_train.csv")

# Define features and target
categorical_features = ['language']
numerical_features = ['degree', 'harmonic', 'betweenness', 'pagerank', 'eigenvector', 'eccentricity']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_id']  # Group by sentence ID to avoid leakage

# Initialize model
model = RandomForestClassifier(
    n_estimators=20,          
    max_depth=15,             
    class_weight='balanced',
    n_jobs=-1,               # Use all CPU cores
    random_state=42
)

# Define F1 scorer for class 1
f1_scorer = make_scorer(f1_score, pos_label=1)  # Focus on root class (1)

# Grouped Cross-Validation (5 folds)
group_kfold = GroupKFold(n_splits=5)
cv_scores = cross_val_score(
    model, X, y,
    cv=group_kfold,
    groups=groups,
    scoring=f1_scorer  # Use F1 for class 1
)

# Print results
print(f"F1 Scores per fold: {cv_scores}")
print(f"Mean F1 (Class 1): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")



"""
HYPERPARAMETERS CODE

# Define hyperparameter search space
param_dist = {
    'n_estimators': randint(100, 500),       # Number of trees: 100-500
    'max_depth': [None] + list(range(10, 60, 10)),  # Depth limits
    'min_samples_split': randint(2, 20),     # Splitting threshold
    'min_samples_leaf': randint(1, 10),      # Leaf node requirements
    'max_features': ['sqrt', 'log2', 0.8],   # Feature sampling
    'class_weight': ['balanced', {0:1, 1:5}] # Handle class imbalance
}

# Initialize randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,                  # Trade-off: computation vs. coverage
    cv=GroupKFold(n_splits=5),   # Grouped cross-validation
    scoring=f1_scorer,           # Optimize for F1 of class 1
    n_jobs=-1,                   # Use all CPU cores
    verbose=1,
    random_state=42
)

# Run hyperparameter search
random_search.fit(X, y, groups=groups)

# Show results
print("\n=== Best Hyperparameters ===")
print(random_search.best_params_)

print("\n=== Best F1 Score (Class 1) ===")
print(f"Validation: {random_search.best_score_:.3f}")

# Train final model with best parameters
best_model = random_search.best_estimator_
best_model.fit(X, y)  # Optional: train on full dataset
"""