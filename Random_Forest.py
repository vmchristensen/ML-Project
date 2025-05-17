import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
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
groups = data['sentence_id']  # Group by sentence ID

# Initialize model
model = RandomForestClassifier(
    n_estimators=20,          
    max_depth=15,             
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# Custom cross-validation for sentence-level accuracy
group_kfold = GroupKFold(n_splits=5)
accuracies = []

for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
    print(f"Fold {fold_idx + 1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups.iloc[test_idx]  # Get sentence IDs for test fold
    
    model.fit(X_train, y_train)
    # Predict probabilities for class 1 (root)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Create DataFrame with test data indices
    test_data = data.iloc[test_idx].copy()
    test_data['proba'] = proba
    
    # Group by sentence_id and select node with highest probability
    selected_roots = test_data.groupby('sentence_id').apply(
        lambda x: x.loc[x['proba'].idxmax()]
    )
    
    # Calculate accuracy
    accuracy = (selected_roots['target'] == 1).mean()
    accuracies.append(accuracy)
    print(f"Fold {fold_idx + 1} Sentence-Level Accuracy: {accuracy:.3f}")

print("\n=== Cross-Validation Results ===")
print(f"Sentence-Level Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")

# Mean Accuracy: 0.344 (±0.044)


# Hyperparameter Tuning (Optional: Uncomment and adjust as needed)
"""
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(10, 60, 10)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.8],
    'class_weight': ['balanced', {0:1, 1:5}]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    cv=GroupKFold(n_splits=5),
    scoring=make_scorer(f1_score, pos_label=1),  # Optional: Replace with custom scorer
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X, y, groups=groups)
best_model = random_search.best_estimator_
"""

"""# After training, use this to predict on test data:
def predict_test_set(model, preprocessor, test_csv_path):
    test_data = pd.read_csv(test_csv_path)
    # Preprocess test data (ensure same preprocessing as training)
    X_test = preprocessor.transform(test_data)
    test_data['proba'] = model.predict_proba(X_test)[:, 1]
    
    # Group by sentence_id and select root with highest probability
    predictions = test_data.groupby('sentence_id').apply(
        lambda x: x.loc[x['proba'].idxmax()]
    ).reset_index()
    
    # Format submission (adjust column names as per your data)
    submission = predictions[['id', 'root']]  # Replace 'root' with the correct node ID column
    submission.columns = ['id', 'root']
    return submission

# Example usage:
# best_model.fit(X, y)  # Train on full data
# submission = predict_test_set(best_model, preprocessor, "preprocessed_test.csv")
# submission.to_csv("submission.csv", index=False)"""