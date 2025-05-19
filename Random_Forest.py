import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, f1_score 

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
groups = data['sentence_group']  

# Get feature names after preprocessing
lang_encoder = preprocessor.named_transformers_['lang']
lang_features = lang_encoder.get_feature_names_out(categorical_features)
num_features = numerical_features
all_features = np.concatenate([lang_features, num_features])

# Initialize model
model = RandomForestClassifier(
    n_estimators=20,          
    max_depth=30, 
    min_samples_split=10,
    min_samples_leaf=4,
    max_features=0.8,            
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# Custom cross-validation for sentence-level accuracy
group_kfold = GroupKFold(n_splits=5)
accuracies = []
instance_f1_0 = []  
instance_f1_1 = []  
feature_importances = []

for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
    print(f"Fold {fold_idx + 1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)

    # ---- Instance-Level Metrics ----
    y_pred = model.predict(X_test)
    print("\nInstance-Level Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-root', 'Root'], digits=3))
    
    # Store F1 scores
    instance_f1_0.append(f1_score(y_test, y_pred, pos_label=0))
    instance_f1_1.append(f1_score(y_test, y_pred, pos_label=1))
    
    # ---- Feature Importance ----
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    feature_importances.append(importance_df)
    print(f"\nTop 10 Features for Fold {fold_idx + 1}:")
    print(importance_df.head(10).to_string(index=False))

    # Predict probabilities for class 1 (root)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Create DataFrame with test data indices
    test_data = data.iloc[test_idx].copy()
    test_data['proba'] = proba
    
    # Group by sentence_id and select node with highest probability
    selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]
    
    # Calculate accuracy
    accuracy = (selected_roots['target'] == 1).mean()
    accuracies.append(accuracy)
    print(f"Fold {fold_idx + 1} Sentence-Level Accuracy: {accuracy:.3f}")

# ---- Aggregate Results ----
print("\n=== Cross-Validation Results ===")
print(f"Sentence-Level Accuracies: {[round(acc, 3) for acc in accuracies]}")
print(f"Mean Sentence-Level Accuracy: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")

print("\n=== Instance-Level Metrics ===")
print(f"Non-Root F1 Scores: {[round(f1, 3) for f1 in instance_f1_0]}")
print(f"Mean Non-Root F1: {np.mean(instance_f1_0):.3f} (±{np.std(instance_f1_0):.3f})")
print(f"\nRoot F1 Scores: {[round(f1, 3) for f1 in instance_f1_1]}")
print(f"Mean Root F1: {np.mean(instance_f1_1):.3f} (±{np.std(instance_f1_1):.3f})")

# ---- Average Feature Importance ----
average_importance = (
    pd.concat(feature_importances)
    .groupby('feature')['importance']
    .mean()
    .sort_values(ascending=False)
)
print("\n=== Average Feature Importance Across Folds ===")
print(average_importance.head(20).to_string())

"""
=== Best Hyperparameters ===
{'n_estimators': 20, 'max_depth': 30, 'min_samples_split': 18, 'min_samples_leaf': 7, 'max_features': 3, 'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42}    
Validation Accuracy: 0.370
🔍 Testing parameters: {'n_estimators': 50, 'max_depth': 20, 'min_samples_split': 17, 'min_samples_leaf': 8, 'max_features': 'log2', 'class_weight': None, 'n_jobs': -1, 'random_state': 42}
✅ Iteration result: Score = 0.382
🔍 Testing parameters: {'n_estimators': 20, 'max_depth': 30, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 0.8, 'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42}
✅ Iteration result: Score = 0.376
Testing parameters: {'n_estimators': 20, 'max_depth': 40, 'min_samples_split': 14, 'min_samples_leaf': 7, 'max_features': 0.8, 'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42}
✅ Iteration result: Score = 0.376
"""