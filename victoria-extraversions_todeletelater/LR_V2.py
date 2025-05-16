
# Using the original preprocessed dataset with all features
import pandas as pd
from sklearn.linear_model import LogisticRegression # Import Logistic Regression
from sklearn.preprocessing import OneHotEncoder, StandardScaler # StandardScaler for numerical features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # To chain preprocessing and model
from sklearn.model_selection import GroupKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, balanced_accuracy_score
from scipy.stats import uniform, loguniform # For hyperparameter search

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_train.csv") # USING THE ORIGINAL FULL FEATURE SET
# data = pd.read_csv("v3_preprocessed_train_selected_features.csv")
# data = pd.read_csv("v4_preprocessed_train_degree_eccentricity.csv")
print(f"Loaded data with shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")


# Define features and target
categorical_features = ['language']

# Adjust numerical_features based on the CSV loaded
# This logic should correctly pick up all 6 centralities from "preprocessed_train.csv"
if 'pagerank' in data.columns and 'eigenvector' in data.columns: # From preprocessed_train.csv
    numerical_features = ['degree', 'harmonic', 'betweenness', 'pagerank', 'eigenvector', 'eccentricity']
elif 'harmonic' in data.columns and 'betweenness' in data.columns and 'pagerank' not in data.columns: # From v3_...
    numerical_features = ['degree', 'harmonic', 'betweenness', 'eccentricity']
elif 'degree' in data.columns and 'eccentricity' in data.columns and 'harmonic' not in data.columns: # From v4_... (degree, eccentricity)
    numerical_features = ['degree', 'eccentricity']
else:
    # Fallback or raise error if columns are unexpected
    potential_num_cols = [col for col in data.columns if col not in categorical_features + ['target', 'sentence_group', 'vertex_id', 'sentence_id', 'n']]
    print(f"Automatically detected numerical features: {potential_num_cols}")
    if not potential_num_cols:
        raise ValueError("No numerical features detected or defined. Please check your CSV and feature definitions.")
    numerical_features = potential_num_cols

print(f"Using categorical features: {categorical_features}")
print(f"Using numerical features: {numerical_features}")


# --- Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features) # Scale numerical features
    ])

# Define the target and groups
y = data['target']
groups = data['sentence_group'] # Use the combined group identifier

# --- Initialize Logistic Regression Model ---
log_reg_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])

# --- Define Scorers ---
f1_scorer_class1 = make_scorer(f1_score, pos_label=1, average='binary')
roc_auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
balanced_accuracy_scorer_custom = make_scorer(balanced_accuracy_score) # Renamed to avoid conflict if sklearn also has one

scoring_metrics = {
    'f1_class1': f1_scorer_class1,
    'roc_auc': roc_auc_scorer,
    'balanced_accuracy': balanced_accuracy_scorer_custom
}

# --- Grouped Cross-Validation (5 folds) ---
print("\n--- Logistic Regression Cross-Validation (Default Params) ---")
group_kfold = GroupKFold(n_splits=5)
cv_scores_all_metrics = {}

for metric_name, scorer_func in scoring_metrics.items():
    print(f"Calculating CV for metric: {metric_name}")
    current_cv_scores = cross_val_score(
        log_reg_model,
        data, # Pass full dataframe, pipeline handles selection and preprocessing
        y,
        cv=group_kfold,
        groups=groups,
        scoring=scorer_func,
        n_jobs=-1
    )
    cv_scores_all_metrics[metric_name] = current_cv_scores
    print(f"{metric_name} Scores per fold: {current_cv_scores}")
    print(f"Mean {metric_name}: {current_cv_scores.mean():.4f} (±{current_cv_scores.std():.4f})\n")


# --- Hyperparameter Tuning for Logistic Regression ---
print("\n--- Logistic Regression Hyperparameter Tuning (RandomizedSearchCV) ---")

param_dist_log_reg = {
    'classifier__C': loguniform(1e-4, 1e3), # Expanded range for C
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear'] # Kept as liblinear as it supports both penalties
    # Consider adding 'classifier__class_weight': ['balanced', {0:1, 1:N} for different N values] if default 'balanced' isn't optimal
}

random_search_log_reg = RandomizedSearchCV(
    estimator=log_reg_model,
    param_distributions=param_dist_log_reg,
    n_iter=50, # Increased iterations for a more thorough search
    cv=GroupKFold(n_splits=5),
    scoring=f1_scorer_class1, # Optimize for F1 of class 1
    refit=True,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Pass the original DataFrame (X features) to fit, pipeline handles preprocessing
random_search_log_reg.fit(data, y, groups=groups)

print("\n=== Best Hyperparameters for Logistic Regression ===")
print(random_search_log_reg.best_params_)

print("\n=== Best F1 Score (Class 1) from Tuning ===")
print(f"Validation F1 (Class 1): {random_search_log_reg.best_score_:.4f}")

best_log_reg_model = random_search_log_reg.best_estimator_

print("\n--- Re-evaluating Best Tuned Logistic Regression Model on CV ---")
for metric_name, scorer_func in scoring_metrics.items():
    current_cv_scores_best = cross_val_score(
        best_log_reg_model,
        data,
        y,
        cv=group_kfold,
        groups=groups,
        scoring=scorer_func,
        n_jobs=-1
    )
    print(f"Best Tuned Model Mean {metric_name}: {current_cv_scores_best.mean():.4f} (±{current_cv_scores_best.std():.4f})")