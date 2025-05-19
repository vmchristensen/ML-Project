import pandas as pd
from joblib import load
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_test.csv"
test_data = pd.read_csv(data_path)

# Load model and preprocessor
model_path = script_dir.parent / "05_Final Model" / "model.joblib"
preprocessor_path = script_dir.parent / "05_Final Model" / "preprocessor.joblib"
model = load(model_path)
preprocessor = load(preprocessor_path)

# Transform features
X_test = preprocessor.transform(test_data)

# Predict probabilities
proba = model.predict_proba(X_test)[:, 1]
test_data['proba'] = proba

# Select root with highest probability per sentence
selected_roots = test_data.loc[test_data.groupby('id')['proba'].idxmax()]

# Create submission file
submission = selected_roots[['id', 'vertex_id']].rename(
    columns={'id': 'id', 'vertex_id': 'root'}
)
output_path = script_dir.parent / "01_Data" / "submission.csv"
submission.to_csv(output_path, index=False)