import pandas as pd
from joblib import load

# Load preprocessed test data
test_data = pd.read_csv('preprocessed_test.csv')

# Load model and preprocessor
preprocessor = load('preprocessor.joblib')
model = load('model.joblib')

# Transform features
X_test = preprocessor.transform(test_data)

# Predict probabilities
proba = model.predict_proba(X_test)[:, 1]
test_data['proba'] = proba

# Select root with highest probability per sentence
selected_roots = test_data.groupby('sentence_id', group_keys=False).apply(
    lambda x: x.loc[x['proba'].idxmax()]
)

# Create submission file
submission = selected_roots[['sentence_id', 'vertex_id']].rename(
    columns={'sentence_id': 'id', 'vertex_id': 'root'}
)
submission.to_csv('submission.csv', index=False)