import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from vars import (TRAINDIR, TESTDIR, COLS, 
                  TARGET_COL, TESTSIZE, RANDOMSTATE, MODEL_NAME)

# Define a custom transformer to convert 'Data' column to binary features
class DataToBinaryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if 'Data' in X.columns:  # Check if 'Data' column exists
            X_copy = X.copy()
            X_copy['Data'] = X_copy['Data'].apply(lambda lst: [1 if val in lst else 0 for val in range(1, max(lst) + 1)])
            return X_copy
        else:
            return X  # No 'Data' column, return X as is

# Read Data
df_train = pd.read_csv(TRAINDIR, sep=';')
df_test = pd.read_csv(TESTDIR, sep=';')

# Data Preprocessing
df_train['Data'] = df_train.Data.apply(lambda s: list(map(int, s.split(','))))
df_train['Target'] = df_train.Target.apply(lambda s: list(map(int, s.split(','))))
df_test['Data'] = df_test.Data.apply(lambda s: list(map(int, s.split(','))))

# Select Columns
df_train = df_train.loc[:, COLS]

if 'Data' in df_train.columns:
    # Create binary columns for each unique integer in the 'Data' column
    binary_data_columns = []

    for val in range(1, max(df_train['Data'].apply(max)) + 1):
        binary_data_columns.append(pd.Series([int(val in lst) for lst in df_train['Data']], name=f'Data_{val}'))

    # Concatenate all binary columns to the original DataFrame
    df_train = pd.concat([df_train] + binary_data_columns, axis=1)

    # Drop the original 'Data' column
    df_train.drop(['Data'], axis=1, inplace=True)

# Repeat the same for the test set
binary_data_columns_test = []

for val in range(1, max(df_test['Data'].apply(max)) + 1):
    binary_data_columns_test.append(pd.Series([int(val in lst) for lst in df_test['Data']], name=f'Data_{val}'))

df_test = pd.concat([df_test] + binary_data_columns_test, axis=1)

# Drop the original 'Data' column for the test set
df_test.drop(['Data'], axis=1, inplace=True)

# Split Data
y = df_train[TARGET_COL]
x = df_train.drop([TARGET_COL], axis=1)
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE)
test_y = test_y.to_frame(name="Target")
train_y = train_y.to_frame(name="Target")

# Create the pipeline with the custom transformer and RandomForestClassifier
pipeline = Pipeline([
    ("data_to_binary", DataToBinaryTransformer()),  # Custom transformer
    ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOMSTATE)),
])

# Fit the pipeline
pipeline.fit(train_X, train_y)

# Predict probabilities
pred = pipeline.predict_proba(test_X)

# Top 10 Predictions
top_10_predictions = []

for probs in pred:
    top_10_indices = (-probs).argsort()[:10]
    top_10_predictions.append(top_10_indices)

# Print the average precision score
print(average_precision_score(test_y, top_10_predictions, average='macro'))
