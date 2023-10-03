import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from vars import (TRAINDIR, TESTDIR, COLS, REAL_COLS, 
                  TARGET_COL, TESTSIZE, RANDOMSTATE, MODEL_NAME)

df_train = pd.read_csv(TRAINDIR, sep=';')
df_test = pd.read_csv(TESTDIR, sep=';')

df_train['Data'] = df_train.Data.apply(lambda s: list(map(int, s.split(','))))
df_train['Target'] = df_train.Target.apply(lambda s: list(map(int, s.split(','))))
df_test['Data'] = df_test.Data.apply(lambda s: list(map(int, s.split(','))))

print(df_train.head())
# df_train = df_train.loc[:, COLS]

# y = df_train[TARGET_COL]
# x = df_train.drop([TARGET_COL], axis=1)

# train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE)

# train_y = train_y.to_frame(name="Target")
# test_y = test_y.to_frame(name="Target")

# transforms = ColumnTransformer([
#     ("mlb", MultiLabelBinarizer(), ["Data"]),
# ])

# pipeline = Pipeline([
#     ("transforms", transforms),
#     ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOMSTATE)),
# ])

# # print(test_y.head())
# pipeline.fit(train_X, train_y)
# pred = pipeline.predict_proba(test_X)

# top_10_predictions = []

# for probs in pred:
#     top_10_indices = (-probs).argsort()[:10]  # Индексы 10 наибольших вероятностей
#     top_10_transactions = pipeline.classes_[top_10_indices]  # MCC-коды для 10 наиболее вероятных транзакций
#     top_10_predictions.append(top_10_transactions)

# print(average_precision_score(test_y, top_10_predictions, average='macro'))

# with open(MODEL_NAME, "wb") as f:
#     pickle.dump(pipeline, f)
