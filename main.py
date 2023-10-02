import os
import pandas as pd
import numpy as np
from sklearn import OneHotEncoder

TRAINDIR = "C:/Users/Kivaari/Desktop/KAGEL/data/df_train.csv"
TESTDIR = "C:/Users/Kivaari/Desktop/KAGEL/data/df_test.csv"
COLS = ["Id", "Data", "Target"]
TARGET_COL = "Target"

TESTSIZE = 0.3
RANDOMSTATE = 1488

df_train = pd.read_csv(TRAINDIR, sep=';')
df_test = pd.read_csv(TESTDIR, sep=';')

df_train['Data'] = df_train.Data.apply(lambda s: list(map(int, s.split(','))))
df_train['Target'] = df_train.Target.apply(lambda s: list(map(int, s.split(','))))
df_test['Data'] = df_test.Data.apply(lambda s: list(map(int, s.split(','))))

y = df_train[TARGET_COL]
x = df_train.drop([TARGET_COL], axis = 1)

