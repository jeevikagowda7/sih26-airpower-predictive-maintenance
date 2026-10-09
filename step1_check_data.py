import pandas as pd

   # Names for the 26 columns (the file has no header, so we give names)
cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

   # unit = engine number, cycle = flight number, op = flight settings, s1..s21 = sensors
train = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul_true = pd.read_csv("data/RUL_FD001.txt", header=None, names=["true_rul"])

print("TRAIN rows, columns:", train.shape)
print("TEST rows, columns:", test.shape)
print("Engines in train:", train["unit"].nunique())
print("Engines in test:", test["unit"].nunique())

   # How long did each training engine live before failing?
life = train.groupby("unit")["cycle"].max()
print("\nTraining engine life (flights until failure):")
print(life.describe().round(1))

print("\nFirst 5 rows of engine 1:")
print(train[train["unit"] == 1].head())

print("\nTrue remaining flights for first 5 test engines:")
print(rul_true.head())