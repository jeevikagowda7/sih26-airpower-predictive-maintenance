import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- 1. Load the data (same as Step 1) ----------
cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul_true = pd.read_csv("data/RUL_FD001.txt", header=None, names=["true_rul"])

# ---------- 2. Make the answer column (RUL = flights left) ----------
# In training, every engine ran until it broke, so we KNOW the flights left:
# flights left = (last flight of this engine) - (this flight)
max_cycle = train.groupby("unit")["cycle"].transform("max")
train["rul"] = max_cycle - train["cycle"]

# A young engine is "healthy", so 300 vs 250 flights left makes no real difference.
# Common practice: cap the answer at 125 ("125 or more = healthy").
CAP = 125
train["rul_capped"] = train["rul"].clip(upper=CAP)

# ---------- 3. Keep only sensors that change (the others are flat lines) ----------
sensors = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
           "s13", "s14", "s15", "s17", "s20", "s21"]
print("Sensors used:", sensors)

# ---------- 4. Smooth the noise ----------
# Sensors are noisy. Average of the last 5 flights of the SAME engine = calmer line.
def add_smooth(df):
    df = df.copy()
    for s in sensors:
        df[s + "_avg"] = df.groupby("unit")[s].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
    return df

train = add_smooth(train)
test = add_smooth(test)
features = [s + "_avg" for s in sensors]

# ---------- 5. Train a Random Forest ----------
X_train = train[features]
y_train = train["rul_capped"]

model = RandomForestRegressor(n_estimators=200, max_depth=12,
                              min_samples_leaf=5, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# ---------- 6. Test on the 100 unseen engines ----------
# For each test engine we only look at its LAST flight (= "today").
last_rows = test.groupby("unit").tail(1).reset_index(drop=True)
pred = model.predict(last_rows[features])

true_rul = rul_true["true_rul"].values
true_capped = np.minimum(true_rul, CAP)

def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name:<38} RMSE = {rmse:6.1f} flights | MAE = {mae:6.1f} flights")

print("\n=== HONEST RESULTS on 100 unseen test engines ===")
# Baseline = a lazy guesser who always says "the average"
lazy = np.full(len(true_rul), y_train.mean())
report("Lazy guess (always the average)", true_capped, lazy)
report("Our Random Forest (vs capped truth)", true_capped, pred)
report("Our Random Forest (vs raw truth)", true_rul, pred)

# How often are we badly wrong in the DANGEROUS direction (we say safe, truth is near failure)?
dangerous = np.sum((pred > true_rul + 20))
print(f"\nEngines where we were 20+ flights too optimistic: {dangerous} of 100")

# ---------- 7. Save everything for the dashboard ----------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
joblib.dump({"model": model, "features": features, "sensors": sensors, "cap": CAP},
            "models/rul_model.pkl", compress=3)

out = pd.DataFrame({
    "unit": last_rows["unit"],
    "flights_flown": last_rows["cycle"],
    "pred_rul": pred.round(1),
    "true_rul": true_rul,
})
out.to_csv("outputs/test_predictions.csv", index=False)
print("\nSaved: models/rul_model.pkl and outputs/test_predictions.csv")
print(out.head(8).to_string(index=False))