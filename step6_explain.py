import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Plain-English names for the sensors (from the NASA C-MAPSS paper, Table 3)
# ---------------------------------------------------------------------
NAMES = {
    "s2": "LPC outlet temperature (T24)",
    "s3": "HPC outlet temperature (T30)",
    "s4": "LPT outlet temperature (T50)",
    "s7": "HPC outlet pressure (P30)",
    "s8": "Physical fan speed (Nf)",
    "s9": "Physical core speed (Nc)",
    "s11": "HPC outlet static pressure (Ps30)",
    "s12": "Fuel flow / Ps30 ratio (phi)",
    "s13": "Corrected fan speed (NRf)",
    "s14": "Corrected core speed (NRc)",
    "s15": "Bypass ratio (BPR)",
    "s17": "Bleed enthalpy (htBleed)",
    "s20": "HPT coolant bleed (W31)",
    "s21": "LPT coolant bleed (W32)",
}
sensors = list(NAMES.keys())

cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=cols)
fleet = pd.read_csv("outputs/fleet_status.csv")

# ---------------------------------------------------------------------
# 1. "Noise unit" for each sensor = how much it normally jiggles when healthy.
#    We measure it from the first 20 flights of every TRAINING engine.
# ---------------------------------------------------------------------
early_train = train[train["cycle"] <= 20]
noise = early_train.groupby("unit")[sensors].std().mean()

# ---------------------------------------------------------------------
# 2. For every aircraft: how far has each sensor drifted from ITS OWN early life?
#    drift = (average of last 5 flights - average of first 20 flights) / noise unit
# ---------------------------------------------------------------------
rows = []
for unit, g in test.groupby("unit"):
    g = g.sort_values("cycle")
    start = g[sensors].iloc[:20].mean()
    now = g[sensors].iloc[-5:].mean()
    drift = (now - start) / noise
    rec = {"unit": unit}
    for s in sensors:
        rec["drift_" + s] = round(float(drift[s]), 2)
    rows.append(rec)
drift_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 3. Top 3 sensors that moved the most + one plain sentence
# ---------------------------------------------------------------------
def make_text(row):
    d = {s: row["drift_" + s] for s in sensors}
    top = sorted(d, key=lambda s: abs(d[s]), reverse=True)[:3]
    parts = []
    for s in top:
        direction = "rose" if d[s] > 0 else "fell"
        parts.append(f"{NAMES[s]} {direction} {abs(d[s]):.1f}x noise")
    return pd.Series({"top1": top[0], "top2": top[1], "top3": top[2],
                      "why_text": "; ".join(parts)})

drift_df = pd.concat([drift_df, drift_df.apply(make_text, axis=1)], axis=1)

out = fleet[["aircraft", "unit", "status"]].merge(drift_df, on="unit")
out["main_reason"] = out["top1"].map(NAMES)
out.to_csv("outputs/explanations.csv", index=False)

# ---------------------------------------------------------------------
# 4. Print a few examples + honest summary
# ---------------------------------------------------------------------
print("Saved: outputs/explanations.csv\n")
for status in ["RED", "YELLOW", "GREEN"]:
    ex = out[out["status"] == status].head(2)
    for _, r in ex.iterrows():
        print(f"[{status}] {r['aircraft']}: {r['why_text']}")

print("\nMost common #1 reason among RED aircraft:")
print(out[out["status"] == "RED"]["main_reason"].value_counts().head(5).to_string())