import os
import numpy as np
import pandas as pd

# =====================================================================
# SETTINGS (our assumptions - you can change these and explain them to judges)
# =====================================================================
SAFETY_MARGIN = 10      # flights we subtract from the model's guess to be extra careful
RED_BELOW = 30          # planning flights-left at or below this = RED
YELLOW_BELOW = 60       # planning flights-left at or below this = YELLOW
FLIGHTS_PER_DAY = 3     # ASSUMPTION: each aircraft flies about 3 flights per day

# ---------------------------------------------------------------------
# 1. Load the model's predictions (from Step 3)
# ---------------------------------------------------------------------
fleet = pd.read_csv("outputs/test_predictions.csv")
fleet["aircraft"] = "AC-" + fleet["unit"].astype(str).str.zfill(3)

# Planning RUL = model guess minus safety margin (never below 0)
fleet["plan_rul"] = (fleet["pred_rul"] - SAFETY_MARGIN).clip(lower=0).round(1)
fleet["days_left"] = (fleet["plan_rul"] / FLIGHTS_PER_DAY).round(1)

# ---------------------------------------------------------------------
# 2. Green / Yellow / Red
# ---------------------------------------------------------------------
def status(rul):
    if rul <= RED_BELOW:
        return "RED"
    if rul <= YELLOW_BELOW:
        return "YELLOW"
    return "GREEN"

fleet["status"] = fleet["plan_rul"].apply(status)

# Failure-risk score 0..100 (0 = far from failure, 100 = about to fail)
fleet["risk_score"] = (100 * (1 - fleet["plan_rul"].clip(upper=80) / 80)).round(0)

# ---------------------------------------------------------------------
# 3. SIMULATED data: bases, spare parts, maintenance slots
#    (NOT real. Made up by us to show how data can be joined together.)
# ---------------------------------------------------------------------
rng = np.random.default_rng(42)
bases = ["Base-A", "Base-B", "Base-C"]
fleet["base"] = rng.choice(bases, size=len(fleet))

# Spare "compressor module" stock at each base, and days to get a new one
spares = pd.DataFrame({
    "base": bases,
    "spares_in_stock": [3, 1, 0],
    "resupply_days": [2, 4, 6],
})

# Free hangar slots per base for the next 7 days
slot_table = {
    "Base-A": [1, 2, 2, 1, 2, 2, 1],
    "Base-B": [0, 1, 1, 0, 1, 1, 1],
    "Base-C": [1, 0, 1, 1, 0, 1, 1],
}
slots = pd.DataFrame(slot_table, index=range(1, 8))
slots.index.name = "day"

os.makedirs("data", exist_ok=True)
spares.to_csv("data/SIMULATED_spares.csv", index=False)
slots.reset_index().to_csv("data/SIMULATED_slots.csv", index=False)

# ---------------------------------------------------------------------
# 4. Mini scheduler: give spares and slots to the riskiest aircraft first
# ---------------------------------------------------------------------
stock = dict(zip(spares["base"], spares["spares_in_stock"]))
resupply = dict(zip(spares["base"], spares["resupply_days"]))
free = {b: slots[b].tolist() for b in bases}      # free[b][0] = day 1

def first_free_day(base, from_day):
    """Earliest day (1..7) at or after from_day with a free slot, else None."""
    for d in range(from_day, 8):
        if free[base][d - 1] > 0:
            return d
    return None

fleet["spare_status"] = "-"
fleet["slot_day"] = np.nan
fleet["action"] = "No action"

needs_work = fleet[fleet["status"].isin(["RED", "YELLOW"])].sort_values(
    "risk_score", ascending=False)

for idx, row in needs_work.iterrows():
    b = row["base"]
    if stock[b] > 0:
        stock[b] -= 1
        fleet.at[idx, "spare_status"] = "In stock"
        start_day = 1
    else:
        fleet.at[idx, "spare_status"] = f"SHORT (arrives day {resupply[b]})"
        start_day = resupply[b]

    day = first_free_day(b, start_day)
    if day is None:
        fleet.at[idx, "action"] = "No slot in next 7 days - ESCALATE"
    else:
        free[b][day - 1] -= 1
        fleet.at[idx, "slot_day"] = day
        fleet.at[idx, "action"] = f"Maintain on day {day}"
        # Will it likely fail before its slot?
        if day > row["days_left"]:
            fleet.at[idx, "action"] += "  (!! may fail before slot)"

# ---------------------------------------------------------------------
# 5. Priority score (0..100): risk + extra push when spares are SHORT
# ---------------------------------------------------------------------
supply_pressure = fleet["spare_status"].str.startswith("SHORT").astype(int) * 100
fleet["priority_score"] = np.where(
    fleet["status"] == "GREEN", 0,
    (0.7 * fleet["risk_score"] + 0.3 * supply_pressure).round(0))

fleet = fleet.sort_values("priority_score", ascending=False).reset_index(drop=True)
cols = ["aircraft", "base", "status", "plan_rul", "days_left", "risk_score",
        "spare_status", "action", "priority_score", "pred_rul", "true_rul",
        "flights_flown", "unit"]
fleet = fleet[cols]
fleet.to_csv("outputs/fleet_status.csv", index=False)

# ---------------------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------------------
print("=== FLEET SUMMARY (100 aircraft) ===")
print(fleet["status"].value_counts().to_string())

print("\n=== Honest check: did our RED alert catch the really-at-risk engines? ===")
truly_risky = fleet["true_rul"] <= RED_BELOW
caught = (fleet["status"] == "RED") & truly_risky
false_red = (fleet["status"] == "RED") & ~truly_risky
print(f"Engines truly within {RED_BELOW} flights of failure: {truly_risky.sum()}")
print(f"  Caught as RED: {caught.sum()}   Missed: {(truly_risky & ~caught).sum()}")
print(f"False RED alarms: {false_red.sum()}")
print("\n=== TOP 10 MAINTENANCE PRIORITIES ===")
show = ["aircraft", "base", "status", "plan_rul", "spare_status", "action", "priority_score"]
print(fleet[show].head(10).to_string(index=False))

print("\nSaved: outputs/fleet_status.csv, data/SIMULATED_spares.csv, data/SIMULATED_slots.csv")