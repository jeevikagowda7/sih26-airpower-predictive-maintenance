import numpy as np
import pandas as pd

FLIGHTS_PER_DAY = 3      # ASSUMPTION (same as Step 4)
HORIZON = 7              # we forecast 7 days


def make_plan(fleet, spares, slots, extra_resupply=0, extra_spares=0, slots_lost=0):
    """Same scheduler as Step 4, but with 'what-if' knobs.

    fleet  : table from outputs/fleet_status.csv
    spares : data/SIMULATED_spares.csv
    slots  : data/SIMULATED_slots.csv  (columns: day, Base-A, Base-B, Base-C)
    """
    plan = fleet.sort_values("unit").reset_index(drop=True).copy()   # original order
    bases = list(spares["base"])

    stock = {b: int(s) + int(extra_spares)
             for b, s in zip(spares["base"], spares["spares_in_stock"])}
    resupply = {b: int(r) + int(extra_resupply)
                for b, r in zip(spares["base"], spares["resupply_days"])}
    free = {b: [max(0, int(v) - int(slots_lost)) for v in slots[b].tolist()]
            for b in bases}

    def first_free_day(base, from_day):
        for d in range(from_day, HORIZON + 1):
            if d >= 1 and free[base][d - 1] > 0:
                return d
        return None

    plan["spare_status"] = "-"
    plan["slot_day"] = np.nan
    plan["action"] = "No action"

    needs = plan[plan["status"].isin(["RED", "YELLOW"])].sort_values(
        "risk_score", ascending=False)

    for idx, row in needs.iterrows():
        b = row["base"]
        if stock[b] > 0:
            stock[b] -= 1
            plan.at[idx, "spare_status"] = "In stock"
            start_day = 1
        else:
            plan.at[idx, "spare_status"] = f"SHORT (arrives day {resupply[b]})"
            start_day = resupply[b]
        day = first_free_day(b, start_day)
        if day is None:
            plan.at[idx, "action"] = "No slot in next 7 days - ESCALATE"
        else:
            free[b][day - 1] -= 1
            plan.at[idx, "slot_day"] = day
            plan.at[idx, "action"] = f"Maintain on day {day}"
            if day > row["days_left"]:
                plan.at[idx, "action"] += "  (!! may fail before slot)"
    plan["true_days_left"] = plan["true_rul"] / FLIGHTS_PER_DAY
    return plan


def availability(plan, basis="days_left", planned=True, maint_days=2, unplanned_extra=3):
    """Count aircraft that are UP on each of the next 7 days.

    basis    : 'days_left' (our estimate) or 'true_days_left' (NASA's true answer)
    planned  : True  = follow our maintenance plan
               False = do nothing (fix only after failure)
    maint_days      : days an aircraft is in the hangar for planned maintenance
    unplanned_extra : extra days lost when a failure is a surprise
    Returns (list of 7 counts, number of unplanned failures in the week)
    """
    M, U = int(maint_days), int(unplanned_extra)
    up = np.ones((len(plan), HORIZON), dtype=bool)
    unplanned = 0
    for i in range(len(plan)):
        fail_day = int(np.floor(plan[basis].iloc[i])) + 1     # first day it is past its life
        slot = plan["slot_day"].iloc[i] if planned else np.nan
        fixed_in_time = (not np.isnan(slot)) and slot <= fail_day
        if fixed_in_time:
            down = range(int(slot), int(slot) + M)
        elif fail_day <= HORIZON:
            down = range(fail_day, fail_day + M + U)
            unplanned += 1
        else:
            down = []
        for d in down:
            if 1 <= d <= HORIZON:
                up[i, d - 1] = False
    return up.sum(axis=0).tolist(), unplanned