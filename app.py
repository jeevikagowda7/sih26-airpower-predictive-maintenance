import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from planner import make_plan, availability

st.set_page_config(page_title="Air Power - Fleet Readiness", layout="wide")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    fleet = pd.read_csv("outputs/fleet_status.csv")
    expl = pd.read_csv("outputs/explanations.csv")
    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    sensors = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=cols)
    spares = pd.read_csv("data/SIMULATED_spares.csv")
    slots = pd.read_csv("data/SIMULATED_slots.csv")
    fleet = fleet.merge(expl[["aircraft", "main_reason"]], on="aircraft", how="left")
    return fleet, expl, sensors, spares, slots

fleet, expl, sensors, spares, slots = load_data()

COLORS = {"GREEN": "#2e9e5b", "YELLOW": "#e0a800", "RED": "#d64545"}

NAMES = {
    "s2": "LPC outlet temp (T24)", "s3": "HPC outlet temp (T30)",
    "s4": "LPT outlet temp (T50)", "s7": "HPC outlet pressure (P30)",
    "s8": "Fan speed (Nf)", "s9": "Core speed (Nc)",
    "s11": "HPC static pressure (Ps30)", "s12": "Fuel flow ratio (phi)",
    "s13": "Corrected fan speed (NRf)", "s14": "Corrected core speed (NRc)",
    "s15": "Bypass ratio (BPR)", "s17": "Bleed enthalpy (htBleed)",
    "s20": "HPT coolant bleed (W31)", "s21": "LPT coolant bleed (W32)",
}

def color_status(val):
    c = COLORS.get(val)
    return f"background-color: {c}; color: white; font-weight: bold" if c else ""

def style_status(df):
    styler = df.style.format(precision=1)
    # pandas 2.1+ uses .map ; older versions use .applymap
    if hasattr(styler, "map"):
        return styler.map(color_status, subset=["status"])
    return styler.applymap(color_status, subset=["status"])

# ---------------------------------------------------------------------
# Header + honesty banner
# ---------------------------------------------------------------------
st.title("Air Power: Predictive Maintenance & Fleet Availability")
st.caption("Smart India Hackathon 2026 | PS 26249 | Prototype")

st.info(
    "DATA HONESTY: Engine sensor data is the PUBLIC, SIMULATED NASA C-MAPSS (FD001) "
    "dataset, not real Air Force data. Bases, spare-part stock and maintenance slots are "
    "SIMULATED by us to show how fragmented data can be joined together."
)

# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------
st.sidebar.header("Filters")
base_pick = st.sidebar.multiselect("Base", sorted(fleet["base"].unique()),
                                   default=sorted(fleet["base"].unique()))
status_pick = st.sidebar.multiselect("Status", ["RED", "YELLOW", "GREEN"],
                                     default=["RED", "YELLOW", "GREEN"])
view = fleet[fleet["base"].isin(base_pick) & fleet["status"].isin(status_pick)]

# ---------------------------------------------------------------------
# Top tiles: Fleet Availability
# ---------------------------------------------------------------------
n_total = len(fleet)
n_green = int((fleet["status"] == "GREEN").sum())
n_yellow = int((fleet["status"] == "YELLOW").sum())
n_red = int((fleet["status"] == "RED").sum())
ready = n_green + n_yellow

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total aircraft", n_total)
c2.metric("GREEN (healthy)", n_green)
c3.metric("YELLOW (watch)", n_yellow)
c4.metric("RED (act now)", n_red)
c5.metric("Ready tomorrow", f"{ready} / {n_total}", f"{100 * ready / n_total:.0f}%")
st.caption("Ready tomorrow = GREEN + YELLOW. RED aircraft are treated as not ready "
           "(close to failure, maintenance needed).")

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
["Fleet table", "Alerts & priorities", "Aircraft detail", "Simulated data & accuracy","7-day forecast & what-if"])
# ---- Tab 1: fleet table
with tab1:
    st.subheader("Fleet table")
    show = ["aircraft", "base", "status", "plan_rul", "days_left", "main_reason",
            "spare_status", "action", "priority_score"]
    st.dataframe(style_status(view[show]), height=520)

    status_counts = fleet["status"].value_counts().reindex(["GREEN", "YELLOW", "RED"]).fillna(0)
    fig = go.Figure(go.Bar(x=status_counts.index, y=status_counts.values,
                           marker_color=[COLORS[s] for s in status_counts.index]))
    fig.update_layout(height=300, title="Aircraft by health status",
                      yaxis_title="Number of aircraft")
    st.plotly_chart(fig)

# ---- Tab 2: alerts
with tab2:
    st.subheader("Alerts: who to maintain first")
    red = fleet[fleet["status"] == "RED"]
    may_fail = red[red["action"].str.contains("may fail")]
    escalate = red[red["action"].str.contains("ESCALATE")]
    short = red[red["spare_status"].str.startswith("SHORT")]

    a1, a2, a3 = st.columns(3)
    a1.metric("RED aircraft with spares SHORT", len(short))
    a2.metric("Slot booked AFTER likely failure", len(may_fail))
    a3.metric("No slot in next 7 days", len(escalate))

    st.write("**Top 15 priorities** (priority score = 70% failure risk + 30% spares shortage)")
    top = fleet.head(15)[["aircraft", "base", "status", "plan_rul", "main_reason",
                          "spare_status", "action", "priority_score"]]
    st.dataframe(style_status(top))

# ---- Tab 3: aircraft detail
with tab3:
    st.subheader("Aircraft detail")
    pick = st.selectbox("Choose an aircraft", fleet["aircraft"].tolist())
    row = fleet[fleet["aircraft"] == pick].iloc[0]
    ex = expl[expl["aircraft"] == pick].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Status", row["status"])
    d2.metric("Estimated flights left", f"{row['plan_rul']:.0f}")
    d3.metric("Estimated days left", f"{row['days_left']:.1f}")
    d4.metric("Priority score", f"{row['priority_score']:.0f}")
    st.write(f"**Base:** {row['base']}  |  **Spares:** {row['spare_status']}  |  "
             f"**Action:** {row['action']}")

    # ---------- Why is it this colour? ----------
    st.markdown(f"### Why is {pick} {row['status']}?")
    st.write(ex["why_text"])
    st.caption("How to read this: we compare each sensor's last 5 flights with this "
               "aircraft's OWN first 20 flights, in units of normal sensor noise. "
               "2x noise or less is ordinary wobble. 5x or more is a clear drift. "
               "This shows which sensors moved most. It is a pointer for engineers, "
               "not a proven cause.")

    unit = int(row["unit"])
    sensor_ids = list(NAMES.keys())
    drift_vals = [float(ex["drift_" + s]) for s in sensor_ids]
    order = np.argsort(np.abs(drift_vals))
    bar_colors = [COLORS["RED"] if abs(drift_vals[i]) >= 5
                  else COLORS["YELLOW"] if abs(drift_vals[i]) >= 3
                  else "#8a8f98" for i in order]
    fig = go.Figure(go.Bar(
        x=[drift_vals[i] for i in order],
        y=[NAMES[sensor_ids[i]] for i in order],
        orientation="h", marker_color=bar_colors))
    fig.update_layout(height=460, title="Sensor drift from this aircraft's own early life "
                                        "(in noise units)",
                      xaxis_title="Drift (negative = fell, positive = rose)")
    st.plotly_chart(fig)

    # ---------- Sensor history ----------
    hist = sensors[sensors["unit"] == unit].copy()
    options = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s20", "s21"]
    chosen = st.multiselect("Sensors to plot (smoothed, 5-flight average)", options,
                            default=[ex["top1"]] if ex["top1"] in options else ["s4"])
    fig = go.Figure()
    for s in chosen:
        smooth = hist[s].rolling(5, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=hist["cycle"], y=smooth, mode="lines", name=s))
        base_line = hist[s].iloc[:20].mean()   # this aircraft's own early "healthy" level
        fig.add_hline(y=base_line, line_dash="dot", opacity=0.4)
    fig.update_layout(height=420, xaxis_title="Flight number",
                      yaxis_title="Sensor value",
                      title=f"{pick}: sensor history (dotted line = its own early-life level)")
    st.plotly_chart(fig)

# ---- Tab 4: simulated data + accuracy
with tab4:
    st.subheader("Simulated spares stock (SIMULATED)")
    st.dataframe(spares)
    st.subheader("Free maintenance slots, next 7 days (SIMULATED)")
    st.dataframe(slots)

    st.subheader("Model accuracy on 100 unseen NASA test engines")
    err = (fleet["pred_rul"] - fleet["true_rul"])
    mae = err.abs().mean()
    rmse = float(np.sqrt((err ** 2).mean()))
    truly = fleet["true_rul"] <= 30
    caught = int(((fleet["status"] == "RED") & truly).sum())
    false_red = int(((fleet["status"] == "RED") & ~truly).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average error (MAE)", f"{mae:.1f} flights")
    m2.metric("RMSE", f"{rmse:.1f} flights")
    m3.metric("At-risk engines caught as RED", f"{caught} / {int(truly.sum())}")
    m4.metric("False RED alarms", false_red)
    st.caption("Honest note: the model is not perfect. Some at-risk engines are missed, ""and the 10-flight safety margin was chosen after looking at test results.")

    # ---- Tab 5: forecast + what-if
with tab5:
    st.subheader("7-day fleet availability forecast + what-if")
    st.caption("Move the sliders to test 'what if' questions. Everything here uses our "
               "SIMULATED spares and slots, plus assumptions listed at the bottom.")

    s1, s2, s3 = st.columns(3)
    extra_spares = s1.slider("Extra spare modules per base", 0, 4, 0)
    extra_delay = s2.slider("Spares delivery delay (extra days)", 0, 5, 0)
    slots_lost = s3.slider("Hangar slots lost per day per base", 0, 2, 0)
    s4, s5, s6 = st.columns(3)
    maint_days = s4.slider("Days in hangar per planned maintenance", 1, 4, 2)
    unplanned_extra = s5.slider("Extra days lost if a failure is a surprise", 0, 6, 3)

    plan = make_plan(fleet, spares, slots, extra_resupply=extra_delay,
                     extra_spares=extra_spares, slots_lost=slots_lost)

    est_nothing, u_est_nothing = availability(plan, "days_left", False,
                                              maint_days, unplanned_extra)
    est_plan, u_est_plan = availability(plan, "days_left", True,
                                        maint_days, unplanned_extra)
    true_nothing, u_true_nothing = availability(plan, "true_days_left", False,
                                                maint_days, unplanned_extra)
    true_plan, u_true_plan = availability(plan, "true_days_left", True,
                                          maint_days, unplanned_extra)

    days = list(range(1, 8))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=est_nothing, mode="lines+markers",
                             name="Do nothing (forecast)",
                             line=dict(color=COLORS["RED"])))
    fig.add_trace(go.Scatter(x=days, y=est_plan, mode="lines+markers",
                             name="Our plan (forecast)",
                             line=dict(color=COLORS["GREEN"])))
    fig.add_trace(go.Scatter(x=days, y=true_plan, mode="lines+markers",
                             name="Our plan (backtest on NASA true RUL)",
                             line=dict(color=COLORS["GREEN"], dash="dot")))
    fig.update_layout(height=420, xaxis_title="Day from today",
                      yaxis_title="Aircraft available (out of 100)",
                      yaxis=dict(range=[60, 101]),
                      title="Fleet availability over the next 7 days")
    st.plotly_chart(fig)

    st.write("**Surprise (unplanned) failures in the next 7 days**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Forecast: do nothing", u_est_nothing)
    k2.metric("Forecast: our plan", u_est_plan, u_est_plan - u_est_nothing,
              delta_color="inverse")
    k3.metric("Backtest (true RUL): do nothing", u_true_nothing)
    k4.metric("Backtest (true RUL): our plan", u_true_plan, u_true_plan - u_true_nothing,
              delta_color="inverse")

    red_plan = plan[plan["status"] == "RED"]
    st.write("**Supply and slot problems among RED aircraft**")
    r1, r2, r3 = st.columns(3)
    r1.metric("Spares SHORT", int(red_plan["spare_status"].str.startswith("SHORT").sum()))
    r2.metric("No slot in 7 days", int(red_plan["action"].str.contains("ESCALATE").sum()))
    r3.metric("Slot after likely failure", int(red_plan["action"].str.contains("may fail").sum()))

    st.caption(
        "How to read this: the dotted line is a backtest. It replays the plan against the "
        "true remaining life that NASA provides for the 100 test engines. The forecast lines "
        "start lower than the backtest because our estimate includes a 10-flight safety "
        "margin, so it is deliberately cautious. ASSUMPTIONS: 3 flights per aircraft per "
        "day; an aircraft is unavailable while in the hangar; an aircraft that fails before "
        "its slot loses the extra 'surprise' days; repaired aircraft do not fail again "
        "within the week; 'do nothing' means fixing only after a failure."
    )

    with st.expander("See the maintenance plan behind these numbers"):
        cols_show = ["aircraft", "base", "status", "plan_rul", "spare_status", "action",
                     "priority_score"]
        st.dataframe(style_status(plan[plan["status"] != "GREEN"]
                                  .sort_values("priority_score", ascending=False)[cols_show]))