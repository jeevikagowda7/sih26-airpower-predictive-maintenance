import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from planner import make_plan, availability

st.set_page_config(page_title="Air Power - Fleet Readiness", page_icon="✈️", layout="wide")

# ---------------------------------------------------------------------
# Look and feel (light theme)
# ---------------------------------------------------------------------
INK = "#0b0b0b"          # main text
INK2 = "#52514e"         # softer text
LINE = "#e4e3de"         # borders and grid lines
BLUE = "#2a78d6"         # main accent
ORANGE = "#eb6834"       # second colour for comparisons
STATUS = {"GREEN": "#0ca30c", "YELLOW": "#fab219", "RED": "#d03b3b"}
PILL = {                 # soft background + dark text, always with the word itself
    "GREEN": ("#e2f4e2", "#0a5c0a"),
    "YELLOW": ("#fff1c9", "#7a5200"),
    "RED": ("#fbe1e1", "#9e1f1f"),
}

# Make every Plotly chart light and clean (wrapped so a plotly version quirk cannot crash the app)
try:
    pio.templates["sih"] = go.layout.Template(layout=dict(
        font=dict(color=INK, family="Inter, Segoe UI, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[BLUE, ORANGE, "#1baf7a", "#4a3aa7"],
        xaxis=dict(gridcolor="#ecebe6", linecolor=LINE, zerolinecolor=LINE),
        yaxis=dict(gridcolor="#ecebe6", linecolor=LINE, zerolinecolor=LINE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    ))
    pio.templates.default = "sih"
except Exception:
    pass

st.markdown("""<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.hero h1 {font-size: 2.15rem; font-weight: 800; margin: 0; color: #0b0b0b; letter-spacing: -0.02em;}
.hero p {color: #52514e; margin: 0.3rem 0 0.7rem; font-size: 1.05rem;}
.chip {display: inline-block; background: #ffffff; border: 1px solid #e4e3de; color: #52514e; border-radius: 999px; padding: 2px 12px; font-size: 0.8rem; margin-right: 8px; font-weight: 600;}
.note {background: #fff8e6; border: 1px solid #f1d98a; border-left: 5px solid #fab219; border-radius: 10px; padding: 0.7rem 1rem; color: #4a3b00; font-size: 0.92rem; margin-top: 0.9rem;}
.kpi-row {display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin: 1.1rem 0 0.4rem;}
.kpi {background: #ffffff; border: 1px solid #e4e3de; border-radius: 14px; padding: 14px 16px 12px;}
.kpi-label {font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: #52514e; font-weight: 700;}
.kpi-value {font-size: 2.4rem; font-weight: 800; line-height: 1.15; color: #0b0b0b;}
.kpi-sub {font-size: 0.82rem; color: #52514e;}
.bar {height: 8px; background: #ecebe6; border-radius: 6px; margin-top: 8px; overflow: hidden;}
.bar div {height: 100%; background: #2a78d6; border-radius: 6px;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e4e3de; border-radius: 12px; padding: 12px 14px;}
button[data-baseweb="tab"] {font-weight: 600;}
@media (max-width: 900px) {.kpi-row {grid-template-columns: repeat(2, 1fr);}}
</style>""", unsafe_allow_html=True)


def kpi(label, value, sub, accent, bar=None):
    label = label.replace("●", f'<span style="color:{accent}">●</span>')
    bar_html = f'<div class="bar"><div style="width:{bar}%"></div></div>' if bar is not None else ""
    return (f'<div class="kpi" style="border-top:4px solid {accent}">'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>{bar_html}</div>')


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

NAMES = {
    "s2": "LPC outlet temp (T24)", "s3": "HPC outlet temp (T30)",
    "s4": "LPT outlet temp (T50)", "s7": "HPC outlet pressure (P30)",
    "s8": "Fan speed (Nf)", "s9": "Core speed (Nc)",
    "s11": "HPC static pressure (Ps30)", "s12": "Fuel flow ratio (phi)",
    "s13": "Corrected fan speed (NRf)", "s14": "Corrected core speed (NRc)",
    "s15": "Bypass ratio (BPR)", "s17": "Bleed enthalpy (htBleed)",
    "s20": "HPT coolant bleed (W31)", "s21": "LPT coolant bleed (W32)",
}

def pill(val):
    p = PILL.get(val)
    if not p:
        return ""
    return f"background-color: {p[0]}; color: {p[1]}; font-weight: 700; text-align: center"

def style_status(df):
    styler = df.style.format(precision=1)
    # pandas 2.1+ uses .map ; older versions use .applymap
    if hasattr(styler, "map"):
        return styler.map(pill, subset=["status"])
    return styler.applymap(pill, subset=["status"])

# ---------------------------------------------------------------------
# Header + honesty banner
# ---------------------------------------------------------------------
st.markdown("""<div class="hero"><h1>✈️ Air Power: Predictive Maintenance &amp; Fleet Availability</h1>
<p>Which aircraft to fix first, why, and how many will be ready this week.</p>
<span class="chip">Smart India Hackathon 2026</span><span class="chip">PS 26249 · Ministry of Defence</span><span class="chip">Prototype</span></div>
<div class="note"><b>Data honesty:</b> engine sensor data is the public, <b>simulated</b> NASA C-MAPSS (FD001) dataset, not real Air Force data. Bases, spare-part stock and maintenance slots are <b>simulated by us</b> to show how fragmented data can be joined together.</div>""",
            unsafe_allow_html=True)

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
pct = round(100 * ready / n_total)

st.markdown(
    '<div class="kpi-row">'
    + kpi("Total aircraft", n_total, "in the fleet", BLUE)
    + kpi("● Green · healthy", n_green, "more than 60 flights left", STATUS["GREEN"])
    + kpi("● Yellow · watch", n_yellow, "31 to 60 flights left", STATUS["YELLOW"])
    + kpi("● Red · act now", n_red, "30 flights or fewer left", STATUS["RED"])
    + kpi("Ready tomorrow", f"{ready} / {n_total}", f"{pct}% of the fleet (Green + Yellow)", BLUE, bar=pct)
    + '</div>', unsafe_allow_html=True)
st.caption("Ready tomorrow = Green + Yellow. Red aircraft are treated as not ready "
           "(close to failure, maintenance needed).")

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Fleet table", "Alerts & priorities", "Aircraft detail", "Simulated data & accuracy",
     "7-day forecast & what-if"])

# ---- Tab 1: fleet table
with tab1:
    st.subheader("Fleet table")
    show = ["aircraft", "base", "status", "plan_rul", "days_left", "main_reason",
            "spare_status", "action", "priority_score"]
    st.dataframe(style_status(view[show]), height=520)

    status_counts = fleet["status"].value_counts().reindex(["GREEN", "YELLOW", "RED"]).fillna(0)
    fig = go.Figure(go.Bar(x=status_counts.index, y=status_counts.values,
                           marker_color=[STATUS[s] for s in status_counts.index],
                           text=status_counts.values.astype(int), textposition="outside"))
    fig.update_layout(height=320, title="Aircraft by health status",
                      yaxis_title="Number of aircraft", showlegend=False,
                      yaxis=dict(range=[0, float(status_counts.max()) * 1.18]))
    st.plotly_chart(fig)

# ---- Tab 2: alerts
with tab2:
    st.subheader("Alerts: who to maintain first")
    red = fleet[fleet["status"] == "RED"]
    may_fail = red[red["action"].str.contains("may fail")]
    escalate = red[red["action"].str.contains("ESCALATE")]
    short = red[red["spare_status"].str.startswith("SHORT")]

    a1, a2, a3 = st.columns(3)
    a1.metric("Red aircraft with spares SHORT", len(short))
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
    bar_colors = [STATUS["RED"] if abs(drift_vals[i]) >= 5
                  else STATUS["YELLOW"] if abs(drift_vals[i]) >= 3
                  else "#c9c8c1" for i in order]
    fig = go.Figure(go.Bar(
        x=[drift_vals[i] for i in order],
        y=[NAMES[sensor_ids[i]] for i in order],
        orientation="h", marker_color=bar_colors))
    fig.update_layout(height=460, showlegend=False,
                      title="Sensor drift from this aircraft's own early life (in noise units)",
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
    st.caption("Honest note: the model is not perfect. Some at-risk engines are missed, "
               "and the 10-flight safety margin was chosen after looking at test results.")

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
                             name="Do nothing (forecast)", line=dict(color=ORANGE)))
    fig.add_trace(go.Scatter(x=days, y=est_plan, mode="lines+markers",
                             name="Our plan (forecast)", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=days, y=true_plan, mode="lines+markers",
                             name="Our plan (backtest on NASA true RUL)",
                             line=dict(color=BLUE, dash="dot")))
    fig.update_layout(height=430, xaxis_title="Day from today",
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

st.markdown("---")
st.caption("Source code: github.com/jeevikagowda7/sih26-airpower-predictive-maintenance  |  "
           "Engine data: NASA C-MAPSS (public, simulated)  |  Spares and slots: simulated by the team")