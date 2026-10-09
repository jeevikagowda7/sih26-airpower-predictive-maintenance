# Air Power: Predictive Maintenance & Fleet Availability

**Smart India Hackathon 2026 | PS 26249 | Ministry of Defence (Defence Services Staff College)**

A working prototype that answers three questions for a fleet commander:
1. **Which aircraft are close to failure?** (AI estimate of Remaining Useful Life)
2. **Which should we fix first, given spare parts and hangar slots?** (maintenance priority score and a simple scheduler)
3. **How many aircraft will be ready this week, and what if things change?** (7-day forecast with what-if sliders)

## Data: what is real and what is simulated
| Data | Source | Status |
|---|---|---|
| Engine sensor data | NASA C-MAPSS turbofan dataset, subset FD001 | Public, **simulated** by NASA. Not real Air Force data |
| Bases, spare-part stock, maintenance slots | Created by us in `step4_fleet_priority.py` | **Simulated** to demonstrate data integration |
| Assumptions (3 flights per day, 2 hangar days, etc.) | Our choice | Adjustable in the dashboard |

The design lets real maintenance, spares and health-monitoring data replace these tables.

## Results (measured on 100 unseen NASA test engines)
| Measure | Value |
|---|---|
| Average error (MAE) of the Random Forest | **13.1 flights** vs capped truth (14.2 vs raw truth). Lazy "always average" guess: 34.8 |
| RMSE | **18.5 flights** vs capped truth (19.5 vs raw truth). Lazy guess: 41.9 |
| Engines truly within 30 flights of failure that we flagged RED | **21 of 25** (4 missed) |
| False RED alarms | **1** |
| Engines where the model was 20+ flights too optimistic | 17 of 100 (this is why we add a 10-flight safety margin) |

Backtest of the 7-day plan against NASA's true remaining life (default assumptions): surprise failures fall from **16 (do nothing) to 10 (our plan)**. In our run, adding 3 spare modules per base in the what-if tab reduced them further to 5. The plan does not raise availability inside the first week, because planned maintenance takes aircraft out of service early. The benefit is fewer surprise breakdowns.

## How it works
| Step | File | What it does |
| 1    | `step1_check_data.py` | Loads and checks the NASA files |
| 3 | `step3_train_model.py` | Trains the Random Forest and reports honest accuracy |
| 4 | `step4_fleet_priority.py` | Green/Yellow/Red status, simulated spares and slots, scheduler, priority score |
| 6 | `step6_explain.py` | "Why is it RED": sensors that drifted most from the aircraft's own early life |
| 7 | `planner.py` | What-if scheduler and 7-day availability forecast |
| 8 | `app.py` | Streamlit dashboard |

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```
The generated files are already in `outputs/` and `data/`, so the dashboard runs straight away. To rebuild everything from scratch:
```
python step3_train_model.py
python step4_fleet_priority.py
python step6_explain.py
streamlit run app.py
```

## Honest limitations
- NASA FD001 is simulated data with one fault type (high-pressure compressor degradation). Real fleets need real data and more fault types.
- The 10-flight safety margin was chosen after looking at test results. In real use it should be chosen on separate validation data.
- The "why is it red" feature shows which sensors drifted most. It is a pointer for engineers, not a proven cause.
- The 7-day forecast is a simple simulation with stated assumptions, not a full digital twin.
- The priority score weights (70% risk, 30% spares shortage) are our assumption.

## Future work
Train on the harder NASA sets (FD002 to FD004), add model-based explanations (SHAP), use real health-monitoring and maintenance records, and connect to live spares and hangar data.

## Reference
A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proc. 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, 2008. Data: NASA Prognostics Center of Excellence.
