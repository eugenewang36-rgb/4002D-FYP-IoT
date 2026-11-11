"""Pre-cooling add-on (planned Semester 2 feature): simulated for (a) arrival home and (b) the workout.

The reactive system (folder 12) never switches the aircon on before the user is in the room.
Pre-cooling is the add-on that does: the aircon starts N minutes before a time the schedule/habit
learner predicts. This script re-uses the folder 12 room model and ceiling-fan evening exactly
(real hourly Singapore temperatures from folder 17) and only adds the pre-cool windows.

Story evening: home 6 pm, studying/resting to 8:30 pm, workout 8:30-10 pm, sleep to 6 am.
 (a) Arrival pre-cool: aircon ON PRE_ARR min before 6 pm, fan off (room empty), setpoint ARR_SET.
 (b) Workout pre-cool: aircon ON PRE_EX min before 8:30 pm while the user studies, setpoint =
     the workout setpoint (25 °C), fan as the reactive rule picks it.
Start temperature for any run = real average Singapore air temperature at the run's first minute
(linear between hours of folder 17). Outputs: precooling_results.csv, run.log.
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "12_activity_setpoints", "activity_setpoints.py")
src = open(SRC, encoding="utf-8").read()
src = src.split("fields = []")[0]                      # stop before the csv/chart code
src = src.replace("os.chdir(os.path.dirname(os.path.abspath(__file__)))", "pass")
sys.argv = ["x", "ceiling"]                              # the household's existing DC ceiling fan
ns = {"__name__": "folder12", "__file__": SRC}
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, SRC, "exec"), ns)

simulate, comfort_minutes, our_system = ns["simulate"], ns["comfort_minutes"], ns["our_system"]
NOW, t_out, SYSTEMS = ns["NOW"], ns["t_out"], ns["SYSTEMS"]
TARIFF, DAYS, V_STILL, FLOOR = ns["TARIFF"], ns["DAYS_PER_YEAR"], ns["V_STILL"], ns["WORKOUT_FLOOR_C"]
ARR, WORK = 18.0, 20.5

def sg(h):  # real hourly average, interpolated (same helper the model uses for outdoor air)
    return t_out(h)

state = {"arr": 0, "ex": 0, "arr_set": 26.0, "ex_set": FLOOR}

def strat(act, t_m, t_a):
    h = NOW[0]
    if act == "Away":                                    # empty room before arrival
        return state["arr_set"], V_STILL
    if act == "Sitting" and state["ex"] and WORK - state["ex"] / 60 - 1e-9 <= h < WORK:
        ts, v = our_system("Exercising", t_m, t_a)       # aircon at the workout setpoint while still at the desk
        return state["ex_set"], ns["best"]["Sitting"][1]   # same study fan as the reactive rule (high)
    return our_system(act, t_m, t_a)

ns["precool"] = strat                                    # simulate() starts earlier for this strategy
rows, log = [], []
cases = [("Reactive (no pre-cool)", 0, 0, 26.0)]
for m in (15, 30, 60):
    cases.append((f"Arrival pre-cool {m} min (26 °C)", m, 0, 26.0))
for m in (15, 30, 60, 90):
    cases.append((f"Workout pre-cool {m} min (25 °C)", 0, m, 26.0))
cases += [(f"Workout pre-cool {m} min (23 °C, colder)", 0, m, 26.0, 23.0) for m in (30, 60, 90)]
cases.append(("Both: arrival 30 min + workout 30 min", 30, 30, 26.0))
base = {}
for c in cases:
    label, a, e, aset = c[:4]
    state.update(arr=a, ex=e, arr_set=aset, ex_set=(c[4] if len(c) > 4 else FLOOR))
    ns["PRECOOL_MIN"] = a
    start = ARR - a / 60
    ns["T_START"] = sg(start)
    for sysl, cop in SYSTEMS:
        ns["STUDY_AC_ON"][0] = False
        recs, ac, fan, ex_kwh = simulate(strat, cop)
        _, _, mins = comfort_minutes(recs)
        tot = ac + fan
        if label.startswith("Reactive"):
            base[sysl] = tot
        sgd = tot * TARIFF * DAYS
        warm = {k: v[0] for k, v in mins.items()}; cold = {k: v[1] for k, v in mins.items()}
        row = dict(case=label, aircon=sysl, kwh_evening=round(tot, 3), sgd_year=round(sgd, 1),
                   extra_pct=round(100 * (tot / base[sysl] - 1), 1),
                   exercising_warm=warm["Exercising"], exercising_cold=cold["Exercising"],
                   sitting_warm=warm["Sitting"], sitting_cold=cold["Sitting"],
                   sleeping_warm=warm["Sleeping"], sleeping_cold=cold["Sleeping"],
                   uncomfortable_total=sum(warm.values()) + sum(cold.values()))
        rows.append(row); log.append(row)
        print({k: int(v) if hasattr(v,'item') else v for k,v in row.items()})
with open(os.path.join(HERE, "precooling_results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# Reactive baselines from the same folder 12 run (ceiling fan, real temperatures), for the report tables
res = ns["results"]
with open(os.path.join(HERE, "reactive_baselines.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["strategy", "aircon", "kwh_evening", "sgd_year", "warm_min", "cold_min"])
    for label, d in res.items():
        m = d["mins"]
        for sysl, _ in SYSTEMS:
            tot = d[sysl][0]
            w.writerow([label, sysl, round(tot, 3), round(tot * TARIFF * DAYS, 1),
                        sum(int(v[0]) for v in m.values()), sum(int(v[1]) for v in m.values())])
w2 = {a: (round(t, 2), v) for a, (t, v) in ns["best"].items()}
print("best settings per activity:", w2)

# What does the camera add? Fan 0.5 m/s + fixed 26 °C, no activity detection (same evening, real temperatures)
nocam = lambda a, tm, ta: (26.0, 0.5) if a != "Away" else (None, V_STILL)
ns["precool"] = nocam
ns["PRECOOL_MIN"] = 0
ns["T_START"] = sg(18.0)
for sysl, cop in SYSTEMS:
    recs, ac, fan, _ = simulate(nocam, cop)
    _, _, mins = comfort_minutes(recs)
    print("NO CAMERA", sysl, round(ac + fan, 3), round((ac + fan) * TARIFF * DAYS, 1),
          "warm", sum(int(v[0]) for v in mins.values()), "cold", sum(int(v[1]) for v in mins.values()))
