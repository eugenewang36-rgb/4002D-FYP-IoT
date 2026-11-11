"""What would raise the evening saving? Design levers tested on the folder 12 evening model
(ceiling fan, real Singapore outdoor temperatures, room starts 29.1 °C, home 8 pm study, workout 8:30-10 pm,
sleep 10 pm-6 am). Each lever is a different control rule; nothing about the room or weather is changed.

Levers (ASSUMED rules, to be judged on comfort as well as energy):
  A  workout floor 25 °C: during the workout the aircon is not set below 25 °C; fan stays on high (1.21 m/s)
  A24 same with a 24 °C floor
  B  faster sleep fan: 0.85 m/s while asleep (setpoint re-solved for SET 25) instead of the 0.5 m/s cap
  A+B both
Baseline for "saving": aircon alone at a fixed 25 °C (the user's usual setting), older aircon (COP 3.5).
Output: savings_levers.csv next to this script.
"""
import contextlib
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "12_activity_setpoints"))
sys.argv = [sys.argv[0], "ceiling"]
cwd = os.getcwd()
os.chdir(os.path.join(HERE, "..", "12_activity_setpoints"))
with contextlib.redirect_stdout(io.StringIO()):
    import activity_setpoints as m
os.chdir(cwd)
import matplotlib.pyplot as plt
plt.close("all")

best = m.best                                     # activity -> (setpoint, fan) chosen by our system
met_s, clo_s = m.ACTIVITIES["Sleeping"][:2]
sleep_085 = (m.setpoint_for(met_s, clo_s, 0.85), 0.85)


def rule(workout_floor=None, sleep_fast=False, ramp_per_h=0.0, ramp_cap=1.5, set_ramp_per_h=0.0, set_ramp_cap=0.4):
    """Factory: returns a fresh control rule (it keeps a counter, so make a new one per run).
    ramp_per_h: raise the sleep setpoint by this many °C per hour after sleep starts (ASSUMED rule),
    up to ramp_cap °C. The controller is called every m.CONTROL_STEP seconds, so calls count time."""
    state = {"sleep_calls": 0}

    def strat(act, tm, ta):
        if act == "Exercising" and workout_floor is not None:
            t, v = best[act]
            return max(t, workout_floor), v
        if act == "Sleeping":
            t, v = sleep_085 if sleep_fast else best[act]
            hours = state["sleep_calls"] * m.CONTROL_STEP / 3600
            state["sleep_calls"] += 1
            if set_ramp_per_h:   # in-band ramp: raise the comfort TARGET (SET) slowly, re-solve the setpoint
                t = m.setpoint_for(met_s, clo_s, v, target=m.KNOB + min(set_ramp_per_h * hours, set_ramp_cap))
            return min(t + min(ramp_per_h * hours, ramp_cap), m.SET_MAX), v
        return best[act]
    return strat


CASES = [("Aircon alone, fixed 25 °C (baseline)", lambda: m.STRATEGIES[1][1]),
         ("Our system today (sleep fan 0.5, workout 23.2 °C + fan high)", lambda: m.STRATEGIES[3][1]),
         ("B: sleep fan 0.85 m/s", lambda: rule(sleep_fast=True)),
         ("B + ramp 0.25 °C/h", lambda: rule(sleep_fast=True, ramp_per_h=0.25)),
         ("B + ramp 0.5 °C/h", lambda: rule(sleep_fast=True, ramp_per_h=0.5)),
         ("A + B: + workout aircon 25 °C, fan high", lambda: rule(workout_floor=25.0, sleep_fast=True)),
         ("A + B + ramp 0.25 °C/h", lambda: rule(workout_floor=25.0, sleep_fast=True, ramp_per_h=0.25)),
         ("A + B + ramp 0.5 °C/h", lambda: rule(workout_floor=25.0, sleep_fast=True, ramp_per_h=0.5)),
         ("B + in-band ramp (target SET 25.0 -> 25.4)", lambda: rule(sleep_fast=True, set_ramp_per_h=0.05)),
         ("A + B + in-band ramp", lambda: rule(workout_floor=25.0, sleep_fast=True, set_ramp_per_h=0.05))]

rows, base = [], None
for name, make in CASES:
    recs, ac, fan, ex = m.simulate(make(), 3.5)
    _, _, mins = m.comfort_minutes(recs)
    sgd = (ac + fan) * m.TARIFF * m.DAYS_PER_YEAR
    _, ac5, fan5, _ = m.simulate(make(), 5.5)
    sgd5 = (ac5 + fan5) * m.TARIFF * m.DAYS_PER_YEAR
    if base is None:
        base, base5 = sgd, sgd5
    unc = sum(w + c for w, c, _ in mins.values())
    ex_set = mins["Exercising"][2]
    rows.append({"case": name, "sgd_year_cop35": round(sgd, 1), "saving_vs_fixed25_pct": round(100 * (1 - sgd / base), 1),
                 "saving_5tick_pct": round(100 * (1 - sgd5 / base5), 1), "uncomfortable_min": unc,
                 "workout_avg_set_C": round(ex_set, 2), "sleep_min_too_warm": mins["Sleeping"][0],
                 "sleep_min_too_cold": mins["Sleeping"][1], "sleep_avg_set_C": round(mins["Sleeping"][2], 2)})
    print(f"{name:40s} S${sgd:5.0f}/yr  saving {100 * (1 - sgd / base):5.1f}% (5-tick {100 * (1 - sgd5 / base5):5.1f}%)  "
          f"uncomfortable {unc:3d} min (sleep warm/cold {mins['Sleeping'][0]}/{mins['Sleeping'][1]})  workout SET {ex_set:.1f} °C")
print(f"sleep at 0.85 m/s: setpoint {sleep_085[0]:.2f} °C")
with open(os.path.join(HERE, "savings_levers.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
