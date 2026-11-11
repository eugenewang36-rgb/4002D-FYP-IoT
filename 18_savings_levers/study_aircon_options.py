"""Studying / resting (6-8:30 pm): realistic aircon options instead of a 29.5 °C setpoint.
  A. Aircon off, fan only (fan speed picked to bring SET closest to the target).
  B. Aircon cycles at 26 °C: off while the fan alone keeps the user comfortable. When the room air gets
     so warm that even the fan on high cannot (SET above the threshold), the aircon goes on at 26 °C. It
     goes off again once the room air is down to 26 °C. Fan speed picked as in A.
Everything else as folder 12 (real 6 pm start 28.4 °C, real outdoor temperatures, ceiling fan).
Output: study_aircon_options.csv next to this script."""
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

OFF = None   # aircon off
base = m.our_system


def fan_for(t_a, act="Sitting"):
    met, clo, _, vmax = m.ACTIVITIES[act]
    levels = [lv for lv in m.FAN_LEVELS if lv <= vmax + 1e-9]
    return min(levels, key=lambda lv: abs(m.set_value(t_a, t_a, lv, met, clo) - m.KNOB))


def option_off(act, t_m=None, t_a=None):
    if act != "Sitting":
        return base(act, t_m, t_a)
    return OFF, fan_for(t_a)


def option_old(act, t_m=None, t_a=None):
    """The earlier rule: aircon at the lowest-power setpoint (29.5 °C) + fan 0.85 m/s with the step-down."""
    if act != "Sitting":
        return base(act, t_m, t_a)
    t, v = m.best["Sitting"]
    return t, (fan_for(t_a) if t_a < t - 0.1 else v)


def make_cycle(on_above):
    state = {"on": False}

    def option_cycle(act, t_m=None, t_a=None):
        if act != "Sitting":
            state["on"] = False
            return base(act, t_m, t_a)
        met, clo, _, vmax = m.ACTIVITIES[act]
        if m.set_value(t_a, t_a, vmax, met, clo) > on_above:
            state["on"] = True
        elif t_a <= 26.1:
            state["on"] = False
        return (26.0 if state["on"] else OFF), fan_for(t_a)
    return option_cycle, state


def run(strat, reset=None):
    out = {}
    for sys_label, cop in m.SYSTEMS:
        if reset:
            reset["on"] = False
        pa = {}
        recs, ac, fan, _ = m.simulate(strat, cop, pa)
        out[sys_label] = ((ac + fan) * m.TARIFF * m.DAYS_PER_YEAR, pa, recs)
    _, _, mins = m.comfort_minutes(out[m.SYSTEMS[0][0]][2])
    return out, mins


fixed = run(m.STRATEGIES[1][1])
cases = [("Earlier rule: aircon 29.5 °C + fan", option_old, None),
         ("Folder 12 now: aircon off + 26 °C safety cycle", base, None), ("A. aircon off, fan only", option_off, None)]
for thr in (m.KNOB, m.KNOB + m.TOL):
    f, st = make_cycle(thr)
    cases.append((f"B. 26 °C on when fan-on-high SET > {thr:.1f}", f, st))

rows = []
for name, strat, st in cases:
    res, mins = run(strat, st)
    recs = res[m.SYSTEMS[0][0]][2]
    study = [r for r in recs if r[1] == "Sitting"]
    on_min = sum(1 for r in study if r[4] is not None)
    starts = sum(1 for a, b in zip(study, study[1:]) if a[4] is None and b[4] is not None) + (study[0][4] is not None)
    for sys_label, _ in m.SYSTEMS:
        o, po, _ = res[sys_label]
        fx, pf, _ = fixed[0][sys_label]
        row = {"option": name, "aircon": sys_label, "sgd_year": round(o), "fixed25_sgd_year": round(fx),
               "saving_vs_fixed25_pct": round(100 * (1 - o / fx), 1),
               "study_kwh": round(po["Sitting"], 3), "study_saving_pct": round(100 * (1 - po["Sitting"] / pf["Sitting"]), 1),
               "workout_kwh": round(po["Exercising"], 3), "workout_saving_pct": round(100 * (1 - po["Exercising"] / pf["Exercising"]), 1),
               "sleep_saving_pct": round(100 * (1 - po["Sleeping"] / pf["Sleeping"]), 1),
               "study_warm_min": mins["Sitting"][0], "study_cold_min": mins["Sitting"][1],
               "study_avg_set": round(mins["Sitting"][2], 2), "workout_warm_min": mins["Exercising"][0],
               "uncomfortable_min": sum(w + c for w, c, _ in mins.values()),
               "study_aircon_on_min": on_min, "study_aircon_starts": starts,
               "room_at_830pm": round(study[-1][2], 2), "room_max_study": round(max(r[2] for r in study), 2)}
        rows.append(row)
        print(row)
with open(os.path.join(HERE, "study_aircon_options.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
