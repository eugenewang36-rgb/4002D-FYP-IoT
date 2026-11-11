"""Two extra runs on the whole-evening model of folder 12 (existing DC ceiling fan), 8 Nov 2026.

1. EXTRA ARM, "comfort-optimised, no activity detection": the controller picks the aircon setting and fan speed that
   hold the comfort target with the least power (the same SET search as our system), but it has no camera, so it
   assumes the occupant is seated at the desk all evening. The occupant really studies, works out and sleeps, so
   the settings are wrong for two of the three activities. This separates what coordinated optimisation gives
   (fan + warmer setpoint) from what the camera adds. (Definition proposed by Eugene, to be confirmed with Saran.)

2. MORE THAN ONE OCCUPANT in the bedroom. Each extra person adds their own body heat to the room (same model as
   folder 12). Cases:
   a. 1, 2 and 3 people doing the same thing all evening (sitting, workout, sleep together).
   b. Two people during the workout (8:30-10 pm): A works out, B studies. Rule 1 = settings follow the most active
      person (A), rule 2 = settings follow the least active person (B). B's comfort is checked with B's own
      activity (seated, 1.0 met, 0.4 clo).

Run:  python extra_arm_and_occupants.py        (uses the ceiling fan, importing folder 12 re-runs it, ~2 min)
Outputs next to this script: extra_arm_and_occupants.csv, extra_arm_and_occupants.log
"""
import contextlib, csv, io, os, sys

import numpy as np
from pythermalcomfort.models import set_tmp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "12_activity_setpoints"))
sys.argv = [sys.argv[0], "ceiling"]
cwd = os.getcwd()
os.chdir(os.path.join(HERE, "..", "12_activity_setpoints"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import activity_setpoints as m
os.chdir(cwd)
import matplotlib.pyplot as plt
plt.close("all")

DAYS, TARIFF = m.DAYS_PER_YEAR, m.TARIFF
rows = []
OLDER, NEW = m.SYSTEMS[0], m.SYSTEMS[1]


def minutes_bad(recs):
    _, _, mins = m.comfort_minutes(recs)
    warm = sum(w for w, c, a in mins.values())
    cold = sum(c for w, c, a in mins.values())
    return warm, cold, mins


def sgd(kwh):
    return kwh * TARIFF * DAYS


def log(msg=""):
    print(msg)
    LOG.append(msg)


LOG = []

# ------------------------------------------------------------------ 1. extra arm
log("1. Extra arm: comfort-optimised, no activity detection (assumes seated all evening)")
NOACT = lambda a, tm, ta: m.our_system("Sitting", tm, ta)
all_arms = [(label, strat) for label, strat in m.STRATEGIES] + [("Comfort-optimised, no activity detection", NOACT)]
for label, strat in all_arms:
    recs, _, _, _ = m.simulate(strat, OLDER[1])
    warm, cold, mins = minutes_bad(recs)
    out = {"case": "arm", "name": label, "too_warm_min": warm, "too_cold_min": cold, "uncomfortable_min": warm + cold}
    for sys_label, cop in m.SYSTEMS:
        _, ac, fan, ex = m.simulate(strat, cop)
        out[sys_label + " kWh"] = round(ac + fan, 3)
        out[sys_label + " S$/yr"] = round(sgd(ac + fan), 1)
    rows.append(out)
    log(f"  {label}: {out['Older aircon (COP 3.5) kWh']} kWh = S${out['Older aircon (COP 3.5) S$/yr']}/yr (older), "
        f"5-tick S${out['New 5-tick aircon (COP 5.5) S$/yr']}/yr; {warm} min too warm, {cold} min too cold")

# ------------------------------------------------------------------ 2. more than one occupant
orig_heat = m.person_heat


def activity2(hour, lead_act):
    """Second person B: studies while A works out (8:30-10 pm). Otherwise does what A does."""
    if lead_act == "Exercising":
        return "Sitting"
    return lead_act


def set_for(person_act, recs, hours_filter=None):
    met, clo, _, _ = m.ACTIVITIES[person_act]
    sel = [r for r in recs if (hours_filter is None or hours_filter(r))]
    s = set_tmp(tdb=[r[2] for r in sel], tr=[r[3] for r in sel], v=[r[5] for r in sel], rh=m.RH, met=met, clo=clo,
                position="sitting", limit_inputs=False, round_output=False).set
    return np.asarray(s)


log("\n2a. 1, 2 and 3 people doing the same thing all evening (our system)")
our = m.STRATEGIES[3][1]
for n in (1, 2, 3):
    def heat(act, n=n):
        s, l = orig_heat(act)
        return s * n, l * n
    m.person_heat = heat
    recs, _, _, _ = m.simulate(our, OLDER[1])
    warm, cold, _ = minutes_bad(recs)
    out = {"case": "occupants same activity", "name": f"{n} people", "too_warm_min": warm, "too_cold_min": cold,
           "uncomfortable_min": warm + cold}
    for sys_label, cop in m.SYSTEMS:
        _, ac, fan, ex = m.simulate(our, cop)
        out[sys_label + " kWh"] = round(ac + fan, 3)
        out[sys_label + " S$/yr"] = round(sgd(ac + fan), 1)
    rows.append(out)
    log(f"  {n} people: {out['Older aircon (COP 3.5) kWh']} kWh = S${out['Older aircon (COP 3.5) S$/yr']}/yr (older), "
        f"{warm} min too warm, {cold} min too cold (each person has the same SET)")
m.person_heat = orig_heat

log("\n2b. Two people during the workout: A works out, B studies (8:30-10 pm)")
for rule, follow in (("settings follow the most active person (A)", "A"), ("settings follow the least active person (B)", "B")):
    def heat2(act):
        s, l = orig_heat(act)
        s2, l2 = orig_heat(activity2(m.NOW[0], act))
        return s + s2, l + l2
    m.person_heat = heat2
    if follow == "A":
        strat = our
    else:
        strat = lambda a, tm, ta: our("Sitting" if a == "Exercising" else a, tm, ta)
    recs, _, _, _ = m.simulate(strat, OLDER[1])
    win = lambda r: r[1] == "Exercising"
    set_a = set_for("Exercising", recs, win)
    set_b = set_for("Sitting", recs, win)
    a_warm = int((set_a > m.KNOB + m.TOL).sum())
    b_cold = int((set_b < m.KNOB - m.TOL).sum())
    b_warm = int((set_b > m.KNOB + m.TOL).sum())
    out = {"case": "two people, A works out and B studies", "name": rule,
           "A_min_too_warm": a_warm, "B_min_too_cold": b_cold, "B_min_too_warm": b_warm,
           "A_avg_SET": round(float(set_a.mean()), 2), "B_avg_SET": round(float(set_b.mean()), 2)}
    for sys_label, cop in m.SYSTEMS:
        _, ac, fan, ex = m.simulate(strat, cop)
        out[sys_label + " kWh"] = round(ac + fan, 3)
        out[sys_label + " S$/yr"] = round(sgd(ac + fan), 1)
    rows.append(out)
    log(f"  {rule}: A {a_warm} min too warm (avg SET {set_a.mean():.1f} °C); B {b_cold} min too cold, {b_warm} min too warm "
        f"(avg SET {set_b.mean():.1f} °C); {out['Older aircon (COP 3.5) kWh']} kWh = S${out['Older aircon (COP 3.5) S$/yr']}/yr (older)")
m.person_heat = orig_heat

fields = []
for r in rows:
    fields += [k for k in r if k not in fields]
with open(os.path.join(HERE, "extra_arm_and_occupants.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
open(os.path.join(HERE, "extra_arm_and_occupants.log"), "w", encoding="utf-8").write("\n".join(LOG))
print("saved")
