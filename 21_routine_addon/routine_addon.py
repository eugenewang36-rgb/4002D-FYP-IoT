"""Add-on features on top of the 6 pm to 6 am system (whole-evening model of folder 12, 9 Nov 2026).

The 6 pm to 6 am results (S$288 a year, 34% less than fixed 25 °C, 128 uncomfortable minutes) are NOT changed.
This script re-runs the same model with three extra time-based features and reports what each one adds:

  1. Pre-cooling      aircon starts 15 min before the expected arrival (published advice, section 2.6)
  2. Pre-shutdown     aircon switches off 30 min before the expected departure (published advice, section 2.6)
                      The base system already does this at 5:30 am (departure / wake at 6 am); here the length is swept.
  3. Occupancy        the system shuts down when no occupant has been detected for 15 min (project target)
                      Tested with the occupant out of the bedroom for part of the evening or the whole night.

Assumptions (all ASSUMED, not measured):
  - the user's routine is already known (arrival 6 pm, departure 6 am); learning it is a Semester 2 task
  - while the room is empty and the occupancy rule is OFF, the system keeps the last setting it used
  - when the room is empty the person adds no body heat; the fixed 25 °C aircon-only baseline keeps running
  - the occupant coming back is recognised at once (the 45 s confirmation is not modelled here)
Run:  python routine_addon.py     (importing folder 12 re-runs it, about 2 minutes)
Outputs next to this script: routine_addon.csv, routine_addon.log
"""
import contextlib, csv, io, os, sys

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
import numpy as np

OURS = m.STRATEGIES[3][1]
FIXED = m.STRATEGIES[1][1]
DAYS, TARIFF, KNOB, TOL = m.DAYS_PER_YEAR, m.TARIFF, m.KNOB, m.TOL
BASE_START, BASE_END = 18.0, 30.0
log, rows = [], []


def out(s=""):
    print(s)
    log.append(s)


def run(strategy, act_fn, start, end, cop, t0=None, wake_min=30):
    """One run of the folder-12 model on a chosen timeline. Returns (records, kWh)."""
    m.START, m.END = start, end
    m.activity_at = act_fn
    m.T_START = t0 if t0 is not None else m.t_out(start)
    m.WAKE_MIN = wake_min
    recs, ac, fan, _ = m.simulate(strategy, cop)
    return recs, ac + fan


def uncomfortable(recs, upto, frm=0.0):
    """Minutes the occupant is in the room and SET is outside KNOB +- TOL, counted between frm and upto (clock hours)."""
    recs = [r for r in recs if r[1] in m.ACTIVITIES and frm <= r[0] < upto]
    if not recs:
        return 0, 0
    sets = []
    for act in m.ACTIVITIES:
        idx = [i for i, r in enumerate(recs) if r[1] == act]
        if not idx:
            continue
        met, clo, _, _ = m.ACTIVITIES[act]
        s = m.set_tmp(tdb=[recs[i][2] for i in idx], tr=[recs[i][3] for i in idx], v=[recs[i][5] for i in idx],
                      rh=m.RH, met=met, clo=clo, position="sitting", limit_inputs=False, round_output=False).set
        sets += list(zip(idx, s))
    s = np.array([x for _, x in sorted(sets)])
    return int((s > KNOB + TOL).sum()), int((s < KNOB - TOL).sum())


def sgd(kwh):
    return kwh * TARIFF * DAYS


def base_act(hour):
    if hour < 18.0 - 1e-9:
        return "Away"
    return m.activity_at.__wrapped__(hour) if hasattr(m.activity_at, "__wrapped__") else story(hour)


def story(hour):
    if hour < 18.0 - 1e-9:
        return "Away"
    if hour < 20.5:
        return "Sitting"
    if hour < 22.0:
        return "Exercising"
    return "Sleeping"


# ---------------------------------------------------------------- reference: the base evening (must stay unchanged)
out("Reference: the 6 pm to 6 am system as in folder 12 (unchanged)")
ref = {}
for lab, cop in m.SYSTEMS:
    r_o, k_o = run(OURS, story, BASE_START, BASE_END, cop, m.t_out(18.0))
    r_f, k_f = run(FIXED, story, BASE_START, BASE_END, cop, m.t_out(18.0))
    ref[cop] = (r_o, k_o, k_f)
    w, c = uncomfortable(r_o, BASE_END - 0.5)
    out(f"  {lab}: ours S${sgd(k_o):.0f}/yr, fixed 25 °C S${sgd(k_f):.0f}/yr, saving {100 * (1 - k_o / k_f):.1f}%, {w + c} uncomfortable min")
    rows.append({"feature": "base", "case": "6 pm to 6 am, no add-on", "aircon": lab, "ours_kwh": round(k_o, 3), "fixed_kwh": round(k_f, 3),
                 "ours_sgd_yr": round(sgd(k_o)), "fixed_sgd_yr": round(sgd(k_f)), "uncomfortable_min": w + c})
r_o35, k_o35, k_f35 = ref[3.5]
base_unc = sum(uncomfortable(r_o35, BASE_END - 0.5))

# ---------------------------------------------------------------- 1. pre-cooling
out("\n1. Pre-cooling: aircon starts before the 6 pm arrival (empty room, fan off)")
out("   Setpoint while the room is empty = the user's knob, 25 °C (older aircon). Energy counted from the start of pre-cooling.")


def precool_strategy(set_c, base):
    def f(act, tm, ta):
        if act == "Away":
            return set_c, m.V_STILL
        return base(act, tm, ta)
    return f


def arrival_comfort(recs):
    """Minutes uncomfortable in the first 30 minutes after arrival (18:00-18:30) and SET at 6:00 pm."""
    w, c = uncomfortable(recs, 18.5, 18.0)
    return w, c


for who, strat in (("ours", OURS), ("fixed 25 °C aircon only", FIXED)):
    for lead in (0, 15, 30, 60):
        for lab, cop in m.SYSTEMS[:1]:
            start = BASE_START - lead / 60
            st = precool_strategy(25.0, strat) if lead else strat
            r, k = run(st, story, start, BASE_END, cop, m.t_out(start))
            w1, c1 = arrival_comfort(r)
            wt, ct = uncomfortable(r, BASE_END - 0.5)
            ref_k = k_o35 if who == "ours" else k_f35
            out(f"   {who:24s} pre-cool {lead:2d} min: S${sgd(k):4.0f}/yr (extra S${sgd(k - ref_k):+.1f}), first 30 min after arrival: {w1} warm / {c1} cool, whole evening {wt + ct} min")
            rows.append({"feature": "pre-cooling", "case": f"{who}, {lead} min", "aircon": lab, "ours_kwh": round(k, 3),
                         "extra_sgd_yr": round(sgd(k - ref_k), 1), "uncomfortable_first30": w1 + c1, "uncomfortable_min": wt + ct})

out("   Same 15-minute pre-cooling with a higher setpoint while the room is empty (older aircon):")
for who, strat, ref_k in (("ours", OURS, k_o35), ("fixed 25 °C aircon only", FIXED, k_f35)):
    for set_c in (25.0, 26.0, 27.0, 28.0):
        start = BASE_START - 0.25
        r, k = run(precool_strategy(set_c, strat), story, start, BASE_END, 3.5, m.t_out(start))
        w1, c1 = arrival_comfort(r)
        wt, ct = uncomfortable(r, BASE_END - 0.5)
        out(f"   {who:24s} 15 min at {set_c:.0f} °C: extra S${sgd(k - ref_k):+.1f}/yr, first 30 min after arrival {w1} warm / {c1} cool, whole evening {wt + ct} min")
        rows.append({"feature": "pre-cooling setpoint", "case": f"{who}, 15 min at {set_c:.0f} C", "aircon": "Older aircon (COP 3.5)", "ours_kwh": round(k, 3),
                     "extra_sgd_yr": round(sgd(k - ref_k), 1), "uncomfortable_first30": w1 + c1, "uncomfortable_min": wt + ct})

# ---------------------------------------------------------------- 2. pre-shutdown
out("\n2. Pre-shutdown: aircon switched off before the 6 am departure (fan stays at its sleep speed)")
out("   The base system uses 30 min (aircon off at 5:30 am). Comfort is counted to 6 am here, not excluded.")
for lab, cop in m.SYSTEMS:
    for lead in (0, 15, 30, 45, 60, 90):
        r, k = run(OURS, story, BASE_START, BASE_END, cop, m.t_out(18.0), wake_min=lead)
        w, c = uncomfortable(r, BASE_END)
        wl, cl = uncomfortable(r, BASE_END, BASE_END - max(lead, 1) / 60)
        out(f"   {lab}: pre-shutdown {lead:2d} min: S${sgd(k):4.0f}/yr, whole night to 6 am {w + c} uncomfortable min (of which in the last {lead} min: {wl + cl})")
        rows.append({"feature": "pre-shutdown", "case": f"{lead} min", "aircon": lab, "ours_kwh": round(k, 3),
                     "ours_sgd_yr": round(sgd(k)), "uncomfortable_min": w + c, "uncomfortable_in_window": wl + cl})
m.WAKE_MIN = 30

# ---------------------------------------------------------------- 3. occupancy shutdown
out("\n3. Occupancy shutdown: system off after 15 min with no occupant detected")
out("   Without the rule, the system keeps the last setting while the room is empty. 'Fixed 25 °C' keeps running.")


def occupancy_strategy(base, rule_min):
    state = {"last": "Sitting", "away_since": None}

    def f(act, tm, ta):
        if act != "Away":
            state["last"], state["away_since"] = act, None
            return base(act, tm, ta)
        if state["away_since"] is None:
            state["away_since"] = m.NOW[0]
        if rule_min is not None and (m.NOW[0] - state["away_since"]) * 60 >= rule_min - 1e-9:
            return None, m.V_STILL
        return base(state["last"], tm, ta)
    return f


def act_with_absence(a0, a1):
    def f(hour):
        if a0 <= hour < a1:
            return "Away"
        return story(hour)
    return f


CASES = [("Out of the bedroom 7:00-8:00 pm (while sitting)", 19.0, 20.0),
         ("Out 9:00-9:45 pm (workout interrupted)", 21.0, 21.75),
         ("Bedroom empty all night (10 pm-6 am)", 22.0, 30.0)]
for name, a0, a1 in CASES:
    fn = act_with_absence(a0, a1)
    for lab, cop in m.SYSTEMS:
        r_n, k_n = run(occupancy_strategy(OURS, None), fn, BASE_START, BASE_END, cop, m.t_out(18.0))
        r_r, k_r = run(occupancy_strategy(OURS, 15), fn, BASE_START, BASE_END, cop, m.t_out(18.0))
        r_x, k_x = run(occupancy_strategy(FIXED, None), fn, BASE_START, BASE_END, cop, m.t_out(18.0))
        cut = BASE_END - 0.5
        un, ur, ux = (sum(uncomfortable(x, cut)) for x in (r_n, r_r, r_x))
        out(f"   {name} | {lab}")
        out(f"      ours, no rule: S${sgd(k_n):.0f}/yr if every evening ({k_n:.2f} kWh), {un} min | ours + 15-min rule: {k_r:.2f} kWh, {ur} min | fixed 25 °C: {k_x:.2f} kWh, {ux} min")
        out(f"      rule saves {k_n - k_r:.2f} kWh (S${(k_n - k_r) * TARIFF:.2f} per such evening); extra uncomfortable minutes after return: {ur - un}")
        rows.append({"feature": "occupancy", "case": name, "aircon": lab, "ours_kwh": round(k_n, 3), "with_rule_kwh": round(k_r, 3),
                     "fixed_kwh": round(k_x, 3), "saved_kwh": round(k_n - k_r, 3), "saved_sgd_per_evening": round((k_n - k_r) * TARIFF, 2),
                     "uncomfortable_no_rule": un, "uncomfortable_with_rule": ur, "uncomfortable_fixed": ux})

# restore the module and write outputs
m.START, m.END, m.activity_at = BASE_START, BASE_END, story
keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(HERE, "routine_addon.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
open(os.path.join(HERE, "routine_addon.log"), "w", encoding="utf-8").write("\n".join(log))
print("saved")
