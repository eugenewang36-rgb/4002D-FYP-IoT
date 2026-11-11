"""Scenario: the user studies/rests for longer. Home at 6 pm and at the desk until 8:30 pm, workout 8:30-10 pm,
sleep 10 pm-6 am. Same room, real Singapore hourly outdoor temperatures, room and walls start at 29.1 °C,
same control rules as folder 12 (our system). Compares with the story evening (home at 8 pm).
Output: longer_study.csv next to this script."""
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


def schedule(home):
    def activity_at(hour):
        if hour < home - 1e-9:
            return "Away"
        if hour < 20.5:
            return "Sitting"
        if hour < 22.0:
            return "Exercising"
        return "Sleeping"
    return activity_at


rows = []
for label, home in (("Story evening: home 8 pm", 20.0), ("Longer study: home 6 pm", 18.0)):
    m.START, m.activity_at = home, schedule(home)        # simulate() and comfort_minutes() read these globals
    res = {}
    for name, strat in m.STRATEGIES[:4]:
        out = {}
        for sys_label, cop in m.SYSTEMS:
            pa = {}
            recs, ac, fan, _ = m.simulate(strat, cop, pa)
            out[sys_label] = ((ac + fan) * m.TARIFF * m.DAYS_PER_YEAR, pa)
        _, _, mins = m.comfort_minutes(recs)
        out["unc"] = sum(w + c for w, c, _ in mins.values())
        res[name] = out
    ours, fixed, matched = m.STRATEGIES[3][0], m.STRATEGIES[1][0], m.STRATEGIES[2][0]
    for sys_label, _ in m.SYSTEMS:
        o, f, mt = res[ours][sys_label][0], res[fixed][sys_label][0], res[matched][sys_label][0]
        po, pf = res[ours][sys_label][1], res[fixed][sys_label][1]
        row = {"scenario": label, "aircon": sys_label, "ours_sgd_year": round(o), "fixed25_sgd_year": round(f),
               "saving_vs_fixed25_pct": round(100 * (1 - o / f), 1), "saving_vs_matched_pct": round(100 * (1 - o / mt), 1),
               "ours_uncomfortable_min": res[ours]["unc"], "fixed25_uncomfortable_min": res[fixed]["unc"]}
        for act in ("Sitting", "Exercising", "Sleeping"):
            row[f"{act.lower()}_saving_vs_fixed25_pct"] = round(100 * (1 - po[act] / pf[act]), 1)
        rows.append(row)
        print(row)
with open(os.path.join(HERE, "longer_study.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
