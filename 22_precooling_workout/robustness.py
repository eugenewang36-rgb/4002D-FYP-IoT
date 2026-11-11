"""Robustness runs requested in review (9 Nov 2026). Same evening and room model as folder 12.

1. Grid: fan power on high 25 W (base) / 50 W (Mayer datasheet) / 50 W with the datasheet airflow (about 4,600 CFM
   against the 5,000 CFM base, so air speed x0.92 for the comfort calculation) x aircon (older / 5-tick) x
   sitting cooled as usual (A) / not cooled by anyone (B) x controller power (Pi averaging 3 / 5 / 8 W).
   Saving is against the fixed 25 C aircon-only baseline run in the same case.
2. Misclassification: the first 15% of each confirmed activity block is read as a neighbouring activity
   (sleeping -> sitting, exercising -> sitting, sitting -> sleeping), alone and all together (assumption: errors
   come at the start of a block, which is the worst place because the settings then run for the whole block's
   first minutes, this is a deliberately simple stress test, not a classifier model).
Outputs: robustness_grid.csv, robustness_misclass.csv
"""
import contextlib, csv, io, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "12_activity_setpoints", "activity_setpoints.py")
src = open(SRC, encoding="utf-8").read().split("fields = []")[0].replace("os.chdir(os.path.dirname(os.path.abspath(__file__)))", "pass")
sys.argv = ["x", "ceiling"]
ns = {"__name__": "folder12", "__file__": SRC}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, SRC, "exec"), ns)
simulate, comfort_minutes, our_system = ns["simulate"], ns["comfort_minutes"], ns["our_system"]
SYSTEMS, TARIFF, DAYS, V_STILL, NOW = ns["SYSTEMS"], ns["TARIFF"], ns["DAYS_PER_YEAR"], ns["V_STILL"], ns["NOW"]
ns["PRECOOL_MIN"] = 0
BASE_FW = ns["fan_watts"]
PI_COST = {3: 17.2, 5: 23.5, 8: 32.9}       # S$/yr, 24 h running, from net_savings.py

def fixed25(a, tm, ta, off):
    return (None, V_STILL) if (off and a == "Sitting") else (25.0, V_STILL)

def ours(a, tm, ta, off):
    return our_system(a, tm, ta)

def run(fn, off, cop, vscale=1.0):
    ns["STUDY_AC_ON"][0] = False
    strat = lambda a, tm, ta: fn(a, tm, ta, off)
    ns["precool"] = strat
    recs, ac, fan, _ = simulate(strat, cop)
    recs = [(h, a, ta, tm, ts, v * (vscale if a in ("Sitting", "Exercising", "Sleeping") else 1.0)) for (h, a, ta, tm, ts, v) in recs]
    _, _, mins = comfort_minutes(recs)
    return (ac + fan) * TARIFF * DAYS, sum(int(v[0]) + int(v[1]) for v in mins.values())

grid = []
for fan_label, fp, vs in (("25 W (base)", 25.0, 1.0), ("50 W", 50.0, 1.0), ("50 W with 4,600 CFM datasheet airflow", 50.0, 0.92)):
    ns["fan_watts"] = (lambda v, fp=fp: fp * (v / 1.21) ** 3) if fp != 25.0 else BASE_FW
    for case, off in (("A sitting cooled as usual", False), ("B sitting not cooled", True)):
        for sysl, cop in SYSTEMS:
            fx, _ = run(fixed25, off, cop)
            ou, mn = run(ours, off, cop, vs)
            gross = fx - ou
            for pi, cost in PI_COST.items():
                grid.append(dict(fan=fan_label, case=case, aircon=sysl, pi_w=pi, fixed25_sgd=round(fx, 1), ours_sgd=round(ou, 1),
                                 gross_pct=round(100 * gross / fx, 1), controller_sgd=cost, net_sgd=round(gross - cost, 1),
                                 net_pct=round(100 * (gross - cost) / fx, 1), uncomfortable_min=mn))
            print(fan_label, case, sysl[:6], "gross", round(100 * gross / fx, 1), "net Pi5", round(100 * (gross - 23.5) / fx, 1), "min", mn)
ns["fan_watts"] = BASE_FW
with open(os.path.join(HERE, "robustness_grid.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)

BLOCKS = {"Sitting": (18.0, 20.5), "Exercising": (20.5, 22.0), "Sleeping": (22.0, 30.0)}
SCN = [("No errors", {}), ("Sleeping read as sitting (first 15% of the night)", {"Sleeping": "Sitting"}),
       ("Exercising read as sitting (first 15% of the workout)", {"Exercising": "Sitting"}),
       ("Sitting read as sleeping (first 15% of the sitting period)", {"Sitting": "Sleeping"}),
       ("All three errors together", {"Sleeping": "Sitting", "Exercising": "Sitting", "Sitting": "Sleeping"})]
mis = []
for name, scn in SCN:
    def mfn(a, tm, ta, off, scn=scn):
        h = NOW[0]
        s, e = BLOCKS[a]
        a2 = scn[a] if (a in scn and (h - s) < 0.15 * (e - s)) else a
        return our_system(a2, tm, ta)
    for sysl, cop in SYSTEMS:
        fx, _ = run(fixed25, False, cop)
        ou, mn = run(mfn, False, cop)
        mis.append(dict(scenario=name, aircon=sysl, ours_sgd=round(ou, 1), fixed25_sgd=round(fx, 1), saving_pct=round(100 * (1 - ou / fx), 1), uncomfortable_min=mn))
        print(mis[-1])
with open(os.path.join(HERE, "robustness_misclass.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(mis[0])); w.writeheader(); w.writerows(mis)
