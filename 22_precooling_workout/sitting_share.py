"""How much of the saving comes from the sitting period? (review feedback, 9 Nov 2026)

Same evening and room model as folder 12 (ceiling fan, real hourly temperatures, 13-minute sensor cycle).
Case A: the household cools the sitting period as it does the rest of the evening (the report's headline case).
Case B: the household would not cool the sitting period anyway: the aircon is OFF from 6 pm to 8:30 pm in every
        strategy (fixed 25 C: no fan; fixed 26 C + 0.5 m/s fan: fan only; ours: aircon off, fan as chosen).
Strategies: fixed 25 C aircon only | fixed 26 C + 0.5 m/s fan, no camera | our system.
Outputs: sitting_share.csv and the printed table.
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
SYSTEMS, TARIFF, DAYS, V_STILL = ns["SYSTEMS"], ns["TARIFF"], ns["DAYS_PER_YEAR"], ns["V_STILL"]
ns["PRECOOL_MIN"] = 0
ns["precool"] = None

def fixed25(a, tm, ta, off_sitting=False):
    if off_sitting and a == "Sitting":
        return None, V_STILL
    return 25.0, V_STILL

def nocam(a, tm, ta, off_sitting=False):
    if off_sitting and a == "Sitting":
        return None, 0.5
    return 26.0, 0.5

def ours(a, tm, ta, off_sitting=False):
    return our_system(a, tm, ta)          # already aircon-off while sitting

rows = []
for case, off in (("A: sitting period cooled as usual", False), ("B: sitting period not cooled by anyone", True)):
    for sysl, cop in SYSTEMS:
        res = {}
        for name, fn in (("Fixed 25 C aircon only", fixed25), ("Fixed 26 C + 0.5 m/s fan, no camera", nocam), ("Our system", ours)):
            ns["STUDY_AC_ON"][0] = False
            strat = (lambda a, tm, ta, fn=fn, off=off: fn(a, tm, ta, off))
            ns["precool"] = strat
            recs, ac, fan, _ = simulate(strat, cop)
            _, _, mins = comfort_minutes(recs)
            kwh = ac + fan
            res[name] = (kwh, kwh * TARIFF * DAYS, sum(int(v[0]) + int(v[1]) for v in mins.values()))
        base = res["Fixed 25 C aircon only"][0]
        for name, (kwh, sgd, m) in res.items():
            rows.append(dict(case=case, aircon=sysl, strategy=name, kwh_evening=round(kwh, 3), sgd_year=round(sgd, 1),
                             saving_vs_fixed25_pct=round(100 * (1 - kwh / base), 1), uncomfortable_min=m))
            print(rows[-1])
with open(os.path.join(HERE, "sitting_share.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
