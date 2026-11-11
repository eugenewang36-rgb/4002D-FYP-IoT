"""How much do the assumed inputs matter? (whole-evening model of folder 12, existing DC ceiling fan, 8 Nov 2026)

Each assumed input is changed on its own and the evening is re-run for (a) our system and (b) the fixed 25 °C
aircon-only baseline. We report the yearly electricity, the saving against the baseline, and the minutes the
occupant is uncomfortable with our system. The control settings per activity (best[...] in folder 12) are kept
as chosen for the base case.

Inputs varied (base value in brackets):
  UA, heat leak through the envelope (60 W/K)                       -> 40 and 90 W/K
  other internal heat, sensible + latent (190 W + 230 W)            -> x0.5 and x1.5
  warm-wall offset, MASS_OFFSET (2 °C)                              -> 0 and 4 °C
  fan power on high (25 W; cube law below high speed)               -> 50 W (Mayer Typhoon 52 DC datasheet) and 66 W (a retailer listing)
  starting room temperature (28.4 °C, real 6 pm average)            -> 29.1 °C (measured HDB average)

Run:  python assumption_sensitivity.py     (importing folder 12 re-runs it, about 2 minutes)
Outputs next to this script: assumption_sensitivity.csv, assumption_sensitivity.log
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

OURS = m.STRATEGIES[3][1]
FIXED = m.STRATEGIES[1][1]
DAYS, TARIFF = m.DAYS_PER_YEAR, m.TARIFF
BASE = dict(UA=m.UA, QS=m.Q_OTHER_SENS, QL=m.Q_OTHER_LAT, MO=m.MASS_OFFSET, T0=m.T_START, FP=25.0, fw=m.fan_watts,
            RH=m.RH, SIT=m.ACTIVITIES["Sitting"])
# Comfort-only changes (9 Nov 2026, review feedback): the controller keeps the base-case settings, but the occupant's
# real conditions differ: air speed at the desk and workout spot, room humidity, and radiant temperature.
CMP = dict(vscale=1.0, rh=BASE["RH"], mrt=0.0)


def apply(UA=None, qscale=1.0, MO=None, T0=None, FP=None, vscale=1.0, rh=None, sit_met=None, mrt=0.0):
    m.UA = UA if UA is not None else BASE["UA"]
    m.G_OUT = 1 / (1 / m.UA - 1 / m.G_IN)
    m.Q_OTHER_SENS = BASE["QS"] * qscale
    m.Q_OTHER_LAT = BASE["QL"] * qscale
    m.MASS_OFFSET = MO if MO is not None else BASE["MO"]
    m.T_START = T0 if T0 is not None else BASE["T0"]
    CMP.update(vscale=vscale, rh=rh if rh is not None else BASE["RH"], mrt=mrt)
    met0, clo0, sh0, vmax0 = BASE["SIT"]
    m.ACTIVITIES["Sitting"] = (sit_met if sit_met is not None else met0, clo0, sh0, vmax0)
    if FP is None:
        m.fan_watts = BASE["fw"]
    else:
        m.fan_watts = lambda v, FP=FP: FP * (v / 1.21) ** 3


CASES = [("Base case", {}),
         ("UA 40 W/K (leakier envelope less)", dict(UA=40)), ("UA 90 W/K", dict(UA=90)),
         ("Other internal heat x0.5", dict(qscale=0.5)), ("Other internal heat x1.5", dict(qscale=1.5)),
         ("Warm-wall offset 0 °C", dict(MO=0.0)), ("Warm-wall offset 4 °C", dict(MO=4.0)),
         ("Fan power on high 50 W", dict(FP=50.0)), ("Fan power on high 66 W", dict(FP=66.0)),
         ("Start temperature 29.1 °C", dict(T0=29.1)),
         ("Air speed at desk and workout spot x0.7", dict(vscale=0.7)), ("Air speed at desk and workout spot x0.5", dict(vscale=0.5)),
         ("Room humidity 80%", dict(rh=80)), ("Sitting at 1.2 met", dict(sit_met=1.2)),
         ("Radiant temperature 1.5 °C above the wall model", dict(mrt=1.5))]
rows, log = [], []
for name, kw in CASES:
    apply(**kw)
    out = {"case": name}
    for sys_label, cop in m.SYSTEMS:
        _, ac_o, fan_o, _ = m.simulate(OURS, cop)
        _, ac_f, fan_f, _ = m.simulate(FIXED, cop)
        o, f = ac_o + fan_o, ac_f + fan_f
        tag = "older" if cop == 3.5 else "5-tick"
        out[f"ours S$/yr {tag}"] = round(o * TARIFF * DAYS, 1)
        out[f"fixed25 S$/yr {tag}"] = round(f * TARIFF * DAYS, 1)
        out[f"saving % {tag}"] = round(100 * (1 - o / f), 1)
    recs, _, _, _ = m.simulate(OURS, 3.5)
    recs = [(h, a, ta, tm + CMP["mrt"], ts, v * (CMP["vscale"] if a in ("Sitting", "Exercising") else 1.0))
            for (h, a, ta, tm, ts, v) in recs]
    rh_keep = m.RH
    m.RH = CMP["rh"]
    _, _, mins = m.comfort_minutes(recs)
    m.RH = rh_keep
    out["uncomfortable min (ours)"] = sum(w + c for w, c, a in mins.values())
    # the same occupant conditions for the fixed 25 °C aircon-only baseline (no fan, so no air-speed change)
    recs_f, _, _, _ = m.simulate(FIXED, 3.5)
    recs_f = [(h, a, ta, tm + CMP["mrt"], ts, v) for (h, a, ta, tm, ts, v) in recs_f]
    m.RH = CMP["rh"]
    _, _, mins_f = m.comfort_minutes(recs_f)
    m.RH = rh_keep
    out["uncomfortable min (fixed 25 °C)"] = sum(w + c for w, c, a in mins_f.values())
    rows.append(out)
    line = (f"{name:38s} ours S${out['ours S$/yr older']:6.1f} (fixed S${out['fixed25 S$/yr older']:6.1f}) saving {out['saving % older']:5.1f}% older, "
            f"{out['saving % 5-tick']:5.1f}% 5-tick; {out['uncomfortable min (ours)']} min uncomfortable (fixed 25 °C: {out['uncomfortable min (fixed 25 °C)']})")
    print(line)
    log.append(line)
apply()
m.ACTIVITIES["Sitting"] = BASE["SIT"]

with open(os.path.join(HERE, "assumption_sensitivity.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
open(os.path.join(HERE, "assumption_sensitivity.log"), "w", encoding="utf-8").write("\n".join(log))
print("saved")
