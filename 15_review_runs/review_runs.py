"""Review follow-up runs (2026-10-25).

Two questions from the full-deck review, answered with the folder 12 evening model
(same room, same evening, same comfort rule; nothing in folder 12 is changed):

  Q1. What does the camera itself save?
      Folder 12 compares our system (camera + fan) with aircon-only strategies, so
      the saving mixes two things: the fan and the activity detection. Here we add
      "fan, but no camera" strategies: a fan at a fixed speed all evening and a
      fixed aircon setpoint (a temperature-only controller cannot know the user
      is exercising or asleep). We sweep that fixed setpoint, so we get the whole
      cost-vs-comfort trade-off a no-camera system can reach.

  Q2. Does the saving survive whole-degree setpoints?
      The optimiser picks setpoints like 26.07 °C or 29.76 °C; a real remote only
      takes 1 °C steps (some take 0.5 °C). We rerun our strategies with the
      setpoint rounded (a) to the nearest step and (b) down to the next step
      (cooler, the comfort-safe choice).

Variables (all from folder 12 unless stated)
  KNOB        = 25.0 °C, the feels-like target (target SET) for every activity
  TOL         = 0.5 °C, comfortable = SET within KNOB ± TOL
  V_STILL     = 0.1 m/s, air speed with the fan off
  FAN_NO_CAM  = 0.5 m/s, fan speed of the no-camera strategies. ASSUMPTION: the
                sleeping cap (noise), since without a camera the fan must be safe
                for a sleeper all night; at 0.5 m/s it draws 4.5 W (assumed fan law
                P = 3 + 12 v^3 W, folder 12)
  FIXED_SPS   = 24-28 °C in 1 °C steps, the fixed setpoints swept for Q1
  STEP        = 1.0 or 0.5 °C, the remote's setpoint step for Q2
  cop         = 3.5 (older aircon) or 5.5 (new 5-tick); results quoted for 3.5
  S$/yr       = kWh per night x TARIFF (S$0.3116/kWh) x 365 nights
  uncomfortable minutes = minutes the user is in the room with SET outside
                KNOB ± TOL (too warm + too cold), SET from the real wall temperature

Outputs (saved next to this script): review_runs.csv, review_runs.png
"""
import contextlib
import csv
import importlib.util
import io
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
F12 = os.path.join(HERE, "..", "12_activity_setpoints", "activity_setpoints.py")


def load_folder12():
    """Run folder 12 quietly (it rewrites its own outputs unchanged) and return it."""
    spec = importlib.util.spec_from_file_location("activity_setpoints", F12)
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(F12))
    try:
        with plt.rc_context(matplotlib.rcParamsDefault), contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    plt.close("all")
    return mod


m = load_folder12()
# "python review_runs.py ceiling" reruns everything with the existing DC ceiling fan
# (folder 12 reads the same command-line option); outputs get "_ceiling".
SUFFIX = "_ceiling" if "ceiling" in sys.argv[1:] else ""
FAN_NO_CAM = 0.5
FIXED_SPS = [24.0, 25.0, 26.0, 27.0, 28.0]
COP = 3.5


def run(strategy, cop=COP):
    """(S$/yr, minutes too warm, minutes too cold) for one strategy."""
    recs, ac, fan, _ = m.simulate(strategy, cop)
    _, _, mins = m.comfort_minutes(recs)
    warm = sum(w for w, c, _ in mins.values())
    cold = sum(c for w, c, _ in mins.values())
    return (ac + fan) * m.TARIFF * m.DAYS_PER_YEAR, warm, cold


def rounded(strategy, step, mode):
    """Wrap a strategy so its setpoint is rounded to the remote's step.
    mode 'nearest' = nearest step; 'down' = next step down (cooler, comfort-safe)."""
    f = round if mode == "nearest" else math.floor

    def s(act, t_m, t_a=None):
        t, v = strategy(act, t_m, t_a)
        return (None if t is None else f(t / step) * step), v   # None = aircon off
    if strategy is m.precool:   # simulate() starts earlier only for the precool function itself
        return None
    return s


rows = []


def add(group, label, strategy):
    cost, warm, cold = run(strategy)
    rows.append({"group": group, "strategy": label, "sgd_year_cop35": round(cost, 1),
                 "min_too_warm": warm, "min_too_cold": cold, "min_uncomfortable": warm + cold})
    print(f"{group:<22} {label:<52} S${cost:6.0f}/yr  {warm + cold:4d} min uncomfortable "
          f"({warm} warm, {cold} cold)")


print("Older aircon (COP 3.5), same evening as folder 12 (8 pm desk, 8:30 pm workout, 10 pm sleep to 6 am)\n")
# Reference rows straight from folder 12
for label, strat in m.STRATEGIES:
    if strat is not m.precool:
        add("folder 12", label, strat)

# Q1: fan, no camera: fixed fan speed + fixed setpoint all evening
for sp in FIXED_SPS:
    add("Q1 fan, no camera", f"Fixed {sp:.0f} °C + fan {FAN_NO_CAM} m/s all evening",
        lambda a, tm, ta, sp=sp: (sp, FAN_NO_CAM))
# and the same without a fan, so the fan's own share is visible too
for sp in FIXED_SPS:
    add("Q1 no fan, no camera", f"Fixed {sp:.0f} °C, fan off", lambda a, tm, ta, sp=sp: (sp, m.V_STILL))

# and no camera but WITH the wall sensor: the controller assumes one activity all
# evening (it cannot see the user) and compensates for warm walls, fan at that
# activity's best speed. ASSUMPTION: it assumes "Sleeping" (most of the evening)
# or "Sitting".
# (wall-sensor runs removed 3 Nov 2026: the wall sensor is no longer part of the design)

# Q2: our strategies with whole-degree (and half-degree) setpoints
ours = [("Our system", m.STRATEGIES[3][1])]
for step in (1.0, 0.5):
    for mode in ("nearest", "down"):
        for name, strat in ours:
            add(f"Q2 {step:g} °C steps", f"{name}, setpoint rounded {mode}", rounded(strat, step, mode))

with open(os.path.join(HERE, f"review_runs{SUFFIX}.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved review_runs{SUFFIX}.csv")

# -----------------------------------------------------------------------------
# Chart: cost vs uncomfortable minutes. One axis pair, one point per strategy.
# Lower-left is better. The no-camera sweeps are drawn as lines (one per fan
# setting) so the trade-off a no-camera system can reach is visible.
# -----------------------------------------------------------------------------
NAVY, BLUE, GREEN, GREY, INK, INK2, GRID = "#14365D", "#2A78D6", "#2F9E6E", "#9AA5B1", "#1A1A1A", "#55606B", "#E3E8EE"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.patch.set_facecolor("white")


def pick(group):
    return [r for r in rows if r["group"] == group]


for group, color, name in [("Q1 no fan, no camera", GREY, "No fan, no camera (fixed 24–28 °C)"),
                           ("Q1 fan, no camera", BLUE, f"Fan {FAN_NO_CAM} m/s, no camera (fixed 24–28 °C)")]:
    rs = pick(group)
    ax.plot([r["min_uncomfortable"] for r in rs], [r["sgd_year_cop35"] for r in rs], "o", color=color,
            ms=10, mec="white", mew=2, label=name, zorder=2)
    for r in rs:
        sp = r["strategy"].split()[1]
        ax.annotate(sp, (r["min_uncomfortable"], r["sgd_year_cop35"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=11, color=INK2)
for r, label in [(pick("folder 12")[3], "Our system (camera + fan)")]:
    ax.plot(r["min_uncomfortable"], r["sgd_year_cop35"], "D", color=GREEN if "wall" not in label else NAVY,
            ms=11, mec="white", mew=2, zorder=3, label=label)

ax.set_xlabel("Uncomfortable minutes in the evening (SET outside 25 ± 0.5 °C)", color=INK)
ax.set_ylabel("Electricity, S$ per year (older aircon)", color=INK)
ax.set_title("What the camera adds: cost vs comfort, same evening (simulated)", loc="left", color=INK, fontsize=15)
ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
ax.legend(frameon=False, loc="lower left", fontsize=11)
ax.text(0.0, -0.16, "Lower-left is better. Numbers next to the dots are the fixed aircon setpoint (°C). "
        "Model: folder 12 two-node bedroom, fan law assumed.", transform=ax.transAxes, fontsize=10, color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"review_runs{SUFFIX}.png"), facecolor="white")
print("Saved review_runs.png")
