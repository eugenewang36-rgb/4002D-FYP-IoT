"""Activity detection followed by our system's action (ceiling fan + aircon), common bedroom.

Scenario (one evening, 5-second steps):
  8:00 pm   the user comes home to the unconditioned bedroom (air and walls 29.0 °C, as folders 11-12)
            and sits down to study at the desk.
  8:30 pm   the user starts a home workout (calisthenics) and keeps going until 10 pm, then sleeps.

What the system does at each change of activity (timings are ASSUMPTIONS from report 2.4.3 and
the deck's response target):
  - pose model (YOLO-pose) samples frames at 10 per second and looks at a 3 s window.
  - the new activity must hold for 45 s before it is confirmed (team decision, 6 Nov 2026, all activities) -> confirmed at +48 s
  - the Pi picks the lowest-power setting, publishes it over MQTT and the
    ESP32 fires the IR commands (12 s after confirmation)                   -> applied at +60 s
    (the response target is <= 45 s after the change is confirmed, it was 18 s / 30 s with a 15 s hold)
Before the user arrives the room is empty: aircon and fan off.

Settings (from folder 12, ceiling-fan run, knob = feels like 25 °C):
  studying / resting: aircon OFF, fan 0.5-0.85 m/s (26 °C safety cycle never starts on this evening)
  exercising       : aircon held at 25 °C + fan 1.21 m/s (high, 25 W)

Compared with:
  - aircon alone at 25 °C, switched on when the user walks in (no fan, no camera).
  - "no activity detection": our sitting setting kept through the workout.

Variables: SET = Standard Effective Temperature (°C), what the user feels. Target 25 ± 0.5 °C.
t_a = room air temperature (°C), t_m = wall temperature (°C), v = air speed at the user (m/s).
Outputs (next to this script): detection_response.csv, s_study_detect.png, s_exercise_detect.png
"""
import contextlib
import csv
import glob
import io
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "12_activity_setpoints"))
sys.argv = [sys.argv[0], "ceiling"]                       # folder 12 in ceiling-fan mode
cwd = os.getcwd()
os.chdir(os.path.join(HERE, "..", "12_activity_setpoints"))
with contextlib.redirect_stdout(io.StringIO()):
    import activity_setpoints as m12                       # room model, SET, fan power, best settings
plt.close("all")
os.chdir(cwd)

DT = 5                                                     # s, time step
ARRIVE, EX_START, EX_END = 18.0, 20.5, 22.0   # clock hours: home 6 pm, studying/resting to 8:30, workout 8:30-10 pm
CONFIRM_S, APPLY_S = 48, 60                                # s after a change: confirmed, command applied
KNOB, TOL = m12.KNOB, m12.TOL
BEST = m12.best                                            # activity -> (setpoint °C, fan m/s)


def true_activity(h):
    if h < ARRIVE:
        return "Away"
    return "Exercising" if EX_START <= h < EX_END else "Sitting"


def run(mode, cop=3.5):
    """mode: 'ours' (detect + act), 'no_detect' (sitting setting kept), 'aircon_only' (25 °C, no fan)."""
    t_a = t_m = m12.T_START
    h, end = ARRIVE - 0.25, EX_END + 0.25
    applied = "Away"                                       # activity the controls are currently set for
    change_at = None
    recs, kwh = [], 0.0
    step_no, seen_m, seen_a = 0, t_m, t_a            # sensor reading held between cycles (13 min, Kai Feng)
    while h < end - 1e-9:
        if m12.SENSOR_CYCLE_S == 0 or (step_no * DT) % m12.SENSOR_CYCLE_S == 0:
            seen_m, seen_a = t_m, t_a
        step_no += 1
        act = true_activity(h)
        if act != (recs[-1]["act"] if recs else "Away"):
            change_at = h
        m12.NOW[0] = h
        if mode == "ours" and change_at is not None and (h - change_at) * 3600 >= APPLY_S:
            applied = act
        if mode == "no_detect" and act != "Away" and change_at is not None and (h - change_at) * 3600 >= APPLY_S:
            applied = "Sitting"
        if mode == "aircon_only":
            t_set, v = (25.0, m12.V_STILL) if act != "Away" else (None, m12.V_STILL)
        else:
            t_set, v = m12.our_system(applied, seen_m, seen_a) if applied != "Away" else (None, m12.V_STILL)
        sens, lat = m12.person_heat(act)
        to = m12.t_out(h % 24)
        q_wall = m12.G_IN * (t_m - t_a)
        q_gain = q_wall + m12.Q_OTHER_SENS + sens + m12.fan_watts(v)
        q_ac = 0.0 if t_set is None else min(max(m12.C_A * (t_a - t_set) / DT + q_gain, 0.0), m12.Q_MAX)
        p_ac = 0.0
        if q_ac > 0:
            p_ac = (q_ac + m12.Q_OTHER_LAT + lat) / (cop * m12.carnot(t_a, to) / m12.carnot(25.0, to))
        kwh += (p_ac + m12.fan_watts(v)) * DT / 3.6e6
        s = None
        if act != "Away":
            met, clo, _, _ = m12.ACTIVITIES[act]
            s = m12.set_value(t_a, t_m, v, met, clo)
        recs.append({"h": h, "act": act, "applied": applied, "t_a": t_a, "t_m": t_m,
                     "t_set": t_set, "v": v, "set": s, "p_w": p_ac + m12.fan_watts(v), "kwh": kwh})
        t_a += (q_gain - q_ac) * DT / m12.C_A
        t_m += (m12.G_OUT * (to + m12.MASS_OFFSET - t_m) - q_wall) * DT / m12.C_M
        h += DT / 3600
    return recs


runs = {m: run(m) for m in ("ours", "no_detect", "aircon_only")}


def window(recs, a, b):
    return [r for r in recs if a - 1e-9 <= r["h"] < b - 1e-9]


def summary(recs, a, b, act):
    w = [r for r in window(recs, a, b) if r["act"] == act]
    mins = DT / 60
    warm = sum(r["set"] > KNOB + TOL for r in w) * mins
    cold = sum(r["set"] < KNOB - TOL for r in w) * mins
    first_ok = next((r["h"] for r in w if abs(r["set"] - KNOB) <= TOL), None)
    kwh = w[-1]["kwh"] - w[0]["kwh"]
    return {"too_warm_min": round(warm, 1), "too_cold_min": round(cold, 1),
            "minutes_to_comfort": None if first_ok is None else round((first_ok - a) * 60, 1),
            "kwh": round(kwh, 3), "avg_set": round(sum(r["set"] for r in w) / len(w), 2)}


rows = []
for label, (a, b, act) in {"Studying / resting 6:00-8:30 pm": (ARRIVE, EX_START, "Sitting"),
                           "Workout 8:30-10:00 pm": (EX_START, EX_END, "Exercising")}.items():
    for m, recs in runs.items():
        s = summary(recs, a, b, act)
        rows.append({"period": label, "strategy": m, **s})
        print(f"{label:24s} {m:12s} {s}")
with open(os.path.join(HERE, "detection_response.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# ---------------------------------------------------------------- charts (deck style)
for f in glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts", "OpenSans-*.ttf")):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Open Sans", "DejaVu Sans"], "font.size": 16})
INK, INK_2, GRID = "#1A1A1A", "#4A4A4A", "#E4E3DF"
BLUE, NAVY, GREY, ORANGE, GREEN = "#2A78D6", "#14365D", "#9AA3AE", "#C2410C", "#2E9E6B"


def clock(h):
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh % 12 or 12}:{mm:02d}"


def chart(a, b, change, act_label, fname, lines, note_xy, legend_loc="upper right"):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.75), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    for m, color, lw, ls, lab in lines:
        w = [r for r in window(runs[m], a, b) if r["set"] is not None]
        ax.plot([r["h"] for r in w], [r["set"] for r in w], color=color, lw=lw, ls=ls, label=lab)
    ax.axhspan(KNOB - TOL, KNOB + TOL, color=GREEN, alpha=0.12, lw=0)
    ax.text(b - 0.005, KNOB + TOL + 0.04, "comfortable (target 25 ± 0.5 °C)", ha="right", va="bottom", fontsize=14, color=GREEN)
    ax.text(note_xy[0], note_xy[1], f"0 s   {act_label} starts\n{CONFIRM_S} s   pose model confirms it\n{APPLY_S} s   command sent",
            fontsize=14, color=INK, va="center", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID))
    for dt, txt in ((0, f"{act_label}\nstarts"), (CONFIRM_S, "pose model\nconfirms"), (APPLY_S, "command\nsent")):
        x = change + dt / 3600
        ax.axvline(x, color=INK_2, lw=1, ls=":")
        ax2.axvline(x, color=INK_2, lw=1, ls=":")
    ax.set_ylabel("What the user feels\n(SET, °C)", color=INK_2)
    ax.legend(loc=legend_loc, frameon=False, fontsize=14, **({"bbox_to_anchor": (1, 0.63)} if legend_loc == "center right" else {}))
    w = window(runs["ours"], a, b)
    ax2.step([r["h"] for r in w], [r["t_set"] if r["t_set"] is not None else float("nan") for r in w],
             where="post", color=NAVY, lw=3, label="aircon setpoint (°C)")
    ax2.set_ylabel("Aircon\nsetpoint (°C)", color=NAVY)
    if all(r["t_set"] is None for r in w):              # aircon off for the whole window (studying / resting)
        ax2.set_ylim(20, 30)
        ax2.text(a + 0.3 * (b - a), 21.6, "Aircon off: the fan does the work", color=NAVY, fontsize=15,
                 va="center", fontweight="bold")
    ax3 = ax2.twinx()
    ax3.step([r["h"] for r in w], [r["v"] if r["v"] > 0.11 else 0 for r in w], where="post", color=BLUE, lw=3)
    ax3.set_ylabel("Fan (m/s)", color=BLUE)
    ax3.set_ylim(0, 1.4)
    for a_ in (ax, ax2, ax3):
        for side in ("top",):
            a_.spines[side].set_visible(False)
        a_.tick_params(colors=INK_2)
    ax.grid(axis="y", color=GRID); ax2.grid(axis="y", color=GRID)
    step = 1 / 12 if b - a <= 0.75 else 0.25                    # 5-minute ticks, or 15 for long windows
    first = (int(a / step) + 1) * step
    ticks = [first + k * step for k in range(13) if first + k * step <= b + 1e-9]
    ax2.set_xticks(ticks); ax2.set_xticklabels([clock(t) + " pm" for t in ticks])
    ax2.set_xlim(a, b)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=200)
    plt.close(fig)
    print("saved", fname)


chart(ARRIVE - 3 / 60, EX_START, ARRIVE, "studying / resting", "s_study_detect.png",
      [("aircon_only", GREY, 3, "-", "Aircon alone, 25 °C"), ("ours", BLUE, 4, "-", "Our system: detects studying / resting, then acts")],
      (ARRIVE + 35 / 60, 26.25))
chart(EX_START - 3 / 60, EX_END, EX_START, "workout", "s_exercise_detect.png",
      [("aircon_only", GREY, 3, "-", "Aircon alone, 25 °C"), ("no_detect", ORANGE, 3, "--", "No activity detection (desk setting kept)"),
       ("ours", BLUE, 4, "-", "Our system: detects exercise, then acts")],
      (EX_START + 12 / 60, 30.05), legend_loc="center right")
