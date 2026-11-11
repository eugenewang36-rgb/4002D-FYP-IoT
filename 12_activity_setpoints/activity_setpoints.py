"""Activity recognition -> aircon + fan settings -> energy.

This ties the comfort model to the project's goal: the camera detects what the
user is doing (sleeping, sitting, exercising), and the system picks the
aircon setpoint and fan level that hold the user's chosen comfort for that
activity with the least electricity.

Part 1: for each activity, the aircon setpoint needed at each fan speed to
        feel the same (equal SET), and the lowest-power combination.
Part 2: a realistic evening in a common bedroom (workout, then desk, then
        sleep), simulated minute by minute with the two-node room model from
        folder 11, comparing four control strategies:
          1. aircon only, set cold for the workout and left there
          2. aircon only, fixed 25 °C (Go 25)
          3. aircon only, setpoint changes with the detected activity
          4. our system: activity-aware aircon setpoint + fan level

The comfort target follows the team's knob design (deck slide "Knob Setting
-> Room Goal"): the knob sets a feels-like temperature, read as a target SET
(Standard Effective Temperature) for whatever the user is doing. Default
knob = 25 °C. "Comfortable" = SET within ±0.5 °C of the target, as in the
team's optimiser.

Fan options (command line):
    python activity_setpoints.py            DC standing fan pointed at the user
                                            (original model, kept for comparison)
    python activity_setpoints.py ceiling    the household's existing DC ceiling
                                            fan (the system's design from Nov 2026)

Outputs (saved next to this script, "_ceiling" added for the ceiling fan):
    activity_setpoints.png   part 1: setpoint vs fan speed per activity
    activity_evening.png     part 2: the evening, energy and comfort
    activity_results.csv     every number behind both charts
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # save outputs next to this script

import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq  # finds the input value that makes a function equal zero

from pythermalcomfort.models import set_tmp

plt.rcParams["text.parse_math"] = False  # treat "$" as plain text, not a maths formula

# =============================================================================
# 1. Activities, comfort target and fan (ASSUMPTIONS, stated)
# =============================================================================
RH = 60  # %, relative humidity with the aircon on
KNOB = 25.0  # °C, the feels-like target the user sets = target SET for every activity
TOL = 0.5  # °C, comfortable = SET within ±TOL of the target (team optimiser rule)

# Activities: name -> (met, clo, sensible share of body heat, max fan speed m/s)
#   met = metabolic rate: 0.7 asleep, 1.0 sitting, 3.5 home workout
#         (calisthenics / skipping, ASHRAE compendium range ~3-4.5)
#   clo = insulation: 0.9 in bed without covers, 0.4 T-shirt and shorts.
#         0.36 light sportswear
#   sensible share = part of the body's heat that warms the room air. The
#         rest leaves as sweat/breath moisture (ASSUMPTION: 0.7 at rest,
#         0.5 when exercising because of sweating)
#   max fan = the team's limit: sleeping capped at level 1 (0.5 m/s, draught
#         and noise). Otherwise up to level 3 (1.5 m/s)
ACTIVITIES = {
    "Sleeping": (0.7, 0.9, 0.7, 0.85),   # sleep fan up to 0.85 m/s (4 Nov 2026, noise not modelled)
    "Sitting": (1.0, 0.4, 0.7, 1.5),
    "Exercising": (3.5, 0.36, 0.5, 1.5),
}
FAN = "ceiling" if "ceiling" in sys.argv[1:] else "standing"
SUFFIX = "_ceiling" if FAN == "ceiling" else ""
EXTRA_EXERCISE = [("Light exercise (2.5 met)", 2.5), ("Hard exercise (4.5 met)", 4.5)]  # chart only
BODY_AREA = 1.8  # m², skin area of an average adult (1 met = 58.2 W/m²)

# Standing fan levels (team design): air speed at the user (m/s) -> power (W).
# ASSUMPTION: DC-motor fan, 15 W at 1.0 m/s. Power ~ (speed)^3 plus 3 W for
# the electronics: 0.5 m/s 4.5 W, 1.0 m/s 15 W, 1.5 m/s 43.5 W.
V_STILL = 0.1
FAN_LEVELS = [0.1, 0.5, 1.0, 1.5]  # 0.1 = fan off (still air)


def fan_watts(v):
    return 0.0 if v <= V_STILL else 3.0 + 12.0 * v ** 3


if FAN == "ceiling":
    # Existing 52" DC ceiling fan (folder 11): 1.21 m/s on high = Raftery et al.
    # (2019) average over the common bedroom's open floor, 25 W on high
    # (ASSUMPTION, listings 18-33 W). Turned down, air speed is taken to scale
    # with fan speed and power with speed cubed (fan law):
    #   P = 25 W x (v / 1.21)^3  ->  0.5 m/s 1.8 W, 0.85 m/s 8.7 W, 1.21 m/s 25 W.
    # The fan cannot go above 1.21 m/s, so sitting and exercising are capped
    # there. Sleeping keeps the 0.5 m/s cap.
    V_CEIL_MAX, CEIL_W_MAX = 1.21, 25.0
    FAN_LEVELS = [0.1, 0.5, 0.85, V_CEIL_MAX]
    ACTIVITIES = {a: (m, c, s, min(vm, V_CEIL_MAX)) for a, (m, c, s, vm) in ACTIVITIES.items()}

    def fan_watts(v):
        return 0.0 if v <= V_STILL else CEIL_W_MAX * (v / V_CEIL_MAX) ** 3


SET_MIN, SET_MAX = 16.0, 30.0  # °C, aircon remote range


def set_value(tdb, tr, v, met, clo):
    """SET (°C). tdb = air temperature, tr = wall (mean radiant) temperature,
    v = air speed. Sitting posture (the SET model has no lying option)."""
    return float(set_tmp(tdb=tdb, tr=tr, v=v, rh=RH, met=met, clo=clo, position="sitting",
                         limit_inputs=False, round_output=False).set)


def setpoint_for(met, clo, v, target=KNOB):
    """Aircon setpoint (°C) at air speed v that gives SET = target (walls at air
    temperature), clipped to the remote's range."""
    t = brentq(lambda x: set_value(x, x, v, met, clo) - target, 5, 45)
    return min(max(t, SET_MIN), SET_MAX)


# =============================================================================
# 2. Room model (common bedroom, from folder 11)
# =============================================================================
TARIFF = 0.3116  # S$ per kWh, SP Group Oct-Dec 2026 incl. 9% GST
DAYS_PER_YEAR = 365
SYSTEMS = [("Older aircon (COP 3.5)", 3.5), ("New 5-tick aircon (COP 5.5)", 5.5)]
L, W, H_C = 3.3, 3.0, 2.6  # m, common bedroom and ceiling height
AREA = L * W
UA = 60  # W/K, heat leak through the envelope (as in folders 03 and 11)
MASS_OFFSET = 2.0  # °C, sun-warmed outside walls act as if 2 °C hotter outside
Q_OTHER_SENS = 190  # W, lights, phone/laptop, solar through the window at night (ASSUMPTION)
Q_OTHER_LAT = 230  # W, moisture from outside air leaking in, as heat (ASSUMPTION)
#   with a sleeper these give ~240 W sensible + ~250 W latent, matching folder 03
C_AIR_PER_M2, MASS_DEPTH, RHO_CP, H_IN = 25e3, 0.05, 2.2e6, 8.0  # see folder 11
Q_MAX = 2640  # W, 9,000 BTU/h bedroom fan coil
# Starting room temperature (3 Nov 2026): 29.1 °C, the MEASURED average indoor air temperature of
# Singapore's naturally ventilated HDB flats during the day and early evening ("Adaptive behaviour and
# thermal comfort in Singapore's naturally ventilated housing", ResearchGate 339297268). Walls start at
# the same temperature (ASSUMPTION).
# 4 Nov 2026: the story evening starts at 6 pm, so the room and walls start at the REAL average air temperature
# at 6 pm (folder 17, data.gov.sg stations, 2025). This is an outdoor reading. An unventilated room is usually
# a little warmer (29.1 °C measured day/early-evening average above), so this start is slightly conservative.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "17_singapore_temperature",
                       "hourly_temperature.csv"), encoding="utf-8") as _f6:
    T_START = [float(r["mean_C"]) for r in csv.DictReader(_f6)][18]   # °C at 6 pm
DT = 30  # s

surfaces = 2 * AREA + 2 * (L + W) * H_C
C_A = C_AIR_PER_M2 * AREA
C_M = surfaces * MASS_DEPTH * RHO_CP
G_IN = H_IN * surfaces
G_OUT = 1 / (1 / UA - 1 / G_IN)


import csv as _csv
import os as _os
_SG = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "17_singapore_temperature", "hourly_temperature.csv")
with open(_SG, encoding="utf-8") as _f:
    SG_HOURLY = [float(r["mean_C"]) for r in _csv.DictReader(_f)]   # °C, hours 0-23 (real averages)


def t_out(hour):
    """Outdoor air temperature (°C) at a clock hour: the REAL island-wide average for that hour
    (data.gov.sg weather stations, 2025, folder 17_singapore_temperature/hourly_temperature.csv),
    linearly interpolated between whole hours. House rule: simulations use real Singapore averages."""
    h = hour % 24
    i = int(h) % 24
    j = (i + 1) % 24
    f = h - int(h)
    return SG_HOURLY[i] * (1 - f) + SG_HOURLY[j] * f


def carnot(t_set, t_o):
    t_evap, t_cond = t_set - 12 + 273.15, t_o + 12 + 273.15
    return t_evap / (t_cond - t_evap)


def person_heat(activity):
    """(sensible W, latent W) from the user doing this activity (0 when away)."""
    if activity == "Away":
        return 0.0, 0.0
    met, _, sens, _ = ACTIVITIES[activity]
    total = met * 58.2 * BODY_AREA
    return total * sens, total * (1 - sens)


def steady_power(activity, t_set, v, cop_25, t_o=28.0):
    """Electricity (W) to hold t_set with fan speed v once the room has settled."""
    sens, lat = person_heat(activity)
    load = UA * (t_o + MASS_OFFSET - t_set) + Q_OTHER_SENS + sens + fan_watts(v) + Q_OTHER_LAT + lat
    return max(load, 0) / (cop_25 * carnot(t_set, t_o) / carnot(25.0, t_o)) + fan_watts(v)


# =============================================================================
# 3. Part 1: setpoint per activity and fan speed. Lowest-power combination
# =============================================================================
rows = []
best = {}  # activity -> (setpoint, fan speed) chosen by our system
print(f"Part 1: knob = feels like {KNOB} °C (target SET), {RH}% RH\n")
for act, (met, clo, _, vmax) in ACTIVITIES.items():
    print(f"{act} ({met} met, {clo} clo), fan allowed up to {vmax} m/s")
    options = []
    for v in FAN_LEVELS:
        t = setpoint_for(met, clo, v)
        allowed = v <= vmax + 1e-9
        p = steady_power(act, t, v, 3.5)
        options.append((p, t, v, allowed))
        print(f"  fan {v:>3.1f} m/s ({fan_watts(v):4.1f} W): setpoint {t:4.1f} °C, "
              f"steady power {p:4.0f} W (older aircon){'' if allowed else '  [over the fan cap]'}")
        rows.append({"part": "1 setpoint", "activity": act, "met": met, "clo": clo, "fan_m_s": v,
                     "fan_w": round(fan_watts(v), 1), "setpoint_C": round(t, 2),
                     "steady_power_w_cop35": round(p), "allowed": allowed})
    p, t, v, _ = min(o for o in options if o[3])
    best[act] = (t, v)
    print(f"  -> lowest power: {t:.1f} °C with fan {v} m/s\n")

# Design rules adopted 4 Nov 2026 (folder 18 tested them):
#  - workout: the fan runs on high and the aircon is held at WORKOUT_FLOOR_C instead of being chased colder
#    (the user is above target for the whole workout either way, colder air costs much more for little gain)
#  - sleep: the comfort target rises slowly through the night (in-band ramp, stays inside KNOB ± TOL)
#  - wake-up: for the last WAKE_MIN minutes the target rises to WAKE_TOP_SET so the warming room nudges the
#    user awake. These minutes are intended and are not counted as discomfort for any strategy
WORKOUT_FLOOR_C = 25.0   # °C, lowest aircon setpoint during a workout (design rule)
SLEEP_START = 22.0       # clock hour sleep starts (story timeline)
RAMP_PER_H = 0.05        # °C of target SET per hour of sleep (design rule)
RAMP_CAP = 0.4           # °C, the ramp stops here, inside the ±TOL comfort band
WAKE_MIN = 30            # min, wake-up ramp length before 6 am
WAKE_TOP_SET = 27.0      # °C, target SET at 6 am (warm enough to wake, ASSUMED)
# Decision 8 Nov 2026: the headline system switches the aircon OFF at 5:30 am (fan keeps its sleep speed), so the room
# warms and wakes the user. Run "python activity_setpoints.py ceiling ramp" to get the older behaviour (target SET
# ramping up to WAKE_TOP_SET with the aircon still on) for comparison.
WAKE_AIRCON_OFF = "ramp" not in sys.argv[1:]
t_ex, v_ex = best["Exercising"]
best["Exercising"] = (max(t_ex, WORKOUT_FLOOR_C), v_ex)
print(f"Rules: workout {best['Exercising'][0]:.1f} °C + fan {v_ex} m/s; sleep fan {best['Sleeping'][1]} m/s at "
      f"{best['Sleeping'][0]:.2f} °C, target SET +{RAMP_PER_H} °C/h up to +{RAMP_CAP}; wake ramp to SET "
      f"{WAKE_TOP_SET} °C over the last {WAKE_MIN} min\n")

# =============================================================================
# 4. Part 2: an evening in the common bedroom
# =============================================================================
START, END = 18.0, 30.0  # 6 pm to 6 am (story timeline, changed 4 Nov 2026)


def activity_at(hour):
    """The user's evening (story timeline): home at 6 pm, studying or resting at the desk until 8:30 pm
    (one activity, "Sitting", 1.0 met), works out 8:30-10 pm, then sleeps until 6 am."""
    if hour < START - 1e-9:
        return "Away"
    if hour < 20.5:
        return "Sitting"
    if hour < 22.0:
        return "Exercising"
    return "Sleeping"


no_fan = {a: setpoint_for(m, c, V_STILL) for a, (m, c, _, _) in ACTIVITIES.items()}


def with_wall_sensor(act, t_m, t_a=None):
    """Our system + a wall-temperature sensor (e.g. an IR thermometer pointed at
    the wall): the setpoint is solved with the MEASURED wall temperature as the
    radiant temperature, so warm walls are compensated by cooler air until they
    cool down. Fan-down rule (review follow-up): an aircon cannot heat, so if
    the room air t_a is already colder than needed (more than 0.5 °C below the
    setpoint), the fan steps down to the level that brings SET closest to the
    target instead of making the user colder."""
    met, clo, _, vmax = ACTIVITIES[act]
    v = best[act][1]
    t = min(max(brentq(lambda x: set_value(x, t_m, v, met, clo) - KNOB, 0, 45), SET_MIN), SET_MAX)
    if t_a is not None and t_a < t - 0.5:
        levels = [lv for lv in FAN_LEVELS if lv <= vmax + 1e-9]
        v = min(levels, key=lambda lv: abs(set_value(t_a, t_m, lv, met, clo) - KNOB))
    return t, v


NOW = [START]   # clock hour of the current simulation step (set by simulate())

# Studying / resting (4 Nov 2026): nobody sets an aircon to 29.5 °C, so the aircon is OFF and the fan does the
# work. Safety cycle: if even the fan on high can no longer keep the user comfortable (SET above KNOB + TOL,
# walls taken at air temperature), the aircon goes on at STUDY_AC_C. It goes off again once the room air is
# down to STUDY_AC_C. folder 18 (study_aircon_options.csv) tested this: on the story evening the room only
# reaches 29.6 °C, so the cycle never starts. Switching on earlier (at SET > KNOB) overcools and adds discomfort.
STUDY_AC_C = 26.0        # °C, aircon setpoint when the safety cycle runs (a setting people actually use)
STUDY_AC_ON = [False]    # cycle state, kept between controller updates


def fan_closest(act, t_a, target=KNOB):
    """Fan level (allowed for this activity) that brings SET closest to the target, walls at air temperature."""
    met, clo, _, vmax = ACTIVITIES[act]
    levels = [lv for lv in FAN_LEVELS if lv <= vmax + 1e-9]
    return min(levels, key=lambda lv: abs(set_value(t_a, t_a, lv, met, clo) - target))


def our_system(act, t_m=None, t_a=None):
    """Our system's rule: best setting per activity, the sleep ramp, the wake-up ramp, and a fan step-down.
    Returns (aircon setpoint °C, or None = aircon off, fan m/s).
    Studying / resting: aircon off, fan picked to bring SET closest to the target, with the 26 °C safety cycle.
    Fan step-down (4 Nov 2026): an aircon cannot heat, so if the room air t_a (room sensor, SHT31) is already
    below the setpoint, the fan drops to the speed that brings SET closest to the target. The controller
    has no wall sensor, so it takes the wall (radiant) temperature as equal to the air temperature.
    The step-down applies whenever the room is more than 0.1 °C below the setpoint."""
    if act == "Sitting":
        if t_a is None:
            return None, best[act][1]
        met, clo, _, vmax = ACTIVITIES[act]
        if set_value(t_a, t_a, vmax, met, clo) > KNOB + TOL:
            STUDY_AC_ON[0] = True
        elif t_a <= STUDY_AC_C + 0.1:
            STUDY_AC_ON[0] = False
        return (STUDY_AC_C if STUDY_AC_ON[0] else None), fan_closest(act, t_a)
    STUDY_AC_ON[0] = False
    t, v = best[act]
    met, clo, _, vmax = ACTIVITIES[act]
    target = KNOB
    if act == "Sleeping":
        target = KNOB + min(RAMP_PER_H * max(NOW[0] - SLEEP_START, 0.0), RAMP_CAP)
        wake_start = END - WAKE_MIN / 60
        if NOW[0] >= wake_start:
            if WAKE_AIRCON_OFF:
                return None, v          # aircon off from 5:30 am, fan stays at the sleep speed
            f = min((NOW[0] - wake_start) / (WAKE_MIN / 60), 1.0)
            target = KNOB + RAMP_CAP + f * (WAKE_TOP_SET - KNOB - RAMP_CAP)
        t = min(setpoint_for(met, clo, v, target), SET_MAX)
    if t_a is not None and t_a < t - 0.1:
        levels = [lv for lv in FAN_LEVELS if lv <= vmax + 1e-9]
        v = min(levels, key=lambda lv: abs(set_value(t_a, t_a, lv, met, clo) - target))
    return t, v


# Each strategy maps (activity, wall temperature °C, air temperature °C) ->
# (aircon setpoint °C, fan m/s). Only the last two use the wall temperature
# (the others measure air temperature only).
STRATEGIES = [
    (f"Aircon only, set for the workout ({no_fan['Exercising']:.0f} °C all night)",
     lambda a, tm, ta: (round(no_fan["Exercising"]), V_STILL)),
    ("Aircon only, fixed 25 °C", lambda a, tm, ta: (25.0, V_STILL)),
    ("Aircon only, changes with activity", lambda a, tm, ta: (no_fan[a], V_STILL)),
    ("Our system: activity-aware aircon + fan", lambda a, tm, ta: our_system(a, tm, ta)),
    # wall-temperature sensor removed from the design (3 Nov 2026)
]
CONTROL_STEP = 300  # s, how often the controller recalculates (every 5 minutes)
# Sensor cycle (8 Nov 2026, section 2.3.2): each sensor node is awake 3 min and in deep sleep 10 min, so the controller gets
# ONE room-air temperature per 13-minute cycle and holds it until the next. Run with 'truesensor' to see the true temperature.
SENSOR_CYCLE_S = 0 if "truesensor" in sys.argv[1:] else 13 * 60  # s (0 = controller sees the true temperature at every update)

# Strategy 6 (review follow-up): pre-cool before a scheduled workout. The
# workout time is known from the schedule (team objective 5), so the aircon
# starts PRECOOL_MIN before the user arrives, aiming for the workout setting
# (wall-compensated, fan off while the room is empty). After arrival it behaves
# like "our system + wall-temperature sensor".
# Tested: 30 min leaves all 60 workout minutes too warm (walls still hot),
# 60 min leaves 40, 90 min gives 0 (fully comfortable), 120 min costs more.
PRECOOL_MIN = 90  # min, how early the aircon starts before the workout (ASSUMPTION)


def precool(act, t_m, t_a=None):
    if act == "Away":
        return with_wall_sensor("Exercising", t_m)[0], V_STILL
    return with_wall_sensor(act, t_m, t_a)


PRECOOL_LABEL = f"Our system + wall sensor + pre-cool {PRECOOL_MIN} min"
# Not run any more (3 Nov 2026): the project studies a reactive system only, so pre-cooling is out of scope.
# STRATEGIES.append((PRECOOL_LABEL, precool))


def simulate(strategy, cop_25, per_act=None):
    """Minute-by-minute run. Returns per-minute records and energy (kWh)."""
    t_a = t_m = T_START
    ac_kwh = fan_kwh = 0.0
    ex_kwh = 0.0  # energy during the workout hour
    recs = []
    last_act = None
    t_seen = t_a   # room temperature the controller last received from the sensor nodes
    start = START - PRECOOL_MIN / 60 if strategy is precool else START  # pre-cool starts earlier
    n = int(round((END - start) * 3600 / DT))
    for i in range(n):
        hour = start + i * DT / 3600
        NOW[0] = hour            # lets time-based rules (sleep ramp, wake-up ramp) see the clock
        act = activity_at(hour)
        if SENSOR_CYCLE_S == 0 or (i * DT) % SENSOR_CYCLE_S == 0:   # a new sensor reading arrives
            t_seen = t_a
        if i % int(CONTROL_STEP / DT) == 0 or act != last_act:  # controller update (uses the held reading)
            t_set, v = strategy(act, t_m, t_seen)
            last_act = act
        sens, lat = person_heat(act)
        to = t_out(hour % 24)
        q_wall = G_IN * (t_m - t_a)
        q_gain = q_wall + Q_OTHER_SENS + sens + fan_watts(v)
        q_ac = 0.0 if t_set is None else min(max(C_A * (t_a - t_set) / DT + q_gain, 0.0), Q_MAX)   # None = off
        step = 0.0
        if q_ac > 0:
            step = (q_ac + Q_OTHER_LAT + lat) / (cop_25 * carnot(t_a, to) / carnot(25.0, to)) * DT / 3.6e6
        ac_kwh += step
        fan_kwh += fan_watts(v) * DT / 3.6e6
        if act == "Exercising":
            ex_kwh += step + fan_watts(v) * DT / 3.6e6
        if per_act is not None and act in ACTIVITIES:   # electricity used while doing each activity
            per_act[act] = per_act.get(act, 0.0) + step + fan_watts(v) * DT / 3.6e6
        if i % 2 == 0:  # record once a minute
            recs.append((hour, act, t_a, t_m, t_set, v))
        t_a += (q_gain - q_ac) * DT / C_A
        t_m += (G_OUT * (to + MASS_OFFSET - t_m) - q_wall) * DT / C_M
    return recs, ac_kwh, fan_kwh, ex_kwh


def comfort_minutes(recs):
    """Minutes too warm / too cold per activity, from SET with the real wall temperature.
    Only minutes when the user is in the room count."""
    recs = [r for r in recs if r[1] in ACTIVITIES and r[0] < END - WAKE_MIN / 60]   # wake-up window not counted
    hours = [r[0] for r in recs]
    acts = [r[1] for r in recs]
    sets = []
    for act in ACTIVITIES:
        idx = [i for i, a in enumerate(acts) if a == act]
        met, clo, _, _ = ACTIVITIES[act]
        s = set_tmp(tdb=[recs[i][2] for i in idx], tr=[recs[i][3] for i in idx], v=[recs[i][5] for i in idx],
                    rh=RH, met=met, clo=clo, position="sitting", limit_inputs=False, round_output=False).set
        sets += list(zip(idx, s))
    sets = [s for _, s in sorted(sets)]
    out = {}
    for act in ACTIVITIES:
        vals = [s for s, a in zip(sets, acts) if a == act]
        # (minutes too warm, minutes too cold, average SET °C during this activity)
        out[act] = (sum(s > KNOB + TOL for s in vals), sum(s < KNOB - TOL for s in vals), float(np.mean(vals)))
    return hours, sets, out


print("Part 2: evening in a common bedroom (8 pm desk, 8:30 pm workout, 10 pm sleep, until 6 am)")
print(f"Room and walls start at {T_START} °C\n")
results = {}
for label, strat in STRATEGIES:
    recs, _, _, _ = simulate(strat, 3.5)
    hours, sets, mins = comfort_minutes(recs)
    results[label] = {"recs": recs, "sets": sets, "mins": mins}
    print(label)
    print("  " + " · ".join(f"{a}: avg SET {m:.1f} °C, {w} min too warm, {c} min too cold"
                            for a, (w, c, m) in mins.items()))
    for sys_label, cop in SYSTEMS:
        pa = {}
        _, ac, fan, ex = simulate(strat, cop, pa)
        tot = ac + fan
        results[label][sys_label] = (tot, ex)
        results[label][sys_label + " per activity"] = pa
        print(f"  {sys_label}: {tot:.2f} kWh/night (workout hour {ex:.2f}) = S${tot * TARIFF * DAYS_PER_YEAR:.0f}/yr")
        rows.append({"part": "2 evening", "activity": label, "aircon": sys_label,
                     "kwh_night": round(tot, 3), "kwh_workout_hour": round(ex, 3),
                     "sgd_year": round(tot * TARIFF * DAYS_PER_YEAR, 2),
                     **{f"{a.lower()}_avg_set_C": round(m, 2) for a, (w, c, m) in mins.items()},
                     **{f"{a.lower()}_min_too_warm": w for a, (w, c, m) in mins.items()},
                     **{f"{a.lower()}_min_too_cold": c for a, (w, c, m) in mins.items()}})
    print()

for sys_label, _ in SYSTEMS:
    for ours_label, _ in STRATEGIES[3:]:
        ours = results[ours_label][sys_label][0]
        for label, _ in STRATEGIES[:3]:
            other = results[label][sys_label][0]
            print(f"{sys_label}: '{ours_label}' vs '{label}': saves {other - ours:.2f} kWh/night "
                  f"= S${(other - ours) * TARIFF * DAYS_PER_YEAR:.0f}/yr ({(other - ours) / other:.0%})")

# Part 3: saving for each activity (kWh used while doing it), our system against two aircon-only baselines
OURS, FIXED, MATCHED = STRATEGIES[3][0], STRATEGIES[1][0], STRATEGIES[2][0]
print("\nPart 3: saving for each activity (our system vs aircon alone)")
for sys_label, _ in SYSTEMS:
    for act in ACTIVITIES:
        o = results[OURS][sys_label + " per activity"][act]
        fx = results[FIXED][sys_label + " per activity"][act]
        mt = results[MATCHED][sys_label + " per activity"][act]
        print(f"  {sys_label}, {act}: ours {o:.3f} kWh, fixed 25 °C {fx:.3f} ({1 - o / fx:+.0%} saving), "
              f"aircon set for the same comfort {mt:.3f} ({1 - o / mt:+.0%} saving)")
        rows.append({"part": "3 per activity", "activity": act, "aircon": sys_label, "kwh_night": round(o, 3),
                     "kwh_fixed25": round(fx, 3), "kwh_matched_aircon_only": round(mt, 3),
                     "saving_vs_fixed25_pct": round(100 * (1 - o / fx), 1),
                     "saving_vs_matched_pct": round(100 * (1 - o / mt), 1)})

fields = []
for r in rows:
    fields += [k for k in r if k not in fields]
with open(f"activity_results{SUFFIX}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved activity_results{SUFFIX}.csv")

# =============================================================================
# 5. Charts
# =============================================================================
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACT_COLORS = {"Sleeping": "#5b4a9e", "Sitting": "#2a78d6", "Exercising": "#c2410c",
              "Light exercise (2.5 met)": "#eb9a6a", "Hard exercise (4.5 met)": "#7a2a08"}
STRAT_COLORS = ["#8f8e88", "#c9c8c2", "#86b6ef", "#2f6b3a", "#c08a1e", "#0f766e"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)


# Chart 1: setpoint vs fan speed for each activity
speeds = np.linspace(0.1, 1.5, 29)
fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
fig.patch.set_facecolor(SURFACE)
curves = [(a, m, c) for a, (m, c, _, _) in ACTIVITIES.items()] + [(n, m, 0.36) for n, m in EXTRA_EXERCISE]
for name, met, clo in curves:
    ts = [setpoint_for(met, clo, v) for v in speeds]
    main = name in ACTIVITIES
    ax.plot(speeds, ts, color=ACT_COLORS[name], lw=2.4 if main else 1.4, ls="-" if main else "--",
            label=f"{name} ({met} met, {clo} clo)" if main else name)
    if main:
        t, v = best[name]
        ax.scatter([v], [t], s=60, color=ACT_COLORS[name], edgecolor=SURFACE, linewidth=2, zorder=3)
        at_edge = v >= 1.4  # keep labels near the right edge inside the chart
        ax.annotate(f"chosen: {t:.1f} °C, fan {v} m/s", (v, t), textcoords="offset points",
                    xytext=(-8, 10) if at_edge else (8, -14), ha="right" if at_edge else "left",
                    fontsize=8.5, color=ACT_COLORS[name])
ax.axvline(0.5, color=ACT_COLORS["Sleeping"], lw=1, ls=":")
ax.text(0.52, 15.4, "sleeping fan cap", fontsize=8, color=ACT_COLORS["Sleeping"])
style(ax)
ax.set_xlim(0.1, 1.5)
ax.set_ylim(15, 31)
ax.set_xlabel("Fan air speed at the user (m/s)", color=INK_2)
ax.set_ylabel(f"Aircon setpoint for SET = {KNOB} °C", color=INK_2)
ax.set_title("Exercising needs a much colder room, and the fan only helps it above ~1 m/s",
             loc="left", fontsize=12, color=INK, pad=22)
ax.text(0, 1.03, f"Knob = feels like {KNOB} °C for every activity · {RH}% RH · walls at air temperature · "
        "remote range 16-30 °C · dots = lowest-power choice", transform=ax.transAxes, fontsize=8.5, color=INK_2)
ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout()
fig.savefig(f"activity_setpoints{SUFFIX}.png")
print(f"Saved activity_setpoints{SUFFIX}.png")

# Chart 2: the evening
fig = plt.figure(figsize=(13, 8.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.42, wspace=0.25)
ax_t = fig.add_subplot(gs[0, :])
ax_e = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

# Top: room air temperature through the evening, with activity bands
for (label, _), color in zip(STRATEGIES, STRAT_COLORS):
    recs = results[label]["recs"]
    ax_t.plot([r[0] for r in recs], [r[2] for r in recs], color=color, lw=2.2 if "Our" in label else 1.6,
              label=label)
for a, (h0, h1) in {"Sitting": (18.0, 20.5), "Exercising": (20.5, 22.0), "Sleeping": (22.0, 30.0)}.items():
    ax_t.axvspan(h0, h1, color=ACT_COLORS[a], alpha=0.07, lw=0)
    ax_t.text((h0 + h1) / 2, 31.3, a, ha="center", fontsize=9, color=ACT_COLORS[a])
style(ax_t)
ticks = [18.0 + k for k in range(13)]
ax_t.set_xticks(ticks)
ax_t.set_xticklabels([f"{(int(h % 24) - 1) % 12 + 1}:{int(round((h % 1) * 60)):02d} {'am' if h % 24 < 12 else 'pm'}"
                      for h in ticks])
ax_t.set_xlim(START, 30)
ax_t.set_ylim(19, 32)
ax_t.set_ylabel("Room air temperature (°C)", color=INK_2)
ax_t.set_title(f"Room air through the evening (common bedroom, starting at {T_START} °C)", loc="left", fontsize=10.5,
               color=INK)
ax_t.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK, ncol=2)

# Bottom left: energy per year
bw = 0.13
for k, ((label, _), color) in enumerate(zip(STRATEGIES, STRAT_COLORS)):
    vals = [results[label][s][0] * TARIFF * DAYS_PER_YEAR for s, _ in SYSTEMS]
    xs = [j + (k - 2.5) * bw for j in range(len(SYSTEMS))]
    ax_e.bar(xs, vals, bw * 0.92, color=color)
    for x_, v_ in zip(xs, vals):
        ax_e.text(x_, v_ + 3, f"S${v_:.0f}", ha="center", fontsize=7.8, color=INK)
style(ax_e)
ax_e.set_xticks(range(len(SYSTEMS)))
ax_e.set_xticklabels([s.replace(" (", "\n(") for s, _ in SYSTEMS], color=INK, fontsize=8.5)
ax_e.set_ylabel("S$ per year (this routine every day)", color=INK_2)
ax_e.set_title("Electricity, 8 pm to 6 am", loc="left", fontsize=10.5, color=INK)

# Bottom right: discomfort minutes (too warm / too cold), per strategy
labels_short = ["Set for workout", "Fixed 25 °C", "Activity-aware\naircon only", "Our system"]
warm = [sum(m[0] for m in results[l]["mins"].values()) for l, _ in STRATEGIES]
cold = [sum(m[1] for m in results[l]["mins"].values()) for l, _ in STRATEGIES]
y = np.arange(len(STRATEGIES))
ax_c.barh(y, warm, 0.6, color="#c2410c", label="too warm")
ax_c.barh(y, cold, 0.6, left=warm, color="#2a78d6", label="too cold")
for i, (w_, c_) in enumerate(zip(warm, cold)):
    ax_c.text(w_ + c_ + 5, i, f"{w_ + c_} min", va="center", fontsize=8, color=INK)
ax_c.set_yticks(y)
ax_c.set_yticklabels(labels_short, color=INK, fontsize=8.5)
ax_c.invert_yaxis()
style(ax_c)
ax_c.grid(axis="x", color=GRID, lw=0.8)
ax_c.grid(axis="y", visible=False)
ax_c.set_xlabel(f"Minutes outside the comfort band (SET {KNOB} ± {TOL} °C)", color=INK_2)
ax_c.set_title("Comfort over the evening (warm walls included)", loc="left", fontsize=10.5, color=INK)
ax_c.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
ax_c.set_xlim(0, max(w + c for w, c in zip(warm, cold)) * 1.25)

fig.suptitle("Activity-aware control: colder only during the workout, fan instead of aircon where it works",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, 0.012, f"Knob = feels like {KNOB} °C · workout 3.5 met (8:30-10 pm), desk 1.0 met, sleep 0.7 met · "
         + ("existing DC ceiling fan 1.8/8.7/25 W at 0.5/0.85/1.21 m/s" if FAN == "ceiling" else
            "DC standing fan 4.5/15/43.5 W at 0.5/1.0/1.5 m/s") + ", sleep fan up to 0.85 m/s (noise not modelled) · two-node room model "
         f"(folder 11) · S${TARIFF}/kWh", fontsize=7.8, color=INK_2)
fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08)
fig.savefig(f"activity_evening{SUFFIX}.png")
print(f"Saved activity_evening{SUFFIX}.png")
