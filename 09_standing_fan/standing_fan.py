"""Standing (pedestal) fan + aircon: oscillating vs pointed at the user.

Question: with a typical standing fan, how much warmer can the aircon be set
for the same comfort as 25 °C aircon alone, and how much electricity and
money does that save for an older (COP 3.5) and a new 5-tick (COP 5.5)
aircon?

Method (same as the rest of the project)
----------------------------------------
- Comfort is measured with SET (Standard Effective Temperature), the model
  ASHRAE 55 uses for air faster than 0.2 m/s ("elevated air speed" method).
- Equal comfort = equal SET: the setpoint with the fan is the air temperature
  that gives the same SET as 25 °C with still air (0.1 m/s).
- As a check, the ASHRAE 55 PMV (comfort vote, -3 cold to +3 hot) and PPD
  (% of people dissatisfied) are printed for every case. Equal SET should give
  about the same PMV as the 25 °C baseline.
- Cost: the household cooling model from 03_household_savings, run once per
  aircon efficiency, with the standing fan's power added. Because a standing
  fan uses much more power than a DC ceiling fan, two fan types are compared:
  a typical AC-motor fan and an energy-saving DC-motor fan.

Part 2 adds a ceiling fan directly above the user, alone and together with
the standing fan (oscillating or pointed), to see whether running both fans
saves more than either one.

Outputs (saved next to this script):
    standing_fan.png            standing fan only: setpoints and monthly cost
    ceiling_plus_standing.png   all cases incl. the ceiling fan: yearly saving
    standing_fan_results.csv    every number behind both charts
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # save outputs next to this script

import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq  # finds the input value that makes a function equal zero

from pythermalcomfort.models import pmv_ppd_ashrae, set_tmp

plt.rcParams["text.parse_math"] = False  # treat "$" as plain text, not a maths formula

# =============================================================================
# 1. The standing fan (ASSUMPTIONS, stated)
# =============================================================================
# A typical 16-inch (40 cm) pedestal fan on MEDIUM, about 1.5 m from the user.
# Fan power on medium (W). Oscillating adds ~1-2 W for the swing motor, ignored.
FAN_TYPES = [("AC motor", 45),  # typical 16" AC-motor fan (box ratings 40-60 W at high)
             ("DC motor", 15)]  # energy-saving DC-motor fan (box ratings ~25-35 W at high)
V_STILL = 0.1  # m/s, room air speed with no fan (still air)
V_JET = 1.0  # m/s, air speed at the user when the fan points straight at them
#             (1.5 m away, medium). ASSUMPTION; on high it can be 1.5-2 m/s.
V_EDGE = 0.15  # m/s, air speed at the user while an oscillating fan points away
#               (some mixed room air still reaches them)

# Oscillating: the head sweeps about 90° and the jet is about 30° wide at the
# user, so the jet is on the user about 1/3 of the time.
IN_JET_SHARE = 30 / 90  # share of time the jet hits the user (ASSUMPTION)


def steady_equivalent(v_high, v_low, share_high):
    """Steady air speed (m/s) that removes the same heat as air that switches
    between v_high (a share_high fraction of the time) and v_low.
    Heat removed by moving air grows with v^0.53 (the convective heat transfer
    coefficient h_c = 8.6 v^0.53 used in SET), so we average v^0.53, not v.
    ASSUMPTION: this ignores any extra cooling people feel from the gusts
    themselves; lab studies suggest gusty air can feel a little cooler than
    steady air, so this is on the cautious side."""
    mean = share_high * v_high ** 0.53 + (1 - share_high) * v_low ** 0.53
    return mean ** (1 / 0.53)


V_OSC = steady_equivalent(V_JET, V_EDGE, IN_JET_SHARE)  # ≈ 0.35 m/s

# Air-speed cases: (label, air speed at the user in m/s)
AIR_CASES = [("Aircon only, 25 °C", V_STILL), ("Oscillating", V_OSC), ("Pointed at user", V_JET)]
# Scenarios: (label, air case, fan power W): no fan, then each fan mode with each fan type
SCENARIOS = [("Aircon only, 25 °C", AIR_CASES[0][0], 0)] + [
    (f"{air}, {ft} fan", air, fw) for air, _ in AIR_CASES[1:] for ft, fw in FAN_TYPES]

# -----------------------------------------------------------------------------
# 1b. Adding a ceiling fan directly above the user
# -----------------------------------------------------------------------------
# Ceiling fan: a 52-inch DC fan on high, the user directly under the blades
# (same values as 03_household_savings and 07_fan_distance_and_comfort_target).
V_CEILING = 0.8  # m/s, air speed at the user from the ceiling fan alone
CEILING_W = 15  # W, DC-motor ceiling fan


def combine(v_down, v_side):
    """Air speed (m/s) when the ceiling fan's downward air meets the standing
    fan's sideways air. ASSUMPTION: the two streams are at right angles, so
    their speeds add like the sides of a right-angled triangle:
    v = sqrt(v_down^2 + v_side^2). In a real room the jets also mix and can
    partly cancel, so treat this as a best case. The background still air
    (V_STILL) is already inside each fan's speed, so it is not added again."""
    return math.hypot(v_down, v_side)


# With the ceiling fan on, the oscillating fan swings between "jet on the user"
# (both fans) and "jet away" (ceiling fan + a little room air).
V_CEIL_OSC = steady_equivalent(combine(V_CEILING, V_JET), combine(V_CEILING, V_EDGE), IN_JET_SHARE)
V_CEIL_PNT = combine(V_CEILING, V_JET)  # ≈ 1.28 m/s

CEILING_AIR_CASES = [("Ceiling fan only", V_CEILING),
                     ("Ceiling + oscillating", V_CEIL_OSC),
                     ("Ceiling + pointed", V_CEIL_PNT)]
# Fan power = ceiling fan + standing fan (if any)
CEILING_SCENARIOS = [("Ceiling fan only", "Ceiling fan only", CEILING_W)] + [
    (f"{air}, {ft} fan", air, CEILING_W + fw) for air, _ in CEILING_AIR_CASES[1:] for ft, fw in FAN_TYPES]

ALL_AIR_CASES = AIR_CASES + CEILING_AIR_CASES
ALL_SCENARIOS = SCENARIOS + CEILING_SCENARIOS
# NOTE: above about 1 m/s, loose paper moves and some people find the draught
# annoying. ASHRAE 55 sets no air speed limit when the user controls the fans.

# =============================================================================
# 2. Comfort (SET and ASHRAE 55 PMV)
# =============================================================================
RH = 60  # %, relative humidity indoors with the aircon on
BASE_T = 25.0  # °C, the baseline: aircon alone at 25 °C (Go 25 guidance)
MAX_SETPOINT = 30  # °C, the highest setting on most aircon remotes
# Situations: (name, met, clo)
#   met = metabolic rate: 1.0 sitting relaxed, 0.7 asleep
#   clo = insulation: 0.4 T-shirt and shorts; 0.9 in bed without covers
SITTING = ("Sitting (1.0 met, 0.4 clo)", 1.0, 0.4)
IN_BED = ("In bed, no covers (0.7 met, 0.9 clo)", 0.7, 0.9)
# NOTE: sleeping (0.7 met) is outside ASHRAE 55-2023's range (1.0-4.0 met); bed
# results are indicative. A fan pointed at a sleeper all night may also cause
# dry eyes or draught complaints, so oscillating is the likelier choice in bed.


def set_value(tdb, v, met, clo):
    """SET (°C) at air temperature tdb and air speed v. Walls at air temperature;
    sitting posture (the SET model has no lying option)."""
    return set_tmp(tdb=tdb, tr=tdb, v=v, rh=RH, met=met, clo=clo, position="sitting",
                   limit_inputs=False, round_output=False).set


def setpoint_equal_set(v, met, clo):
    """Aircon setpoint (°C) at air speed v that feels the same as BASE_T with still air."""
    if v <= V_STILL:
        return BASE_T
    target = set_value(BASE_T, V_STILL, met, clo)
    return brentq(lambda t: set_value(t, v, met, clo) - target, BASE_T - 1, 40)


def pmv_ppd(tdb, v, met, clo):
    """ASHRAE 55 PMV and PPD (%); airspeed_control=True because the user controls the fan."""
    r = pmv_ppd_ashrae(tdb=tdb, tr=tdb, vr=v, rh=RH, met=met, clo=clo, airspeed_control=True,
                       limit_inputs=False, round_output=False)
    return r.pmv, r.ppd


# =============================================================================
# 3. Household electricity model (same as 03_household_savings)
# =============================================================================
TARIFF = 0.3116  # S$ per kWh, SP Group Oct-Dec 2026 incl. 9% GST
DAYS = 30  # days per month
MASS_OFFSET = 2.0  # °C, warm concrete walls give off heat as if it were 2 °C hotter outside
# COP = cooling delivered ÷ electricity used, at 25 °C:
#   3.5 = older system (estimate); 5.5 = new 5-tick multi-split (NEA minimum)
SYSTEMS = [("Older aircon (COP 3.5)", 3.5), ("New 5-tick aircon (COP 5.5)", 5.5)]
# Rooms: hours = clock hours the aircon runs (range(22, 30) = 10 pm to 6 am);
# UA = heat leaking in per °C of indoor-outdoor difference (W/K);
# q_int = heat made inside (W); q_lat = moisture the aircon removes, as heat (W);
# who = which comfort situation applies. One standing fan per room.
ROOMS = [
    {"name": "Master bedroom", "hours": range(22, 30), "UA": 60, "q_int": 250, "q_lat": 250, "who": IN_BED},
    {"name": "Second bedroom", "hours": range(22, 30), "UA": 60, "q_int": 250, "q_lat": 250, "who": IN_BED},
    {"name": "Living room", "hours": range(19, 23), "UA": 100, "q_int": 400, "q_lat": 300, "who": SITTING},
]
FAN_HOURS = sum(len(r["hours"]) for r in ROOMS) * DAYS  # fan-hours per month, all rooms together


def t_out(hour):
    """Outdoor temperature (°C) at a given hour: 29 °C average, ±3 °C, hottest at 2 pm."""
    return 29.0 + 3.0 * math.cos(2 * math.pi * (hour - 14) / 24)


def carnot(t_set, t_o):
    """Ideal-aircon efficiency, used only to scale the real COP with setpoint.
    Indoor coil 12 °C below the room, outdoor coil 12 °C above outside (kelvin)."""
    t_evap, t_cond = t_set - 12 + 273.15, t_o + 12 + 273.15
    return t_evap / (t_cond - t_evap)


def room_kwh_per_day(room, t_set, cop_25, fan_w):
    """Electricity (kWh) one room uses per day: aircon at t_set (°C) plus fan (W)."""
    kwh = 0.0
    for h in room["hours"]:
        to = t_out(h % 24)
        load = room["UA"] * (to + MASS_OFFSET - t_set) + room["q_int"] + room["q_lat"]  # cooling load (W)
        cop = cop_25 * carnot(t_set, to) / carnot(25.0, to)
        kwh += max(load, 0) / cop / 1000 + fan_w / 1000
    return kwh


# =============================================================================
# 4. Run the scenarios
# =============================================================================
print("Standing fan + aircon, same comfort (equal SET) as aircon alone at 25 °C")
print(f"Fan: 16-inch on medium, AC {FAN_TYPES[0][1]} W or DC {FAN_TYPES[1][1]} W · pointed: {V_JET} m/s · "
      f"oscillating: jet on user {IN_JET_SHARE:.0%} of the time -> steady equivalent {V_OSC:.2f} m/s")
print(f"Ceiling fan overhead: {V_CEILING} m/s, {CEILING_W} W · with oscillating fan -> {V_CEIL_OSC:.2f} m/s · "
      f"with pointed fan -> {V_CEIL_PNT:.2f} m/s\n")

setpoints = {}  # (air case, situation name) -> setpoint °C
for label, v in ALL_AIR_CASES:
    print(label)
    for who in (SITTING, IN_BED):
        t_exact = setpoint_equal_set(v, who[1], who[2])
        t = min(t_exact, MAX_SETPOINT)
        setpoints[(label, who[0])] = t
        p, d = pmv_ppd(t, v, who[1], who[2])
        cap = " (capped at remote max)" if t_exact > MAX_SETPOINT else ""
        print(f"  {who[0]:<38} setpoint {t:4.1f} °C{cap} · ASHRAE 55 PMV {p:+.2f}, PPD {d:3.0f}%")

rows, costs = [], {}  # costs[(system, scenario)] = S$ per month
print()
for sys_label, cop in SYSTEMS:
    print(sys_label)
    base_kwh = None
    for label, air, fan_w in ALL_SCENARIOS:
        v = dict(ALL_AIR_CASES)[air]
        kwh = sum(room_kwh_per_day(r, setpoints[(air, r["who"][0])], cop, fan_w) for r in ROOMS) * DAYS
        if base_kwh is None:
            base_kwh = kwh  # first scenario = aircon only
        cost = kwh * TARIFF
        costs[(sys_label, label)] = cost
        save_kwh = base_kwh - kwh
        print(f"  {label:<34} {kwh:6.1f} kWh/month  S${cost:6.2f}/month  "
              f"saves {save_kwh:5.1f} kWh ({save_kwh / base_kwh:4.0%}) = S${save_kwh * TARIFF * 12:6.2f}/year")
        breakeven = None
        if fan_w:
            # Break-even fan power: the wattage at which the fan's own electricity
            # uses up the whole aircon saving. Each watt less saves FAN_HOURS Wh a month.
            breakeven = fan_w + save_kwh * 1000 / FAN_HOURS
            print(f"  {'':<34} break-even total fan power {breakeven:4.0f} W (fans using less than this in total save money)")
        rows.append({"aircon": sys_label, "cop_at_25C": cop, "scenario": label,
                     "air_speed_m_s": round(v, 3), "fan_w": fan_w,
                     "living_setpoint_C": round(setpoints[(air, SITTING[0])], 2),
                     "bedroom_setpoint_C": round(setpoints[(air, IN_BED[0])], 2),
                     "kwh_per_month": round(kwh, 1), "cost_sgd_per_month": round(cost, 2),
                     "saving_kwh_per_month": round(save_kwh, 1),
                     "saving_pct": round(100 * save_kwh / base_kwh, 1),
                     "saving_sgd_per_year": round(save_kwh * TARIFF * 12, 2),
                     "breakeven_fan_w": round(breakeven, 1) if breakeven else ""})

with open("standing_fan_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nSaved standing_fan_results.csv")

# =============================================================================
# 5. Chart: setpoints (left) and monthly cost for both aircon types (right)
# =============================================================================
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AIR_COLORS = ["#8f8e88", "#86b6ef", "#2a78d6"]  # aircon only, oscillating, pointed
# Cost bars: grey = aircon only; light blue = oscillating; dark blue = pointed.
# AC-motor fan bars are hatched, DC-motor fan bars are solid.
SC_COLORS = ["#8f8e88", "#86b6ef", "#86b6ef", "#2a78d6", "#2a78d6"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=150, gridspec_kw={"width_ratios": [1, 1.5]})
fig.patch.set_facecolor(SURFACE)

# Left: setpoint for the same comfort, sitting and in bed
x = np.arange(2)  # 0 = sitting, 1 = in bed
w = 0.26
for i, (label, v) in enumerate(AIR_CASES):
    vals = [setpoints[(label, SITTING[0])], setpoints[(label, IN_BED[0])]]
    bars = ax1.bar(x + (i - 1) * w, vals, w * 0.92, color=AIR_COLORS[i], label=f"{label} ({v:.2f} m/s)")
    for b, val in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, val + 0.08, f"{val:.1f}", ha="center", fontsize=8.5, color=INK)
style(ax1)
ax1.set_xticks(x)
ax1.set_xticklabels(["Sitting\n(living room)", "In bed, no covers*\n(bedrooms)"], color=INK)
ax1.set_ylim(22, 31)
ax1.set_ylabel("Aircon setpoint for the same comfort (°C)", color=INK_2)
ax1.set_title("Setpoint that feels like 25 °C aircon alone", loc="left", fontsize=11, color=INK)
ax1.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)

# Right: monthly cost, grouped by aircon type
x2 = np.arange(len(SYSTEMS))
w2 = 0.17
for i, (label, _, _) in enumerate(SCENARIOS):
    vals = [costs[(s, label)] for s, _ in SYSTEMS]
    is_ac = "AC motor" in label
    bars = ax2.bar(x2 + (i - 2) * w2, vals, w2 * 0.9, color=SC_COLORS[i], label=label,
                   hatch="///" if is_ac else None, edgecolor=SURFACE if is_ac else "none", linewidth=0)
    for b, val, (s, _) in zip(bars, vals, SYSTEMS):
        base = costs[(s, SCENARIOS[0][0])]
        txt = f"S${val:.0f}" if i == 0 else f"S${val:.0f}\n{(val - base) * 12:+.0f}/yr"
        ax2.text(b.get_x() + b.get_width() / 2, val + 0.6, txt, ha="center", fontsize=7.5, color=INK)
style(ax2)
ax2.set_xticks(x2)
ax2.set_xticklabels([s for s, _ in SYSTEMS], color=INK)
ax2.set_ylim(0, max(costs.values()) * 1.45)
ax2.set_ylabel("Aircon + fan cost per month (S$)", color=INK_2)
ax2.set_title("Monthly cost, 4-room HDB (S$/yr = change vs aircon only; minus = saving)",
              loc="left", fontsize=11, color=INK)
ax2.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK, ncol=1)

fig.suptitle("Fan power decides it: a 15 W DC standing fan saves in every case; "
             "a 45 W AC fan can cost more than it saves", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, 0.015, f"Equal SET to 25 °C at 0.1 m/s · {RH}% RH · 16\" fan on medium, AC {FAN_TYPES[0][1]} W (hatched) "
         f"or DC {FAN_TYPES[1][1]} W (solid) · oscillating = jet on user {IN_JET_SHARE:.0%} of the time · "
         f"S${TARIFF}/kWh · * sleeping is outside ASHRAE 55's range (indicative)", fontsize=7.8, color=INK_2)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))
fig.savefig("standing_fan.png")
print("Saved standing_fan.png")

# =============================================================================
# 6. Chart 2: ceiling fan, standing fan, and both together (yearly saving)
# =============================================================================
# One panel per aircon type. Each bar = yearly saving vs 25 °C aircon only
# (negative = costs more). Bar labels give the setpoints (sitting / in bed).
PLOT_SCENARIOS = ALL_SCENARIOS[1:]  # everything except the aircon-only baseline
GROUP_COLORS = {"Oscillating": "#86b6ef", "Pointed at user": "#2a78d6", "Ceiling fan only": "#5e8f6a",
                "Ceiling + oscillating": "#9cc5a5", "Ceiling + pointed": "#2f6b3a"}

fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), dpi=150, sharey=True)
fig.patch.set_facecolor(SURFACE)
names = [s[0] for s in PLOT_SCENARIOS][::-1]  # top-to-bottom in list order
for ax, (sys_label, _) in zip(axes, SYSTEMS):
    base = costs[(sys_label, ALL_SCENARIOS[0][0])]
    savings = [(base - costs[(sys_label, s[0])]) * 12 for s in PLOT_SCENARIOS][::-1]
    for i, (s, val) in enumerate(zip(PLOT_SCENARIOS[::-1], savings)):
        label, air, fan_w = s
        is_ac = "AC motor" in label
        ax.barh(i, val, 0.68, color=GROUP_COLORS[air], hatch="///" if is_ac else None,
                edgecolor=SURFACE if is_ac else "none", linewidth=0)
        sp = f"{setpoints[(air, SITTING[0])]:.1f} / {setpoints[(air, IN_BED[0])]:.1f} °C"
        txt = f"{val:+.0f}  ({sp}, {fan_w} W)"
        # labels sit right of the bar end, or right of zero for negative bars (clear of the row names)
        ax.text(max(val, 0) + 3, i, txt, va="center", ha="left", fontsize=7.8, color=INK)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_xlim(-60, 215)
    ax.set_title(sys_label, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Saving per year vs 25 °C aircon alone (S$; minus = costs more)", color=INK_2)
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names, color=INK, fontsize=8.5)
fig.suptitle("With a ceiling fan overhead, adding a standing fan lowers the saving: +0.2-0.7 °C does not pay for a second fan",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, 0.015, f"Same comfort as 25 °C aircon alone (equal SET) · ceiling fan overhead {V_CEILING} m/s, "
         f"{CEILING_W} W DC · standing fan AC {FAN_TYPES[0][1]} W (hatched) or DC {FAN_TYPES[1][1]} W · "
         "combined air speed assumes the streams meet at right angles (best case)\n"
         "Bar labels: S$/yr  (setpoint sitting / in bed, total fan power) · * sleeping is outside ASHRAE 55's range "
         "(indicative)", fontsize=7.8, color=INK_2)
fig.tight_layout(rect=(0, 0.055, 1, 0.95))
fig.savefig("ceiling_plus_standing.png")
print("Saved ceiling_plus_standing.png")
