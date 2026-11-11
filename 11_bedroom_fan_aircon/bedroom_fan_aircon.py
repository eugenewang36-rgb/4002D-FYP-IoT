"""HDB bedroom with furniture: ceiling fan + aircon at equal SET comfort, and
the net electricity saving.

Scenario (the user's assumptions)
---------------------------------
- A typical HDB bedroom has a wardrobe and built-in features (a study desk
  or dressing table with shelves) that take up floor space.
- The ceiling fan is placed centrally over the bed, so the user feels more
  than 1 m/s.
- The aircon runs at night while the user sleeps. With the fan on, the aircon
  can be set warmer for the same comfort (equal SET) as aircon alone.

How it works
------------
1. Air speed felt by the user, v_user (m/s): the Raftery et al. (2019) model
   from folder 10, applied to the OPEN floor area left after the wardrobe and
   built-ins. ASSUMPTION: furniture against the walls shrinks the space the
   fan has to stir, like a smaller room. v_user = the model's AVERAGE speed
   over that open area (the user is directly under the fan, where it is
   faster still, so this is on the cautious side). The script checks that
   v_user > 1 m/s, as assumed.
2. Comfort: SET (Standard Effective Temperature). Fan setpoint = air
   temperature at v_user with the same SET as the baseline setpoint at still
   air (0.1 m/s). ASHRAE 55 PMV/PPD is printed as a check.
3. Electricity: the bedroom cooling model from 03_household_savings, 10 pm to
   6 am, for an older (COP 3.5) and a new 5-tick (COP 5.5) aircon.
4. Net saving = aircon electricity saved − fan electricity used.

Part 2 (sections 8-10): COOL-DOWN when the user walks into a hot room.
   A minute-by-minute model of the room air and the warm walls shows how soon
   the user feels comfortable, with and without the fan, and the energy used
   from switch-on to 6 am. See section 8 for the method and assumptions.

Part 3 (section 11): REVIEW CHECKS added after review feedback
   reviews: fan speed below high, wall heat-storage depth, the team's actual
   hardware (DC standing fan at the 0.5 m/s sleep cap), humidity 60-80 % and
   posture (sitting vs standing in the SET model).

Outputs (saved next to this script):
    bedroom_plans.png          floor plans of the two bedrooms with the fan
    bedroom_net_savings.png    setpoints and net savings
    bedroom_results.csv        every number behind the charts
    bedroom_cooldown.png       cool-down: temperatures, comfort and energy
    cooldown_results.csv       every number behind the cool-down chart
    review_checks.csv          every number from the part 3 checks
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # save outputs next to this script

import csv
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import brentq  # finds the input value that makes a function equal zero

from pythermalcomfort.models import pmv_ppd_ashrae, set_tmp

plt.rcParams["text.parse_math"] = False  # treat "$" as plain text, not a maths formula

# =============================================================================
# 1. Rooms and furniture (ASSUMPTIONS from typical HDB floor plans)
# =============================================================================
# Each item: (label, x, y, width, depth) in metres, measured from the room's
# bottom-left corner. The bed is shown on the plan but does NOT count as lost
# floor space: the fan's air flows over it and the user lies on it.
ROOMS = [
    {"name": "Common bedroom", "L": 3.3, "W": 3.0,
     "built_in": [("Wardrobe", 0.0, 2.4, 2.0, 0.6),  # 2.0 m wide, 0.6 m deep, along the top wall
                  ("Study desk + shelves", 2.8, 0.3, 0.5, 1.2)],  # built-in along the right wall
     "bed": ("Super-single bed", 0.5, 0.25, 1.07, 1.9)},
    {"name": "Master bedroom", "L": 4.0, "W": 3.4,
     "built_in": [("Wardrobe", 0.0, 2.8, 2.4, 0.6),
                  ("Dressing table + shelves", 3.5, 0.3, 0.5, 1.2)],
     "bed": ("Queen bed", 1.0, 0.3, 1.52, 1.9)},
]

# =============================================================================
# 2. Ceiling fan and air speed (model from 10_room_size_vs_ceiling_fan)
# =============================================================================
CFM_TO_M3S = 0.000471947  # 1 cubic foot per minute = 0.000471947 m³/s
D = 1.32  # m, 52-inch DC ceiling fan
CFM = 5000  # fan airflow on high (typical 52" DC fan; ENERGY STAR list 4,365-5,741 CFM)
FAN_W = 25  # W, typical power of a 52" DC fan on high (listings 18-33 W)
H_C, H_B = 2.6, 2.3  # m, ceiling height and blade height (as in folder 10)
V_STILL = 0.1  # m/s, still air with no fan
V_ASSUMED_MIN = 1.0  # m/s, the user's assumption: the user feels more than this


def room_speeds(open_area, aspect_ratio, cfm=CFM):
    """(lowest, average, highest) seated air speed in m/s from the Raftery et al.
    (2019) model, for one fan serving `open_area` m² with the given aspect ratio."""
    s_f = cfm * CFM_TO_M3S / (math.pi * D ** 2 / 4)  # fan air speed = airflow ÷ blade disc area
    r = math.sqrt(open_area)
    lowest = s_f * (0.90 * D / r - 0.017 * H_C / D + 0.11 * D / 1.7 + 0.024 + 0.047) / aspect_ratio
    average = s_f * (0.99 * D / r - 0.060 * H_C / D + 0.11 * D / 1.7 + 0.024 + 0.25)
    highest = s_f * (1.3 - 0.18 * H_B / D - 0.10)
    return lowest, average, highest


for room in ROOMS:
    lost = sum(w * d for _, _, _, w, d in room["built_in"])  # m², floor taken by wardrobe + built-ins
    room["area"] = room["L"] * room["W"]
    room["open_area"] = room["area"] - lost
    # Open space shape: the wardrobe takes its depth off the room's width
    wardrobe_depth = room["built_in"][0][4]
    room["aspect"] = max(room["L"], room["W"] - wardrobe_depth) / min(room["L"], room["W"] - wardrobe_depth)
    room["v_low"], room["v_user"], room["v_peak"] = room_speeds(room["open_area"], room["aspect"])
    # Fan centred over the bed
    _, bx, by, bw, bd = room["bed"]
    room["fan_xy"] = (bx + bw / 2, by + bd / 2)

# =============================================================================
# 3. Comfort: SET for a sleeper
# =============================================================================
RH = 60  # %, relative humidity with the aircon on
# Sleep situations: (name, met, clo). met 0.7 = asleep;
# clo 0.9 = mattress + sleepwear (no covers); 1.5 = plus a thin blanket.
SLEEP_CASES = [("No covers (0.9 clo)", 0.7, 0.9), ("Thin blanket (1.5 clo)", 0.7, 1.5)]
MAIN_CASE = SLEEP_CASES[0]  # used for the cost results, as in the rest of the project
BASELINES = [22, 23, 24, 25, 26]  # °C, aircon-alone settings the user might use now
MAIN_BASE = 25  # °C, Go 25 guidance
MAX_SETPOINT = 30  # °C, the highest setting on most remotes


def set_value(tdb, v, met, clo):
    """SET (°C); walls at air temperature; sitting posture (the SET model has
    no lying option); limit_inputs=False so the sleeping case can run."""
    return set_tmp(tdb=tdb, tr=tdb, v=v, rh=RH, met=met, clo=clo, position="sitting",
                   limit_inputs=False, round_output=False).set


def fan_setpoint(base, v, met, clo):
    """Setpoint (°C) with the fan at speed v that has the same SET as `base` °C at still air."""
    target = set_value(base, V_STILL, met, clo)
    t = brentq(lambda x: set_value(x, v, met, clo) - target, base - 1, 40)
    return min(t, MAX_SETPOINT)


def pmv_ppd(tdb, v, met, clo):
    """ASHRAE 55 PMV and PPD (%); the user controls the fan (airspeed_control=True)."""
    r = pmv_ppd_ashrae(tdb=tdb, tr=tdb, vr=v, rh=RH, met=met, clo=clo, airspeed_control=True,
                       limit_inputs=False, round_output=False)
    return r.pmv, r.ppd


# =============================================================================
# 4. Electricity (bedroom model from 03_household_savings)
# =============================================================================
TARIFF = 0.3116  # S$ per kWh, SP Group Oct-Dec 2026 incl. 9% GST
DAYS_PER_YEAR = 365
HOURS = range(22, 30)  # 10 pm to 6 am (hours past midnight written as 24-29)
UA, Q_INT, Q_LAT = 60, 250, 250  # W/K heat leak, W internal heat, W moisture load (as in folder 03)
MASS_OFFSET = 2.0  # °C, warm concrete walls give off heat as if it were 2 °C hotter outside
SYSTEMS = [("Older aircon (COP 3.5)", 3.5), ("New 5-tick aircon (COP 5.5)", 5.5)]
FAN_PRICE = 350  # S$, a 52" DC ceiling fan bought and installed (ASSUMPTION; roughly S$250-450)


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
    """Ideal-aircon efficiency, used only to scale COP with setpoint (coils 12 °C from room/outside)."""
    t_evap, t_cond = t_set - 12 + 273.15, t_o + 12 + 273.15
    return t_evap / (t_cond - t_evap)


def aircon_kwh_per_night(t_set, cop_25):
    """Aircon electricity (kWh) for one bedroom over one night at setpoint t_set (°C)."""
    kwh = 0.0
    for h in HOURS:
        to = t_out(h % 24)
        load = UA * (to + MASS_OFFSET - t_set) + Q_INT + Q_LAT  # cooling load (W)
        kwh += max(load, 0) / (cop_25 * carnot(t_set, to) / carnot(25.0, to)) / 1000
    return kwh


FAN_KWH_PER_NIGHT = FAN_W * len(HOURS) / 1000  # fan on all night

# =============================================================================
# 5. Run
# =============================================================================
rows = []
print(f"52\" DC ceiling fan on high: {CFM:,} CFM, {FAN_W} W, centred over the bed\n")
for room in ROOMS:
    ok = "yes" if room["v_user"] > V_ASSUMED_MIN else "NO"
    print(f"{room['name']}: {room['L']} x {room['W']} m = {room['area']:.1f} m²; open floor after built-ins "
          f"{room['open_area']:.1f} m²")
    print(f"  Air speed: user (open-area average) {room['v_user']:.2f} m/s (> {V_ASSUMED_MIN} m/s? {ok}) · "
          f"slowest spot {room['v_low']:.2f} · directly under fan {room['v_peak']:.2f} m/s")
    for case in SLEEP_CASES:
        _, met, clo = case
        t = fan_setpoint(MAIN_BASE, room["v_user"], met, clo)
        p0, d0 = pmv_ppd(MAIN_BASE, V_STILL, met, clo)
        p1, d1 = pmv_ppd(t, room["v_user"], met, clo)
        print(f"  {case[0]:<24} 25 °C aircon alone -> {t:4.1f} °C with fan "
              f"(ASHRAE PMV {p0:+.2f} -> {p1:+.2f})")
    print()

print("Net saving per bedroom (no covers, 10 pm-6 am every night), baseline = aircon alone\n")
results = {}  # (room, system, baseline) -> dict
for room in ROOMS:
    for sys_label, cop in SYSTEMS:
        print(f"{room['name']}, {sys_label}")
        for base in BASELINES:
            t_fan = fan_setpoint(base, room["v_user"], MAIN_CASE[1], MAIN_CASE[2])
            ac_base = aircon_kwh_per_night(base, cop) * DAYS_PER_YEAR
            ac_fan = aircon_kwh_per_night(t_fan, cop) * DAYS_PER_YEAR
            fan = FAN_KWH_PER_NIGHT * DAYS_PER_YEAR
            saved, net = ac_base - ac_fan, ac_base - ac_fan - fan  # kWh per year
            payback = FAN_PRICE / (net * TARIFF) if net > 0 else float("inf")
            r = {"room": room["name"], "aircon": sys_label, "cop_at_25C": cop,
                 "open_area_m2": round(room["open_area"], 2), "v_user_m_s": round(room["v_user"], 2),
                 "baseline_setpoint_C": base, "fan_setpoint_C": round(t_fan, 2),
                 "aircon_only_kwh_yr": round(ac_base, 1), "aircon_with_fan_kwh_yr": round(ac_fan, 1),
                 "fan_kwh_yr": round(fan, 1), "aircon_saved_kwh_yr": round(saved, 1),
                 "net_saved_kwh_yr": round(net, 1), "net_saved_pct": round(100 * net / ac_base, 1),
                 "net_saved_sgd_yr": round(net * TARIFF, 2), "payback_years": round(payback, 1)}
            results[(room["name"], sys_label, base)] = r
            rows.append(r)
            mark = "  <- Go 25" if base == MAIN_BASE else ""
            print(f"  {base} °C -> {t_fan:4.1f} °C: aircon saved {saved:5.0f} kWh, fan uses {fan:3.0f} kWh, "
                  f"net {net:5.0f} kWh/yr ({100 * net / ac_base:3.0f}%) = S${net * TARIFF:5.2f}/yr, "
                  f"fan pays back in {payback:4.1f} yr{mark}")
        print()

with open("bedroom_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Saved bedroom_results.csv")

# =============================================================================
# 6. Chart 1: floor plans
# =============================================================================
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, LIGHT_BLUE, WOOD, BED = "#2a78d6", "#86b6ef", "#c9b79c", "#e9e4da"

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150, gridspec_kw={"width_ratios": [3.3, 4.0]})
fig.patch.set_facecolor(SURFACE)
for ax, room in zip(axes, ROOMS):
    ax.set_facecolor(SURFACE)
    ax.add_patch(Rectangle((0, 0), room["L"], room["W"], fill=False, ec=INK, lw=2.5))
    for label, x, y, w, d in room["built_in"]:
        ax.add_patch(Rectangle((x, y), w, d, fc=WOOD, ec=INK_2, lw=0.8))
        ax.text(x + w / 2, y + d / 2, label.replace(" + ", "\n+ "), ha="center", va="center", fontsize=7.5,
                color=INK, rotation=90 if d > w else 0)
    label, x, y, w, d = room["bed"]
    ax.add_patch(Rectangle((x, y), w, d, fc=BED, ec=INK_2, lw=0.8))
    ax.text(x + w / 2, y + 0.15, label, ha="center", va="bottom", fontsize=7.5, color=INK_2)
    fx, fy = room["fan_xy"]
    ax.add_patch(Circle((fx, fy), D / 2, fc=LIGHT_BLUE, alpha=0.35, ec=BLUE, lw=1.2, ls="--"))
    ax.plot([fx], [fy], marker="+", color=BLUE, ms=10, mew=2)
    ax.text(fx, fy + 0.12, f"52\" fan\n{room['v_user']:.2f} m/s felt", ha="center", va="bottom", fontsize=8,
            color=BLUE)
    ax.set_xlim(-0.1, room["L"] + 0.1)
    ax.set_ylim(-0.1, room["W"] + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{room['name']}: {room['L']} × {room['W']} m ({room['area']:.1f} m²)\n"
                 f"open floor {room['open_area']:.1f} m² · slowest spot {room['v_low']:.2f} m/s",
                 loc="left", fontsize=9.5, color=INK)
fig.suptitle("Fan centred over the bed: the user feels over 1 m/s in both bedrooms", x=0.01, ha="left",
             fontsize=12, color=INK)
fig.text(0.01, 0.02, f"52\" DC fan on high ({CFM:,} CFM) · speed felt = Raftery et al. (2019) average over the open "
         "floor (wardrobe and built-ins excluded) · dashed circle = blade span", fontsize=7.8, color=INK_2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig("bedroom_plans.png")
print("Saved bedroom_plans.png")

# =============================================================================
# 7. Chart 2: setpoint and net saving
# =============================================================================


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150, gridspec_kw={"width_ratios": [1, 1.35]})
fig.patch.set_facecolor(SURFACE)

# Left: fan setpoint vs aircon-alone setting
ax1.plot(BASELINES, BASELINES, color="#8f8e88", lw=1.5, ls="--", label="Aircon alone")
for room, color in zip(ROOMS, (BLUE, LIGHT_BLUE)):
    t = [results[(room["name"], SYSTEMS[0][0], b)]["fan_setpoint_C"] for b in BASELINES]
    ax1.plot(BASELINES, t, color=color, lw=2.2, marker="o", ms=6,
             label=f"{room['name']} + fan ({room['v_user']:.2f} m/s)")
    for b, tt in zip(BASELINES, t):
        if room is ROOMS[0]:
            ax1.annotate(f"{tt:.1f}", (b, tt), textcoords="offset points", xytext=(0, 7), ha="center",
                         fontsize=8, color=INK)
style(ax1)
ax1.set_xticks(BASELINES)
ax1.set_xlabel("Aircon-alone setting the user is used to (°C)", color=INK_2)
ax1.set_ylabel("Setting with the fan for the same comfort (°C)", color=INK_2)
ax1.set_title("Same SET comfort, asleep without covers", loc="left", fontsize=10.5, color=INK)
ax1.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)

# Right: net yearly saving per bedroom at 25 °C baseline: aircon saved, minus fan use
groups = [(room["name"], s) for room in ROOMS for s, _ in SYSTEMS]
x = range(len(groups))
for i, (rn, s) in enumerate(groups):
    r = results[(rn, s, MAIN_BASE)]
    saved_s = r["aircon_saved_kwh_yr"] * TARIFF
    fan_s = r["fan_kwh_yr"] * TARIFF
    net_s = r["net_saved_sgd_yr"]
    ax2.bar(i - 0.2, saved_s, 0.36, color=LIGHT_BLUE, label="Aircon electricity saved" if i == 0 else None)
    ax2.bar(i + 0.2, net_s, 0.36, color=BLUE, label="Net saving (after fan)" if i == 0 else None)
    ax2.text(i - 0.2, saved_s + 0.6, f"S${saved_s:.0f}", ha="center", fontsize=8, color=INK)
    ax2.text(i + 0.2, net_s + 0.6, f"S${net_s:.0f}\n({r['net_saved_pct']:.0f}%)", ha="center", fontsize=8,
             color=INK)
    ax2.text(i, -4.5, f"fan uses S${fan_s:.0f}\npays back {r['payback_years']:.0f} yr", ha="center", va="top",
             fontsize=7.5, color=INK_2)
style(ax2)
ax2.axhline(0, color=INK_2, lw=1)
ax2.set_xticks(list(x))
ax2.set_xticklabels([f"{rn}\n{s.split(' (')[1].rstrip(')')}" for rn, s in groups], color=INK, fontsize=8.5)
ax2.tick_params(axis="x", pad=34)
top = max(results[(rn, s, MAIN_BASE)]["aircon_saved_kwh_yr"] * TARIFF for rn, s in groups)
ax2.set_ylim(-13, top * 1.3)
ax2.set_ylabel("S$ per year, per bedroom", color=INK_2)
ax2.set_title("Yearly saving vs 25 °C aircon alone (S$; % = share of aircon bill)", loc="left", fontsize=10.5,
              color=INK)
ax2.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)

r_c = results[(ROOMS[0]["name"], SYSTEMS[0][0], MAIN_BASE)]
fig.suptitle(f"Fan over the bed lets the aircon go from 25 °C to {r_c['fan_setpoint_C']:.1f} °C; "
             "after the fan's own electricity, each bedroom still saves money", x=0.01, ha="left",
             fontsize=12.5, color=INK)
fig.text(0.01, 0.015, f"Equal SET · sleeping 0.7 met, no covers 0.9 clo, {RH}% RH (below ASHRAE 55-2023's 1.0-4.0 met range: "
         f"indicative) · aircon 10 pm-6 am, 365 nights · DC fan {FAN_W} W on high · S${TARIFF}/kWh · "
         f"fan S${FAN_PRICE} installed (assumed)", fontsize=7.8, color=INK_2)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))
fig.savefig("bedroom_net_savings.png")
print("Saved bedroom_net_savings.png")

# =============================================================================
# 8. Part 2: cool-down when the user walks into a hot room
# =============================================================================
# The question: the user comes home at 10 pm to a bedroom that has heated up
# all day. With the fan, they feel comfortable much sooner. Does that also
# save energy?
#
# Key physics: a fan does not cool the air (its motor even adds a little heat).
# It makes the PERSON feel cooler, so:
#   - the user feels comfortable before the room has cooled down, and
#   - the aircon can aim for a warmer setpoint, so it has to pull less heat
#     out of the warm concrete walls, floor and ceiling.
#
# Model: two "nodes" (lumped heat stores), stepped forward every DT seconds.
#   T_a = room air + furniture + bedding temperature (°C), heat capacity C_a (J/K)
#   T_m = inner layer of the concrete walls/floor/ceiling (°C), heat capacity C_m (J/K)
#   Heat flows (W):
#     walls -> air:     G_in × (T_m − T_a), with G_in = H_IN × surface area
#     outside -> walls: G_out × (T_out + MASS_OFFSET − T_m)
#     people, lights, fan motor -> air: Q_INT (+ fan watts when the fan is on)
#     aircon removes Q_ac from the air, up to its capacity Q_MAX
#   G_out is chosen so that, in steady state, the two-node model gives the
#   same heat gain as the UA = 60 W/K used everywhere else in the project.
# Comfort: SET with the air at T_a and the walls (mean radiant temperature)
# at T_m. Warm walls keep the user feeling warm even after the air has cooled.
# Electricity: heat the aircon removes (plus the moisture load Q_LAT while it
# runs) ÷ COP, plus the fan.

DT = 30  # s, time step
C_AIR_PER_M2 = 25e3  # J/(K·m² floor): room air (~3) + furniture, mattress, bedding (~22) (ASSUMPTION)
MASS_DEPTH = 0.05  # m, depth of concrete that takes part in a few-hours cool-down (ASSUMPTION)
RHO_CP_CONCRETE = 2.2e6  # J/(m³·K), volumetric heat capacity of concrete
H_IN = 8.0  # W/(m²·K), heat transfer from inside surfaces to room air (convection + radiation)
Q_MAX = 2640  # W, cooling capacity of a bedroom fan coil: 9,000 BTU/h (typical for HDB bedrooms)
# Starting room temperature (3 Nov 2026): 29.1 °C, the MEASURED average indoor air temperature of
# Singapore's naturally ventilated HDB flats during the day and early evening ("Adaptive behaviour and
# thermal comfort in Singapore's naturally ventilated housing", ResearchGate 339297268). Walls start at
# the same temperature (ASSUMPTION).
T_START = 29.1  # °C, bedroom air and walls when the user arrives (measured average, see above)
ARRIVE = 22.0  # clock hour the user walks in (10 pm)
END = 30.0  # 6 am (hours past midnight written as 24-30)
PRECOOL_MIN = 30  # min, how early a "pre-cool" user switches on the aircon (e.g. by app or timer)
SIT = (1.0, 0.4)  # (met, clo): the user is awake at first, sitting in T-shirt and shorts
COMFORT_BASE = 25.0  # °C, the reference: aircon alone at 25 °C once the room has fully settled
SET_TOLERANCE = 0.5  # °C: "comfortable" = SET within 0.5 °C of that settled reference (ASSUMPTION;
#                      a difference people barely notice)
OFF_EARLY_MIN = 30  # min: in the "early off" case the aircon switches off this long before the
#                     user wakes at 6 am (the fan keeps running)


def room_mass(room):
    """Heat capacities (J/K) and conductances (W/K) for one room (see notes above)."""
    surfaces = 2 * room["area"] + 2 * (room["L"] + room["W"]) * H_C  # floor + ceiling + walls (m²)
    c_a = C_AIR_PER_M2 * room["area"]
    c_m = surfaces * MASS_DEPTH * RHO_CP_CONCRETE
    g_in = H_IN * surfaces  # walls <-> air
    g_out = 1 / (1 / UA - 1 / g_in)  # outside <-> walls, so g_out and g_in in series = UA
    return c_a, c_m, g_in, g_out


def cooldown(room, setpoint, fan_on, start_hour, cop_25, off_hour=None, fan_w=None):
    """Simulate from start_hour to 6 am. If off_hour is given, the aircon switches
    off at that hour (the fan, if on, keeps running). fan_w = fan power (W) when
    on; default FAN_W. Returns times (h), T_a and T_m lists (°C), and the aircon
    and fan electricity (kWh)."""
    c_a, c_m, g_in, g_out = room_mass(room)
    t_a = t_m = T_START
    times, ta_list, tm_list = [], [], []
    ac_kwh = fan_kwh = 0.0
    fan_w = (FAN_W if fan_w is None else fan_w) if fan_on else 0
    n = int(round((END - start_hour) * 3600 / DT))
    for i in range(n + 1):
        hour = start_hour + i * DT / 3600
        times.append(hour)
        ta_list.append(t_a)
        tm_list.append(t_m)
        if i == n:
            break
        to = t_out(hour % 24)
        q_wall = g_in * (t_m - t_a)  # W, walls warming the air
        q_gain = q_wall + Q_INT + fan_w  # W, all heat entering the air
        # Ideal thermostat with a capacity limit: remove what is needed to bring
        # the air to the setpoint in this step, but no more than Q_MAX.
        q_need = c_a * (t_a - setpoint) / DT + q_gain
        q_ac = min(max(q_need, 0.0), Q_MAX)
        if off_hour is not None and hour >= off_hour - 1e-9:
            q_ac = 0.0  # aircon switched off: the room starts to warm up again
        if q_ac > 0:
            cop = cop_25 * carnot(t_a, to) / carnot(25.0, to)
            ac_kwh += (q_ac + Q_LAT) / cop * DT / 3.6e6
        fan_kwh += fan_w * DT / 3.6e6
        t_a += (q_gain - q_ac) * DT / c_a
        t_m += (g_out * (to + MASS_OFFSET - t_m) - q_wall) * DT / c_m
    return times, ta_list, tm_list, ac_kwh, fan_kwh


def set_series(times, ta, tm, v, every=2):
    """SET (°C) of the awake user along a run, once a minute (every 2nd 30 s step)."""
    idx = list(range(0, len(times), every))
    s = set_tmp(tdb=[ta[i] for i in idx], tr=[tm[i] for i in idx], v=[v] * len(idx), rh=RH,
                met=SIT[0], clo=SIT[1], position="sitting", limit_inputs=False, round_output=False).set
    return [times[i] for i in idx], list(s)


def settled_wall_temp(room, t_air):
    """Wall temperature (°C) once the room has fully settled with the air held at
    t_air: heat in from outside = heat out to the air (night-average outdoor temp)."""
    _, _, g_in, g_out = room_mass(room)
    t_o = sum(t_out(h % 24) for h in HOURS) / len(HOURS) + MASS_OFFSET
    return (g_out * t_o + g_in * t_air) / (g_out + g_in)


def target_set(room):
    """SET (°C) the user must reach to count as comfortable: the settled 25 °C
    aircon-alone SET (air 25 °C, walls slightly warmer, still air) + tolerance."""
    tr = settled_wall_temp(room, COMFORT_BASE)
    ref = set_tmp(tdb=COMFORT_BASE, tr=tr, v=V_STILL, rh=RH, met=SIT[0], clo=SIT[1], position="sitting",
                  limit_inputs=False, round_output=False).set
    return ref + SET_TOLERANCE


TARGET_SET = target_set(ROOMS[0])


def sleep_target_set(room):
    """Same idea for the sleeper (0.7 met, no covers): the settled 25 °C
    aircon-alone SET + tolerance. Used to check the last 30 min before waking."""
    _, met, clo = MAIN_CASE
    tr = settled_wall_temp(room, COMFORT_BASE)
    ref = set_tmp(tdb=COMFORT_BASE, tr=tr, v=V_STILL, rh=RH, met=met, clo=clo, position="sitting",
                  limit_inputs=False, round_output=False).set
    return ref + SET_TOLERANCE


def sleep_set_series(times, ta, tm, v, start_h, end_h=END, every=2):
    """SET (°C) of the sleeper between start_h and end_h, once a minute."""
    idx = [i for i in range(0, len(times), every) if start_h - 1e-9 <= times[i] <= end_h + 1e-9]
    _, met, clo = MAIN_CASE
    s = set_tmp(tdb=[ta[i] for i in idx], tr=[tm[i] for i in idx], v=[v] * len(idx), rh=RH,
                met=met, clo=clo, position="sitting", limit_inputs=False, round_output=False).set
    return [times[i] for i in idx], list(s)


SLEEP_TARGET_SET = sleep_target_set(ROOMS[0])


def minutes_to_comfort(set_times, sets):
    """Minutes after arrival until the user's SET first falls to TARGET_SET (0 if already)."""
    for t, s in zip(set_times, sets):
        if t >= ARRIVE - 1e-9 and s <= TARGET_SET:
            return max(0.0, (t - ARRIVE) * 60)
    return float("nan")


# =============================================================================
# 9. Run the cool-down scenarios (common bedroom)
# =============================================================================
room = ROOMS[0]
FAN_SP = results[(room["name"], SYSTEMS[0][0], MAIN_BASE)]["fan_setpoint_C"]  # 25 °C equivalent with fan
OFF_AT = END - OFF_EARLY_MIN / 60  # 5:30 am
# (label, aircon setpoint °C, fan on?, aircon switch-on hour, aircon switch-off hour or None = 6 am)
COOL_CASES = [
    ("Aircon alone, on at arrival (25 °C)", 25.0, False, ARRIVE, None),
    (f"Aircon alone, pre-cooled {PRECOOL_MIN} min (25 °C)", 25.0, False, ARRIVE - PRECOOL_MIN / 60, None),
    (f"Fan + aircon, on at arrival ({FAN_SP:.1f} °C)", FAN_SP, True, ARRIVE, None),
    (f"Fan + aircon, aircon off at 5:30 am ({FAN_SP:.1f} °C)", FAN_SP, True, ARRIVE, OFF_AT),
]
c_a, c_m, g_in, g_out = room_mass(room)
print(f"\nCool-down: {room['name']}, air and walls at {T_START} °C when the user arrives at 10 pm")
print(f"  C_air {c_a / 1e3:.0f} kJ/K, C_walls {c_m / 1e6:.1f} MJ/K, walls<->air {g_in:.0f} W/K, "
      f"outside<->walls {g_out:.0f} W/K, aircon capacity {Q_MAX / 1000:.2f} kW")
print(f"  Comfortable = SET {TARGET_SET:.1f} °C or lower (awake, sitting): within {SET_TOLERANCE} °C of how "
      f"25 °C aircon alone feels once settled\n")

print(f"  Asleep: comfortable = SET {SLEEP_TARGET_SET:.1f} °C or lower (0.7 met, no covers), same rule\n")

cool_rows, runs = [], {}
for label, sp, fan_on, start, off in COOL_CASES:
    v = room["v_user"] if fan_on else V_STILL
    times, ta, tm, _, _ = cooldown(room, sp, fan_on, start, SYSTEMS[0][1], off)
    st, sets = set_series(times, ta, tm, v)
    mins = minutes_to_comfort(st, sets)
    # The last hour before waking: how warm the sleeper feels (matters for the early-off case)
    wt, wsets = sleep_set_series(times, ta, tm, v, END - 1.0)
    last30 = [s for t, s in zip(wt, wsets) if t >= OFF_AT - 1e-9]
    over = sum(1 for s in last30 if s > SLEEP_TARGET_SET)  # minutes above the sleep comfort limit
    runs[label] = (times, ta, tm, st, sets, mins, wt, wsets)
    i_arr = min(range(len(times)), key=lambda k: abs(times[k] - ARRIVE))
    print(label)
    print(f"  at arrival: air {ta[i_arr]:.1f} °C, walls {tm[i_arr]:.1f} °C · comfortable after {mins:.0f} min")
    print(f"  5:30-6:00 am: air {ta[-61]:.1f} -> {ta[-1]:.1f} °C · sleeper's SET up to {max(last30):.1f} °C "
          f"(limit {SLEEP_TARGET_SET:.1f}) · {over} min above the limit")
    for sys_label, cop in SYSTEMS:
        _, _, _, ac, fan = cooldown(room, sp, fan_on, start, cop, off)
        cool_rows.append({"room": room["name"], "case": label, "aircon": sys_label, "setpoint_C": round(sp, 2),
                          "fan_on": fan_on, "aircon_on_at_h": round(start % 24, 2),
                          "aircon_off_at_h": round((off if off else END) % 24, 2),
                          "minutes_to_comfort_after_arrival": round(mins, 1),
                          "max_sleep_set_last_30min_C": round(max(last30), 2),
                          "minutes_above_sleep_limit_last_30min": over,
                          "aircon_kwh_night": round(ac, 3), "fan_kwh_night": round(fan, 3),
                          "total_kwh_night": round(ac + fan, 3),
                          "total_sgd_year": round((ac + fan) * TARIFF * DAYS_PER_YEAR, 2)})
        print(f"  {sys_label}: aircon {ac:.2f} + fan {fan:.2f} = {ac + fan:.2f} kWh/night "
              f"= S${(ac + fan) * TARIFF * DAYS_PER_YEAR:.0f}/yr")

print()
for sys_label, _ in SYSTEMS:
    tot = {r["case"]: r["total_kwh_night"] for r in cool_rows if r["aircon"] == sys_label}
    # (fan case, compared against)
    for fan_case, other in ((COOL_CASES[2][0], COOL_CASES[0][0]), (COOL_CASES[2][0], COOL_CASES[1][0]),
                            (COOL_CASES[3][0], COOL_CASES[0][0]), (COOL_CASES[3][0], COOL_CASES[2][0])):
        d = tot[other] - tot[fan_case]
        print(f"{sys_label}: '{fan_case}' vs '{other}': net saving {d:.2f} kWh/night = "
              f"S${d * TARIFF * DAYS_PER_YEAR:.0f}/yr ({d / tot[other]:.0%})")

with open("cooldown_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(cool_rows[0].keys()))
    w.writeheader()
    w.writerows(cool_rows)
print("Saved cooldown_results.csv")

# =============================================================================
# 10. Chart: cool-down
# =============================================================================
# Colours: aircon on at arrival, pre-cooled, fan + aircon, fan + aircon with early off
CASE_COLORS = ["#8f8e88", "#c2410c", BLUE, "#2f6b3a"]
fig = plt.figure(figsize=(13, 9.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.55, wspace=0.22)
ax_t = fig.add_subplot(gs[0, 0])
ax_s = fig.add_subplot(gs[1, 0], sharex=ax_t)
ax_w = fig.add_subplot(gs[2, 0])  # the last hour before waking
ax_e = fig.add_subplot(gs[:, 1])


def clock(h):
    """Clock label for an hour value, e.g. 22.5 -> '10:30 pm'."""
    h24 = h % 24
    hh, mm = int(h24), int(round((h24 % 1) * 60))
    return f"{(hh - 1) % 12 + 1}:{mm:02d} {'am' if hh < 12 else 'pm'}"


WINDOW = (ARRIVE - PRECOOL_MIN / 60 - 0.05, ARRIVE + 2.5)  # show about 9:30 pm to 12:30 am
for (label, sp, fan_on, start, off), color in zip(COOL_CASES, CASE_COLORS):
    times, ta, tm, st, sets, mins, wt, wsets = runs[label]
    if off is not None:
        continue  # identical to "fan + aircon" until 5:30 am; shown in the bottom panel instead
    ax_t.plot(times, ta, color=color, lw=2, label=label)
    ax_t.plot(times, tm, color=color, lw=1.2, ls=":")
    ax_s.plot(st, sets, color=color, lw=2)
    t_ok = ARRIVE + mins / 60
    if t_ok <= WINDOW[1]:
        ax_s.scatter([t_ok], [TARGET_SET], s=45, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax_s.annotate(f"comfortable after {mins:.0f} min", (t_ok, TARGET_SET), textcoords="offset points",
                      xytext=(6, 6), fontsize=8.5, color=color)
# The aircon-alone cases only become comfortable after the chart window: one shared label
m_on, m_pre = runs[COOL_CASES[0][0]][5], runs[COOL_CASES[1][0]][5]
ax_s.text(WINDOW[1], 27.0, f"aircon alone: comfortable only after {m_on / 60:.1f} h\n"
          f"({m_pre / 60:.1f} h if pre-cooled): the walls are still warm", ha="right", fontsize=8.5, color=INK_2)
for ax in (ax_t, ax_s):
    style(ax)
    ax.axvline(ARRIVE, color=INK_2, lw=1, ls="--")
    ax.set_xlim(*WINDOW)
ax_t.text(ARRIVE - 0.02, 25.6, "user\narrives", fontsize=8, color=INK_2, ha="right")
ax_t.set_ylabel("Temperature (°C)", color=INK_2)
ax_t.set_title("Room air (solid) and walls (dotted): the walls stay warm for hours", loc="left",
               fontsize=10.5, color=INK)
ax_t.legend(loc="lower right", bbox_to_anchor=(1, 0.12), frameon=False, fontsize=8, labelcolor=INK)
ax_s.axhline(TARGET_SET, color=INK_2, lw=1)
ax_s.text(WINDOW[1], TARGET_SET + 0.1, "comfortable (= 25 °C aircon alone)", ha="right", va="bottom",
          fontsize=8, color=INK_2)
ax_s.set_ylabel("User's SET, awake (°C)", color=INK_2)
ax_s.set_title("How warm the user feels after walking in", loc="left", fontsize=10.5, color=INK)
ticks = [ARRIVE - 0.5 + 0.5 * k for k in range(6)]
ax_s.set_xticks(ticks)
ax_s.set_xticklabels([clock(h) for h in ticks])
plt.setp(ax_t.get_xticklabels(), visible=False)

# Bottom panel: the last hour before waking (5-6 am), sleeper's SET, all cases
for (label, sp, fan_on, start, off), color in zip(COOL_CASES, CASE_COLORS):
    wt, wsets = runs[label][6], runs[label][7]
    ax_w.plot(wt, wsets, color=color, lw=2.2 if off else 1.6)
style(ax_w)
ax_w.axhline(SLEEP_TARGET_SET, color=INK_2, lw=1)
ax_w.text(END - 1 + 0.01, SLEEP_TARGET_SET + 0.02, "comfortable asleep (limit)", ha="left", va="bottom",
          fontsize=8, color=INK_2)
ax_w.axvline(OFF_AT, color="#2f6b3a", lw=1, ls="--")
ax_w.text(OFF_AT - 0.01, SLEEP_TARGET_SET - 0.52, "aircon off,\nfan stays on", ha="right", va="bottom",
          fontsize=8, color="#2f6b3a")
early = runs[COOL_CASES[3][0]]
last = [s for t, s in zip(early[6], early[7]) if t >= OFF_AT - 1e-9]
over_min = sum(1 for s in last if s > SLEEP_TARGET_SET)
ax_w.text(END - 1 + 0.01, max(last) - 0.02, f"Green = aircon off at 5:30 am: the sleeper's SET rises to {max(last):.1f} °C,\n"
          f"{max(last) - SLEEP_TARGET_SET:.1f} °C over the limit for the last {over_min} min",
          ha="left", va="top", fontsize=8.5, color="#2f6b3a")
wticks = [END - 1 + 0.25 * k for k in range(5)]
ax_w.set_xticks(wticks)
ax_w.set_xticklabels([clock(h) for h in wticks])
ax_w.set_xlim(END - 1, END)
ax_w.set_ylabel("Sleeper's SET (°C)", color=INK_2)
ax_w.set_title("Before waking (5-6 am): how warm the sleeper feels", loc="left", fontsize=10.5, color=INK)

# Energy panel: whole night (switch-on to 6 am), S$ per year, both aircon types
bw = 0.2
for k, ((label, _, _, _, _), color) in enumerate(zip(COOL_CASES, CASE_COLORS)):
    vals = [next(r["total_sgd_year"] for r in cool_rows if r["case"] == label and r["aircon"] == s)
            for s, _ in SYSTEMS]
    xs = [j + (k - 1.5) * bw for j in range(len(SYSTEMS))]
    ax_e.bar(xs, vals, bw * 0.92, color=color, label=label)
    for x_, v_ in zip(xs, vals):
        ax_e.text(x_, v_ + 1.5, f"S${v_:.0f}", ha="center", fontsize=7.8, color=INK)
ax_e.legend(loc="upper right", frameon=False, fontsize=7.5, labelcolor=INK)
style(ax_e)
ax_e.set_xticks(range(len(SYSTEMS)))
ax_e.set_xticklabels([s.replace(" (", "\n(") for s, _ in SYSTEMS], color=INK, fontsize=8.5)
ax_e.set_ylabel("Aircon + fan electricity, S$ per year (one bedroom)", color=INK_2)
ax_e.set_title("Energy, switch-on to 6 am every night", loc="left", fontsize=10.5, color=INK)
ax_e.set_ylim(0, max(r["total_sgd_year"] for r in cool_rows) * 1.35)

fig.suptitle("Arriving to a hot room: how soon the user is comfortable, and the energy used",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, 0.012, f"{room['name']}, air and walls {T_START} °C at 10 pm · {Q_MAX / 1000:.2f} kW aircon · "
         f"awake user 1.0 met, 0.4 clo, {RH}% RH; walls set the radiant temperature · fan "
         f"{room['v_user']:.2f} m/s, {FAN_W} W · S${TARIFF}/kWh · two-node room model (assumed capacities, see script)",
         fontsize=7.8, color=INK_2)
fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.07)
fig.savefig("bedroom_cooldown.png")
print("Saved bedroom_cooldown.png")

# =============================================================================
# 11. Part 3: review checks (review follow-ups)
# =============================================================================
# (a) Fan speed: the ceiling fan run below high (power ~ speed cubed, fan law).
# (b) Wall heat-storage depth: the cool-down saving for 2-8 cm of concrete.
# (c) The 0.5 m/s sleep cap (team design: draught and noise) with each fan:
#     - the existing DC ceiling fan turned down to give 0.5 m/s over the bed.
#       Fan law (power ~ speed cubed): FAN_W x (0.5 / v_user)^3, about 1.8 W
#       in the common bedroom (ASSUMPTION: air speed scales with fan speed).
#     - a DC standing fan pointed at the bed, 4.5 W at 0.5 m/s (folder 12
#       model: P = 3 + 12 v^3 W), and its uncapped 1.0 m/s / 15 W case.
#     The system now controls the existing ceiling fan; the standing fan is
#     kept for comparison (backup slides).
# (d) Humidity: 60 / 70 / 80 % RH (the aircon may dehumidify less at warm
#     setpoints). The energy model keeps the moisture load fixed, so only the
#     setpoint changes here.
# (e) Posture: the SET model has no lying option; compare "sitting" (used)
#     with "standing" as a bracket.
print("\nPart 3: review checks (common bedroom, asleep without covers, baseline 25 °C aircon alone)")
room = ROOMS[0]
_, MET_S, CLO_S = MAIN_CASE
review = []


def set_rh(tdb, tr, v, met, clo, rh, position="sitting"):
    """SET (°C) with humidity and posture as inputs."""
    return float(set_tmp(tdb=tdb, tr=tr, v=v, rh=rh, met=met, clo=clo, position=position,
                         limit_inputs=False, round_output=False).set)


def setpoint_rh(base, v, met, clo, rh, position="sitting"):
    """Setpoint (°C) at air speed v with the same SET as `base` °C at still air."""
    target = set_rh(base, base, V_STILL, met, clo, rh, position)
    t = brentq(lambda x: set_rh(x, x, v, met, clo, rh, position) - target, base - 1, 40)
    return min(t, MAX_SETPOINT)


def steady_net_sgd(t_set, cop, fan_w):
    """Net yearly saving (S$) vs 25 °C aircon alone, one bedroom, steady model."""
    kwh = (aircon_kwh_per_night(MAIN_BASE, cop) - aircon_kwh_per_night(t_set, cop)
           - fan_w * len(HOURS) / 1000) * DAYS_PER_YEAR
    return kwh * TARIFF


# (a) fan speed
print("\n(a) Ceiling fan speed (steady model)")
for frac in (1.0, 0.85, 0.7, 0.55):
    v = room_speeds(room["open_area"], room["aspect"], CFM * frac)[1]
    fw = FAN_W * frac ** 3
    t = fan_setpoint(MAIN_BASE, v, MET_S, CLO_S)
    nets = {s: steady_net_sgd(t, cop, fw) for s, cop in SYSTEMS}
    print(f"  {frac:.0%} speed: {v:.2f} m/s, {fw:4.1f} W, setpoint {t:.1f} °C · "
          + " · ".join(f"{s.split(' (')[0]}: S${n:.0f}/yr" for s, n in nets.items()))
    for s, n in nets.items():
        review.append({"check": "a fan speed", "case": f"{frac:.0%} speed", "air_speed_m_s": round(v, 2),
                       "fan_w": round(fw, 1), "setpoint_C": round(t, 2), "aircon": s, "net_sgd_year": round(n, 2)})

# (b) wall heat-storage depth (cool-down model)
print("\n(b) Wall heat-storage depth (cool-down model, fan + aircon vs aircon alone at arrival)")
DEPTH_USED = MASS_DEPTH
for depth in (0.02, 0.035, 0.05, 0.065, 0.08):
    MASS_DEPTH = depth  # room_mass() reads this module-level value
    times, ta, tm, _, _ = cooldown(room, 25.0, False, ARRIVE, 3.5)
    st, sets = set_series(times, ta, tm, V_STILL)
    mins_alone = minutes_to_comfort(st, sets)
    for s, cop in SYSTEMS:
        _, _, _, ac0, _ = cooldown(room, 25.0, False, ARRIVE, cop)
        _, _, _, ac1, f1 = cooldown(room, FAN_SP, True, ARRIVE, cop)
        save = (ac0 - ac1 - f1) * TARIFF * DAYS_PER_YEAR
        review.append({"check": "b wall depth", "case": f"{depth * 100:.1f} cm", "aircon": s,
                       "net_sgd_year": round(save, 2), "pct_saving": round(100 * (ac0 - ac1 - f1) / ac0, 1),
                       "aircon_alone_minutes_to_comfort": round(mins_alone, 0)})
        print(f"  {depth * 100:4.1f} cm, {s.split(' (')[0]}: saves S${save:.0f}/yr "
              f"({(ac0 - ac1 - f1) / ac0:.0%}); aircon alone comfortable after {mins_alone / 60:.1f} h")
MASS_DEPTH = DEPTH_USED

# (c) the 0.5 m/s sleep cap: ceiling fan (our system) and standing fan (comparison)
print("\n(c) Sleep setting 0.85 m/s (ceiling fan), and a DC standing fan for comparison")
SLEEP_V = 0.85  # m/s at the bed while asleep (design rule from 4 Nov 2026; was a 0.5 m/s cap)
CEIL_CAP_W = FAN_W * (SLEEP_V / room["v_user"]) ** 3  # W, fan law
for check, label, v, fw in (
        ("c ceiling fan", f"ceiling fan at {SLEEP_V} m/s sleep setting, {CEIL_CAP_W:.1f} W", SLEEP_V, CEIL_CAP_W),
        ("c standing fan", "standing fan at 0.5 m/s sleep cap, 4.5 W", 0.5, 4.5),
        ("c standing fan", "standing fan at 1.0 m/s (no cap), 15 W", 1.0, 15.0)):
    t = fan_setpoint(MAIN_BASE, v, MET_S, CLO_S)
    times, ta, tm, _, _ = cooldown(room, t, True, ARRIVE, 3.5, fan_w=fw)
    st, sets = set_series(times, ta, tm, v)
    mins = minutes_to_comfort(st, sets)
    print(f"  {label}: setpoint {t:.1f} °C, awake user comfortable after {mins:.0f} min")
    for s, cop in SYSTEMS:
        steady = steady_net_sgd(t, cop, fw)
        _, _, _, ac0, _ = cooldown(room, 25.0, False, ARRIVE, cop)
        _, _, _, ac1, f1 = cooldown(room, t, True, ARRIVE, cop, fan_w=fw)
        cool = (ac0 - ac1 - f1) * TARIFF * DAYS_PER_YEAR
        print(f"    {s}: steady model S${steady:.0f}/yr · cool-down model S${cool:.0f}/yr "
              f"({(ac0 - ac1 - f1) / ac0:.0%})")
        review.append({"check": check, "case": label, "air_speed_m_s": v, "fan_w": round(fw, 2),
                       "setpoint_C": round(t, 2), "aircon": s, "net_sgd_year": round(steady, 2),
                       "cooldown_net_sgd_year": round(cool, 2), "pct_saving": round(100 * (ac0 - ac1 - f1) / ac0, 1),
                       "minutes_to_comfort": round(mins, 0)})

# (d) humidity and (e) posture
print("\n(d) Humidity and (e) posture: setpoint with the fan for the same SET as 25 °C")
for label, v, fw in (("ceiling fan 1.21 m/s, 25 W", room["v_user"], FAN_W), ("standing fan 0.5 m/s, 4.5 W", 0.5, 4.5)):
    for rh in (60, 70, 80):
        for pos in ("sitting", "standing"):
            if pos == "standing" and rh != 60:
                continue
            t = setpoint_rh(MAIN_BASE, v, MET_S, CLO_S, rh, pos)
            n = steady_net_sgd(t, 3.5, fw)
            print(f"  {label}, {rh}% RH, SET posture {pos}: {t:.1f} °C, net S${n:.0f}/yr (older aircon, steady)")
            review.append({"check": "d humidity / e posture", "case": f"{label}; {rh}% RH; {pos}",
                           "air_speed_m_s": round(v, 2), "fan_w": fw, "setpoint_C": round(t, 2),
                           "aircon": SYSTEMS[0][0], "net_sgd_year": round(n, 2)})

fields = []
for r in review:
    fields += [k for k in r if k not in fields]
with open("review_checks.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(review)
print("\nSaved review_checks.csv")
