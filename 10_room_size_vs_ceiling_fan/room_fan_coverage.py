"""Can one ceiling fan give at least 1 m/s everywhere in a typical HDB room?

The project's critical assumption: a DC ceiling fan moves enough air that a
person feels at least 1 m/s anywhere in the room. This script tests it by
modelling only the air speeds that a typical DC ceiling fan produces in rooms
of different sizes. (Comfort and cost come later.)

Model: Raftery et al. (2019), "Ceiling fans: Predicting indoor air speeds
based on full scale laboratory measurements", Building and Environment.
From 78 full-scale lab tests they fitted simple equations for the LOWEST,
AVERAGE (area-weighted) and HIGHEST air speed in the occupied part of a room.
The equations below are copied from CBE's open-source Ceiling Fan Design Tool
(github.com/CenterForTheBuiltEnvironment/fan-tool, objects.js), which uses them.

Symbols (all SI units):
    Q    = fan airflow on the chosen speed (m³/s)
    D    = fan blade diameter (m)
    S_F  = "fan air speed" = Q ÷ (π D² / 4), the airflow spread over the blade disc (m/s)
    A    = floor area served by the fan (m²), here one fan per room, so A = room area
    r    = √A, the room's characteristic width (m)
    AR   = aspect ratio = long side ÷ short side (≥ 1)
    H_c  = ceiling height (m),  H_b = blade height above the floor (m)
    seated = 1 for a seated person (speed averaged over 0.1, 0.6 and 1.1 m
             above the floor, the ASHRAE 55 heights), 0 for standing
    Dimensionless groups: D/r, H_c/D, D/1.7, H_b/D

    lowest  = S_F × (0.90 D/r − 0.017 H_c/D + 0.11 D/1.7 + 0.024 seated + 0.047) ÷ AR
    average = S_F × (0.99 D/r − 0.060 H_c/D + 0.11 D/1.7 + 0.024 seated + 0.25)
    highest = S_F × (1.3 − 0.18 H_b/D − 0.10 seated)

Reported accuracy over 56 downward-blowing lab tests: median error 0.03 m/s
(lowest), 0.05 m/s (average), 0.12 m/s (highest).

Outputs (saved next to this script):
    room_fan_coverage.png   lowest/average/highest speed vs room size, and the
                            fan airflow needed for 1 m/s everywhere
    room_fan_coverage.csv   the numbers for each HDB room and fan speed setting
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # save outputs next to this script

import csv
import math

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.parse_math"] = False  # treat "$" as plain text, not a maths formula

# =============================================================================
# 1. Inputs (ASSUMPTIONS, stated)
# =============================================================================
CFM_TO_M3S = 0.000471947  # 1 cubic foot per minute = 0.000471947 m³/s

# Typical 52-inch DC-motor ceiling fan
D = 1.32  # m, 52-inch blade span
# High-speed airflow of 52" DC fans on the US ENERGY STAR list ranges from
# about 4,365 to 5,741 CFM (manufacturer ratings). Typical value used: 5,000 CFM.
CFM_LOW, CFM_TYPICAL, CFM_HIGH = 4365, 5000, 5741
FAN_W_HIGH = 25  # W, typical power of such a fan on high (listings: 18-33 W)

# HDB room geometry
H_C = 2.6  # m, floor-to-ceiling height (ASSUMPTION: typical HDB flat)
H_B = 2.3  # m, blade height: fan on a short downrod, blades 0.3 m below the
#           ceiling (0.23 D, above the 0.2 D needed to avoid starving the fan,
#           and above the 2.1 m minimum blade height)
SEATED = 1  # seated person (also the closest case for someone lying in bed)
TARGET = 1.0  # m/s, the project's assumption: at least this everywhere

# Typical HDB rooms (ASSUMPTIONS from typical floor plans, actual flats vary):
# (name, length m, width m)
ROOMS = [
    ("Common bedroom (~10 m²)", 3.3, 3.0),
    ("Master bedroom (~14 m²)", 4.0, 3.4),
    ("Living/dining, 4-room (~25 m²)", 7.0, 3.6),
]
# Fan speed settings: airflow scales in proportion to fan speed (CBE guidebook:
# "airflow and air speed scale linearly" with rotational speed).
SPEEDS = [("High (100%)", 1.0), ("Medium (70%)", 0.7), ("Low (40%)", 0.4)]


# =============================================================================
# 2. The Raftery et al. (2019) model
# =============================================================================
def fan_air_speed(cfm, d=D):
    """S_F (m/s): airflow divided by the area of the blade disc."""
    return cfm * CFM_TO_M3S / (math.pi * d ** 2 / 4)


def room_speeds(cfm, length, width, d=D, h_c=H_C, h_b=H_B, seated=SEATED):
    """(lowest, average, highest) air speed in m/s in a room with one fan in the middle."""
    s_f = fan_air_speed(cfm, d)
    r = math.sqrt(length * width)
    ar = max(length, width) / min(length, width)
    lowest = s_f * (0.90 * d / r - 0.017 * h_c / d + 0.11 * d / 1.7 + 0.024 * seated + 0.047) / ar
    average = s_f * (0.99 * d / r - 0.060 * h_c / d + 0.11 * d / 1.7 + 0.024 * seated + 0.25)
    highest = s_f * (1.3 - 0.18 * h_b / d - 0.10 * seated)
    return lowest, average, highest


def cfm_needed(length, width, target=TARGET):
    """Fan airflow (CFM) for the LOWEST speed in the room to reach `target` m/s.
    Every speed is proportional to airflow, so scale from 1 CFM."""
    return target / room_speeds(1.0, length, width)[0]


def largest_square_room(cfm, target=TARGET):
    """Largest square room (m², one fan) whose lowest speed is still >= target.
    Solved from the 'lowest' equation with AR = 1: D/r = (target/S_F - rest)/0.9."""
    s_f = fan_air_speed(cfm)
    rest = -0.017 * H_C / D + 0.11 * D / 1.7 + 0.024 * SEATED + 0.047
    d_over_r = (target / s_f - rest) / 0.90
    return (D / d_over_r) ** 2 if d_over_r > 0 else float("inf")


# =============================================================================
# 3. Results for the HDB rooms
# =============================================================================
print(f"Typical 52\" DC ceiling fan: D = {D} m, {CFM_TYPICAL} CFM on high "
      f"(S_F = {fan_air_speed(CFM_TYPICAL):.2f} m/s); ceiling {H_C} m, blades at {H_B} m; seated")
print(f"Target: lowest air speed in the room >= {TARGET} m/s\n")
rows = []
for name, L, W in ROOMS:
    print(f"{name}: {L} x {W} m = {L * W:.1f} m², aspect ratio {max(L, W) / min(L, W):.2f}")
    for sp_name, frac in SPEEDS:
        lo, av, hi = room_speeds(CFM_TYPICAL * frac, L, W)
        ok = "MEETS 1 m/s" if lo >= TARGET else "below 1 m/s"
        print(f"  {sp_name:<13} lowest {lo:4.2f} · average {av:4.2f} · highest {hi:4.2f} m/s  -> {ok}")
        rows.append({"room": name, "length_m": L, "width_m": W, "area_m2": round(L * W, 1),
                     "speed_setting": sp_name, "airflow_cfm": round(CFM_TYPICAL * frac),
                     "lowest_m_s": round(lo, 2), "average_m_s": round(av, 2), "highest_m_s": round(hi, 2),
                     "lowest_meets_1_m_s": lo >= TARGET})
    need = cfm_needed(L, W)
    print(f"  Airflow needed for >= {TARGET} m/s everywhere: {need:,.0f} CFM "
          f"({need / CFM_TYPICAL:.0%} of a typical 52\" DC fan on high)\n")

for cfm in (CFM_LOW, CFM_TYPICAL, CFM_HIGH):
    print(f"{cfm:,} CFM on high: largest square room with >= {TARGET} m/s everywhere = "
          f"{largest_square_room(cfm):.1f} m²")

with open("room_fan_coverage.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nSaved room_fan_coverage.csv")

# =============================================================================
# 4. Chart
# =============================================================================
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, LIGHT_BLUE, RED = "#2a78d6", "#86b6ef", "#c2410c"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150)
fig.patch.set_facecolor(SURFACE)

# Left: speeds vs room area (square rooms), typical fan on high, with a band for the fan range
areas = np.linspace(6, 30, 97)
sides = np.sqrt(areas)
for idx, (label, color, lw) in enumerate((("Highest", LIGHT_BLUE, 1.5), ("Average", BLUE, 2.2),
                                          ("Lowest", INK, 2.2))):
    i = 2 - idx  # room_speeds returns (lowest, average, highest)
    typ = [room_speeds(CFM_TYPICAL, s, s)[i] for s in sides]
    lo_band = [room_speeds(CFM_LOW, s, s)[i] for s in sides]
    hi_band = [room_speeds(CFM_HIGH, s, s)[i] for s in sides]
    ax1.fill_between(areas, lo_band, hi_band, color=color, alpha=0.15, lw=0)
    ax1.plot(areas, typ, color=color, lw=lw, label=f"{label} speed in room")
ax1.axhline(TARGET, color=RED, lw=1.2, ls="--")
ax1.text(29.8, TARGET + 0.03, "1 m/s target", color=RED, ha="right", va="bottom", fontsize=8.5)
# Mark the HDB rooms on the "lowest" line (their real shape, not square)
for name, L, W in ROOMS:
    lo = room_speeds(CFM_TYPICAL, L, W)[0]
    ax1.scatter([L * W], [lo], s=42, color=INK, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax1.annotate(f"{name.split(' (')[0]}\n{lo:.2f} m/s", (L * W, lo), textcoords="offset points",
                 xytext=(0, -26), ha="center", fontsize=7.8, color=INK)
style(ax1)
ax1.set_xlim(6, 30)
ax1.set_ylim(0, 2.0)
ax1.set_xlabel("Room floor area served by one fan (m²)", color=INK_2)
ax1.set_ylabel("Air speed at a seated person (m/s)", color=INK_2)
ax1.set_title(f"52\" DC fan on high ({CFM_TYPICAL:,} CFM; band = {CFM_LOW:,}-{CFM_HIGH:,} CFM)",
              loc="left", fontsize=10.5, color=INK)
ax1.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
ax1.text(6.4, 0.06, "Lines = square rooms. Dots = the HDB rooms' real shape:\n"
         "a longer room has a slower far end, so its dot sits below the line.", ha="left", fontsize=7.8,
         color=INK_2)

# Right: airflow needed for >= 1 m/s everywhere vs room area, compared with real fans
need_sq = [cfm_needed(s, s) for s in sides]
ax2.plot(areas, need_sq, color=INK, lw=2.2)
ax2.text(29.7, need_sq[-1] + 350, "needed (square room)", ha="right", fontsize=8.5, color=INK)
ax2.axhspan(CFM_LOW, CFM_HIGH, color=BLUE, alpha=0.12, lw=0)
ax2.text(29.7, (CFM_LOW + CFM_HIGH) / 2, f"Real 52\" DC fans on high ({CFM_LOW:,}-{CFM_HIGH:,} CFM)",
         va="center", ha="right", fontsize=8.5, color=BLUE)
for name, L, W in ROOMS:
    need = cfm_needed(L, W)
    ax2.scatter([L * W], [need], s=42, color=INK, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax2.annotate(f"{name.split(' (')[0]}\n{need:,.0f} CFM", (L * W, need), textcoords="offset points",
                 xytext=(-6, 8), ha="right", fontsize=7.8, color=INK)
style(ax2)
ax2.set_xlim(6, 30)
ax2.set_ylim(0, max(need_sq + [cfm_needed(L, W) for _, L, W in ROOMS]) * 1.12)
ax2.set_xlabel("Room floor area served by one fan (m²)", color=INK_2)
ax2.set_ylabel("Fan airflow for >= 1 m/s everywhere (CFM)", color=INK_2)
ax2.set_title("Airflow one fan needs so the slowest spot still gets 1 m/s", loc="left", fontsize=10.5,
              color=INK)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

fig.suptitle("One 52\" DC fan on high gives 1 m/s everywhere only in rooms up to about 7-10 m²; HDB rooms fall short",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, 0.015, f"Model: Raftery et al. (2019) lab regressions, as used in the CBE Ceiling Fan Design Tool · "
         f"ceiling {H_C} m, blades {H_B} m · seated (0.1/0.6/1.1 m average) · one fan in the room centre · "
         "empty room (furniture lowers speeds)", fontsize=7.8, color=INK_2)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))
fig.savefig("room_fan_coverage.png")
print("Saved room_fan_coverage.png")
