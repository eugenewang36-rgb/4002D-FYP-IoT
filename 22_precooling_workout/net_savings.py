"""Net saving after the controller's own electricity (Raspberry Pi, camera, illuminator, ESP32 nodes).

Gross saving = fixed 25 C aircon-only bill minus our system's bill for the 6 pm to 6 am evening (folder 12, ceiling
fan, real hourly Singapore temperatures, 13-minute sensor cycle). The values below are those printed by
21_routine_addon (older aircon: ours S$288, fixed 25 C S$440, 5-tick: ours S$189, fixed S$280).
The controller runs all day, so its energy over a whole year is subtracted from the evening-only saving. That is
conservative: the aircon bill it is compared with covers 6 pm to 6 am only.
Every power value is an ASSUMPTION or a figure calculated in section 2.3.2, to be measured with USB power meters in
Semester 2 (Chapter 3, test B7). Pi 5 measured at 3.3 W idle and 9.8 W under a CPU stress test (report ref [88]).
"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
TARIFF, HOURS = 0.3116, 8760
BILLS = {"Older aircon (COP 3.5)": (440.0, 288.0), "New 5-tick aircon (COP 5.5)": (280.0, 189.0)}  # (fixed 25 C, ours) S$/yr
PSU = 1.15  # 15% power-supply loss


def budget(pi_w, pi_hours=24):
    """Average power (W) over 24 h. The Pi, camera and illuminator run pi_hours a day. The ESP32 boards run 24 h."""
    pi = pi_w * pi_hours / 24
    cam = 0.8 * pi_hours / 24
    ir = 1.5 * min(pi_hours, 12) / 24      # 1.5 W for the dark hours (12 h)
    ctrl = 0.4                              # control node (section 2.3.2/2.3.1)
    sens = 0.54                             # three sensor nodes with sensors (section 2.3.2)
    parts = [("Raspberry Pi 5 running pose detection", pi), ("Camera module (0.8 W)", cam),
             ("Infrared illuminator (1.5 W for 12 dark hours)", ir), ("Control node ESP32", ctrl),
             ("Three sensor nodes with sensors", sens)]
    raw = sum(w for _, w in parts)
    parts.append(("Power-supply loss (15%)", raw * (PSU - 1)))
    return parts, raw * PSU


out = []
for label, pi, hrs in (("Pi averaging 3 W", 3.0, 24), ("central: Pi averaging 5 W", 5.0, 24), ("Pi averaging 8 W", 8.0, 24),
                       ("central, Pi/camera/illuminator on 12 h only", 5.0, 12)):
    parts, tot = budget(pi, hrs)
    cost = tot * HOURS / 1000 * TARIFF
    for ac, (fx, ou) in BILLS.items():
        gross = {"Older aircon (COP 3.5)": 151.0, "New 5-tick aircon (COP 5.5)": 91.0}[ac]  # as in the report (unrounded bills)
        net = gross - cost
        out.append(dict(case=label, controller_w=round(tot, 2), controller_sgd_year=round(cost, 1), aircon=ac,
                        fixed25_sgd=fx, ours_sgd=ou, gross_saving_sgd=gross, gross_pct=round(100 * gross / fx, 1),
                        net_saving_sgd=round(net, 1), net_pct=round(100 * net / fx, 1)))
        print(out[-1])
parts, tot = budget(5.0)
for n, w in parts:
    print(f"  {w:5.2f} W  {n}")
print(f"  total {tot:.2f} W = {tot * HOURS / 1000:.1f} kWh/yr = S${tot * HOURS / 1000 * TARIFF:.1f}/yr")
with open(os.path.join(HERE, "net_savings.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
