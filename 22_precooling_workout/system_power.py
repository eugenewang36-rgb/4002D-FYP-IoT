"""Overall system saving: time-resolved simulation of the controller's own electricity (1-minute steps, 24 h).

Components (power values are ASSUMPTIONS / datasheet figures, to be measured with USB power meters in Semester 2):
  Raspberry Pi 5 : constant average (3, 5 or 8 W, measured 3.3 W idle, 9.8 W stress test, ref [88])
  Camera         : 0.8 W while the Pi runs
  IR illuminator : 1.5 W during the 12 dark hours (18:00-06:00), off by day
  Control node   : 0.4 W, 24 h (ESP32, section 2.3.1)
  Sensor nodes   : 3 nodes, each 13-minute cycle = 3 min awake at 0.6369 W + 10 min deep sleep at 0.0434 W (section 2.3.2)
  PSU loss       : 15% on the total
Overall system saving = (fixed 25 C aircon bill - (our aircon + fan bill + whole-system electricity)) / fixed 25 C bill.
Aircon + fan bills are the folder 12 / 21 values for the 18:00-06:00 evening (older S$440 -> 288, 5-tick S$280 -> 189).
Two accountings are given: ALL DAY (controller runs 24 h, S$ per year, conservative, the report's headline) and
EVENING ONLY (controller energy inside the same 12 h window as the aircon bill, like-for-like).
"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
TARIFF, PSU = 0.3116, 1.15
AWAKE_W, SLEEP_W, CYCLE, AWAKE_MIN = 0.6369132, 0.0434082, 13, 3
BILLS = {"Older aircon (COP 3.5)": (440.0, 288.0), "New 5-tick aircon (COP 5.5)": (280.0, 189.0)}
GROSS = {"Older aircon (COP 3.5)": 151.0, "New 5-tick aircon (COP 5.5)": 91.0}  # unrounded gross savings, as in the report


def simulate(pi_w, pi_hours=24):
    """Return per-component energy (Wh) for a 24 h day, minute by minute, hour 0 = 18:00."""
    e = dict(pi=0.0, camera=0.0, illuminator=0.0, control_node=0.0, sensor_nodes=0.0)
    eve = dict(e)
    for m in range(24 * 60):
        dark = m < 12 * 60                      # 18:00-06:00
        pi_on = (m / 60) < pi_hours if pi_hours < 24 else True
        if pi_hours < 24:
            pi_on = dark and m < pi_hours * 60
        node = 3 * (AWAKE_W if (m % CYCLE) < AWAKE_MIN else SLEEP_W)
        w = dict(pi=pi_w if pi_on else 0.0, camera=0.8 if pi_on else 0.0,
                 illuminator=1.5 if (pi_on and dark) else 0.0, control_node=0.4, sensor_nodes=node)
        for k, v in w.items():
            e[k] += v / 60
            if dark:
                eve[k] += v / 60
    return e, eve


rows = []
for label, pi, hrs in (("Pi averaging 3 W", 3.0, 24), ("Pi averaging 5 W (central)", 5.0, 24), ("Pi averaging 8 W", 8.0, 24),
                       ("Pi, camera, illuminator on 12 h only", 5.0, 12)):
    day, eve = simulate(pi, hrs)
    day_wh, eve_wh = sum(day.values()) * PSU, sum(eve.values()) * PSU
    avg_w = day_wh / 24
    cost_all = day_wh * 365 / 1000 * TARIFF
    cost_eve = eve_wh * 365 / 1000 * TARIFF
    for ac, (fx, ou) in BILLS.items():
        g = GROSS[ac]
        r = dict(case=label, system_avg_w=round(avg_w, 2), system_wh_per_day=round(day_wh, 1), system_wh_evening=round(eve_wh, 1),
                 aircon=ac, fixed25_sgd=fx, ours_aircon_fan_sgd=ou,
                 system_cost_allday_sgd=round(cost_all, 1), overall_saving_allday_pct=round(100 * (g - cost_all) / fx, 1),
                 system_cost_evening_sgd=round(cost_eve, 1), overall_saving_evening_pct=round(100 * (g - cost_eve) / fx, 1))
        rows.append(r)
        print(r)
day, eve = simulate(5.0)
print("central case, Wh per day (x1.15 PSU):", {k: round(v, 1) for k, v in day.items()}, "sensor nodes avg W", round(day['sensor_nodes'] / 24, 3))
with open(os.path.join(HERE, "system_power.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
