"""Real average outdoor air temperature in Singapore for each hour of the day.

Source: data.gov.sg real-time API "air-temperature" (Meteorological Service Singapore weather stations,
one reading per station per minute). https://api-open.data.gov.sg/v2/real-time/api/air-temperature
Sample: the 5th, 15th and 25th of every month of 2025 (36 days), every station, every minute.
For each clock hour h (0-23) the script averages all readings taken from h:00 to h:59 across all
stations and days, so the result is an island-wide, year-round average (°C) for that hour.

Cross-check: Meteorological Service Singapore 1991-2020 climate normals (Changi) give a 24-hour mean of
27.8 °C, a mean daily maximum of 31.6 °C and a mean daily minimum of 25.0 °C (annual averages of the
monthly values, https://www.weather.gov.sg/climate-climate-of-singapore/).

Output: hourly_temperature.csv (hour, mean_C, n_readings) next to this script.
Project rule: every simulation uses these real hourly averages for its time window.
"""
import csv
import json
import os
import time
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://api-open.data.gov.sg/v2/real-time/api/air-temperature?date={d}"
DAYS = [f"2025-{m:02d}-{d:02d}" for m in range(1, 13) for d in (5, 15, 25)]

sums, counts = defaultdict(float), defaultdict(int)


def get(url):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EE4002D-student-project"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as e:  # rate limit or network blip: wait and retry
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"failed: {url}")


for day in DAYS:
    url, pages = URL.format(d=day), 0
    while url:
        d = get(url)["data"]
        for r in d["readings"]:
            hour = int(r["timestamp"][11:13])                 # local time (+08:00)
            for s in r["data"]:
                if s.get("value") is not None:
                    sums[hour] += float(s["value"]); counts[hour] += 1
        tok = d.get("paginationToken")
        url = URL.format(d=day) + f"&paginationToken={tok}" if tok else None
        pages += 1
    print(day, pages, "pages")

with open(os.path.join(HERE, "hourly_temperature.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["hour", "mean_C", "n_readings"])
    for h in range(24):
        w.writerow([h, round(sums[h] / counts[h], 2), counts[h]])
means = [sums[h] / counts[h] for h in range(24)]
print("24-h mean %.2f, min %.2f at %d h, max %.2f at %d h" % (sum(means) / 24, min(means), means.index(min(means)),
                                                             max(means), means.index(max(means))))
for h in range(24):
    print(h, round(means[h], 2))
