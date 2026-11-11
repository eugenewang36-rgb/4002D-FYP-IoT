# Thermal comfort simulations: fan and air conditioner control in a Singapore bedroom

Simulation code and results for EE4002D. The model compares an air conditioner on a fixed 25 °C setting with a system that picks the aircon setting and fan speed for each activity (sitting, workout, sleep). Comfort is measured with the Standard Effective Temperature (SET) from the `pythermalcomfort` package, following ASHRAE 55.

## Main results (evening from 6 pm to 6 am, older aircon with COP 3.5)

| Result | Value | File |
|---|---|---|
| Yearly electricity, fixed 25 °C | S$440 | `22_precooling_workout/sitting_share.csv` (case A) |
| Yearly electricity, activity-based system | S$288 (34% less) | same |
| Uncomfortable minutes, fixed 25 °C and system | 551 and 128 | same |
| Controller power | 8.6 W, S$23.5 a year | `22_precooling_workout/system_power.csv` |
| Saving after the controller's own electricity | 29% (24% with a 5-tick unit) | same |
| Saving if nobody cools the room while sitting | 26% (about 20% after the controller) | `22_precooling_workout/sitting_share.csv` |

All values come from the model and use assumed fan and controller power. None are measured.

## Folders

| Folder | Contents |
|---|---|
| `09_standing_fan` | Fan power against saving for a standing fan |
| `10_room_size_vs_ceiling_fan` | Air speed a ceiling fan gives in different room sizes |
| `11_bedroom_fan_aircon` | Whole-night bedroom model with heat stored in the walls |
| `12_activity_setpoints` | Main evening model, setpoints and fan speeds for each activity |
| `15_review_runs` | What the camera adds compared with a fixed 26 °C and fan |
| `16_detection_response` | Comfort during the time the system waits to confirm an activity |
| `17_singapore_temperature` | Hourly outdoor temperature used by the simulations (2025 weather station data) |
| `18_savings_levers` | Which control choices give the saving |
| `19_extra_arm_and_occupants` | The fixed 26 °C and fan arm, and several people in the room |
| `20_assumption_sensitivity` | Effect of changing each assumed input |
| `21_routine_addon` | Pre-cooling, early shutdown and empty-room shutdown |
| `22_precooling_workout` | Share of the saving from the sitting period, robustness runs, controller power |

Folders 15 to 22 import the model in `12_activity_setpoints/activity_setpoints.py`.

## Running

Python 3.12 is recommended.

```
pip install -r requirements.txt
cd 22_precooling_workout
python system_power.py
```

Each script writes its CSV and figures next to itself. Weather data is read from `17_singapore_temperature/hourly_temperature.csv`.
