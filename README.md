# Thermal comfort simulations: fan and air conditioner control in a Singapore bedroom

This repository holds the simulation code and results for EE4002D. The model compares two ways of cooling one bedroom. The first is an air conditioner (aircon) left on a fixed 25 °C setting. The second is a system that picks the aircon setting and the fan speed for each activity: sitting, working out and sleeping.

Comfort is measured as a "feels like" temperature called the Standard Effective Temperature (SET). It combines air temperature, humidity, air speed, clothing and activity into one number. The `pythermalcomfort` package calculates it, following the ASHRAE 55 comfort standard. A minute counts as uncomfortable when the SET is more than 0.5 °C from the 25 °C target.

## Main results

The evening runs from 6 pm to 6 am. The aircon is an older unit with a coefficient of performance (COP) of 3.5. A higher COP means less electricity for the same cooling. A "5-tick" unit is a new, more efficient aircon with a COP of 5.5.

| Result | Value | File |
|---|---|---|
| Yearly electricity, fixed 25 °C | S$440 | `22_precooling_workout/sitting_share.csv` (case A) |
| Yearly electricity, activity-based system | S$288 (34% less) | same |
| Uncomfortable minutes, fixed 25 °C and system | 551 and 128 | same |
| Controller power | 8.6 W, S$23.5 a year | `22_precooling_workout/system_power.csv` |
| Saving after the controller's own electricity | 29% (24% with a 5-tick unit) | same |
| Saving if nobody cools the room while sitting | 26% (about 20% after the controller) | `22_precooling_workout/sitting_share.csv` |

Every value comes from the model, and it uses assumed fan and controller power. Nothing here is measured yet.

## Folders

| Folder | Contents |
|---|---|
| `09_standing_fan` | Fan power against saving for a standing fan |
| `10_room_size_vs_ceiling_fan` | Air speed a ceiling fan gives in different room sizes |
| `11_bedroom_fan_aircon` | Whole-night bedroom model, including heat stored in the walls |
| `12_activity_setpoints` | Main evening model, with the setpoint and fan speed for each activity |
| `15_review_runs` | What the camera adds compared with a fixed 26 °C and a fan |
| `16_detection_response` | Comfort while the system waits to confirm an activity |
| `17_singapore_temperature` | Hourly outdoor temperature used by the simulations, from 2025 weather station data |
| `18_savings_levers` | Which control choices give the saving |
| `19_extra_arm_and_occupants` | The fixed 26 °C and fan option, and several people in the room |
| `20_assumption_sensitivity` | The effect of changing each assumed input |
| `21_routine_addon` | Pre-cooling, early shutdown and empty-room shutdown |
| `22_precooling_workout` | The share of the saving from the sitting period, robustness runs and controller power |

The scripts in folders 15 to 22 import the model in `12_activity_setpoints/activity_setpoints.py`.

## Running the code

Python 3.12 is recommended.

```
pip install -r requirements.txt
cd 22_precooling_workout
python system_power.py
```

Each script writes its CSV files and figures next to itself. The weather data is read from `17_singapore_temperature/hourly_temperature.csv`.
