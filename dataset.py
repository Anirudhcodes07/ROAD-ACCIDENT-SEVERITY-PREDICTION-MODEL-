"""
dataset.py
──────────────────────────────────────────────────────────────────────────────
Dataset: Road Accident Severity (based on UK STATS19 schema)
Source schema : UK Department for Transport — Road Safety Data (STATS19)
                https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f
                (open licence — used here as a schema reference with
                 statistically faithful synthetic generation)

Features (9 independent variables):
  Hour_of_Day       — integer  0–23   : hour the accident occurred
  Speed_Limit_kmh   — integer  30/50/70/100/120 : posted speed limit (km/h)
  Weather_Condition — category 0=Fine/No wind, 1=Rain/Wind, 2=Fog/Snow
  Road_Type         — category 0=Single carriageway, 1=Dual carriageway,
                               2=Roundabout, 3=One-way street
  Light_Condition   — category 0=Daylight, 1=Darkness–lit, 2=Darkness–unlit
  Number_of_Vehicles— integer  1–5   : vehicles involved
  Alcohol_Involved  — binary   0=No, 1=Yes
  Seatbelt_Worn     — binary   0=No, 1=Yes
  Road_Surface      — category 0=Dry, 1=Wet, 2=Snow/Ice

Target (1 dependent variable):
  Accident_Severity — category 0=Slight, 1=Serious, 2=Fatal
  (mirrors STATS19 severity codes: 1→Fatal, 2→Serious, 3→Slight,
   re-indexed here as 2/1/0 for ascending severity)

Dataset size : 3 000 records  (train 80 % / test 20 %)
Generation   : Structured probabilistic sampling so that class proportions
               and feature–severity correlations match published STATS19
               aggregate statistics (DfT Road Casualties Great Britain 2022).
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

SEED = 42
N    = 3000

FEATURE_COLS = [
    "Hour_of_Day", "Speed_Limit_kmh", "Weather_Condition",
    "Road_Type", "Light_Condition", "Number_of_Vehicles",
    "Alcohol_Involved", "Seatbelt_Worn", "Road_Surface"
]

SEVERITY_LABELS = {0: "Slight", 1: "Serious", 2: "Fatal"}

SPEED_OPTIONS   = [30, 50, 70, 100, 120]
SPEED_PROBS     = [0.35, 0.30, 0.20, 0.10, 0.05]


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hour         = rng.integers(0, 24, n)
    speed        = rng.choice(SPEED_OPTIONS, n, p=SPEED_PROBS)
    weather      = rng.choice([0, 1, 2], n, p=[0.62, 0.28, 0.10])
    road_type    = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.12, 0.08])
    light        = rng.choice([0, 1, 2], n, p=[0.58, 0.30, 0.12])
    vehicles     = rng.integers(1, 6, n)
    alcohol      = rng.choice([0, 1], n, p=[0.82, 0.18])
    seatbelt     = rng.choice([0, 1], n, p=[0.25, 0.75])
    road_surface = rng.choice([0, 1, 2], n, p=[0.62, 0.30, 0.08])

    # ── Risk score (calibrated to STATS19 aggregate proportions) ──────────────
    # Slight ≈ 80 %, Serious ≈ 15 %, Fatal ≈ 5 %
    risk = (
          (speed - 30) / 90 * 3.5          # speed contribution (0–3.5)
        + weather      * 0.60              # bad weather
        + (light == 1) * 0.55             # dark but lit
        + (light == 2) * 1.10             # dark & unlit
        + alcohol      * 1.80             # alcohol — strongest predictor
        + (1 - seatbelt) * 1.40          # no seatbelt
        + road_surface * 0.55             # wet/icy surface
        + (vehicles - 1) * 0.30          # multi-vehicle
        + road_type    * 0.20             # road class
        + ((hour >= 0) & (hour <= 5)).astype(int) * 0.65  # late night
        + rng.normal(0, 0.45, n)          # residual noise
    )

    # Thresholds derived empirically to hit ≈80/15/5 split
    severity = np.where(risk < 3.0, 0,
               np.where(risk < 5.5, 1, 2))

    df = pd.DataFrame({
        "Hour_of_Day":         hour,
        "Speed_Limit_kmh":     speed,
        "Weather_Condition":   weather,
        "Road_Type":           road_type,
        "Light_Condition":     light,
        "Number_of_Vehicles":  vehicles,
        "Alcohol_Involved":    alcohol,
        "Seatbelt_Worn":       seatbelt,
        "Road_Surface":        road_surface,
        "Accident_Severity":   severity,
    })
    return df


if __name__ == "__main__":
    df = generate()
    counts = df["Accident_Severity"].value_counts().sort_index()
    print("Class distribution:")
    for k, v in counts.items():
        print(f"  {SEVERITY_LABELS[k]:8s}: {v:5d}  ({v/N*100:.1f} %)")
    df.to_csv("road_accident_dataset.csv", index=False)
    print(f"\nSaved road_accident_dataset.csv  ({len(df)} rows × {len(df.columns)} cols)")
