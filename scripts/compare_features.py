#!/usr/bin/env python3
"""Compare Python vs Rust feature values for a specific race."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.features import FeatureEngineering, get_feature_columns


def get_python_features(date: int, stadium: int, race: int) -> dict:
    """Compute features using Python."""
    programs_df = pd.read_csv(PROCESSED_DATA_DIR / "programs_entries.csv")
    results_df = pd.read_csv(PROCESSED_DATA_DIR / "results_entries.csv")
    races_df = pd.read_csv(PROCESSED_DATA_DIR / "programs_races.csv")

    # Filter to sample race
    race_programs = programs_df[
        (programs_df["date"] == date) &
        (programs_df["stadium_code"] == stadium) &
        (programs_df["race_no"] == race)
    ].copy()

    if race_programs.empty:
        return {"error": "Race not found in programs"}

    # Merge with results for exhibition_time
    race_results = results_df[
        (results_df["date"] == date) &
        (results_df["stadium_code"] == stadium) &
        (results_df["race_no"] == race)
    ]

    if not race_results.empty:
        merge_keys = ["date", "stadium_code", "race_no", "boat_no"]
        race_programs = race_programs.merge(
            race_results[merge_keys + ["exhibition_time", "course", "rank"]],
            on=merge_keys,
            how="left"
        )

    # Merge race_type
    race_info = races_df[
        (races_df["date"] == date) &
        (races_df["stadium_code"] == stadium) &
        (races_df["race_no"] == race)
    ]
    if not race_info.empty:
        race_programs["race_type"] = race_info["race_type"].iloc[0]

    # Create features
    fe = FeatureEngineering()
    features = fe.create_all_features(race_programs, results_df, include_historical=True)

    # Get feature columns
    feature_cols = get_feature_columns()

    result = {"race_type": race_info["race_type"].iloc[0] if not race_info.empty else None}
    for boat_no in sorted(race_programs["boat_no"].unique()):
        boat_features = features[features["boat_no"] == boat_no]
        if boat_features.empty:
            continue

        row = boat_features.iloc[0]
        boat_data = {}
        for col in feature_cols:
            if col in features.columns:
                val = row[col]
                if pd.isna(val):
                    boat_data[col] = None
                else:
                    boat_data[col] = float(val)
            else:
                boat_data[col] = None
        result[f"boat_{boat_no}"] = boat_data

    return result


def main():
    # Sample race with a semi-final (準優勝戦)
    # Find a race with interesting race_type
    races_df = pd.read_csv(PROCESSED_DATA_DIR / "programs_races.csv")

    # Find a 準優勝戦 in test period
    semi_finals = races_df[
        (races_df["date"] > 20240630) &
        (races_df["race_type"].str.contains("準優", na=False))
    ].head(1)

    if semi_finals.empty:
        # Fall back to any race
        sample = races_df[races_df["date"] > 20240630].head(1)
    else:
        sample = semi_finals

    date = int(sample["date"].iloc[0])
    stadium = int(sample["stadium_code"].iloc[0])
    race = int(sample["race_no"].iloc[0])
    race_type = sample["race_type"].iloc[0]

    print(f"Sample race: date={date}, stadium={stadium}, race={race}")
    print(f"race_type: {race_type}")
    print()

    # Get Python features
    py_features = get_python_features(date, stadium, race)

    if "error" in py_features:
        print(f"Error: {py_features['error']}")
        return

    # Output key features for comparison
    feature_cols = get_feature_columns()

    print("=" * 80)
    print("PYTHON FEATURE VALUES (Boat 1)")
    print("=" * 80)

    boat1 = py_features.get("boat_1", {})
    for col in feature_cols:
        val = boat1.get(col)
        if val is not None:
            print(f"  {col}: {val:.6f}")
        else:
            print(f"  {col}: None")

    # Focus on race context
    print()
    print("=" * 80)
    print("RACE CONTEXT FEATURES (all boats)")
    print("=" * 80)
    print(f"race_type: {py_features.get('race_type')}")
    for boat_no in range(1, 7):
        boat_key = f"boat_{boat_no}"
        if boat_key in py_features:
            boat = py_features[boat_key]
            race_grade = boat.get("race_grade")
            is_final = boat.get("is_final")
            print(f"  Boat {boat_no}: race_grade={race_grade}, is_final={is_final}")


if __name__ == "__main__":
    main()
