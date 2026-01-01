#!/usr/bin/env python3
"""Debug script to compare Python vs Rust feature values."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.features import FeatureEngineering, get_feature_columns


def main():
    """Output feature values for a sample race."""
    # Load data
    programs_df = pd.read_csv(PROCESSED_DATA_DIR / "programs_entries.csv")
    results_df = pd.read_csv(PROCESSED_DATA_DIR / "results_entries.csv")
    races_df = pd.read_csv(PROCESSED_DATA_DIR / "programs_races.csv")

    # Pick a sample race (2024-07-15, stadium 23, race 1)
    sample_date = 20240715
    sample_stadium = 23
    sample_race = 1

    # Filter to sample race
    race_programs = programs_df[
        (programs_df["date"] == sample_date) &
        (programs_df["stadium_code"] == sample_stadium) &
        (programs_df["race_no"] == sample_race)
    ].copy()

    if race_programs.empty:
        # Try to find any race in the test period
        test_programs = programs_df[programs_df["date"] > 20240630].head(6)
        if test_programs.empty:
            print("No test data found")
            return
        race_programs = test_programs
        sample_date = race_programs["date"].iloc[0]
        sample_stadium = race_programs["stadium_code"].iloc[0]
        sample_race = race_programs["race_no"].iloc[0]

    print(f"Sample race: date={sample_date}, stadium={sample_stadium}, race={sample_race}")
    print(f"Boats: {race_programs['boat_no'].tolist()}")
    print()

    # Merge with results for exhibition_time
    race_results = results_df[
        (results_df["date"] == sample_date) &
        (results_df["stadium_code"] == sample_stadium) &
        (results_df["race_no"] == sample_race)
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
        (races_df["date"] == sample_date) &
        (races_df["stadium_code"] == sample_stadium) &
        (races_df["race_no"] == sample_race)
    ]
    if not race_info.empty:
        race_programs["race_type"] = race_info["race_type"].iloc[0]

    # Create features
    fe = FeatureEngineering()
    features = fe.create_all_features(race_programs, results_df, include_historical=True)

    # Print feature values for each boat
    feature_cols = get_feature_columns()

    print("=" * 80)
    print("FEATURE VALUES BY BOAT")
    print("=" * 80)

    for boat_no in sorted(race_programs["boat_no"].unique()):
        boat_features = features[features["boat_no"] == boat_no]
        if boat_features.empty:
            continue

        row = boat_features.iloc[0]
        print(f"\n--- Boat {boat_no} ---")
        print(f"racer_id: {row.get('racer_id', 'N/A')}")

        if "branch" in race_programs.columns:
            branch = race_programs[race_programs["boat_no"] == boat_no]["branch"].iloc[0]
            print(f"branch: {branch}")

        if "racer_class" in race_programs.columns:
            racer_class = race_programs[race_programs["boat_no"] == boat_no]["racer_class"].iloc[0]
            print(f"racer_class: {racer_class}")

        print()
        for col in feature_cols:
            if col in features.columns:
                val = row[col]
                if pd.isna(val):
                    print(f"  {col}: NaN")
                elif isinstance(val, float):
                    print(f"  {col}: {val:.6f}")
                else:
                    print(f"  {col}: {val}")
            else:
                print(f"  {col}: [NOT FOUND]")

    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    for col in feature_cols:
        if col in features.columns:
            print(f"{col}:")
            print(f"  min={features[col].min():.4f}, max={features[col].max():.4f}, "
                  f"mean={features[col].mean():.4f}, std={features[col].std():.4f}")


if __name__ == "__main__":
    main()
