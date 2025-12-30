"""
Pytest fixtures for boatrace-ai tests
"""

import sys
from pathlib import Path
from datetime import date

import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_programs_df():
    """Sample programs entry data"""
    return pd.DataFrame({
        "date": [20240115, 20240115, 20240115, 20240115, 20240115, 20240115],
        "stadium_code": [23, 23, 23, 23, 23, 23],
        "race_no": [1, 1, 1, 1, 1, 1],
        "boat_no": [1, 2, 3, 4, 5, 6],
        "racer_id": [3527, 5036, 5160, 4861, 4876, 4097],
        "racer_name": ["中嶋誠一", "福田翔吾", "藤森陸斗", "田中宏樹", "梅木敬太", "貫地谷直人"],
        "age": [53, 25, 24, 35, 29, 41],
        "branch": ["長崎", "佐賀", "福岡", "福岡", "福岡", "広島"],
        "weight": [51, 52, 54, 53, 53, 55],
        "racer_class": ["A2", "B1", "B1", "B1", "B1", "B1"],
        "national_win_rate": [5.10, 5.09, 4.18, 4.92, 3.66, 4.26],
        "national_in2_rate": [30.40, 32.63, 20.48, 26.03, 16.42, 20.78],
        "local_win_rate": [5.91, 4.33, 2.64, 4.96, 4.84, 4.69],
        "local_in2_rate": [43.18, 33.33, 4.00, 37.04, 31.58, 28.57],
        "motor_no": [55, 51, 27, 54, 26, 18],
        "motor_in2_rate": [15.87, 28.40, 54.17, 24.66, 34.12, 38.37],
        "boat_no_equip": [78, 85, 84, 72, 31, 71],
        "boat_in2_rate": [33.33, 24.72, 37.22, 29.73, 38.78, 35.71],
    })


@pytest.fixture
def sample_results_df():
    """Sample results entry data"""
    return pd.DataFrame({
        "date": [20240115, 20240115, 20240115, 20240115, 20240115, 20240115],
        "stadium_code": [23, 23, 23, 23, 23, 23],
        "race_no": [1, 1, 1, 1, 1, 1],
        "boat_no": [4, 3, 1, 6, 5, 2],
        "racer_id": [4861, 5160, 3527, 4097, 4876, 5036],
        "rank": [1, 2, 3, 4, 5, 6],
        "race_time": ["1.49.6", "1.50.7", "1.51.2", "1.52.9", "1.54.9", "1.57.4"],
        "course": [4, 3, 1, 6, 5, 2],
        "start_timing": [0.05, 0.07, 0.04, 0.09, 0.09, 0.13],
    })


@pytest.fixture
def sample_position_probs():
    """Sample position probability predictions (6 boats × 6 positions)"""
    # Realistic probabilities where boat 1 (1コース) has highest win probability
    probs = np.array([
        [0.45, 0.20, 0.15, 0.10, 0.06, 0.04],  # Boat 1
        [0.15, 0.25, 0.20, 0.18, 0.12, 0.10],  # Boat 2
        [0.12, 0.18, 0.22, 0.20, 0.16, 0.12],  # Boat 3
        [0.10, 0.15, 0.18, 0.22, 0.20, 0.15],  # Boat 4
        [0.10, 0.12, 0.15, 0.18, 0.25, 0.20],  # Boat 5
        [0.08, 0.10, 0.10, 0.12, 0.21, 0.39],  # Boat 6
    ])
    return probs


@pytest.fixture
def sample_odds():
    """Sample exacta odds"""
    return {
        (1, 2): 3.5,
        (1, 3): 5.2,
        (1, 4): 8.0,
        (4, 3): 23.1,
        (4, 1): 15.0,
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary data directory structure"""
    raw_dir = tmp_path / "raw"
    (raw_dir / "results").mkdir(parents=True)
    (raw_dir / "programs").mkdir(parents=True)
    (tmp_path / "processed").mkdir()
    return tmp_path
