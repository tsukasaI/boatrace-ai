"""
Boat Race AI Prediction System - Configuration
"""
from pathlib import Path
from datetime import date

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data collection settings
DATA_CONFIG = {
    # Collection period (2 years + Dec 2025)
    "start_date": date(2023, 1, 1),
    "end_date": date(2025, 12, 30),

    # URL settings (HTTPS - server redirects from HTTP)
    "base_urls": {
        "results": "https://www1.mbrace.or.jp/od2/K/",    # Race results (K = results)
        "programs": "https://www1.mbrace.or.jp/od2/B/",   # Race programs (B = programs)
    },

    # Request interval (seconds) - to reduce server load
    "request_interval": 2,
}

# Stadium codes (all 24 venues)
STADIUM_CODES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "琵琶湖", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

# Racer classes
RACER_CLASSES = ["A1", "A2", "B1", "B2"]

# Model settings
MODEL_CONFIG = {
    # Exacta (2-consecutive) prediction
    "bet_type": "exacta",

    # Expected value threshold
    "expected_value_threshold": 1.0,

    # Feature time decay (days)
    "decay_half_life_days": 90,
}
