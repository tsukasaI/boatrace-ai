"""
競艇AI予測システム - 設定ファイル
"""
from pathlib import Path
from datetime import date

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# データ取得設定
DATA_CONFIG = {
    # 取得期間（2年分）
    "start_date": date(2023, 1, 1),
    "end_date": date(2024, 12, 31),
    
    # URL設定（HTTPS使用 - サーバーがHTTPからリダイレクトする）
    "base_urls": {
        "results": "https://www1.mbrace.or.jp/od2/K/",    # 競走成績（Kはレース結果）
        "programs": "https://www1.mbrace.or.jp/od2/B/",   # 番組表（Bは番組表）
    },
    
    # リクエスト間隔（秒）- サーバー負荷軽減
    "request_interval": 2,
}

# レース場コード（全24場）
STADIUM_CODES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "琵琶湖", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

# 選手クラス
RACER_CLASSES = ["A1", "A2", "B1", "B2"]

# モデル設定
MODEL_CONFIG = {
    # 2連単予測
    "bet_type": "exacta",  # 2連単
    
    # 期待値閾値
    "expected_value_threshold": 1.0,
    
    # 特徴量の時間減衰（日数）
    "decay_half_life_days": 90,
}
