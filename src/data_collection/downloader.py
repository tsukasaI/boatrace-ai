"""
競艇データ取得スクリプト

公式サイトから競走成績と番組表をダウンロード
- 競走成績: http://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh
- 番組表:   http://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh
"""

import sys
import time
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_CONFIG, RAW_DATA_DIR

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BoatraceDataDownloader:
    """競艇公式サイトからデータをダウンロードするクラス"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.base_urls = DATA_CONFIG["base_urls"]
        self.interval = DATA_CONFIG["request_interval"]
        
        # 出力ディレクトリ作成
        (self.output_dir / "results").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "programs").mkdir(parents=True, exist_ok=True)
    
    def _build_url(self, data_type: str, target_date: date) -> str:
        """
        ダウンロードURLを構築
        
        Args:
            data_type: "results" or "programs"
            target_date: 対象日付
            
        Returns:
            ダウンロードURL
        """
        base_url = self.base_urls[data_type]
        yyyymm = target_date.strftime("%Y%m")
        yymmdd = target_date.strftime("%y%m%d")
        
        # K=競走成績, B=番組表
        prefix = "k" if data_type == "results" else "b"
        
        return f"{base_url}{yyyymm}/{prefix}{yymmdd}.lzh"
    
    def _download_file(self, url: str, output_path: Path, max_retries: int = 3) -> bool:
        """
        ファイルをダウンロード（リトライ付き）

        Args:
            url: ダウンロードURL
            output_path: 保存先パス
            max_retries: 最大リトライ回数

        Returns:
            成功した場合True
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60, allow_redirects=True)

                if response.status_code == 200:
                    output_path.write_bytes(response.content)
                    return True
                elif response.status_code == 404:
                    # その日のレースがない場合（年末年始など）- リトライ不要
                    logger.debug(f"No data for {url}")
                    return False
                else:
                    logger.warning(f"HTTP {response.status_code}: {url}")
                    # 5xx系エラーはリトライ
                    if response.status_code >= 500:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** (attempt + 1)  # 2, 4, 8秒
                            logger.info(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                    return False

            except requests.Timeout:
                logger.warning(f"Timeout: {url}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                return False

            except requests.RequestException as e:
                logger.error(f"Download error: {url} - {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                return False

        return False
    
    def download_date_range(
        self, 
        start_date: date, 
        end_date: date, 
        data_types: list = None
    ) -> dict:
        """
        指定期間のデータをダウンロード
        
        Args:
            start_date: 開始日
            end_date: 終了日
            data_types: ["results", "programs"] のリスト
            
        Returns:
            ダウンロード結果の統計
        """
        if data_types is None:
            data_types = ["results", "programs"]
        
        stats = {dt: {"success": 0, "skip": 0, "fail": 0} for dt in data_types}
        
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        
        logger.info(f"Downloading data from {start_date} to {end_date} ({total_days} days)")

        # 日付リストを生成
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        # tqdmでプログレスバー表示
        for current_date in tqdm(dates, desc="Downloading", unit="day"):
            for data_type in data_types:
                output_path = (
                    self.output_dir / data_type /
                    f"{data_type}_{current_date.strftime('%Y%m%d')}.lzh"
                )

                # 既存ファイルはスキップ
                if output_path.exists():
                    stats[data_type]["skip"] += 1
                    continue

                url = self._build_url(data_type, current_date)

                if self._download_file(url, output_path):
                    stats[data_type]["success"] += 1
                else:
                    stats[data_type]["fail"] += 1

                # サーバー負荷軽減のため待機
                time.sleep(self.interval)

        return stats


def main():
    """メイン処理"""
    downloader = BoatraceDataDownloader()
    
    # 設定から期間を取得
    start_date = DATA_CONFIG["start_date"]
    end_date = DATA_CONFIG["end_date"]
    
    # ダウンロード実行
    stats = downloader.download_date_range(start_date, end_date)
    
    # 結果表示
    logger.info("=" * 50)
    logger.info("Download completed!")
    for data_type, counts in stats.items():
        logger.info(
            f"  {data_type}: "
            f"success={counts['success']}, "
            f"skip={counts['skip']}, "
            f"fail={counts['fail']}"
        )


if __name__ == "__main__":
    main()
