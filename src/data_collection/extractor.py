"""
LZHファイル解凍スクリプト

ダウンロードしたLZH形式のファイルを解凍してテキストファイルに変換
"""

import sys
import logging
from pathlib import Path

from tqdm import tqdm

# lhaplusなどのLZH解凍ライブラリ
# Python 3.x では lhafile を使用
try:
    import lhafile
    HAS_LHAFILE = True
except ImportError:
    HAS_LHAFILE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LzhExtractor:
    """LZHファイルを解凍するクラス"""
    
    def __init__(self, input_dir: Path = None, output_dir: Path = None):
        self.input_dir = input_dir or RAW_DATA_DIR
        self.output_dir = output_dir or RAW_DATA_DIR
        
        if not HAS_LHAFILE:
            logger.warning(
                "lhafile module not found. "
                "Install with: pip install lhafile"
            )
    
    def extract_file(self, lzh_path: Path) -> Path | None:
        """
        単一のLZHファイルを解凍

        Args:
            lzh_path: LZHファイルのパス

        Returns:
            解凍されたメインファイルのパス（失敗時はNone）
        """
        if not HAS_LHAFILE:
            return None

        try:
            # 出力ディレクトリを入力と同じ構造で作成
            relative_path = lzh_path.relative_to(self.input_dir)
            output_subdir = self.output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            # LZHファイルを開く
            archive = lhafile.Lhafile(str(lzh_path))
            file_list = archive.infolist()

            if len(file_list) > 1:
                logger.warning(f"Archive contains {len(file_list)} files: {lzh_path}")

            main_output_path = None

            # アーカイブ内のすべてのファイルを展開
            for info in file_list:
                content = archive.read(info.filename)

                # 出力ファイルパス（lzh_stem を使って命名）
                output_path = output_subdir / lzh_path.stem
                output_path = output_path.with_suffix(".txt")
                output_path.write_bytes(content)

                logger.debug(f"Extracted: {output_path.name}")

                if main_output_path is None:
                    main_output_path = output_path

            return main_output_path

        except (OSError, lhafile.LhaError) as e:
            logger.error(f"Extract error: {lzh_path} - {e}")
            return None
    
    def extract_all(self, data_types: list = None) -> dict:
        """
        すべてのLZHファイルを解凍
        
        Args:
            data_types: ["results", "programs"] のリスト
            
        Returns:
            解凍結果の統計
        """
        if data_types is None:
            data_types = ["results", "programs"]
        
        stats = {dt: {"success": 0, "skip": 0, "fail": 0} for dt in data_types}
        
        for data_type in data_types:
            input_subdir = self.input_dir / data_type
            
            if not input_subdir.exists():
                logger.warning(f"Directory not found: {input_subdir}")
                continue
            
            lzh_files = list(input_subdir.glob("*.lzh"))
            logger.info(f"Found {len(lzh_files)} LZH files in {data_type}")

            for lzh_path in tqdm(lzh_files, desc=f"Extracting {data_type}", unit="file"):
                # 既存ファイルはスキップ
                txt_path = lzh_path.with_suffix(".txt")
                if txt_path.exists():
                    stats[data_type]["skip"] += 1
                    continue
                
                if self.extract_file(lzh_path):
                    stats[data_type]["success"] += 1
                else:
                    stats[data_type]["fail"] += 1
        
        return stats


def main():
    """メイン処理"""
    if not HAS_LHAFILE:
        logger.error("Please install lhafile: pip install lhafile")
        sys.exit(1)
    
    extractor = LzhExtractor()
    stats = extractor.extract_all()
    
    logger.info("=" * 50)
    logger.info("Extraction completed!")
    for data_type, counts in stats.items():
        logger.info(
            f"  {data_type}: "
            f"success={counts['success']}, "
            f"skip={counts['skip']}, "
            f"fail={counts['fail']}"
        )


if __name__ == "__main__":
    main()
