"""
競艇データパーサー

固定長テキストファイルを解析してCSVに変換
"""

import re
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Generator

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, STADIUM_CODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RaceInfo:
    """レース情報"""
    date: str
    stadium_code: int
    stadium_name: str
    race_no: int
    race_type: str
    distance: int
    title: str = ""
    day: int = 0  # 開催何日目


@dataclass
class RacerEntry:
    """選手エントリー情報（番組表）"""
    boat_no: int          # 艇番（1-6）
    racer_id: int         # 登録番号
    racer_name: str       # 選手名
    age: int              # 年齢
    branch: str           # 支部
    weight: int           # 体重
    racer_class: str      # 級別（A1, A2, B1, B2）
    
    # 勝率
    national_win_rate: float   # 全国勝率
    national_in2_rate: float   # 全国2連率
    local_win_rate: float      # 当地勝率
    local_in2_rate: float      # 当地2連率
    
    # モーター・ボート
    motor_no: int              # モーター番号
    motor_in2_rate: float      # モーター2連率
    boat_no_equip: int         # ボート番号
    boat_in2_rate: float       # ボート2連率


@dataclass
class RaceResult:
    """レース結果"""
    boat_no: int          # 艇番
    racer_id: int         # 登録番号
    rank: int             # 着順（0=失格等）
    race_time: str        # レースタイム
    course: int           # 進入コース
    start_timing: float   # スタートタイミング


@dataclass
class RacePayouts:
    """レース払戻金"""
    date: str
    stadium_code: int
    race_no: int
    win: dict              # 単勝 {boat: payout}
    place: dict            # 複勝 {boat: payout}
    exacta: dict           # 2連単 {(1st, 2nd): payout}
    quinella: dict         # 2連複 {(a, b): payout}
    wide: dict             # ワイド {(a, b): payout}
    trifecta: dict         # 3連単 {(1st, 2nd, 3rd): payout}
    trio: dict             # 3連複 {(a, b, c): payout}


class ProgramParser:
    """番組表パーサー"""
    
    # レース場識別パターン（例: "22BBGN" → 福岡）
    STADIUM_PATTERN = re.compile(r'^(\d{2})BBGN')
    
    # レース番号パターン（例: "１Ｒ 予選"）
    RACE_PATTERN = re.compile(r'[　\s]*([０-９\d]+)Ｒ\s*(.*?)\s*Ｈ(\d+)')
    
    # 選手行パターン
    RACER_PATTERN = re.compile(
        r'^(\d)\s+'           # 艇番
        r'(\d{4})'            # 登録番号
        r'(.{4})'             # 選手名（4文字固定）
        r'(\d{2})'            # 年齢
        r'(.{2})'             # 支部
        r'(\d{2})'            # 体重
        r'([AB][12])'         # 級別
        r'\s*(\d+\.\d+)'      # 全国勝率
        r'\s*(\d+\.\d+)'      # 全国2連率
        r'\s*(\d+\.\d+)'      # 当地勝率
        r'\s*(\d+\.\d+)'      # 当地2連率
        r'\s*(\d+)'           # モーター番号
        r'\s*(\d+\.\d+)'      # モーター2連率
        r'\s*(\d+)'           # ボート番号
        r'\s*(\d+\.\d+)'      # ボート2連率
    )
    
    def __init__(self, encoding: str = "cp932"):
        self.encoding = encoding
    
    def _normalize_number(self, text: str) -> str:
        """全角数字を半角に変換"""
        trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
        return text.translate(trans_table)
    
    def parse_file(self, file_path: Path) -> Generator[tuple[RaceInfo, list[RacerEntry]], None, None]:
        """
        番組表ファイルをパース
        
        Yields:
            (RaceInfo, list[RacerEntry]) のタプル
        """
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        
        lines = content.split("\n")
        
        # ファイル名から日付を取得（programs_YYYYMMDD.txt）
        date_str = file_path.stem.split("_")[-1]
        
        current_stadium = None
        current_race = None
        current_racers = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # レース場識別
            stadium_match = self.STADIUM_PATTERN.match(line)
            if stadium_match:
                stadium_code = int(stadium_match.group(1))
                current_stadium = {
                    "code": stadium_code,
                    "name": STADIUM_CODES.get(stadium_code, "不明")
                }
                i += 1
                continue
            
            # レース情報
            race_match = self.RACE_PATTERN.search(line)
            if race_match and current_stadium:
                # 前のレースがあれば出力
                if current_race and current_racers:
                    yield current_race, current_racers
                
                race_no = int(self._normalize_number(race_match.group(1)))
                race_type = race_match.group(2).strip()
                distance = int(race_match.group(3))
                
                current_race = RaceInfo(
                    date=date_str,
                    stadium_code=current_stadium["code"],
                    stadium_name=current_stadium["name"],
                    race_no=race_no,
                    race_type=race_type,
                    distance=distance,
                )
                current_racers = []
                i += 1
                continue
            
            # 選手情報（簡易パース）
            if current_race and line.strip() and line[0].isdigit():
                try:
                    racer = self._parse_racer_line(line)
                    if racer:
                        current_racers.append(racer)
                except Exception as e:
                    logger.debug(f"Parse error at line {i}: {e}")
            
            i += 1
        
        # 最後のレース
        if current_race and current_racers:
            yield current_race, current_racers
    
    def _parse_racer_line(self, line: str) -> RacerEntry | None:
        """選手行をパース"""
        # まず正規表現でパースを試みる
        match = self.RACER_PATTERN.match(line)
        if match:
            try:
                return RacerEntry(
                    boat_no=int(match.group(1)),
                    racer_id=int(match.group(2)),
                    racer_name=match.group(3).strip(),
                    age=int(match.group(4)),
                    branch=match.group(5).strip(),
                    weight=int(match.group(6)),
                    racer_class=match.group(7),
                    national_win_rate=float(match.group(8)),
                    national_in2_rate=float(match.group(9)),
                    local_win_rate=float(match.group(10)),
                    local_in2_rate=float(match.group(11)),
                    motor_no=int(match.group(12)),
                    motor_in2_rate=float(match.group(13)),
                    boat_no_equip=int(match.group(14)),
                    boat_in2_rate=float(match.group(15)),
                )
            except (ValueError, IndexError) as e:
                logger.debug(f"Regex parse error: {e}")

        # フォールバック: 固定位置パース
        # Format: "1 3527後藤浩之53滋賀51A2 5.10 30.40..."
        try:
            if len(line) < 20:
                return None

            boat_no = int(line[0])
            racer_id = int(line[2:6])
            racer_name = line[6:10].strip()
            age = int(line[10:12])
            branch = line[12:14].strip()
            weight = int(line[14:16])
            racer_class = line[16:18]

            # 残りは空白区切りの数値
            remaining = line[18:].split()
            if len(remaining) < 8:
                return None

            return RacerEntry(
                boat_no=boat_no,
                racer_id=racer_id,
                racer_name=racer_name,
                age=age,
                branch=branch,
                weight=weight,
                racer_class=racer_class,
                national_win_rate=float(remaining[0]),
                national_in2_rate=float(remaining[1]),
                local_win_rate=float(remaining[2]),
                local_in2_rate=float(remaining[3]),
                motor_no=int(remaining[4]),
                motor_in2_rate=float(remaining[5]),
                boat_no_equip=int(remaining[6]),
                boat_in2_rate=float(remaining[7]),
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Fallback parse error: {e}")

        return None


class ResultParser:
    """競走成績パーサー"""

    STADIUM_PATTERN = re.compile(r'^(\d{2})KBGN')
    # 結果ファイルでは半角数字が使われる: "   1R       朝１戦予選　    H1800m"
    RACE_PATTERN = re.compile(r'^\s*(\d+)R\s+(.*?)\s+H(\d+)')

    # 結果行パターン: "  01  4 4861 田... 54   72  6.80   4    0.05     1.49.6"
    RESULT_LINE_PATTERN = re.compile(
        r'^\s*(\d{2})\s+'     # 着順 (01-06)
        r'(\d)\s+'            # 艇番 (1-6)
        r'(\d{4})'            # 登録番号
    )

    def __init__(self, encoding: str = "cp932"):
        self.encoding = encoding

    def _normalize_number(self, text: str) -> str:
        """全角数字を半角に変換"""
        trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
        return text.translate(trans_table)

    def _parse_result_line(self, line: str) -> RaceResult | None:
        """結果行をパース"""
        match = self.RESULT_LINE_PATTERN.match(line)
        if not match:
            return None

        try:
            rank = int(match.group(1))
            boat_no = int(match.group(2))
            racer_id = int(match.group(3))

            # 残りの部分から数値を抽出
            # Format: ... 54   72  6.80   4    0.05     1.49.6
            remaining = line[match.end():]

            # 空白で分割して数値を取得
            parts = remaining.split()

            # 後ろから探す: レースタイム, ST, 進入コース, 展示タイム, 展示ST, 展示番号
            # 最低限必要なのは: 進入コース, ST, レースタイム
            course = 0
            start_timing = 0.0
            race_time = ""

            if len(parts) >= 3:
                # 最後の値がレースタイム (1.49.6 形式)
                race_time = parts[-1]

                # その前がSTタイミング
                try:
                    start_timing = float(parts[-2])
                except ValueError:
                    pass

                # その前が進入コース
                try:
                    course = int(parts[-3])
                except ValueError:
                    pass

            return RaceResult(
                boat_no=boat_no,
                racer_id=racer_id,
                rank=rank,
                race_time=race_time,
                course=course,
                start_timing=start_timing,
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Result parse error: {e}")
            return None

    def parse_file(self, file_path: Path) -> Generator[tuple[RaceInfo, list[RaceResult]], None, None]:
        """競走成績ファイルをパース"""
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        lines = content.split("\n")
        date_str = file_path.stem.split("_")[-1]

        current_stadium = None
        current_race = None
        current_results = []
        in_results_section = False

        for line in lines:
            # レース場識別
            stadium_match = self.STADIUM_PATTERN.match(line)
            if stadium_match:
                stadium_code = int(stadium_match.group(1))
                current_stadium = {
                    "code": stadium_code,
                    "name": STADIUM_CODES.get(stadium_code, "不明")
                }
                continue

            # レース情報
            race_match = self.RACE_PATTERN.search(line)
            if race_match and current_stadium:
                # 前のレースがあれば出力
                if current_race and current_results:
                    yield current_race, current_results

                race_no = int(self._normalize_number(race_match.group(1)))
                race_type = race_match.group(2).strip()
                distance = int(race_match.group(3))

                current_race = RaceInfo(
                    date=date_str,
                    stadium_code=current_stadium["code"],
                    stadium_name=current_stadium["name"],
                    race_no=race_no,
                    race_type=race_type,
                    distance=distance,
                )
                current_results = []
                in_results_section = False
                continue

            # 区切り線でセクション開始
            if line.startswith("---") and current_race:
                in_results_section = True
                continue

            # 結果行のパース
            if in_results_section and current_race and line.strip():
                result = self._parse_result_line(line)
                if result:
                    current_results.append(result)
                elif line.strip().startswith("単勝") or line.strip().startswith("複勝"):
                    # 払戻金セクションに入ったら終了
                    in_results_section = False

        # 最後のレース
        if current_race and current_results:
            yield current_race, current_results


class PayoutParser:
    """払戻金パーサー"""

    STADIUM_PATTERN = re.compile(r'^(\d{2})KBGN')
    RACE_PATTERN = re.compile(r'^\s*(\d+)R\s+')

    # 払戻金パターン
    WIN_PATTERN = re.compile(r'単勝\s+(\d)\s+(\d+)')
    PLACE_PATTERN = re.compile(r'複勝\s+(\d)\s+(\d+)')
    EXACTA_PATTERN = re.compile(r'２連単\s+(\d)-(\d)\s+(\d+)')
    QUINELLA_PATTERN = re.compile(r'２連複\s+(\d)-(\d)\s+(\d+)')
    WIDE_PATTERN = re.compile(r'(?:ワイド|ﾜｲﾄﾞ)\s+(\d)-(\d)\s+(\d+)')
    TRIFECTA_PATTERN = re.compile(r'３連単\s+(\d)-(\d)-(\d)\s+(\d+)')
    TRIO_PATTERN = re.compile(r'３連複\s+(\d)-(\d)-(\d)\s+(\d+)')

    def __init__(self, encoding: str = "cp932"):
        self.encoding = encoding

    def parse_file(self, file_path: Path) -> Generator[RacePayouts, None, None]:
        """払戻金ファイルをパース"""
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        lines = content.split("\n")
        date_str = file_path.stem.split("_")[-1]

        current_stadium = None
        current_race_no = None
        current_payouts = self._empty_payouts()
        in_payout_section = False

        for line in lines:
            # レース場識別
            stadium_match = self.STADIUM_PATTERN.match(line)
            if stadium_match:
                current_stadium = int(stadium_match.group(1))
                continue

            # レース番号
            race_match = self.RACE_PATTERN.match(line)
            if race_match and current_stadium:
                # 前のレースがあれば出力
                if current_race_no and self._has_payouts(current_payouts):
                    yield RacePayouts(
                        date=date_str,
                        stadium_code=current_stadium,
                        race_no=current_race_no,
                        **current_payouts,
                    )

                current_race_no = int(race_match.group(1))
                current_payouts = self._empty_payouts()
                in_payout_section = False
                continue

            # 払戻金セクション検出
            if "単勝" in line or "２連単" in line or "３連単" in line:
                in_payout_section = True

            # 払戻金パース
            if in_payout_section and current_race_no:
                self._parse_payout_line(line, current_payouts)

        # 最後のレース
        if current_race_no and current_stadium and self._has_payouts(current_payouts):
            yield RacePayouts(
                date=date_str,
                stadium_code=current_stadium,
                race_no=current_race_no,
                **current_payouts,
            )

    def _empty_payouts(self) -> dict:
        """空の払戻金辞書を返す"""
        return {
            "win": {},
            "place": {},
            "exacta": {},
            "quinella": {},
            "wide": {},
            "trifecta": {},
            "trio": {},
        }

    def _has_payouts(self, payouts: dict) -> bool:
        """払戻金データがあるか確認"""
        return any(len(v) > 0 for v in payouts.values())

    def _parse_payout_line(self, line: str, payouts: dict) -> None:
        """払戻金行をパース"""
        # 単勝
        for match in self.WIN_PATTERN.finditer(line):
            boat = int(match.group(1))
            payout = int(match.group(2))
            payouts["win"][boat] = payout

        # 複勝
        for match in self.PLACE_PATTERN.finditer(line):
            boat = int(match.group(1))
            payout = int(match.group(2))
            payouts["place"][boat] = payout

        # 2連単
        for match in self.EXACTA_PATTERN.finditer(line):
            first = int(match.group(1))
            second = int(match.group(2))
            payout = int(match.group(3))
            payouts["exacta"][(first, second)] = payout

        # 2連複
        for match in self.QUINELLA_PATTERN.finditer(line):
            a = int(match.group(1))
            b = int(match.group(2))
            payout = int(match.group(3))
            payouts["quinella"][tuple(sorted([a, b]))] = payout

        # ワイド
        for match in self.WIDE_PATTERN.finditer(line):
            a = int(match.group(1))
            b = int(match.group(2))
            payout = int(match.group(3))
            payouts["wide"][tuple(sorted([a, b]))] = payout

        # 3連単
        for match in self.TRIFECTA_PATTERN.finditer(line):
            first = int(match.group(1))
            second = int(match.group(2))
            third = int(match.group(3))
            payout = int(match.group(4))
            payouts["trifecta"][(first, second, third)] = payout

        # 3連複
        for match in self.TRIO_PATTERN.finditer(line):
            a = int(match.group(1))
            b = int(match.group(2))
            c = int(match.group(3))
            payout = int(match.group(4))
            payouts["trio"][tuple(sorted([a, b, c]))] = payout


def convert_to_csv(input_dir: Path, output_dir: Path, data_type: str = "programs"):
    """
    テキストファイルをCSVに変換
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        data_type: "programs" or "results"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if data_type == "programs":
        parser = ProgramParser()
    else:
        parser = ResultParser()
    
    all_races = []
    all_entries = []
    
    txt_files = list((input_dir / data_type).glob("*.txt"))
    logger.info(f"Processing {len(txt_files)} {data_type} files")

    for txt_path in tqdm(txt_files, desc=f"Parsing {data_type}", unit="file"):
        for race_info, entries in parser.parse_file(txt_path):
            race_dict = {
                "date": race_info.date,
                "stadium_code": race_info.stadium_code,
                "stadium_name": race_info.stadium_name,
                "race_no": race_info.race_no,
                "race_type": race_info.race_type,
                "distance": race_info.distance,
            }
            all_races.append(race_dict)
            
            for entry in entries:
                entry_dict = {
                    "date": race_info.date,
                    "stadium_code": race_info.stadium_code,
                    "race_no": race_info.race_no,
                    **entry.__dict__
                }
                all_entries.append(entry_dict)
    
    # DataFrameに変換して保存
    if all_races:
        races_df = pd.DataFrame(all_races)
        races_df.to_csv(output_dir / f"{data_type}_races.csv", index=False)
        logger.info(f"Saved {len(races_df)} races to {data_type}_races.csv")
    
    if all_entries:
        entries_df = pd.DataFrame(all_entries)
        entries_df.to_csv(output_dir / f"{data_type}_entries.csv", index=False)
        logger.info(f"Saved {len(entries_df)} entries to {data_type}_entries.csv")


def convert_payouts_to_csv(input_dir: Path, output_dir: Path):
    """
    払戻金データをCSVに変換

    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = PayoutParser()
    all_payouts = []

    txt_files = list((input_dir / "results").glob("*.txt"))
    logger.info(f"Processing {len(txt_files)} files for payouts")

    for txt_path in tqdm(txt_files, desc="Parsing payouts", unit="file"):
        for payout in parser.parse_file(txt_path):
            # 2連単のみをフラット化して保存（バックテスト用）
            for combo, pay in payout.exacta.items():
                all_payouts.append({
                    "date": payout.date,
                    "stadium_code": payout.stadium_code,
                    "race_no": payout.race_no,
                    "bet_type": "exacta",
                    "first": combo[0],
                    "second": combo[1],
                    "payout": pay,
                    "odds": pay / 100,  # 100円あたりの払戻金 → オッズ
                })

            # 3連単も保存
            for combo, pay in payout.trifecta.items():
                all_payouts.append({
                    "date": payout.date,
                    "stadium_code": payout.stadium_code,
                    "race_no": payout.race_no,
                    "bet_type": "trifecta",
                    "first": combo[0],
                    "second": combo[1],
                    "third": combo[2] if len(combo) > 2 else 0,
                    "payout": pay,
                    "odds": pay / 100,
                })

    if all_payouts:
        payouts_df = pd.DataFrame(all_payouts)
        payouts_df.to_csv(output_dir / "payouts.csv", index=False)
        logger.info(f"Saved {len(payouts_df)} payout records to payouts.csv")


def main():
    """メイン処理"""
    convert_to_csv(RAW_DATA_DIR, PROCESSED_DATA_DIR, "programs")
    convert_to_csv(RAW_DATA_DIR, PROCESSED_DATA_DIR, "results")
    convert_payouts_to_csv(RAW_DATA_DIR, PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
