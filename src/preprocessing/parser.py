"""
Boat Race Data Parser

Parses fixed-width text files and converts them to CSV format.
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


def normalize_fullwidth_numbers(text: str) -> str:
    """Convert fullwidth digits to halfwidth."""
    trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
    return text.translate(trans_table)


@dataclass
class RaceInfo:
    """Race information."""
    date: str
    stadium_code: int
    stadium_name: str
    race_no: int
    race_type: str
    distance: int
    title: str = ""
    day: int = 0  # Day of the event


@dataclass
class RacerEntry:
    """Racer entry information (from race program)."""
    boat_no: int          # Boat number (1-6)
    racer_id: int         # Registration number
    racer_name: str       # Racer name
    age: int              # Age
    branch: str           # Branch/region
    weight: int           # Weight (kg)
    racer_class: str      # Class (A1, A2, B1, B2)

    # Win rates
    national_win_rate: float   # National win rate
    national_in2_rate: float   # National top-2 finish rate
    local_win_rate: float      # Local venue win rate
    local_in2_rate: float      # Local venue top-2 finish rate

    # Motor and boat
    motor_no: int              # Motor number
    motor_in2_rate: float      # Motor top-2 finish rate
    boat_no_equip: int         # Boat number
    boat_in2_rate: float       # Boat top-2 finish rate


@dataclass
class RaceResult:
    """Race result."""
    boat_no: int          # Boat number
    racer_id: int         # Registration number
    rank: int             # Finishing position (0 = disqualified)
    race_time: str        # Race time
    course: int           # Starting course
    start_timing: float   # Start timing


@dataclass
class RacePayouts:
    """Race payouts."""
    date: str
    stadium_code: int
    race_no: int
    win: dict              # Win {boat: payout}
    place: dict            # Place {boat: payout}
    exacta: dict           # Exacta {(1st, 2nd): payout}
    quinella: dict         # Quinella {(a, b): payout}
    wide: dict             # Wide {(a, b): payout}
    trifecta: dict         # Trifecta {(1st, 2nd, 3rd): payout}
    trio: dict             # Trio {(a, b, c): payout}


class ProgramParser:
    """Race program parser."""

    # Stadium identifier pattern (e.g., "22BBGN" -> Fukuoka)
    STADIUM_PATTERN = re.compile(r'^(\d{2})BBGN')

    # Race number pattern (e.g., "１Ｒ 予選")
    RACE_PATTERN = re.compile(r'[　\s]*([０-９\d]+)Ｒ\s*(.*?)\s*Ｈ(\d+)')

    # Racer line pattern
    RACER_PATTERN = re.compile(
        r'^(\d)\s+'           # Boat number
        r'(\d{4})'            # Registration number
        r'(.{4})'             # Racer name (4 chars fixed)
        r'(\d{2})'            # Age
        r'(.{2})'             # Branch
        r'(\d{2})'            # Weight
        r'([AB][12])'         # Class
        r'\s*(\d+\.\d+)'      # National win rate
        r'\s*(\d+\.\d+)'      # National top-2 rate
        r'\s*(\d+\.\d+)'      # Local win rate
        r'\s*(\d+\.\d+)'      # Local top-2 rate
        r'\s*(\d+)'           # Motor number
        r'\s*(\d+\.\d+)'      # Motor top-2 rate
        r'\s*(\d+)'           # Boat number
        r'\s*(\d+\.\d+)'      # Boat top-2 rate
    )
    
    def __init__(self, encoding: str = "cp932"):
        self.encoding = encoding

    def parse_file(self, file_path: Path) -> Generator[tuple[RaceInfo, list[RacerEntry]], None, None]:
        """
        Parse race program file.

        Yields:
            Tuple of (RaceInfo, list[RacerEntry])
        """
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        lines = content.split("\n")

        # Extract date from filename (programs_YYYYMMDD.txt)
        date_str = file_path.stem.split("_")[-1]

        current_stadium = None
        current_race = None
        current_racers = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Stadium identification
            stadium_match = self.STADIUM_PATTERN.match(line)
            if stadium_match:
                stadium_code = int(stadium_match.group(1))
                current_stadium = {
                    "code": stadium_code,
                    "name": STADIUM_CODES.get(stadium_code, "Unknown")
                }
                i += 1
                continue

            # Race information
            race_match = self.RACE_PATTERN.search(line)
            if race_match and current_stadium:
                # Yield previous race if exists
                if current_race and current_racers:
                    yield current_race, current_racers

                race_no = int(normalize_fullwidth_numbers(race_match.group(1)))
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

            # Racer information (simple parse)
            if current_race and line.strip() and line[0].isdigit():
                try:
                    racer = self._parse_racer_line(line)
                    if racer:
                        current_racers.append(racer)
                except Exception as e:
                    logger.debug(f"Parse error at line {i}: {e}")

            i += 1

        # Last race
        if current_race and current_racers:
            yield current_race, current_racers
    
    def _parse_racer_line(self, line: str) -> RacerEntry | None:
        """Parse racer line."""
        # First try regex parsing
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

        # Fallback: fixed position parsing
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

            # Remaining values are space-separated numbers
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
    """Race result parser."""

    STADIUM_PATTERN = re.compile(r'^(\d{2})KBGN')
    # Result files use half-width numbers: "   1R       朝１戦予選　    H1800m"
    RACE_PATTERN = re.compile(r'^\s*(\d+)R\s+(.*?)\s+H(\d+)')

    # Result line pattern: "  01  4 4861 田... 54   72  6.80   4    0.05     1.49.6"
    RESULT_LINE_PATTERN = re.compile(
        r'^\s*(\d{2})\s+'     # Rank (01-06)
        r'(\d)\s+'            # Boat number (1-6)
        r'(\d{4})'            # Registration number
    )

    def __init__(self, encoding: str = "cp932"):
        self.encoding = encoding

    def _parse_result_line(self, line: str) -> RaceResult | None:
        """Parse result line."""
        match = self.RESULT_LINE_PATTERN.match(line)
        if not match:
            return None

        try:
            rank = int(match.group(1))
            boat_no = int(match.group(2))
            racer_id = int(match.group(3))

            # Extract values from remaining part
            # Format: ... 54   72  6.80   4    0.05     1.49.6
            remaining = line[match.end():]

            # Split by whitespace to get values
            parts = remaining.split()

            # Search from end: race_time, ST, course, exhibition_time, exhibition_ST, exhibition_no
            # Minimum needed: course, ST, race_time
            course = 0
            start_timing = 0.0
            race_time = ""

            if len(parts) >= 3:
                # Last value is race time (1.49.6 format)
                race_time = parts[-1]

                # Previous is ST timing
                try:
                    start_timing = float(parts[-2])
                except ValueError:
                    pass

                # Before that is starting course
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
        """Parse race result file."""
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
            # Stadium identification
            stadium_match = self.STADIUM_PATTERN.match(line)
            if stadium_match:
                stadium_code = int(stadium_match.group(1))
                current_stadium = {
                    "code": stadium_code,
                    "name": STADIUM_CODES.get(stadium_code, "Unknown")
                }
                continue

            # Race information
            race_match = self.RACE_PATTERN.search(line)
            if race_match and current_stadium:
                # Yield previous race if exists
                if current_race and current_results:
                    yield current_race, current_results

                race_no = int(normalize_fullwidth_numbers(race_match.group(1)))
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

            # Section starts at separator line
            if line.startswith("---") and current_race:
                in_results_section = True
                continue

            # Parse result lines
            if in_results_section and current_race and line.strip():
                result = self._parse_result_line(line)
                if result:
                    current_results.append(result)
                elif line.strip().startswith("単勝") or line.strip().startswith("複勝"):
                    # End when entering payout section
                    in_results_section = False

        # Last race
        if current_race and current_results:
            yield current_race, current_results


class PayoutParser:
    """Payout parser."""

    STADIUM_PATTERN = re.compile(r'^(\d{2})KBGN')
    RACE_PATTERN = re.compile(r'^\s*(\d+)R\s+')

    # Payout patterns (Japanese bet type names in source data)
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
        """Parse payout file."""
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
            # Stadium identification
            stadium_match = self.STADIUM_PATTERN.match(line)
            if stadium_match:
                current_stadium = int(stadium_match.group(1))
                continue

            # Race number
            race_match = self.RACE_PATTERN.match(line)
            if race_match and current_stadium:
                # Yield previous race if exists
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

            # Detect payout section
            if "単勝" in line or "２連単" in line or "３連単" in line:
                in_payout_section = True

            # Parse payouts
            if in_payout_section and current_race_no:
                self._parse_payout_line(line, current_payouts)

        # Last race
        if current_race_no and current_stadium and self._has_payouts(current_payouts):
            yield RacePayouts(
                date=date_str,
                stadium_code=current_stadium,
                race_no=current_race_no,
                **current_payouts,
            )

    def _empty_payouts(self) -> dict:
        """Return empty payout dictionary."""
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
        """Check if payout data exists."""
        return any(len(v) > 0 for v in payouts.values())

    def _parse_payout_line(self, line: str, payouts: dict) -> None:
        """Parse payout line."""
        # Win
        for match in self.WIN_PATTERN.finditer(line):
            boat = int(match.group(1))
            payout = int(match.group(2))
            payouts["win"][boat] = payout

        # Place
        for match in self.PLACE_PATTERN.finditer(line):
            boat = int(match.group(1))
            payout = int(match.group(2))
            payouts["place"][boat] = payout

        # Exacta
        for match in self.EXACTA_PATTERN.finditer(line):
            first = int(match.group(1))
            second = int(match.group(2))
            payout = int(match.group(3))
            payouts["exacta"][(first, second)] = payout

        # Quinella
        for match in self.QUINELLA_PATTERN.finditer(line):
            a = int(match.group(1))
            b = int(match.group(2))
            payout = int(match.group(3))
            payouts["quinella"][tuple(sorted([a, b]))] = payout

        # Wide
        for match in self.WIDE_PATTERN.finditer(line):
            a = int(match.group(1))
            b = int(match.group(2))
            payout = int(match.group(3))
            payouts["wide"][tuple(sorted([a, b]))] = payout

        # Trifecta
        for match in self.TRIFECTA_PATTERN.finditer(line):
            first = int(match.group(1))
            second = int(match.group(2))
            third = int(match.group(3))
            payout = int(match.group(4))
            payouts["trifecta"][(first, second, third)] = payout

        # Trio
        for match in self.TRIO_PATTERN.finditer(line):
            a = int(match.group(1))
            b = int(match.group(2))
            c = int(match.group(3))
            payout = int(match.group(4))
            payouts["trio"][tuple(sorted([a, b, c]))] = payout


def convert_to_csv(input_dir: Path, output_dir: Path, data_type: str = "programs"):
    """
    Convert text files to CSV.

    Args:
        input_dir: Input directory
        output_dir: Output directory
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

    # Convert to DataFrame and save
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
    Convert payout data to CSV.

    Args:
        input_dir: Input directory
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = PayoutParser()
    all_payouts = []

    txt_files = list((input_dir / "results").glob("*.txt"))
    logger.info(f"Processing {len(txt_files)} files for payouts")

    for txt_path in tqdm(txt_files, desc="Parsing payouts", unit="file"):
        for payout in parser.parse_file(txt_path):
            # Flatten exacta payouts for backtesting
            for combo, pay in payout.exacta.items():
                all_payouts.append({
                    "date": payout.date,
                    "stadium_code": payout.stadium_code,
                    "race_no": payout.race_no,
                    "bet_type": "exacta",
                    "first": combo[0],
                    "second": combo[1],
                    "payout": pay,
                    "odds": pay / 100,  # Payout per 100 yen -> odds
                })

            # Also save trifecta
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
    """Main entry point."""
    convert_to_csv(RAW_DATA_DIR, PROCESSED_DATA_DIR, "programs")
    convert_to_csv(RAW_DATA_DIR, PROCESSED_DATA_DIR, "results")
    convert_payouts_to_csv(RAW_DATA_DIR, PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
