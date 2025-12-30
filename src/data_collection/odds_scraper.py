#!/usr/bin/env python
"""
Real-time Odds Scraper for Boatrace.jp

Scrapes exacta odds from the official website for accurate backtesting.

Usage:
    uv run python -m src.data_collection.odds_scraper --date 20241230 --stadium 23 --race 1
    uv run python -m src.data_collection.odds_scraper --date 20241230 --stadium 23
"""

import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress XML parsed as HTML warning (the page is valid HTML)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT, STADIUM_CODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directory for scraped odds
ODDS_DIR = PROJECT_ROOT / "data" / "odds"

# Base URLs for odds pages
BASE_URL_EXACTA = "https://www.boatrace.jp/owpc/pc/race/odds2tf"
BASE_URL_TRIFECTA = "https://www.boatrace.jp/owpc/pc/race/odds3t"

# For backwards compatibility
BASE_URL = BASE_URL_EXACTA

# Request headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
}


@dataclass
class ExactaOdds:
    """Exacta odds for a single race"""
    date: int                    # YYYYMMDD
    stadium_code: int            # 1-24
    race_no: int                 # 1-12
    scraped_at: str              # ISO format timestamp
    odds: dict                   # {(first, second): odds_value}

    def to_json_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "date": self.date,
            "stadium_code": self.stadium_code,
            "race_no": self.race_no,
            "scraped_at": self.scraped_at,
            "exacta": {
                f"{k[0]}-{k[1]}": v for k, v in self.odds.items()
            }
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "ExactaOdds":
        """Create from JSON dictionary"""
        odds = {
            tuple(map(int, k.split("-"))): v
            for k, v in data["exacta"].items()
        }
        return cls(
            date=data["date"],
            stadium_code=data["stadium_code"],
            race_no=data["race_no"],
            scraped_at=data["scraped_at"],
            odds=odds,
        )


@dataclass
class TrifectaOdds:
    """Trifecta odds for a single race (3連単)"""
    date: int                    # YYYYMMDD
    stadium_code: int            # 1-24
    race_no: int                 # 1-12
    scraped_at: str              # ISO format timestamp
    odds: dict                   # {(first, second, third): odds_value} - 120 combinations

    def to_json_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "date": self.date,
            "stadium_code": self.stadium_code,
            "race_no": self.race_no,
            "scraped_at": self.scraped_at,
            "trifecta": {
                f"{k[0]}-{k[1]}-{k[2]}": v for k, v in self.odds.items()
            }
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "TrifectaOdds":
        """Create from JSON dictionary"""
        odds = {
            tuple(map(int, k.split("-"))): v
            for k, v in data["trifecta"].items()
        }
        return cls(
            date=data["date"],
            stadium_code=data["stadium_code"],
            race_no=data["race_no"],
            scraped_at=data["scraped_at"],
            odds=odds,
        )


class OddsScraper:
    """Scraper for boatrace.jp odds"""

    def __init__(self, delay: float = 2.0):
        """
        Args:
            delay: Delay between requests in seconds (default: 2.0)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last_request_time = 0

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    def _build_url(self, date: int, stadium_code: int, race_no: int, bet_type: str = "exacta") -> str:
        """Build URL for odds page"""
        # Stadium code needs to be zero-padded to 2 digits
        base = BASE_URL_TRIFECTA if bet_type == "trifecta" else BASE_URL_EXACTA
        return f"{base}?rno={race_no}&jcd={stadium_code:02d}&hd={date}"

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML page with rate limiting and retry"""
        self._wait_for_rate_limit()

        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = "utf-8"
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(self.delay * (attempt + 1))

        return None

    def _parse_exacta_odds(self, html: str) -> dict:
        """
        Parse exacta odds from HTML

        The table structure:
        - Header row: boat 1 | name | boat 2 | name | ... (6 boats as 1st place)
        - Body rows: 2nd boat | odds | 2nd boat | odds | ... (for each 1st place column)

        Returns:
            Dictionary of {(first, second): odds}
        """
        soup = BeautifulSoup(html, "lxml")
        odds = {}

        # Find the 2連単オッズ section (first odds table after the title)
        title = soup.find("span", class_="title7_mainLabel", string="2連単オッズ")
        if title is None:
            logger.warning("Could not find 2連単オッズ title")
            return odds

        # Find the table container after the title
        title_div = title.find_parent("div", class_="title7")
        if title_div is None:
            logger.warning("Could not find title container")
            return odds

        table_div = title_div.find_next_sibling("div", class_="table1")
        if table_div is None:
            logger.warning("Could not find table container")
            return odds

        table = table_div.find("table")
        if table is None:
            logger.warning("Could not find odds table")
            return odds

        # Get header to determine 1st place boats
        thead = table.find("thead")
        if thead is None:
            logger.warning("Could not find table header")
            return odds

        # Extract 1st place boat numbers from header
        header_cells = thead.find_all("th")
        first_boats = []
        for cell in header_cells:
            boat_num = self._get_boat_number(cell)
            if boat_num is not None and boat_num not in first_boats:
                first_boats.append(boat_num)

        if len(first_boats) != 6:
            logger.warning(f"Expected 6 first boats, got {len(first_boats)}")

        # Parse body - each row has pairs of (2nd place boat, odds) for each 1st place column
        tbody = table.find("tbody")
        if tbody is None:
            logger.warning("Could not find table body")
            return odds

        rows = tbody.find_all("tr")

        for row in rows:
            cells = row.find_all("td")

            # Cells come in pairs: (boat number, odds) for each 1st place column
            # 6 columns = 12 cells per row
            cell_idx = 0
            for first_idx, first_boat in enumerate(first_boats):
                if cell_idx + 1 >= len(cells):
                    break

                boat_cell = cells[cell_idx]
                odds_cell = cells[cell_idx + 1]
                cell_idx += 2

                second_boat = self._get_boat_number(boat_cell)
                if second_boat is None:
                    # Try parsing from text
                    text = boat_cell.get_text(strip=True)
                    if text.isdigit():
                        second_boat = int(text)

                if second_boat is None:
                    continue

                odds_value = self._parse_odds_value(odds_cell)
                if odds_value is not None and first_boat != second_boat:
                    odds[(first_boat, second_boat)] = odds_value

        return odds

    def _parse_trifecta_odds(self, html: str) -> dict:
        """
        Parse trifecta odds from HTML (3連単)

        The table structure:
        - Header row: 6 columns (1st place boats)
        - For each column: 5 groups of 2nd place boats (rowspan=4)
        - Each group: 4 rows for 3rd place options
        - Cell triplets: (2nd place with rowspan) (3rd place) (odds)

        Returns:
            Dictionary of {(first, second, third): odds}
        """
        soup = BeautifulSoup(html, "lxml")
        odds = {}

        # Find the 3連単オッズ section
        title = soup.find("span", class_="title7_mainLabel", string="3連単オッズ")
        if title is None:
            logger.warning("Could not find 3連単オッズ title")
            return odds

        # Find the table container after the title
        title_div = title.find_parent("div", class_="title7")
        if title_div is None:
            logger.warning("Could not find title container")
            return odds

        table_div = title_div.find_next_sibling("div", class_="table1")
        if table_div is None:
            logger.warning("Could not find table container")
            return odds

        table = table_div.find("table")
        if table is None:
            logger.warning("Could not find odds table")
            return odds

        # Get header to determine 1st place boats (columns)
        thead = table.find("thead")
        if thead is None:
            logger.warning("Could not find table header")
            return odds

        # Extract 1st place boat numbers from header
        header_cells = thead.find_all("th")
        first_boats = []
        for cell in header_cells:
            boat_num = self._get_boat_number(cell)
            if boat_num is not None and boat_num not in first_boats:
                first_boats.append(boat_num)

        if len(first_boats) != 6:
            logger.warning(f"Expected 6 first boats, got {len(first_boats)}")
            return odds

        # Parse body
        tbody = table.find("tbody")
        if tbody is None:
            logger.warning("Could not find table body")
            return odds

        rows = tbody.find_all("tr")

        # Track current 2nd place boat for each column (due to rowspan)
        current_second = [None] * 6

        for row in rows:
            cells = row.find_all("td")

            # Process cells - structure varies due to rowspan
            # Each column can have: (2nd boat with rowspan, 3rd boat, odds) or (3rd boat, odds)
            cell_idx = 0

            for col_idx, first_boat in enumerate(first_boats):
                if cell_idx >= len(cells):
                    break

                cell = cells[cell_idx]

                # Check if this cell has rowspan (new 2nd place boat)
                rowspan = cell.get("rowspan")
                if rowspan:
                    # This is a 2nd place boat cell
                    second_boat = self._get_boat_number(cell)
                    if second_boat is None:
                        text = cell.get_text(strip=True)
                        if text.isdigit():
                            second_boat = int(text)
                    current_second[col_idx] = second_boat
                    cell_idx += 1

                    # Next cell is 3rd place boat (use TEXT, not class - class shows 2nd place color)
                    if cell_idx >= len(cells):
                        break
                    third_cell = cells[cell_idx]
                    third_boat = self._get_boat_number_from_text(third_cell)
                    cell_idx += 1

                    # Next cell is odds
                    if cell_idx >= len(cells):
                        break
                    odds_cell = cells[cell_idx]
                    odds_value = self._parse_odds_value(odds_cell)
                    cell_idx += 1

                    if second_boat and third_boat and odds_value is not None:
                        odds[(first_boat, second_boat, third_boat)] = odds_value
                else:
                    # No rowspan - use current 2nd place boat
                    second_boat = current_second[col_idx]

                    # This cell is 3rd place boat (use TEXT, not class)
                    third_boat = self._get_boat_number_from_text(cell)
                    cell_idx += 1

                    # Next cell is odds
                    if cell_idx >= len(cells):
                        break
                    odds_cell = cells[cell_idx]
                    odds_value = self._parse_odds_value(odds_cell)
                    cell_idx += 1

                    if second_boat and third_boat and odds_value is not None:
                        odds[(first_boat, second_boat, third_boat)] = odds_value

        return odds

    def _get_boat_number(self, cell) -> Optional[int]:
        """Extract boat number from cell class"""
        if cell is None:
            return None

        classes = cell.get("class", [])
        for cls in classes:
            for i in range(1, 7):
                if f"is-boatColor{i}" in cls or cls == f"is-boatColor{i}":
                    return i

        # Try to parse from text content
        text = cell.get_text(strip=True)
        if text.isdigit() and 1 <= int(text) <= 6:
            return int(text)

        return None

    def _get_boat_number_from_text(self, cell) -> Optional[int]:
        """Extract boat number from cell text (for trifecta 3rd place cells)"""
        if cell is None:
            return None

        text = cell.get_text(strip=True)
        if text.isdigit() and 1 <= int(text) <= 6:
            return int(text)

        return None

    def _parse_odds_value(self, cell) -> Optional[float]:
        """Parse odds value from cell"""
        if cell is None:
            return None

        # Look for oddsPoint class
        odds_span = cell.find(class_="oddsPoint")
        if odds_span:
            text = odds_span.get_text(strip=True)
        else:
            text = cell.get_text(strip=True)

        # Clean and parse
        text = text.replace(",", "").replace("欠場", "").replace("取消", "").strip()

        if not text or text == "-":
            return None

        try:
            return float(text)
        except ValueError:
            return None

    def scrape_exacta(
        self,
        date: int,
        stadium_code: int,
        race_no: int,
    ) -> Optional[ExactaOdds]:
        """
        Scrape exacta odds for a single race

        Args:
            date: Race date (YYYYMMDD)
            stadium_code: Stadium code (1-24)
            race_no: Race number (1-12)

        Returns:
            ExactaOdds object or None if failed
        """
        url = self._build_url(date, stadium_code, race_no)
        logger.info(f"Scraping: {url}")

        html = self._fetch_page(url)
        if html is None:
            logger.error(f"Failed to fetch page for race {race_no}")
            return None

        odds = self._parse_exacta_odds(html)

        if not odds:
            logger.warning(f"No odds found for race {race_no}")
            return None

        # Expect 30 combinations (6 * 5)
        if len(odds) != 30:
            logger.warning(f"Expected 30 combinations, got {len(odds)}")

        return ExactaOdds(
            date=date,
            stadium_code=stadium_code,
            race_no=race_no,
            scraped_at=datetime.now().isoformat(),
            odds=odds,
        )

    def scrape_trifecta(
        self,
        date: int,
        stadium_code: int,
        race_no: int,
    ) -> Optional[TrifectaOdds]:
        """
        Scrape trifecta odds for a single race (3連単)

        Args:
            date: Race date (YYYYMMDD)
            stadium_code: Stadium code (1-24)
            race_no: Race number (1-12)

        Returns:
            TrifectaOdds object or None if failed
        """
        url = self._build_url(date, stadium_code, race_no, bet_type="trifecta")
        logger.info(f"Scraping trifecta: {url}")

        html = self._fetch_page(url)
        if html is None:
            logger.error(f"Failed to fetch trifecta page for race {race_no}")
            return None

        odds = self._parse_trifecta_odds(html)

        if not odds:
            logger.warning(f"No trifecta odds found for race {race_no}")
            return None

        # Expect 120 combinations (6 * 5 * 4)
        if len(odds) != 120:
            logger.warning(f"Expected 120 combinations, got {len(odds)}")

        return TrifectaOdds(
            date=date,
            stadium_code=stadium_code,
            race_no=race_no,
            scraped_at=datetime.now().isoformat(),
            odds=odds,
        )

    def scrape_stadium(
        self,
        date: int,
        stadium_code: int,
        races: range = range(1, 13),
    ) -> list[ExactaOdds]:
        """
        Scrape all races at a stadium

        Args:
            date: Race date (YYYYMMDD)
            stadium_code: Stadium code (1-24)
            races: Race numbers to scrape (default: 1-12)

        Returns:
            List of ExactaOdds objects
        """
        stadium_name = STADIUM_CODES.get(stadium_code, f"Stadium {stadium_code}")
        logger.info(f"Scraping {stadium_name} ({date})...")

        results = []
        for race_no in races:
            odds = self.scrape_exacta(date, stadium_code, race_no)
            if odds:
                results.append(odds)

        logger.info(f"Scraped {len(results)} races from {stadium_name}")
        return results


def save_odds(odds: ExactaOdds, output_dir: Path = None) -> Path:
    """Save exacta odds to JSON file"""
    output_dir = output_dir or ODDS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{odds.date}_{odds.stadium_code:02d}_{odds.race_no:02d}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(odds.to_json_dict(), f, indent=2, ensure_ascii=False)

    return filepath


def save_trifecta_odds(odds: TrifectaOdds, output_dir: Path = None) -> Path:
    """Save trifecta odds to JSON file"""
    output_dir = output_dir or ODDS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use _3t suffix to distinguish from exacta
    filename = f"{odds.date}_{odds.stadium_code:02d}_{odds.race_no:02d}_3t.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(odds.to_json_dict(), f, indent=2, ensure_ascii=False)

    return filepath


def load_odds(
    date: int,
    stadium_code: int,
    race_no: int,
    odds_dir: Path = None,
) -> Optional[ExactaOdds]:
    """Load exacta odds from JSON file"""
    odds_dir = odds_dir or ODDS_DIR

    filename = f"{date}_{stadium_code:02d}_{race_no:02d}.json"
    filepath = odds_dir / filename

    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ExactaOdds.from_json_dict(data)


def load_trifecta_odds(
    date: int,
    stadium_code: int,
    race_no: int,
    odds_dir: Path = None,
) -> Optional[TrifectaOdds]:
    """Load trifecta odds from JSON file"""
    odds_dir = odds_dir or ODDS_DIR

    filename = f"{date}_{stadium_code:02d}_{race_no:02d}_3t.json"
    filepath = odds_dir / filename

    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return TrifectaOdds.from_json_dict(data)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Scrape odds from boatrace.jp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --date 20241230 --stadium 23 --race 1          # Exacta odds
  %(prog)s --date 20241230 --stadium 23 --trifecta        # Trifecta odds
  %(prog)s --date 20241230 --stadium 23 --race 1 --trifecta
  %(prog)s --list-stadiums
        """
    )

    parser.add_argument(
        "--date", "-d",
        type=int,
        help="Race date (YYYYMMDD format)"
    )
    parser.add_argument(
        "--stadium", "-s",
        type=int,
        help="Stadium code (1-24)"
    )
    parser.add_argument(
        "--race", "-r",
        type=int,
        help="Race number (1-12). If not specified, scrape all 12 races."
    )
    parser.add_argument(
        "--trifecta", "-t",
        action="store_true",
        help="Scrape trifecta (3連単) odds instead of exacta (2連単)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help=f"Output directory (default: {ODDS_DIR})"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--list-stadiums",
        action="store_true",
        help="List all stadium codes and names"
    )

    args = parser.parse_args()

    if args.list_stadiums:
        print("\nStadium Codes:")
        print("-" * 40)
        for code, name in sorted(STADIUM_CODES.items()):
            print(f"  {code:2}: {name}")
        return

    if not args.date or not args.stadium:
        parser.print_help()
        return

    # Validate inputs
    if not (1 <= args.stadium <= 24):
        print(f"Error: Stadium code must be 1-24, got {args.stadium}")
        return

    if args.race and not (1 <= args.race <= 12):
        print(f"Error: Race number must be 1-12, got {args.race}")
        return

    # Initialize scraper
    scraper = OddsScraper(delay=args.delay)
    output_dir = args.output or ODDS_DIR
    bet_type = "trifecta" if args.trifecta else "exacta"

    if args.race:
        # Scrape single race
        if args.trifecta:
            odds = scraper.scrape_trifecta(args.date, args.stadium, args.race)
            save_func = save_trifecta_odds
        else:
            odds = scraper.scrape_exacta(args.date, args.stadium, args.race)
            save_func = save_odds

        if odds:
            filepath = save_func(odds, output_dir)
            print(f"\nSaved to: {filepath}")
            print(f"Bet type: {bet_type}")
            print(f"Combinations: {len(odds.odds)}")

            # Show top 5 by odds
            sorted_odds = sorted(odds.odds.items(), key=lambda x: x[1])
            print("\nTop 5 (lowest odds):")
            for combo, o in sorted_odds[:5]:
                combo_str = "-".join(str(c) for c in combo)
                print(f"  {combo_str}: {o:.1f}")
        else:
            print(f"Failed to scrape {bet_type} odds")
    else:
        # Scrape all races at stadium
        count = 0
        for race_no in range(1, 13):
            if args.trifecta:
                odds = scraper.scrape_trifecta(args.date, args.stadium, race_no)
                if odds:
                    filepath = save_trifecta_odds(odds, output_dir)
                    print(f"Saved R{race_no}: {filepath}")
                    count += 1
            else:
                odds = scraper.scrape_exacta(args.date, args.stadium, race_no)
                if odds:
                    filepath = save_odds(odds, output_dir)
                    print(f"Saved R{race_no}: {filepath}")
                    count += 1

        print(f"\nTotal: {count} {bet_type} races saved")


if __name__ == "__main__":
    main()
