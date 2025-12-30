#!/usr/bin/env python
"""
Daily Odds Collection Script

Scrapes exacta odds from all active stadiums for a given date.

Usage:
    uv run python -m src.data_collection.collect_daily --date 20251230
    uv run python -m src.data_collection.collect_daily  # Today's date
    uv run python -m src.data_collection.collect_daily --stadiums 23 24  # Specific stadiums
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import STADIUM_CODES, PROJECT_ROOT
from src.data_collection.odds_scraper import OddsScraper, save_odds, ODDS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def collect_stadium(
    scraper: OddsScraper,
    date: int,
    stadium_code: int,
    skip_existing: bool = True,
) -> dict:
    """
    Collect odds for all races at a stadium.

    Args:
        scraper: OddsScraper instance
        date: Date in YYYYMMDD format
        stadium_code: Stadium code (1-24)
        skip_existing: Skip races that already have saved odds

    Returns:
        Stats dict with success/skip/fail counts
    """
    stats = {"success": 0, "skip": 0, "fail": 0}
    stadium_name = STADIUM_CODES.get(stadium_code, f"Stadium {stadium_code}")

    for race_no in range(1, 13):
        # Check if already exists
        if skip_existing:
            filepath = ODDS_DIR / f"{date}_{stadium_code:02d}_{race_no:02d}.json"
            if filepath.exists():
                stats["skip"] += 1
                continue

        # Scrape odds
        odds = scraper.scrape_exacta(date, stadium_code, race_no)

        if odds and len(odds.odds) == 30:
            save_odds(odds)
            stats["success"] += 1
        elif odds:
            # Partial odds (race may have scratches)
            save_odds(odds)
            stats["success"] += 1
            logger.warning(f"{stadium_name} R{race_no}: Only {len(odds.odds)} combinations")
        else:
            stats["fail"] += 1

    return stats


def collect_all_stadiums(
    date: int,
    stadiums: Optional[List[int]] = None,
    delay: float = 2.0,
    skip_existing: bool = True,
) -> dict:
    """
    Collect odds from all stadiums for a given date.

    Args:
        date: Date in YYYYMMDD format
        stadiums: List of stadium codes (default: all 24)
        delay: Delay between requests in seconds
        skip_existing: Skip races that already have saved odds

    Returns:
        Overall stats dict
    """
    scraper = OddsScraper(delay=delay)
    stadiums = stadiums or list(STADIUM_CODES.keys())

    total_stats = {
        "stadiums_processed": 0,
        "races_success": 0,
        "races_skip": 0,
        "races_fail": 0,
    }

    for stadium_code in tqdm(stadiums, desc="Stadiums", unit="stadium"):
        stadium_name = STADIUM_CODES.get(stadium_code, f"Stadium {stadium_code}")

        stats = collect_stadium(
            scraper, date, stadium_code, skip_existing=skip_existing
        )

        total_stats["stadiums_processed"] += 1
        total_stats["races_success"] += stats["success"]
        total_stats["races_skip"] += stats["skip"]
        total_stats["races_fail"] += stats["fail"]

        if stats["success"] > 0:
            logger.info(
                f"{stadium_name}: {stats['success']} scraped, "
                f"{stats['skip']} skipped, {stats['fail']} failed"
            )

    return total_stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Collect exacta odds from all stadiums",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Collect today's odds
  %(prog)s --date 20251230           # Collect specific date
  %(prog)s --stadiums 23 24          # Only specific stadiums
  %(prog)s --no-skip                 # Re-scrape existing files
        """
    )

    parser.add_argument(
        "--date", "-d",
        type=int,
        default=int(datetime.now().strftime("%Y%m%d")),
        help="Date to collect (YYYYMMDD format, default: today)"
    )
    parser.add_argument(
        "--stadiums", "-s",
        type=int,
        nargs="+",
        help="Stadium codes to collect (default: all 24)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing files, re-scrape all"
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

    # Validate stadiums
    if args.stadiums:
        invalid = [s for s in args.stadiums if s not in STADIUM_CODES]
        if invalid:
            print(f"Error: Invalid stadium codes: {invalid}")
            print("Use --list-stadiums to see valid codes")
            return

    # Ensure output directory exists
    ODDS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect odds
    logger.info(f"Collecting odds for {args.date}")
    logger.info(f"Output directory: {ODDS_DIR}")

    stats = collect_all_stadiums(
        date=args.date,
        stadiums=args.stadiums,
        delay=args.delay,
        skip_existing=not args.no_skip,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("COLLECTION SUMMARY")
    print("=" * 50)
    print(f"Date: {args.date}")
    print(f"Stadiums processed: {stats['stadiums_processed']}")
    print(f"Races scraped: {stats['races_success']}")
    print(f"Races skipped: {stats['races_skip']}")
    print(f"Races failed: {stats['races_fail']}")
    print("=" * 50)

    total_races = stats['races_success'] + stats['races_skip']
    if total_races > 0:
        print(f"\nTotal odds files: {total_races}")
        print(f"Storage: {ODDS_DIR}")


if __name__ == "__main__":
    main()
