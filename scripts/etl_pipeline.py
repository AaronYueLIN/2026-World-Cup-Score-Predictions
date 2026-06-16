#!/usr/bin/env python3
"""ETL Pipeline entry point -- Import match data from CSV or API into PostgreSQL

Usage:
  python scripts/etl_pipeline.py --source csv --file ../data/results.csv
  python scripts/etl_pipeline.py --source api --days 7
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.etl import fetch_api, init_db, load_csv, upsert_matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("etl_pipeline")


def main():
    parser = argparse.ArgumentParser(description="QuantBet-EV ETL Pipeline")
    parser.add_argument(
        "--source", choices=["csv", "api"], required=True, help="Data source"
    )
    parser.add_argument("--file", type=str, help="CSV file path (required when source=csv)")
    parser.add_argument("--days", type=int, default=7, help="API fetch days (source=api)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch write row count")
    args = parser.parse_args()

    # 0. Initialize database
    logger.info("Initializing database tables...")
    init_db()

    # 1. Load data
    if args.source == "csv":
        if not args.file:
            parser.error("--source csv requires --file argument")
        logger.info("Loading from CSV: %s", args.file)
        df = load_csv(args.file)
        upsert_matches(df, batch_size=args.batch_size)

    elif args.source == "api":
        logger.info("Fetching from API (last %d days)", args.days)
        matches = fetch_api(days_back=args.days)
        if matches:
            import pandas as pd

            df = pd.DataFrame(matches)
            upsert_matches(df, batch_size=args.batch_size)
        else:
            logger.info("No new matches")

    logger.info("ETL complete")


if __name__ == "__main__":
    main()
