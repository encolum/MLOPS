"""
Simplified data ingestion module for loading CSV files into PostgreSQL
"""
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import os
import sys
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PIPELINE_DIR = Path(__file__).resolve().parent
LABELED_DIR = DATA_PIPELINE_DIR / "labeled"

# Load environment variables from the data pipeline directory.
load_dotenv(DATA_PIPELINE_DIR / ".env")

def connect_to_db(db_name="postgres"):
    """Create database connection using environment variables"""
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    
    conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(conn_string)

def create_database_if_not_exists(db_name):
    """Create the database if it doesn't exist"""
    db_name = validate_identifier(db_name)
    conn = psycopg2.connect(
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database="postgres",
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cursor.fetchone():
                logger.info(f"Database '{db_name}' already exists.")
                return

            logger.info(f"Creating database '{db_name}'...")
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Database '{db_name}' created.")
    finally:
        conn.close()

def load_data_to_db(csv_file, db_name, table_name, if_exists="replace"):
    """
    Load data from CSV file to PostgreSQL database
    
    Args:
        csv_file: Path to the CSV file
        db_name: Database name
        table_name: Table name
        if_exists: How to handle existing table ('replace', 'append', 'fail')
    
    Returns:
        Number of records loaded
    """
    try:
        # Ensure database exists
        create_database_if_not_exists(db_name)
        
        # Load CSV data
        logger.info(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Convert any date columns to datetime
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        df, duplicate_rows = deduplicate_by_id(df)
        if duplicate_rows:
            logger.info(f"Removed {duplicate_rows} duplicate rows from CSV")

        engine = connect_to_db(db_name)
        df, skipped_rows = filter_existing_ids(engine, df, table_name, if_exists)

        if if_exists == "append" and skipped_rows:
            logger.info(f"Skipped {skipped_rows} rows already present in {db_name}.{table_name}")

        if len(df):
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        elif if_exists == "fail":
            # Preserve pandas' expected failure behavior when the table exists.
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)

        if if_exists == "replace" and not table_exists(engine, table_name):
            # An empty DataFrame still creates the table with pandas' schema.
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)

        logger.info(f"Loaded {len(df)} records into {db_name}.{table_name}")
        return len(df)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def validate_identifier(identifier):
    """Validate a PostgreSQL identifier supplied through the CLI."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(identifier)):
        raise ValueError(f"Invalid PostgreSQL identifier: {identifier}")
    return identifier


def deduplicate_by_id(df):
    """Drop duplicate rows in a CSV by its tweet id."""
    if "id" not in df.columns:
        raise ValueError("CSV must contain an 'id' column for idempotent ingestion")

    original_rows = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    return df, original_rows - len(df)


def table_exists(engine, table_name):
    """Return whether a table exists in the public schema."""
    table_name = validate_identifier(table_name)
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = :table_name
                    """
                ),
                {"table_name": table_name},
            ).fetchone()
        )


def filter_existing_ids(engine, df, table_name, if_exists):
    """For append mode, keep only rows whose id is not already in PostgreSQL."""
    if if_exists != "append" or not table_exists(engine, table_name):
        return df, 0

    table_name = validate_identifier(table_name)
    with engine.connect() as conn:
        existing_ids = set(
            str(value)
            for value in conn.execute(
                text(f'SELECT id FROM public."{table_name}"')
            ).scalars()
        )

    original_rows = len(df)
    df = df[~df["id"].astype(str).isin(existing_ids)].reset_index(drop=True)
    return df, original_rows - len(df)


def get_latest_labeled_file():
        """Get the latest labeled file from the labeled directory"""
        files = [f for f in LABELED_DIR.iterdir() if f.suffix == '.csv'] if LABELED_DIR.exists() else []
        if not files:
            return None
        return max(files, key=lambda path: path.stat().st_mtime)

def test_connection():
    """Test the database connection properly using psycopg2"""
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    
    try:
        # Thử kết nối trực tiếp bằng psycopg2
        conn = psycopg2.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            database="postgres",  # Kết nối đến postgres mặc định
            connect_timeout=3     # Timeout sau 3 giây
        )
        conn.close()
        print(f"Successfully connected to PostgreSQL at {db_host}:{db_port}")
        return True
    except Exception as e:
        print(f"Failed to connect to PostgreSQL at {db_host}:{db_port}")
        print(f"Error message: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. If PostgreSQL runs on Windows and you are using WSL:")
        print("   - Create a .env file with the following contents:")
        print('     DB_HOST=host.docker.internal  # or the Windows IP')
        print('     DB_USER=postgres')
        print('     DB_PASSWORD=your_password')
        print('     DB_PORT=5432')
        print("   - Make sure PostgreSQL accepts remote connections:")
        print("     + Edit pg_hba.conf and add: host all all 0.0.0.0/0 md5")
        print("     + Edit postgresql.conf: listen_addresses = '*'")
        print("2. Use the ingest_sqlite.py tool instead (simpler):")
        print("   python ingest_sqlite.py --file your_file.csv")
        return False

# Chạy test_connection() ở đầu hàm main() để kiểm tra
def main():
    """Main function to load data from CSV to PostgreSQL"""
    import argparse
    from datetime import datetime
    
    # Test connection first - if it fails, suggest alternatives
    if not test_connection():
        print("Could not connect to PostgreSQL. Stopping program.")
        sys.exit(1)
    
    # Get default input file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        default_input = get_latest_labeled_file() or LABELED_DIR / f'labeled_twitter_{timestamp}.csv'
    except Exception as e:
        default_input = LABELED_DIR / f'labeled_twitter_{timestamp}.csv'
        print(f"Could not find default CSV file: {e}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load CSV data into PostgreSQL")
    parser.add_argument(
        "--file", "-f", 
        default=default_input,
        help="Path to the input CSV file (raw data)."
    )
    parser.add_argument(
        "--database", "-d", 
        default="twitter_analysis_tutorial",
        help="Database name"
    )
    parser.add_argument(
        "--table", "-t", 
        default="tweets", 
        help="Table name"
    )
    parser.add_argument(
        "--mode", "-m", 
        default="append", 
        choices=["replace", "append", "fail"],
        help="How to handle existing tables"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    # Load through the shared idempotent implementation.
    try:
        loaded_rows = load_data_to_db(
            csv_file=args.file,
            db_name=args.database,
            table_name=args.table,
            if_exists=args.mode,
        )
        print(
            f"Successfully loaded {loaded_rows} new rows into "
            f"{args.database}.{args.table}"
        )
        
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
