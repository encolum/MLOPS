"""
Data ingestion module to load CSV data into PostgreSQL database.
"""
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_connection_string(db_name=None):
    """
    Create PostgreSQL connection string from environment variables.
    
    Args:
        db_name: Optional database name to override environment variable
        
    Returns:
        Connection string for SQLAlchemy
    """
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "iloveyou044")  # Using your password from the notebook
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    
    # Use provided db_name or default from environment
    if db_name is None:
        db_name = os.getenv("DB_NAME", "Tweet")  # Using your DB name from the notebook
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# def ensure_database_exists(db_name):
#     """
#     Create PostgreSQL database if it doesn't exist.
    
#     Args:
#         db_name: Name of the database to create
#     """
#     # Connect to default 'postgres' database to check if our target database exists
#     conn_string = get_connection_string("postgres")
#     engine = create_engine(conn_string)
    
#     with engine.connect() as connection:
#         # Check if database exists
#         result = connection.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
#         exists = result.fetchone()
        
#         if not exists:
#             # Need to use raw psycopg2 for database creation
#             conn = psycopg2.connect(
#                 user=os.getenv("DB_USER"),
#                 password=os.getenv("DB_PASSWORD"),
#                 host=os.getenv("DB_HOST"),
#                 port=os.getenv("DB_PORT"),
#                 database="postgres"
#             )
#             conn.autocommit = True
            
#             with conn.cursor() as cursor:
#                 cursor.execute(f"CREATE DATABASE {db_name}")
                
#             conn.close()
#             logger.info(f"Database '{db_name}' created successfully.")
#         else:
#             logger.info(f"Database '{db_name}' already exists.")

def ensure_database_exists(db_name):
    """
    Create PostgreSQL database if it doesn't exist.
    
    Args:
        db_name: Name of the database to create
    """
    # Get connection details from environment or defaults
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST") # <-- Get the host here
    db_port = os.getenv("DB_PORT")
    
    # Connect to default 'postgres' database to check if our target database exists
    # Use the actual host for the initial connection string as well
    conn_string_postgres = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres"
    
    try:
        # Use SQLAlchemy to check existence (this should use TCP/IP via the URL)
        engine = create_engine(conn_string_postgres)
        
        with engine.connect() as connection:
            # Check if database exists
            # Note: SQLAlchemy's text() is safer for raw SQL
            from sqlalchemy import text
            result = connection.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            exists = result.fetchone()
            
            if not exists:
                logger.info(f"Database '{db_name}' does not exist. Creating...")
                # Need to use raw psycopg2 for database creation
                # IMPORTANT: Use the actual db_host obtained from env/default
                conn = psycopg2.connect(
                    user=db_user,
                    password=db_password,
                    host=db_host, # <-- Use the variable here!
                    port=db_port,
                    database="postgres"
                )
                conn.autocommit = True
                
                with conn.cursor() as cursor:
                    # Safer to check again within the transaction block or ensure exclusiveness
                    # For simplicity in this example, we assume the check above is sufficient
                    try:
                        cursor.execute(f"CREATE DATABASE {db_name}")
                        logger.info(f"Database '{db_name}' created successfully.")
                    except psycopg2.errors.DuplicateDatabase:
                        # Handle race condition if another process created it between checks
                        logger.info(f"Database '{db_name}' already exists (created by another process?).")
                    
                conn.close()
            else:
                logger.info(f"Database '{db_name}' already exists.")

    except psycopg2.OperationalError as e:
        # Catch the specific connection error here for better messaging
        logger.error(f"Failed to connect to PostgreSQL at {db_host}:{db_port} to check/create database: {e}")
        logger.error("Please ensure the database server is running and accessible from WSL.")
        logger.error(f"If the server is on Windows, check Firewall rules for port {db_port} and listen_addresses='*' in postgresql.conf.")
        # Re-raise the error so the calling function (ingest_data) knows it failed
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while checking/creating database '{db_name}': {e}")
        raise

def ingest_data(raw_csv_file, db_name, table_name="data_table", if_exists="replace"):
    """
    Ingest data from a CSV file into a PostgreSQL database.

    Args:
        raw_csv_file (str): Path to the CSV file as Raw data.
        db_name (str): Name of the PostgreSQL database.
        table_name (str): Name of the table to create/update.
        if_exists (str): What to do if table exists ('replace', 'append', 'fail').
    """
    try:
        # Ensure database exists
        ensure_database_exists(db_name)
        
        # Load data from CSV
        logger.info(f"Loading data from {raw_csv_file}...")
        df = pd.read_csv(raw_csv_file)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns.")
        
        # Handle potential data cleaning/transformation
        # Convert date columns to datetime if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Converted '{col}' to datetime format.")
            except:
                logger.warning(f"Could not convert '{col}' to datetime.")
        
        # Connect to PostgreSQL database
        conn_string = get_connection_string(db_name)
        logger.info(f"Connecting to database '{db_name}'...")
        engine = create_engine(conn_string)
        
        # Write the DataFrame to PostgreSQL
        logger.info(f"Writing data to table '{table_name}' ({if_exists} mode)...")
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        
        records_count = len(df)
        logger.info(f"Successfully ingested {records_count} records into {db_name}.{table_name}")
        return records_count
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Ingest raw data from a CSV file into a PostgreSQL database."
    )
    parser.add_argument(
        "--input", "-i", 
        default='/mnt/d/MLOps2/data/labeled/labeled_twitter_20250423_020616.csv',
        help="Path to the input CSV file (raw data)."
    )
    parser.add_argument(
        "--db_name", "-d",
        default='Tweet',
        help="Name of the PostgreSQL database."
    )
    parser.add_argument(
        "--table", "-t",
        default="tweets",
        help="Name of the table in the PostgreSQL database (default: tweets)."
    )
    parser.add_argument(
        "--mode", "-m",
        default="replace",
        choices=["replace", "append", "fail"],
        help="How to handle existing tables: replace, append, or fail (default: replace)."
    )
    args = parser.parse_args()

    csv_file_path = args.input
    db_name = args.db_name
    table_name = args.table
    if_exists = args.mode

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file {csv_file_path} does not exist.")
        sys.exit(1)

    # Ingest data
    try:
        records_count = ingest_data(csv_file_path, db_name, table_name, if_exists)
        print(f"✅ Successfully ingested {records_count} records into {db_name}.{table_name}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()