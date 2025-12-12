import streamlit as st
import pandas as pd
import duckdb
import os
from typing import List, Optional

# --- Configuration ---
DB_FILE = 'data/stock_analysis.duckdb'
CSV_FILE_PATH = 'data/euronext_sbf120_components.csv'
GLOBAL_DEFAULT_START_DATE = '2015-01-01'


# --- Database Connection and Initialization ---

@st.cache_resource
def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Initializes and returns the DuckDB connection."""
    conn = duckdb.connect(database=DB_FILE, read_only=False)
    return conn


def init_db():
    """Initializes the required tables if they don't exist."""
    conn = get_db_connection()
    # 1. Create the data directory if it doesn't exist (important for Docker mount)
    if not os.path.exists('data'):
        os.makedirs('data')

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            ticker_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            icb_sector VARCHAR
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            ticker_id VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker_id, date)
        );
    """)


# --- Ticker Management Functions ---

def update_tickers_from_local_csv() -> int:
    """
    Reads data from the local CSV file and populates the 'tickers' table.
    """
    conn = get_db_connection()
    try:
        df = conn.read_csv(
            CSV_FILE_PATH,
            header=True,
            dtype={
                'ticker_id': 'VARCHAR',
                'name': 'VARCHAR',
                'icb_sector': 'VARCHAR'
            }
        ).df()

        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM tickers")  # Clear old data before loading new
        conn.register('temp_tickers_df', df)

        conn.execute("""
            INSERT INTO tickers (ticker_id, name, icb_sector)
            SELECT ticker_id, name, icb_sector FROM temp_tickers_df;
        """)

        conn.execute("COMMIT")
        return len(df)

    except Exception as e:
        conn.execute("ROLLBACK")
        st.error(
            f"Error loading CSV data: {e}. Check if '{CSV_FILE_PATH}' exists and has the correct format (ticker_id, name, icb_sector).")
        return 0


def get_ticker_details() -> pd.DataFrame:
    """Fetches all ticker details from the database."""
    # Get the cached connection inside the function that uses it
    conn = get_db_connection()
    return conn.execute("SELECT * FROM tickers ORDER BY ticker_id").df()


def get_all_ticker_ids() -> List[str]:
    """Fetches a list of all ticker IDs from the database."""
    # Get the cached connection inside the function that uses it
    conn = get_db_connection()
    return conn.execute("SELECT ticker_id FROM tickers").fetchdf()['ticker_id'].tolist()


def add_ticker_to_db_and_csv(ticker_id: str, name: str, icb_sector: str) -> bool:
    """
    Inserts a single new ticker into the 'tickers' table and appends it
    to the underlying local CSV file for persistence.
    """
    conn = get_db_connection()
    ticker_id = ticker_id.strip().upper()

    # 1. Update the database table
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            """
            INSERT OR REPLACE INTO tickers (ticker_id, name, icb_sector) 
            VALUES (?, ?, ?);
            """,
            (ticker_id, name.strip(), icb_sector.strip())
        )
        conn.execute("COMMIT")
        st.toast(f"Ticker {ticker_id} added/updated in the database.")

    except Exception as e:
        conn.execute("ROLLBACK")
        st.error(f"Error adding ticker {ticker_id} to the database: {e}")
        return False

    # 2. Update the underlying CSV file for persistence
    try:
        # Format the row to match the CSV file structure
        new_row = f"\n{ticker_id},{name.strip()},{icb_sector.strip()}"

        # The 'a' mode appends to the file
        with open(CSV_FILE_PATH, 'a') as f:
            f.write(new_row)
        st.success(f"Ticker {ticker_id} appended to {CSV_FILE_PATH}.")
        return True

    except Exception as e:
        st.error(f"Error appending ticker to CSV: {e}")
        return False


# --- Historical Data Insertion (for use by fetcher.py) ---

def get_global_min_start_date() -> str:
    """
    Finds the single earliest date we need to fetch data for across all tickers.
    This is the day AFTER the oldest MAX(date) among all tickers that have data.
    If historical_data table is empty, returns the GLOBAL_DEFAULT_START_DATE.
    """
    conn = get_db_connection()
    # Find the OLDEST MAX(date) across all tickers that have data.
    query = """
    SELECT MIN(max_date) FROM (
        SELECT MAX(date) as max_date 
        FROM historical_data 
        GROUP BY ticker_id
    );
    """
    result = conn.execute(query).fetchone()

    if result and result[0] is not None:
        # The date found is the OLDEST last update date. We start the fetch the day after.
        oldest_max_date = pd.to_datetime(result[0])
        start_date = oldest_max_date + pd.Timedelta(days=1)
        return start_date.strftime('%Y-%m-%d')

    # If table is empty, use the defined historical start date
    return GLOBAL_DEFAULT_START_DATE

def insert_historical_data(df: pd.DataFrame) -> int:
    """Inserts historical stock data into the historical_data table."""
    if df.empty:
        return 0

    try:
        conn = get_db_connection()
        conn.execute("COMMIT")
        return len(df)

    except Exception as e:
        conn.execute("ROLLBACK")
        st.error(f"Error inserting historical data: {e}")
        return 0