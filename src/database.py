import streamlit as st
import pandas as pd
import duckdb
import os
from typing import List, Optional, Union

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
    # --- NEW: Dividends Table ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dividends (
            ticker_id VARCHAR,
            date DATE,
            amount DOUBLE,
            PRIMARY KEY (ticker_id, date)
        );
    """)

def get_db_counts():

    conn = get_db_connection()
    ticker_count = len(get_ticker_details())
    historical_count = conn.execute("SELECT COUNT(*) FROM historical_data;").fetchone()[0]
    dividend_count = conn.execute("SELECT COUNT(*) FROM dividends;").fetchone()[0]

    return ticker_count, historical_count, dividend_count

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


# --- Historical Data Insertion ---

def get_global_min_start_date() -> str:
    """
    Finds the earliest missing date across all tickers in BOTH price and dividend tables.
    This sets the start date for the next bulk fetch.
    """
    conn = get_db_connection()

    def get_latest_date_in_table(table_name: str) -> Optional[str]:
        """Helper to find the minimum of the latest dates across all tickers in a table."""
        try:
            # Find the max date for each ticker, then take the minimum of those maximums
            latest_date_across_all = conn.execute(f"""
                SELECT MIN(max_date)
                FROM (
                    SELECT MAX(date) AS max_date
                    FROM {table_name}
                    FROM tickers
                    WHERE ticker_id IN (SELECT DISTINCT ticker_id FROM {table_name})
                    GROUP BY ticker_id
                );
            """).fetchone()[0]
            return latest_date_across_all
        except Exception:
            # Catches error if table is completely empty or non-existent
            return None

    latest_price_date = get_latest_date_in_table('historical_data')
    latest_dividend_date = get_latest_date_in_table('dividends')

    # Default start date (if a table is empty, or no data exists)
    default_start_date = pd.to_datetime(GLOBAL_DEFAULT_START_DATE).date()

    # Determine the required start date for each type
    def calculate_start_date(latest_db_date):
        if latest_db_date is None:
            # If table is empty, start from the absolute beginning
            return default_start_date
        else:
            # Start date is the day AFTER the latest date found
            return (pd.to_datetime(latest_db_date) + pd.Timedelta(days=1)).date()

    start_date_prices = calculate_start_date(latest_price_date)
    start_date_dividends = calculate_start_date(latest_dividend_date)

    # We must fetch from the EARLIEST required start date
    earliest_start_date = min(start_date_prices, start_date_dividends)

    return earliest_start_date.strftime('%Y-%m-%d')


def insert_historical_data(historical_df: pd.DataFrame) -> str:
    """
    Inserts historical price data into the DuckDB 'historical_data' table.
    """
    # 1. Get the cached connection
    conn = get_db_connection()

    if historical_df.empty:
        return "No new historical data to insert."

    # Validate/prepare DataFrame columns (Ensures column order and names are correct)
    required_cols = ['ticker_id', 'date', 'open', 'high', 'low', 'close', 'volume']
    if list(historical_df.columns) != required_cols:
        raise ValueError(f"DataFrame columns must match the schema: {required_cols}")

    rows_to_insert = len(historical_df)
    initial_row_count = conn.execute("SELECT count(*) FROM historical_data").fetchone()[0]

    try:
        conn.execute("BEGIN TRANSACTION")

        # --- A. Register DataFrame as a temporary view/table ---
        conn.register('temp_new_prices', historical_df)  # Use the whole DF

        # --- B. Perform the Insert with Duplicate Check ---
        # *** TABLE NAME IS CORRECTED HERE ***
        conn.execute("""
            INSERT INTO historical_data
            SELECT * FROM temp_new_prices AS tnp
            WHERE NOT EXISTS (
                SELECT 1 
                FROM historical_data AS p
                WHERE p.ticker_id = tnp.ticker_id 
                  AND p.date = tnp.date
            );
        """)

        # --- C. Cleanup and Commit ---
        conn.unregister('temp_new_prices')
        conn.execute("COMMIT")

        # 3. Calculate and return the actual number of inserted rows
        final_row_count = conn.execute("SELECT count(*) FROM historical_data").fetchone()[0]
        actual_inserted_rows = final_row_count - initial_row_count

        return (f"Successfully processed {rows_to_insert} rows. "
                f"Inserted {actual_inserted_rows} new rows into historical_data.")

    except Exception as e:
        # 4. Handle Error and Rollback
        print(f"Transaction failed. Rolling back changes. Error: {e}")
        conn.execute("ROLLBACK")
        raise


# --- Dividend Data Insertion ---

def insert_dividends_data(dividends_df: pd.DataFrame) -> str:
    """
    Inserts dividend data into the DuckDB 'dividends' table.
    Expects DataFrame cols: [ticker_id, date, amount]
    """
    conn = get_db_connection()

    if dividends_df.empty:
        return "No new dividend data to insert."

    required_cols = ['ticker_id', 'date', 'amount']
    if list(dividends_df.columns) != required_cols:
        # Map columns if they are slightly off, or raise error
        raise ValueError(f"Dividend DataFrame columns must match: {required_cols}")

    rows_to_insert = len(dividends_df)
    initial_row_count = conn.execute("SELECT count(*) FROM dividends").fetchone()[0]

    try:
        conn.execute("BEGIN TRANSACTION")

        conn.register('temp_new_dividends', dividends_df)

        # Insert only if (ticker_id, date) doesn't exist
        conn.execute("""
            INSERT INTO dividends
            SELECT * FROM temp_new_dividends AS tnd
            WHERE NOT EXISTS (
                SELECT 1 
                FROM dividends AS d
                WHERE d.ticker_id = tnd.ticker_id 
                  AND d.date = tnd.date
            );
        """)

        conn.unregister('temp_new_dividends')
        conn.execute("COMMIT")

        final_row_count = conn.execute("SELECT count(*) FROM dividends").fetchone()[0]
        actual_inserted = final_row_count - initial_row_count

        return f"Processed {rows_to_insert} dividend records. Inserted {actual_inserted} new."

    except Exception as e:
        conn.execute("ROLLBACK")
        print(f"Dividend Transaction failed: {e}")
        raise e

# --- Data Retrieval ---

def get_ticker_metadata(ticker_id: str) -> dict:
    """Fetches static metadata (name, sector, yield) for a ticker."""
    conn = get_db_connection()

    query = f"""
        SELECT name, icb_sector 
        FROM tickers
        WHERE ticker_id = '{ticker_id}';
    """
    metadata = conn.execute(query).fetchone()

    if metadata:
        return {"name": metadata[0], "sector": metadata[1]}
    return {}

def get_ticker_dividends(ticker_id: str) -> pd.DataFrame:
    """Fetches dividend history for a specific ticker."""
    conn = get_db_connection()
    query = f"""
        SELECT date, amount 
        FROM dividends 
        WHERE ticker_id = '{ticker_id}' 
        ORDER BY date ASC
    """
    df = conn.execute(query).fetchdf()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df


def get_historical_prices_data(
        tickers: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches historical price data for one or more tickers within an optional date range.

    Returns a DataFrame with MultiIndex (date, ticker_id) suitable for analysis.
    """
    conn = get_db_connection()

    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]

    if not tickers:
        return pd.DataFrame()

    ticker_list_str = ', '.join(f"'{t}'" for t in tickers)

    date_filter = ""
    if start_date:
        date_filter += f" AND date >= '{start_date}'"
    if end_date:
        date_filter += f" AND date <= '{end_date}'"

    query = f"""
        SELECT ticker_id, date, open, high, low, close, volume 
        FROM historical_data 
        WHERE ticker_id IN ({ticker_list_str})
        {date_filter}
        ORDER BY date ASC;
    """

    df = conn.execute(query).fetchdf()

    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    # MultiIndex is essential for bulk analysis (Comparator)
    df.set_index(['date', 'ticker_id'], inplace=True)

    return df.sort_index()


def get_dividends_data(
        tickers: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches dividend data for one or more tickers within an optional date range.

    Returns a DataFrame with MultiIndex (date, ticker_id).
    """
    conn = get_db_connection()

    if isinstance(tickers, str):
        tickers = [tickers]

    if not tickers:
        return pd.DataFrame()

    ticker_list_str = ', '.join(f"'{t}'" for t in tickers)

    date_filter = ""
    if start_date:
        date_filter += f" AND date >= '{start_date}'"
    if end_date:
        date_filter += f" AND date <= '{end_date}'"

    query = f"""
        SELECT ticker_id, date, amount 
        FROM dividends 
        WHERE ticker_id IN ({ticker_list_str})
        {date_filter}
        ORDER BY date ASC;
    """
    df = conn.execute(query).fetchdf()
    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date', 'ticker_id'], inplace=True)

    return df.sort_index()