from datetime import date
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
    return duckdb.connect(database=DB_FILE, read_only=False)


def init_db():
    conn = get_db_connection()
    if not os.path.exists('data'):
        os.makedirs('data')

    # Schema definition remains SQL
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tickers (ticker_id VARCHAR PRIMARY KEY, name VARCHAR, icb_sector VARCHAR);")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS historical_data (ticker_id VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT, PRIMARY KEY (ticker_id, date));")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dividends (ticker_id VARCHAR, date DATE, amount DOUBLE, PRIMARY KEY (ticker_id, date));")
    conn.execute("CREATE TABLE IF NOT EXISTS isin_mapping (isin VARCHAR PRIMARY KEY, ticker_id VARCHAR, name VARCHAR);")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS portfolio_history (snapshot_date DATE, ticker_id VARCHAR, quantity DOUBLE, pru DOUBLE, current_value DOUBLE, pnl_percent DOUBLE, PRIMARY KEY (snapshot_date, ticker_id));")


def get_db_counts():
    conn = get_db_connection()
    ticker_count = conn.table("tickers").count("ticker_id").fetchone()[0]
    historical_count = conn.table("historical_data").count("*").fetchone()[0]
    dividend_count = conn.table("dividends").count("*").fetchone()[0]
    return ticker_count, historical_count, dividend_count


# --- Ticker Management Functions ---

def update_tickers_from_local_csv() -> int:
    conn = get_db_connection()
    try:
        # Relational CSV reading
        df_rel = conn.read_csv(CSV_FILE_PATH)
        df = df_rel.project("ticker_id, name, icb_sector").df()

        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM tickers")
        conn.register('temp_tickers', df)
        conn.execute("INSERT INTO tickers SELECT * FROM temp_tickers")
        conn.execute("COMMIT")
        return len(df)
    except Exception as e:
        conn.execute("ROLLBACK")
        st.error(f"Error loading CSV: {e}")
        return 0


def get_ticker_details() -> pd.DataFrame:
    return get_db_connection().table("tickers").order("ticker_id").df()


def get_all_ticker_ids() -> List[str]:
    return get_db_connection().table("tickers").project("ticker_id").df()['ticker_id'].tolist()


def add_ticker_to_db_and_csv(ticker_id: str, name: str, icb_sector: str) -> bool:
    conn = get_db_connection()
    tid = ticker_id.strip().upper()

    try:
        conn.execute("INSERT OR REPLACE INTO tickers VALUES (?, ?, ?)", (tid, name.strip(), icb_sector.strip()))

        with open(CSV_FILE_PATH, 'a') as f:
            f.write(f"\n{tid},{name.strip()},{icb_sector.strip()}")
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False


def get_sector_peers(ticker_id: str) -> list:
    conn = get_db_connection()
    t_rel = conn.table("tickers")
    sector_res = t_rel.filter(f"ticker_id = '{ticker_id}'").project("icb_sector").fetchone()

    if not sector_res or not sector_res[0]: return []

    return t_rel.filter(f"icb_sector = '{sector_res[0]}' AND ticker_id != '{ticker_id}'") \
        .limit(5).project("ticker_id").df()['ticker_id'].tolist()


# --- Data Insertion (Historical & Dividends) ---

def get_global_min_start_date() -> str:
    conn = get_db_connection()

    def get_min_last_date(table_name):
        try:
            # Relational aggregate: find max date per ticker, then min of those
            return conn.table(table_name).aggregate("ticker_id, max(date) as md").aggregate("min(md)").fetchone()[0]
        except:
            return None

    last_p = get_min_last_date('historical_data')
    last_d = get_min_last_date('dividends')

    base_date = pd.to_datetime(GLOBAL_DEFAULT_START_DATE).date()
    p_start = (pd.to_datetime(last_p) + pd.Timedelta(days=1)).date() if last_p else base_date
    d_start = (pd.to_datetime(last_d) + pd.Timedelta(days=1)).date() if last_d else base_date

    return min(p_start, d_start).strftime('%Y-%m-%d')


def _bulk_insert_with_check(df: pd.DataFrame, table_name: str, keys: List[str]):
    conn = get_db_connection()
    if df.empty: return "No data."

    initial = conn.table(table_name).count("*").fetchone()[0]
    conn.execute("BEGIN TRANSACTION")
    conn.register('temp_in', df)

    # Anti-join logic using Relations syntax (sql method on relation)
    where_clause = " AND ".join([f"t.{k} = temp_in.{k}" for k in keys])
    conn.execute(f"""
        INSERT INTO {table_name} 
        SELECT * FROM temp_in 
        WHERE NOT EXISTS (SELECT 1 FROM {table_name} t WHERE {where_clause})
    """)

    conn.execute("COMMIT")
    final = conn.table(table_name).count("*").fetchone()[0]
    return f"Inserted {final - initial} new rows into {table_name}."


def insert_historical_data(df: pd.DataFrame):
    return _bulk_insert_with_check(df, "historical_data", ["ticker_id", "date"])


def insert_dividends_data(df: pd.DataFrame):
    return _bulk_insert_with_check(df, "dividends", ["ticker_id", "date"])


# --- Metadata & Dividends Retrieval ---

def get_ticker_metadata(ticker_id: str) -> dict:
    res = get_db_connection().table("tickers").filter(f"ticker_id='{ticker_id}'").fetchone()
    return {"name": res[1], "sector": res[2]} if res else {}


def get_ticker_dividends(ticker_id: str) -> pd.DataFrame:
    rel = get_db_connection().table("dividends").filter(f"ticker_id='{ticker_id}'").order("date")
    df = rel.project("date, amount").df()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df


def get_historical_prices_data(tickers, start_date=None, end_date=None) -> pd.DataFrame:
    conn = get_db_connection()
    rel = conn.table("historical_data")

    t_list = [tickers] if isinstance(tickers, str) else tickers
    if not t_list: return pd.DataFrame()

    rel = rel.filter(f"ticker_id IN {tuple(t_list) if len(t_list) > 1 else f'({repr(t_list[0])})'}")
    if start_date: rel = rel.filter(f"date >= '{start_date}'")
    if end_date: rel = rel.filter(f"date <= '{end_date}'")

    df = rel.order("date").df()
    if df.empty: return df
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index(['date', 'ticker_id']).sort_index()


# --- Portfolio & ISIN Mapping ---

def get_isin_mapping() -> dict:
    df = get_db_connection().table("isin_mapping").df()
    return dict(zip(df['isin'], df['ticker_id'])) if not df.empty else {}


def save_isin_mapping(isin: str, ticker_id: str, name: str):
    get_db_connection().execute("INSERT OR REPLACE INTO isin_mapping VALUES (?, ?, ?)",
                                (isin.strip(), ticker_id.strip().upper(), name.strip()))


def get_all_tickers_map() -> dict:
    df = get_db_connection().table("tickers").project("name, ticker_id").df()
    return dict(zip(df['name'], df['ticker_id']))


def get_portfolio_valuation(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    conn = get_db_connection()
    # ORM approach: Use a Relation with a QUALIFY window to get latest prices
    latest_prices = conn.table("historical_data").sql("""
        SELECT ticker_id, close as last_price, date as price_date 
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker_id ORDER BY date DESC) = 1
    """)

    enriched = latest_prices.join(conn.table("tickers"), on="ticker_id", how="left").df()
    merged = pd.merge(portfolio_df, enriched, on='ticker_id', how='left')

    merged['current_value'] = merged['quantity'] * merged['last_price']
    merged['total_cost'] = merged['quantity'] * merged['pru']
    merged['unrealized_pnl'] = merged['current_value'] - merged['total_cost']
    merged['pnl_percent'] = (merged['unrealized_pnl'] / merged['total_cost'] * 100).fillna(0)
    return merged


def save_portfolio_snapshot(valued_df: pd.DataFrame):
    conn = get_db_connection()
    snap = valued_df[['ticker_id', 'quantity', 'pru', 'current_value', 'pnl_percent']].copy()
    snap['snapshot_date'] = date.today()

    conn.execute("BEGIN TRANSACTION")
    conn.register('temp_snap', snap)
    conn.execute(
        "INSERT OR REPLACE INTO portfolio_history SELECT snapshot_date, ticker_id, quantity, pru, current_value, pnl_percent FROM temp_snap")
    conn.execute("COMMIT")


def get_latest_portfolio_snapshot():
    conn = get_db_connection()
    max_date = conn.table("portfolio_history").aggregate("max(snapshot_date)").fetchone()[0]
    if not max_date: return pd.DataFrame(), None

    df = conn.table("portfolio_history").filter(f"snapshot_date = '{max_date}'").project(
        "ticker_id, quantity, pru").df()
    return df, max_date


def get_full_snapshot_history() -> pd.DataFrame:
    return get_db_connection().table("portfolio_history").order("snapshot_date").df()


def clear_portfolio_history(only_last=False):
    conn = get_db_connection()
    if only_last:
        max_d = conn.table("portfolio_history").aggregate("max(snapshot_date)").fetchone()[0]
        if max_d: conn.execute(f"DELETE FROM portfolio_history WHERE snapshot_date = '{max_d}'")
    else:
        conn.execute("DELETE FROM portfolio_history")


def get_all_snapshot_dates():
    return \
    get_db_connection().table("portfolio_history").project("snapshot_date").distinct().order("snapshot_date DESC").df()[
        'snapshot_date'].tolist()


def get_portfolio_performance_series(benchmark_ticker: str = "^FCHI", start_date: str = None) -> pd.DataFrame:
    conn = get_db_connection()
    if not start_date:
        res = conn.table("portfolio_history").aggregate("min(snapshot_date)").fetchone()
        start_date = res[0] if res and res[0] else str(date.today())

    # ORM Chaining for the Performance Logic
    calendar = conn.sql(
        f"SELECT CAST(range AS DATE) as date FROM range('{start_date}'::DATE, CURRENT_DATE + INTERVAL '1 day', INTERVAL '1 day')")

    # ASOF Join for Daily Holdings
    holdings = calendar.join(conn.table("portfolio_history"), on="snapshot_date <= date", how="asof")

    # ASOF Join for Daily Prices
    prices = holdings.join(conn.table("historical_data"),
                           condition="portfolio_history.ticker_id = historical_data.ticker_id",
                           how="asof", left_on="date", right_on="date")

    # Calculations
    daily_perf = prices.project("""
        date, 
        CASE WHEN date = snapshot_date THEN current_value ELSE (quantity * close) END as val,
        (quantity * pru) as cost
    """).aggregate("date, sum(val) as portfolio_value, sum(cost) as portfolio_cost")

    # Benchmark
    bench = conn.table("historical_data").filter(f"ticker_id = '{benchmark_ticker}'")

    return daily_perf.join(bench, on="date", how="asof").project("""
        date, portfolio_value, portfolio_cost, close as benchmark_price
    """).order("date").df()