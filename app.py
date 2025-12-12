import streamlit as st
from src.database import (
    get_db_connection,
    init_db,
    update_tickers_from_local_csv,
    get_ticker_details,
    get_all_ticker_ids,
    insert_historical_data,
    get_global_min_start_date, add_ticker_to_db_and_csv)
from src.fetcher import fetch_historical_data_for_tickers, DEFAULT_END_DATE

# Initialize the database and connection
init_db()

# --- Title and Config Updates ---
st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")
st.title("🇫🇷 Market Analysis Dashboard")

# Create a sidebar container for controls
with st.sidebar:
    st.header("Data Management")

    # 1. Load Tickers Button
    if st.button("🚀 Load Tickers & ICB from Local CSV"):
        with st.spinner("Loading data from local CSV..."):
            count = update_tickers_from_local_csv()
        if count > 0:
            st.success(f"✅ Tickers database updated. Total unique tickers processed: {count}")
        else:
            st.warning("⚠️ Tickers database update failed or zero tickers processed.")

    # 2. Fetch Historical Stock Data Button (Now bulk, efficient update)
    if st.button("⬇️ Fetch/Update Historical Stock Data (Bulk)"):
        with st.spinner("Preparing to fetch data..."):
            all_tickers = get_all_ticker_ids()
            if not all_tickers:
                st.warning("No tickers found in the database. Please load tickers from CSV first.")
            else:
                # 1. Get the single earliest date we need to fetch from across all tickers
                start_date_to_fetch = get_global_min_start_date()

                if start_date_to_fetch >= DEFAULT_END_DATE:
                    st.info("All tickers are already up-to-date with today's date.")
                else:
                    st.info(f"Fetching data for {len(all_tickers)} tickers starting from {start_date_to_fetch}...")

                    # 2. Fetch data in one bulk API call
                    historical_df = fetch_historical_data_for_tickers(
                        tickers=all_tickers,
                        start_date=start_date_to_fetch
                    )

                    if historical_df is not None and not historical_df.empty:
                        # 3. Insert/Replace data (this is where "update what's missing" happens efficiently)
                        rows_inserted = insert_historical_data(historical_df)
                        st.success(
                            f"✅ Historical data fetch complete. Total new data points (rows) inserted/updated: {rows_inserted}")
                    elif historical_df is not None:
                        st.info("⚠️ Fetch complete, but no new data points were available to insert.")

    # 3. Add Single Ticker Form
    st.header("Add Single Ticker")
    with st.form("add_ticker_form"):
        new_ticker_id = st.text_input("Ticker ID (e.g., AAPL)", max_chars=10).strip().upper()
        new_name = st.text_input("Company Name (e.g., Apple Inc.)").strip()
        new_icb_sector = st.text_input("ICB Sector (e.g., Technology)").strip()

        submitted = st.form_submit_button("➕ Add Ticker")

        if submitted:
            if not all([new_ticker_id, new_name, new_icb_sector]):
                st.error("All three fields are required.")
            else:
                success = add_ticker_to_db_and_csv(new_ticker_id, new_name, new_icb_sector)
                if success:
                    st.rerun()

# ----------------------------------------------------
# Main Dashboard
# ----------------------------------------------------

st.header("Database Overview")

try:
    ticker_df = get_ticker_details()
    if not ticker_df.empty:
        st.metric("Total Tickers Loaded", value=len(ticker_df))
        st.dataframe(ticker_df, use_container_width=True, height=250)

        # Adding back the historical data count for good measure
        conn = get_db_connection()
        historical_count = conn.execute("SELECT COUNT(*) FROM historical_data;").fetchone()[0]
        st.metric("Total Historical Data Points", value=f"{historical_count:,}")
    else:
        st.info("The 'tickers' table is empty. Click the 'Load Tickers & ICB from Local CSV' button to populate it.")
except Exception as e:
    st.error(f"Could not connect to or query the database: {e}")