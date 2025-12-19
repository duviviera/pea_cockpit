import pandas as pd
import streamlit as st
from src.database import (
    get_all_ticker_ids,
    get_ticker_details,
    get_db_counts,
    get_global_min_start_date,
    insert_historical_data,
    insert_dividends_data,
    get_historical_prices_data,
    get_dividends_data,
    update_tickers_from_local_csv,
    add_ticker_to_db_and_csv
)
from src.fetcher import fetch_historical_data_for_tickers, DEFAULT_END_DATE

st.set_page_config(layout="wide", page_title="Data Management")
st.title("⚙️ Data Management & Inspection")

# --- 1. Sidebar Controls for Data Loading/Fetching ---
with st.sidebar:
    st.header("Data Source Controls")

    # 1. Load Tickers Button
    # Note: Requires update_tickers_from_local_csv to be imported and available.
    if st.button("🚀 Load Tickers & ICB from Local CSV"):
        with st.spinner("Loading data from local CSV..."):
            count = update_tickers_from_local_csv()
        if count > 0:
            st.success(f"✅ Tickers database updated. Total unique tickers processed: {count}")
        else:
            st.warning("⚠️ Tickers database update failed or zero tickers processed.")

    st.markdown("---")

    # 2. Fetch Historical Stock Data Button (Bulk, efficient update)
    if st.button("⬇️ Fetch/Update Historical Stock Data (Bulk)"):
        with st.spinner("Preparing to fetch data..."):
            all_tickers = get_all_ticker_ids()
            if not all_tickers:
                st.warning("No tickers found in the database. Please load tickers from CSV first.")
            else:
                start_date_to_fetch = get_global_min_start_date()

                if start_date_to_fetch >= DEFAULT_END_DATE:
                    st.info("All tickers are already up-to-date with today's date.")
                else:
                    st.info(f"Fetching data for {len(all_tickers)} tickers starting from {start_date_to_fetch}...")

                    prices_df, dividends_df = fetch_historical_data_for_tickers(
                        tickers=all_tickers,
                        start_date=start_date_to_fetch
                    )

                    price_msg = insert_historical_data(prices_df)
                    st.success(f"✅ Prices: {price_msg}")

                    div_msg = insert_dividends_data(dividends_df)
                    st.success(f"💰 Dividends: {div_msg}")

                    if (prices_df is None or prices_df.empty) and (dividends_df is None or dividends_df.empty):
                        st.warning("⚠️ Fetch complete, but no new data points (prices or dividends) were available.")

# --- 2. Database Overview (Metrics) ---
st.header("Database Overview")
ticker_count, historical_count, dividend_count = get_db_counts()

if ticker_count == 0:
    st.warning('Please load tickers from CSV first.')
else:
    col_t, col_p, col_d = st.columns(3)
    col_t.metric("Total Tickers", ticker_count)
    col_p.metric("Price Rows", f"{historical_count}")
    col_d.metric("Dividend Payouts", f"{dividend_count:,}")

# --- 3. Add Ticker Form ---
st.header("Manually Add New Ticker")
with st.form("add_ticker_form", clear_on_submit=True):
    col_id, col_name, col_sector = st.columns(3)
    with col_id:
        new_ticker_id = st.text_input("Ticker ID (e.g., AAPL)", max_chars=10).strip().upper()
    with col_name:
        new_name = st.text_input("Company Name (e.g., Apple Inc.)").strip()
    with col_sector:
        new_icb_sector = st.text_input("ICB Sector (e.g., Technology)").strip()

    submitted = st.form_submit_button("➕ Add Ticker")

    if submitted:
        if not all([new_ticker_id, new_name, new_icb_sector]):
            st.error("All three fields are required.")
        else:
            success = add_ticker_to_db_and_csv(new_ticker_id, new_name, new_icb_sector)
            if success:
                st.success(f"Ticker {new_ticker_id} added successfully (Metadata update required)")
                st.rerun() # to force the reload of the ticker list

st.markdown("---")

# --- 4. Data Inspector ---
st.header("🔍 Inspect Data Rows")

all_tickers = get_all_ticker_ids()
if all_tickers:
    col_tick, col_start, col_end, col_btn = st.columns([2, 2, 2, 1])

    with col_tick:
        selected_ticker = st.selectbox("Select Ticker", options=all_tickers)

    with col_start:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))

    with col_end:
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    with col_btn:
        st.write("")
        st.write("")
        inspect_btn = st.button("Show Rows")

    if inspect_btn and selected_ticker:
        st.write(f"### Data for **{selected_ticker}**")

        tab_prices, tab_divs = st.tabs(["📉 Historical Prices", "💰 Dividends"])

        with tab_prices:
            prices_df_multiindex = get_historical_prices_data([selected_ticker], str(start_date), str(end_date))

            if not prices_df_multiindex.empty:
                prices_df = prices_df_multiindex.reset_index(level='ticker_id', drop=True).reset_index()
                st.dataframe(prices_df, use_container_width=True)
            else:
                st.warning(f"No price data found between {start_date} and {end_date}.")

        with tab_divs:
            divs_df_multiindex = get_dividends_data([selected_ticker], str(start_date), str(end_date))

            if not divs_df_multiindex.empty:
                divs_df = divs_df_multiindex.reset_index(level='ticker_id', drop=True).reset_index()
                st.dataframe(divs_df, use_container_width=True)
            else:
                st.info(f"No dividend data found between {start_date} and {end_date}.")