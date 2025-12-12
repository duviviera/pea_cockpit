import streamlit as st
import pandas as pd
import datetime
from src.database import get_all_ticker_ids, get_historical_prices_data, get_ticker_metadata
from src.analysis import calculate_time_series_indicators, calculate_ytd_return, calculate_dividend_indicators, \
    get_normalized_comparison
from src.charts import plot_candlestick_chart, plot_comparison_chart


st.set_page_config(layout="wide", page_title="Analysis Dashboard")
st.title("📈 Stock Analysis Dashboard")

# --- 1. Sidebar Controls ---
all_tickers = get_all_ticker_ids()
if not all_tickers:
    st.warning("⚠️ No tickers found in the database. Please go to the 'Data Management' page to load or fetch data.")
    st.stop()

max_date = datetime.date.today()

with st.sidebar:
    st.header("Single Ticker Analysis")
    selected_ticker = st.selectbox("Select Ticker", options=all_tickers)

    st.header("Chart Period Selection")

    # --- Start and End Date Inputs (New Feature) ---
    # Define a default start date (e.g., 1 year ago)
    default_start_date = max_date - datetime.timedelta(days=365)

    start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        max_value=max_date,
    )

    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=start_date,  # Ensures a valid range
        max_value=max_date
    )

    # Validation
    if start_date > end_date:
        st.error("Error: Start date cannot be after end date. Please adjust your selection.")
        st.stop()

    st.markdown("---")
    st.header("Comparison Tool")
    comparison_tickers = st.multiselect(
        "Select Tickers for Comparison",
        options=[t for t in all_tickers],
        default=[all_tickers[0]] if len(all_tickers) > 0 else []
    )
    comparison_period = st.slider("Comparison Period (Days)", min_value=30, max_value=730, value=180, step=30)

    st.markdown("---")
    st.info("Use the **⚙️ Data Management** page to load new tickers and fetch data.")

# --- 2. Single Ticker Analysis Tab ---
tab_single, tab_comp = st.tabs(["🚀 Single Ticker Analysis", "⚖️ Comparison Analysis"])

with tab_single:
    # 1. Fetch data and metadata using the YF format
    historical_mi = get_historical_prices_data(selected_ticker)

    if historical_mi.empty:
        st.error(f"No historical price data found for {selected_ticker.split('.')[0]}.")
    else:
        # Data preparation
        historical_df = historical_mi.droplevel('ticker_id')

        # Ensure the index is a DatetimeIndex
        historical_df.index = pd.to_datetime(historical_df.index)

        metadata = get_ticker_metadata(selected_ticker)
        current_price = historical_df['close'].iloc[-1] if not historical_df.empty else 0.0

        # 2. Calculate indicators
        analyzed_df = calculate_time_series_indicators(historical_df)
        ytd_return = calculate_ytd_return(analyzed_df)
        div_metrics = calculate_dividend_indicators(selected_ticker, current_price)

        # --- Metrics Display ---
        # Display full YF ticker ID and company name
        st.header(f"Details for {selected_ticker.split('.')[0]} - {metadata.get('name', 'N/A')}")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Current Close", f"€{current_price:,.2f}")
        col2.metric("YTD Return", f"{ytd_return * 100:,.2f}%")
        col3.metric("Sector", metadata.get('icb_sector', 'N/A'))
        col4.metric("TTM Dividend Yield", f"{div_metrics['yield_ttm'] * 100:,.2f}%")
        col5.metric("5Y Dividend CAGR", f"{div_metrics['cagr_5y'] * 100:,.2f}%")

        st.markdown("---")

        # --- Candlestick Chart with Date Filtering ---
        st.subheader("Price & Technical Indicators")

        # Apply Date Filtering to the analyzed_df using the sidebar date inputs
        # Convert index of analyzed_df to date objects for comparison
        analyzed_df_dates = analyzed_df.index.to_series().dt.date

        chart_data = analyzed_df[
            (analyzed_df_dates >= start_date) &
            (analyzed_df_dates <= end_date)
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning

        if chart_data.empty:
            st.warning(f"No data available in the selected date range: {start_date} to {end_date}.")
        else:
            # Use the filtered data for the plot
            fig = plot_candlestick_chart(chart_data, selected_ticker)
            st.plotly_chart(fig, use_container_width=True, height=600)

        # --- Data Table (Technical Indicators) with Date Formatting ---
        st.subheader("Technical Data Snippet")
        display_cols = ['close', 'Daily_Return', 'EMA_20', 'EMA_50', 'EMA_200']

        # Prepare the DataFrame for display
        df_display = analyzed_df[display_cols].tail(10).copy()

        # Fix Date Format: Convert index to date-only string for display
        df_display.index = df_display.index.strftime('%Y-%m-%d')
        df_display.index.name = 'date'

        st.dataframe(
            df_display.style.format({
                'close': "{:.2f}",
                'Daily_Return': "{:.2%}",
                'EMA_20': "{:.2f}",
                'EMA_50': "{:.2f}",
                'EMA_200': "{:.2f}"
            })
            # The background_gradient requires 'matplotlib' to be installed.
            # Please run 'pip install matplotlib' in your environment.
            ,
            use_container_width=True
        )

# --- 3. Comparison Analysis Tab ---

with tab_comp:
    if comparison_tickers:
        st.header(f"Comparative Performance Over {comparison_period} Days")

        # Use the YF format tickers for fetching data
        tickers_to_compare_yf = comparison_tickers + [selected_ticker]
        comparison_df = get_normalized_comparison(tickers_to_compare_yf, comparison_period)

        if comparison_df.empty:
            st.warning("No data available for the selected tickers/period.")
        else:
            fig_comp = plot_comparison_chart(comparison_df)
            st.plotly_chart(fig_comp, use_container_width=True, height=600)

            # Performance Summary Table
            st.subheader("Performance Summary")

            summary_data = []
            for ticker in comparison_df.columns:
                final_index = comparison_df[ticker].iloc[-1]
                total_return = (final_index / 100.0) - 1.0
                summary_data.append({
                    "Ticker": ticker,
                    f"Return over {comparison_period} days": total_return
                })

            summary_df = pd.DataFrame(summary_data).set_index("Ticker")

            st.dataframe(
                summary_df.style.format({
                    f"Return over {comparison_period} days": "{:.2%}"
                }).background_gradient(cmap='RdYlGn', subset=[f"Return over {comparison_period} days"]),
                use_container_width=True
            )
    else:
        st.info("Select at least one other ticker in the sidebar to run a comparison analysis.")