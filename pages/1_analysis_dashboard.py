import streamlit as st
import pandas as pd
import datetime
from src.database import get_all_ticker_ids, get_historical_prices_data, get_ticker_metadata
from src.analysis import (
    calculate_time_series_indicators,
    calculate_ytd_return,
    calculate_dividend_indicators,
    get_normalized_comparison,
    calculate_single_ticker_summary,
    get_comparison_performance_data,
    calculate_yearly_evolution,  # NEW
    calculate_sector_rankings  # NEW
)
from src.charts import (
    plot_candlestick_chart,
    plot_comparison_chart,
    plot_force_comparison_chart,
    plot_evolution_bar_chart  # NEW
)


# --- Utility Functions for Ticker Handling ---
def clean_ticker_id(ticker_id: str) -> str:
    return ticker_id.upper().replace('.PA', '').strip()


def get_yf_ticker_id(clean_id: str) -> str:
    if '.' not in clean_id and clean_id.isalpha():
        return f"{clean_id}.PA"
    return clean_id


# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Analysis Dashboard")
st.title("📈 Stock Analysis Dashboard")

# --- 1. Data Preparation and Sidebar Controls ---
all_tickers_raw = get_all_ticker_ids()

if not all_tickers_raw:
    st.warning("⚠️ No tickers found in the database. Please go to the 'Data Management' page to load or fetch data.")
    st.stop()

all_tickers_clean = [clean_ticker_id(t) for t in all_tickers_raw]
default_clean_ticker = all_tickers_clean[0]
max_date = datetime.date.today()

with st.sidebar:
    st.header("Analysis Controls")
    selected_ticker_clean = st.selectbox("Select Primary Ticker", options=all_tickers_clean,
                                         index=all_tickers_clean.index(default_clean_ticker))
    comparison_options = [t for t in all_tickers_clean if t != selected_ticker_clean]
    comparison_tickers_clean = st.multiselect("Select Tickers for Comparison", options=comparison_options,
                                              default=comparison_options[0] if comparison_options else [])
    comparison_period = st.slider("Comparison Period (Trading Days)", min_value=90, max_value=365, value=252, step=30)
    st.markdown("---")
    st.header("Chart Period Selection")
    default_start_date = max_date - datetime.timedelta(days=365)
    start_date = st.date_input("Start Date", value=default_start_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=start_date, max_value=max_date)
    if start_date > end_date:
        st.error("Error: Start date cannot be after end date.")
        st.stop()
    st.markdown("---")
    st.info("Use the **⚙️ Data Management** page to load new tickers.")

selected_ticker_yf = get_yf_ticker_id(selected_ticker_clean)
comparison_tickers_yf = [get_yf_ticker_id(t) for t in comparison_tickers_clean]

# --- 2. Single Ticker Analysis Tab ---
tab_single, tab_comp = st.tabs(["🚀 Single Ticker Analysis", "⚖️ Comparison Analysis"])

with tab_single:
    # 1. Fetch all required data
    historical_mi = get_historical_prices_data(selected_ticker_yf)

    if historical_mi.empty:
        st.error(f"No historical price data found for {selected_ticker_yf}.")
    else:
        # Data preparation
        historical_df = historical_mi.droplevel('ticker_id')
        historical_df.index = pd.to_datetime(historical_df.index)

        metadata = get_ticker_metadata(selected_ticker_yf)
        company_name = metadata.get('name', 'N/A')
        sector = metadata.get('sector', 'N/A')
        current_price = historical_df['close'].iloc[-1] if not historical_df.empty else 0.0

        # 2. Calculate core metrics
        analyzed_df = calculate_time_series_indicators(historical_df)
        ytd_return = calculate_ytd_return(analyzed_df)
        div_metrics = calculate_dividend_indicators(selected_ticker_yf, current_price)
        summary_metrics = calculate_single_ticker_summary(analyzed_df, current_price, selected_ticker_yf,
                                                          comparison_period)

        # Calculate Ranks
        ranks = calculate_sector_rankings(selected_ticker_yf, sector)

        # Calculate Evolution Data
        evolution_df = calculate_yearly_evolution(selected_ticker_yf, historical_df)

        st.subheader(f"Details for {selected_ticker_yf} - {company_name}")

        # --- 1. Combined Data Table (Updated with Momentum Score/Rank) ---
        overview_data = {
            "Sector": [sector],
            "Current Price": [f"€{current_price:,.2f}"],
            "Div Yield (TTM)": [f"{div_metrics['yield_ttm'] * 100:,.2f}%"],
            "Sector Rank (Yield)": [ranks["yield_rank"]],
            "1Y Volatility": [f"{summary_metrics['Volatility_1Y'] * 100:,.2f}%"],
            "YTD Return": [f"{ytd_return * 100:,.2f}%"],
            "1Y Return": [f"{summary_metrics['Return_1Y'] * 100:,.2f}%"],
            "Sector Rank (Perf)": [ranks["perf_rank"]],
            "Sector Rank (Momentum)": [ranks["momentum_rank"]]
        }

        df_overview = pd.DataFrame(overview_data)

        st.dataframe(
            df_overview,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Sector Rank (Perf)": st.column_config.TextColumn("Sector Rank (Perf)", help="Ranked by 1Y Return in Sector"),
                "Sector Rank (Yield)": st.column_config.TextColumn("Sector Rank (Yield)",
                                                                   help="Ranked by TTM Yield in Sector"),
                "Sector Rank (Momentum)": st.column_config.TextColumn("Sector Rank (Momentum)",
                                                             help="Ranked by Momentum Score in Sector. Higher Z-Score is better.")
            }
        )

        st.markdown("---")

        st.header(f"📈 - Technical & Fundamental Evolution")

        # 1. Define the columns: 2/3 width for candlestick, 1/3 for vertical bar charts
        col_evolution, col_candlestick = st.columns([1, 2])

        with col_candlestick:
            st.subheader("Price History & Technical Indicators")
            analyzed_df_dates = analyzed_df.index.to_series().dt.date
            chart_data = analyzed_df[(analyzed_df_dates >= start_date) & (analyzed_df_dates <= end_date)].copy()

            if chart_data.empty:
                st.warning(f"No data available in the selected date range: {start_date} to {end_date}.")
            else:
                fig = plot_candlestick_chart(chart_data, selected_ticker_yf)
                st.plotly_chart(fig, use_container_width=True, height=650)

        with col_evolution:
            st.subheader("Yearly Evolution")
            fig_evolution = plot_evolution_bar_chart(evolution_df, selected_ticker_yf)

            # The height is set to match the candlestick chart height (650)
            st.plotly_chart(fig_evolution, use_container_width=True, height=650)

# --- 3. Comparison Analysis Tab ---
with tab_comp:
    if comparison_tickers_yf:
        tickers_to_compare_yf = comparison_tickers_yf + [selected_ticker_yf]
        comparison_summary_df = get_comparison_performance_data(tickers_to_compare_yf, comparison_period)
        comparison_df_hist = get_normalized_comparison(tickers_to_compare_yf, comparison_period)

        if comparison_summary_df.empty and comparison_df_hist.empty:
            st.warning("No performance data available for the selected tickers.")
        else:
            st.header(f"Comparative Performance Over {comparison_period} Trading Days")
            col_historical, col_force = st.columns(2)
            with col_historical:
                st.subheader("Historical Normalized Performance")
                st.markdown("Price action comparison, normalized to 100.")
                if not comparison_df_hist.empty:
                    fig_comp_hist = plot_comparison_chart(comparison_df_hist)
                    st.plotly_chart(fig_comp_hist, use_container_width=True, height=550)
                else:
                    st.info("No historical data to plot.")
            with col_force:
                st.subheader("Risk, Return, and Dividend")
                st.markdown("Volatility (Y) vs. Return (X) with Yield (Size/Color).")
                if not comparison_summary_df.empty:
                    fig_comp_force = plot_force_comparison_chart(comparison_summary_df)
                    st.plotly_chart(fig_comp_force, use_container_width=True, height=550)
                else:
                    st.info("No summary data to plot.")
    else:
        st.info("Select at least one other ticker in the sidebar to run a comparison analysis.")