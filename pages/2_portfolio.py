# pages/2_Portfolio.py
from datetime import timedelta, date

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import calculate_dividend_indicators
from charts import plot_sector_risk_pie
from src.portfolio import parse_bourse_direct_file, suggest_tickers, calculate_marginal_risk
from src.database import get_isin_mapping, get_all_tickers_map, save_isin_mapping, get_portfolio_valuation, \
    save_portfolio_snapshot, get_latest_portfolio_snapshot, get_full_snapshot_history, get_portfolio_performance_series

st.set_page_config(layout="wide", page_title="My Portfolio")

# Professional Styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

st.title("💼 My PEA Portfolio")

# --- Data Loading ---
uploaded_file = st.file_uploader("Upload Bourse Direct positions", type=['xlsx', 'csv'])

last_update_date = None
if uploaded_file:
    # Use the uploaded file data
    port_df = parse_bourse_direct_file(uploaded_file)
    st.info("Using data from uploaded file.")
else:
    # Fallback to the latest database snapshot
    port_df, last_update_date = get_latest_portfolio_snapshot()
    if not port_df.empty:
        st.success("Loaded latest portfolio snapshot from database.")
    else:
        st.warning("No data found. Please upload a Bourse Direct file to initialize your portfolio.")

# --- Logic Execution ---
if not port_df.empty:
    known_mappings = get_isin_mapping()  # Already saved in DB
    db_tickers_map = get_all_tickers_map()  # {Name: TickerID}
    if 'isin' in port_df.columns:
        unmapped_df = port_df[~port_df['isin'].isin(known_mappings.keys())]

        if not unmapped_df.empty:
            st.info("Mapping new securities...")
            suggestions = suggest_tickers(unmapped_df, db_tickers_map)
            with st.form("match_form"):
                user_inputs = {}
                for _, row in unmapped_df.iterrows():
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"**{row['name']}** ({row['isin']})")
                    user_inputs[row['isin']] = col2.text_input("Ticker", value=suggestions.get(row['isin'], ""),
                                                               key=row['isin'])
                if st.form_submit_button("Save Mappings"):
                    for isin, tick in user_inputs.items():
                        if tick:
                            orig_name = port_df[port_df['isin'] == isin]['name'].iloc[0]
                            save_isin_mapping(isin, tick.upper(), orig_name)
                    st.rerun()
            st.stop()
        # Apply mapping to create ticker_id
        port_df['ticker_id'] = port_df['isin'].map(known_mappings)

    mapped_df = port_df.dropna(subset=['ticker_id'])
    print(mapped_df.head())

    if not mapped_df.empty:
        # Enrich Data
        valued_df = get_portfolio_valuation(mapped_df)
        valued_df = calculate_marginal_risk(valued_df)

        # Calculate Overall Dividend Yield
        total_annual_div = 0
        for _, row in valued_df.iterrows():
            div_m = calculate_dividend_indicators(row['ticker_id'], row['last_price'])
            total_annual_div += (div_m['yield_ttm'] * row['current_value'])

        port_yield = (total_annual_div / valued_df['current_value'].sum() * 100) if not valued_df.empty else 0

        # --- Tabs Interface ---
        tab_dash, tab_perf, tab_maint = st.tabs(["📊 Dashboard", "📈 Performance Evolution", "🛠️ Maintenance"])

        with tab_dash:
            # Metrics Header
            total_val = valued_df['current_value'].sum()
            total_cost = valued_df['total_cost'].sum()
            total_pnl_pct = ((total_val / total_cost) - 1) * 100 if total_cost > 0 else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Value", f"{total_val:,.2f} €")
            c2.metric("Total PnL", f"{total_pnl_pct:.2f}%", delta=f"{(total_val - total_cost):,.2f} €")
            c3.metric("Div. Yield (TTM)", f"{port_yield:.2f}%")
            update_label = last_update_date.strftime('%d %b %Y') if last_update_date else "New Upload"
            c4.metric("Last Snapshot", update_label)
            if c5.button("💾 Save Snapshot"):
                save_portfolio_snapshot(valued_df)
                st.rerun()

            st.divider()

            # Charts & Table
            left, right = st.columns([1, 2])
            with left:
                st.plotly_chart(plot_sector_risk_pie(valued_df), use_container_width=True)

            with right:
                st.subheader("Detailed Positions")
                st.dataframe(
                    valued_df[['ticker_id', 'name', 'icb_sector', 'quantity', 'last_price', 'current_value', 'pnl_percent',
                               'marginal_risk']],
                    column_config={
                        "current_value": st.column_config.NumberColumn("Value", format="%.2f €"),
                        "last_price": st.column_config.NumberColumn("Price", format="%.2f €"),
                        "pnl_percent": st.column_config.NumberColumn("PnL %", format="%.2f%%"),
                        "marginal_risk": st.column_config.NumberColumn("Risk Contrib.", format="%.2f%%"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
        with tab_perf:
            # 1. Controls
            c1, c2, c3 = st.columns([1, 1, 1])

            # Get the earliest date in history for the picker
            history_meta = get_full_snapshot_history()
            min_hist_date = history_meta['snapshot_date'].min() if not history_meta.empty else date.today()

            with c1:
                start_date_input = st.date_input("Start Date", value=min_hist_date, min_value=min_hist_date)
            with c2:
                bench_id = st.selectbox("Benchmark", ["^FCHI", "^GSPC", "URTH.PA"])
            with c3:
                view_type = st.radio("View Mode", ["Daily (Market)", "Snapshots Only"], horizontal=True)

            # 2. Fetch Data
            df_perf = get_portfolio_performance_series(benchmark_ticker=bench_id, start_date=str(start_date_input))

            if not df_perf.empty:
                # Calculate PnL %
                df_perf['Portfolio %'] = (df_perf['portfolio_value'] / df_perf['portfolio_cost'] - 1) * 100

                # Normalize Benchmark
                first_bench = df_perf['benchmark_price'].iloc[0]
                first_port_pnl = df_perf['Portfolio %'].iloc[0]
                df_perf['Benchmark %'] = ((df_perf['benchmark_price'] / first_bench) * (
                            1 + first_port_pnl / 100) - 1) * 100

                # 3. Filtering for "Snapshots Only" if selected
                if view_type == "Snapshots Only":
                    # Only keep dates that exist in the portfolio_history table
                    snapshot_dates = history_meta['snapshot_date'].unique()
                    df_perf = df_perf[df_perf['date'].isin(snapshot_dates)]
                    plot_mode = 'lines+markers'
                else:
                    plot_mode = 'lines'

                # 4. Plot
                fig = px.line(df_perf, x='date', y=['Portfolio %', 'Benchmark %'],
                              title="Portfolio Performance vs Benchmark")

                if view_type == "Snapshots Only":
                    fig.update_traces(mode='markers+lines')

                fig.update_layout(hovermode="x unified", yaxis_ticksuffix="%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected range.")

        with tab_maint:
            st.warning("Warning: Deleting history cannot be undone.")

            col_del1, col_del2 = st.columns(2)

            if col_del1.button("🗑️ Delete Last Snapshot"):
                from src.database import clear_portfolio_history

                clear_portfolio_history(only_last=True)
                st.success("Last snapshot removed.")
                st.rerun()

            if col_del2.button("🔥 Clear ALL History"):
                # Double confirmation check
                confirm = st.checkbox("I am sure I want to wipe all portfolio history")
                if confirm:
                    from src.database import clear_portfolio_history

                    clear_portfolio_history(only_last=False)
                    st.success("All history cleared.")
                    st.rerun()