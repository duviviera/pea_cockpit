import pandas as pd
import yfinance as yf
import streamlit as st
from typing import List, Optional, Tuple

# --- Configuration ---
DEFAULT_START_DATE = '2015-01-01'  # Keep this for reference
DEFAULT_END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')


# --- Fetching Logic ---

def fetch_historical_data_for_tickers(
        tickers: List[str],
        start_date: str,
        end_date: str = DEFAULT_END_DATE) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetches both Price history AND Dividends in a bulk call.

    Returns:
        Tuple(prices_df, dividends_df)
    """
    if not tickers:
        st.warning("No tickers provided.")
        return None, None

    try:
        # Fetch with actions=True to get Dividends
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            actions=True,
            progress=False,
            threads=True,
            auto_adjust=True,
        )

        if data.empty:
            return None, None

        prices_list = []
        dividends_list = []

        # yfinance structure varies if 1 ticker vs multiple tickers
        if len(tickers) == 1:
            # Single ticker structure: Index=Date, Cols=[Open, Close, ..., Dividends, Stock Splits]
            ticker = tickers[0]
            df = data.copy()
            df.index.name = 'date'
            df = df.reset_index()

            # 1. Process Prices
            price_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            # Check if columns exist (sometimes Volume is missing if no trades)
            available_price_cols = [c for c in price_cols if c in df.columns]
            p_df = df[available_price_cols].copy()
            p_df['ticker_id'] = ticker
            p_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
                        inplace=True)
            prices_list.append(p_df)

            # 2. Process Dividends
            if 'Dividends' in df.columns:
                d_df = df[['date', 'Dividends']].copy()
                d_df = d_df[d_df['Dividends'] > 0]  # Keep only non-zero payouts
                d_df['ticker_id'] = ticker
                d_df.rename(columns={'Dividends': 'amount'}, inplace=True)
                dividends_list.append(d_df)

        else:
            # Multi-ticker structure: MultiIndex Columns (Ticker, Metric) or (Metric, Ticker)
            # Since we used group_by='ticker', Level 0 is Ticker, Level 1 is Metric
            # Structure: data[('AAPL', 'Close')], data[('AAPL', 'Dividends')]

            for ticker in tickers:
                if ticker not in data.columns.levels[0]:
                    continue

                ticker_data = data[ticker].copy()
                ticker_data.index.name = 'date'
                ticker_data = ticker_data.reset_index()

                # 1. Prices
                # Ensure we have the basic columns
                if 'Close' in ticker_data.columns:
                    p_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    # Handle missing volume
                    sel_cols = [c for c in p_cols if c in ticker_data.columns]
                    p_df = ticker_data[sel_cols].copy()
                    p_df['ticker_id'] = ticker
                    p_df.columns = [c.lower() for c in p_df.columns]  # lowercase
                    prices_list.append(p_df)

                # 2. Dividends
                if 'Dividends' in ticker_data.columns:
                    d_df = ticker_data[['date', 'Dividends']].copy()
                    d_df = d_df[d_df['Dividends'] > 0]
                    if not d_df.empty:
                        d_df['ticker_id'] = ticker
                        d_df.columns = ['date', 'amount', 'ticker_id']
                        dividends_list.append(d_df)

        # --- Finalize Prices DataFrame ---
        if prices_list:
            final_prices = pd.concat(prices_list, ignore_index=True)
            final_prices = final_prices[['ticker_id', 'date', 'open', 'high', 'low', 'close', 'volume']]
            final_prices['date'] = pd.to_datetime(final_prices['date']).dt.date
            final_prices = final_prices.dropna(subset=['close'])
        else:
            final_prices = pd.DataFrame()

        # --- Finalize Dividends DataFrame ---
        if dividends_list:
            final_dividends = pd.concat(dividends_list, ignore_index=True)
            final_dividends = final_dividends[['ticker_id', 'date', 'amount']]
            final_dividends['date'] = pd.to_datetime(final_dividends['date']).dt.date
        else:
            final_dividends = pd.DataFrame()

        return final_prices, final_dividends

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None