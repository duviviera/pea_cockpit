import pandas as pd
import yfinance as yf
import streamlit as st
from typing import List, Optional

# --- Configuration ---
DEFAULT_START_DATE = '2015-01-01'  # Keep this for reference
DEFAULT_END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')


# --- Fetching Logic ---

def fetch_historical_data_for_tickers(
        tickers: List[str],  # Back to a list of tickers
        start_date: str,  # Single start date for the bulk fetch
        end_date: str = DEFAULT_END_DATE
) -> Optional[pd.DataFrame]:
    """
    Fetches historical stock data from Yahoo Finance in a bulk call
    starting from a single calculated date (for maximum efficiency).

    Returns a long-format DataFrame suitable for database insertion.
    """
    if not tickers:
        st.warning("No tickers provided for fetching.")
        return None

    try:
        if start_date >= end_date:
            st.info(f"All tickers are up-to-date as of {end_date}.")
            return pd.DataFrame()

        # Fetch data for all tickers at once (THE BULK CALL)
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            progress=False,
            actions=False,
        )

        if data.empty:
            st.warning("Yahoo Finance returned no data for the requested period.")
            return None

        # Convert wide-format DataFrame to long-format for database
        if len(tickers) == 1:
            df = data.copy()
            df['ticker_id'] = tickers[0]
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'ticker_id']
        else:
            # Handle multi-index columns for multiple tickers
            data = data.stack(level=0).rename_axis(['Date', 'ticker_id']).reset_index()
            df = data
            df.columns = [
                'date', 'ticker_id', 'close', 'high', 'low', 'open', 'volume', 'adj_close'
            ]

        # Select and rename columns
        df = df[['ticker_id', 'date', 'open', 'high', 'low', 'close', 'volume']]

        # Ensure correct data types
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['ticker_id'] = df['ticker_id'].astype(str)
        df = df.dropna()

        return df

    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return None