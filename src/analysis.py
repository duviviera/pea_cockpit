import pandas as pd

from src.database import get_ticker_dividends, get_historical_prices_data


def calculate_time_series_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates moving averages and daily returns for a price DataFrame.
    """
    if df.empty:
        return df

    # --- Moving Averages (EMA 20/50/200) ---
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Daily returns (needed for other potential volatility/return analysis)
    df['Daily_Return'] = df['close'].pct_change()

    return df


def calculate_ytd_return(df: pd.DataFrame) -> float:
    """
    Calculates the Year-to-Date (YTD) return based on the close price.
    """
    if df.empty or len(df) < 2:
        return 0.0

    current_year = df.index[-1].year

    # Find the price on the first trading day of the current year
    ytd_start_price_series = df.loc[df.index.year == current_year, 'close']

    if ytd_start_price_series.empty:
        # Case: Data only started this year but after Jan 1st,
        # or data is missing for start of year. Cannot calculate YTD accurately.
        return 0.0

    start_price = ytd_start_price_series.iloc[0]
    end_price = df['close'].iloc[-1]

    # Calculate return: (End Price / Start Price) - 1
    if start_price > 0:
        return (end_price / start_price) - 1.0

    return 0.0


def calculate_dividend_indicators(ticker_id: str, current_price: float) -> dict:
    """
    Calculates advanced dividend metrics: Yield (TTM), 5Y CAGR, and Consecutive Growth Years.
    """
    div_df = get_ticker_dividends(ticker_id)

    metrics = {
        "yield_ttm": 0.0,
        "cagr_5y": 0.0,
        "years_growth": 0,
        "payout_count": 0
    }

    if div_df.empty or current_price <= 0:
        return metrics

    # 1. Yield TTM (Trailing 12 Months)
    one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
    ttm_divs = div_df[div_df.index >= one_year_ago]
    total_annual_payout = ttm_divs['amount'].sum()

    metrics['yield_ttm'] = (total_annual_payout / current_price) if current_price > 0 else 0.0
    metrics['payout_count'] = len(ttm_divs)

    # 2. Dividend Growth (CAGR & Streaks)
    # Group by year to handle multiple payments per year (quarterly/monthly)
    yearly_divs = div_df.resample('Y')['amount'].sum()
    # Remove current partial year for growth calc, or keep? Usually better to look at full completed years for CAGR.
    # For this simplified version, we'll use all data but be aware last year might be incomplete.

    if len(yearly_divs) >= 5:
        # 5 Year CAGR
        # We take the value 5 years ago and the last full year value
        try:
            latest_full_year = yearly_divs.iloc[-2]  # Last completed year (roughly)
            start_val = yearly_divs.iloc[-6]
            if start_val > 0:
                metrics['cagr_5y'] = ((latest_full_year / start_val) ** (1 / 5)) - 1
        except IndexError:
            metrics['cagr_5y'] = 0.0

    # 3. Consecutive Years of Growth (The "Aristocrat" check)
    streak = 0
    # Reverse iterate (Newest -> Oldest)
    # Note: Logic assumes yearly buckets.
    sorted_years = yearly_divs.sort_index(ascending=False)

    for i in range(len(sorted_years) - 1):
        if sorted_years.iloc[i] >= sorted_years.iloc[i + 1]:  # Allow flat or growth
            # Strict growth: sorted_years.iloc[i] > sorted_years.iloc[i+1]
            if sorted_years.iloc[i] > sorted_years.iloc[i + 1]:
                streak += 1
            else:
                # If it's equal, we might keep counting streak or stop depending on definition.
                # Let's stop if strictly looking for growth.
                pass
        else:
            break

    metrics['years_growth'] = streak

    return metrics


def get_normalized_comparison(tickers: list, period_days: int) -> pd.DataFrame:
    """
    Fetches data for multiple tickers, calculates cumulative returns,
    and normalizes them to a base of 100.
    """

    comparison_dfs = []

    # 1. Determine the start date
    end_date = pd.to_datetime('today').normalize()
    # Add buffer days for weekends/holidays
    start_date = end_date - pd.Timedelta(days=period_days + 15)

    # 2. Fetch all relevant price data in one bulk query from the DB
    historical_data_bulk = get_historical_prices_data(tickers=tickers,
                                                      start_date=start_date.strftime('%Y-%m-%d'))

    if historical_data_bulk.empty:
        return pd.DataFrame()

    # 3. Pivot the data: Index=Date, Columns=Ticker, Values=Close Price
    close_prices = historical_data_bulk['close'].unstack(level='ticker_id')

    # 4. Normalize prices to a base of 100
    normalized_df = pd.DataFrame(index=close_prices.index)

    for ticker in tickers:
        if ticker in close_prices.columns:
            series = close_prices[ticker].dropna()
            if not series.empty:
                # Find the starting value for normalization (first row of the series)
                first_value = series.iloc[0]

                if first_value > 0:
                    # Calculation: (Current Price / Starting Price) * 100
                    normalized_df[ticker] = (series / first_value) * 100

    return normalized_df.dropna(how='all')