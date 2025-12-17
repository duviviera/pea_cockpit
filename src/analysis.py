import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List, Optional
from src.database import get_ticker_dividends, get_historical_prices_data, get_ticker_details


# --- Utility Functions ---

def _calculate_annualized_volatility(df: pd.DataFrame, period_days: int) -> float:
    """Calculates annualized volatility based on daily log returns."""
    if len(df) < 2:
        return 0.0

    # Filter to the required period
    start_date = df.index[-1] - pd.Timedelta(days=period_days)
    period_df = df[df.index >= start_date].copy()

    # Calculate log returns
    period_df['Log_Return'] = np.log(period_df['close'] / period_df['close'].shift(1))

    # Annualize based on 252 trading days
    return period_df['Log_Return'].std() * np.sqrt(252)


# --- Core Analysis Functions ---

def calculate_time_series_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates moving averages, daily returns, and RSI.
    """
    if df.empty:
        return df.copy()

    df_copy = df.copy()

    # --- Moving Averages (EMA 20/50/200) ---
    df_copy['EMA_20'] = df_copy['close'].ewm(span=20, adjust=False).mean()
    df_copy['EMA_50'] = df_copy['close'].ewm(span=50, adjust=False).mean()
    df_copy['EMA_200'] = df_copy['close'].ewm(span=200, adjust=False).mean()

    # Daily returns
    df_copy['Daily_Return'] = df_copy['close'].pct_change()

    # --- RSI (Relative Strength Index) ---
    try:
        # Simple two-point RSI placeholder for demonstration
        delta = df_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
    except:
        df_copy['RSI'] = 50.0  # Fallback

    return df_copy


def calculate_ytd_return(df: pd.DataFrame) -> float:
    """
    Calculates the Year-to-Date (YTD) return based on the close price.
    """
    if df.empty or len(df) < 2:
        return 0.0

    current_year = df.index[-1].year

    ytd_start_price_series = df.loc[df.index.year == current_year, 'close']

    if ytd_start_price_series.empty:
        return 0.0

    start_price = ytd_start_price_series.iloc[0]
    end_price = df['close'].iloc[-1]

    if start_price > 0:
        return (end_price / start_price) - 1.0

    return 0.0


def calculate_dividend_indicators(ticker_id: str, current_price: float) -> dict:
    """
    Calculates advanced dividend metrics: Yield (TTM) and 5Y CAGR.
    """
    # Assuming get_ticker_dividends() returns a DataFrame with DatetimeIndex and 'amount' column
    div_df = get_ticker_dividends(ticker_id)

    metrics = {
        "yield_ttm": 0.0,
        "cagr_5y": 0.0,
    }

    if div_df.empty or current_price <= 0:
        return metrics

    # 1. Yield TTM (Trailing 12 Months) - Robust check to avoid overlap
    one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
    ttm_divs = div_df[div_df.index >= one_year_ago]
    total_annual_payout = ttm_divs['amount'].sum()

    metrics['yield_ttm'] = (total_annual_payout / current_price) if current_price > 0 else 0.0

    # 2. Dividend Growth (CAGR)
    yearly_divs = div_df.resample('Y')['amount'].sum()

    if len(yearly_divs) >= 5:
        try:
            # Look back from the latest available full year dividend sum
            # We take the value 5 years ago and the latest value
            latest_val = yearly_divs.iloc[-1]
            start_val = yearly_divs.iloc[-5]
            if start_val > 0 and latest_val > 0:
                metrics['cagr_5y'] = ((latest_val / start_val) ** (1 / 5)) - 1
        except IndexError:
            metrics['cagr_5y'] = 0.0

    return metrics


def get_normalized_comparison(tickers: list, period_days: int) -> pd.DataFrame:
    """
    Fetches data for multiple tickers, calculates cumulative returns,
    and normalizes them to a base of 100 for the time-series comparison plot.
    """

    end_date = pd.to_datetime('today').normalize()
    start_date = end_date - pd.Timedelta(days=period_days + 15)

    historical_data_bulk = get_historical_prices_data(tickers=tickers,
                                                      start_date=start_date.strftime('%Y-%m-%d'))

    if historical_data_bulk.empty:
        return pd.DataFrame()

    close_prices = historical_data_bulk['close'].unstack(level='ticker_id')
    normalized_df = pd.DataFrame(index=close_prices.index)

    for ticker in tickers:
        if ticker in close_prices.columns:
            series = close_prices[ticker].dropna()
            if not series.empty:
                first_value = series.iloc[0]

                if first_value > 0:
                    normalized_df[ticker] = (series / first_value) * 100

    return normalized_df.dropna(how='all')


def calculate_single_ticker_summary(analyzed_df: pd.DataFrame, current_price: float, ticker_id: str,
                                    period_days: int) -> Dict[str, Any]:
    """
    Calculates 1-year return, annualized volatility, and past dividend amounts
    for the single-ticker summary table.
    """

    summary = {'Return_1Y': 0.0, 'Volatility_1Y': 0.0, 'Div_Current_Year': 0.0, 'Div_Last_Year': 0.0}

    if analyzed_df.empty:
        return summary

    # 1. Volatility and Return (based on 252 days/period_days)
    start_date = analyzed_df.index[-1] - pd.Timedelta(days=period_days)
    period_df = analyzed_df[analyzed_df.index >= start_date].copy()

    if len(period_df) > 1:
        # 1-Year Total Return
        start_price = period_df['close'].iloc[0]
        end_price = period_df['close'].iloc[-1]
        summary['Return_1Y'] = (end_price / start_price) - 1.0

        # 1-Year Annualized Volatility (based on daily log returns)
        summary['Volatility_1Y'] = _calculate_annualized_volatility(analyzed_df, period_days)

    # 2. Current/Last Year Dividend Amounts
    div_df = get_ticker_dividends(ticker_id)  # Need fresh dividend data here

    if not div_df.empty:
        today = datetime.date.today()
        current_year = today.year
        last_year = today.year - 1

        div_df['date'] = pd.to_datetime(div_df.index)

        summary['Div_Current_Year'] = div_df[div_df['date'].dt.year == current_year]['amount'].sum()
        summary['Div_Last_Year'] = div_df[div_df['date'].dt.year == last_year]['amount'].sum()

    return summary

def get_comparison_performance_data(tickers_yf: List[str], period_days: int) -> pd.DataFrame:
    """
    Aggregates Volatility, Return, and Yield for the force comparison plot.
    """

    data = []

    for ticker_id in tickers_yf:
        # 1. Fetch Price Data
        historical_mi = get_historical_prices_data(ticker_id)
        if historical_mi.empty:
            continue

        historical_df = historical_mi.droplevel('ticker_id')
        historical_df.index = pd.to_datetime(historical_df.index)
        current_price = historical_df['close'].iloc[-1]

        # 2. Calculate Return and Volatility
        start_date = historical_df.index[-1] - pd.Timedelta(days=period_days)
        period_df = historical_df[historical_df.index >= start_date].copy()

        if len(period_df) > 1:
            start_price = period_df['close'].iloc[0]
            total_return = (current_price / start_price) - 1.0
            volatility = _calculate_annualized_volatility(historical_df, period_days)
        else:
            total_return = 0.0
            volatility = 0.0

        # 3. Calculate Dividend Yield
        div_metrics = calculate_dividend_indicators(ticker_id, current_price)
        dividend_yield = div_metrics['yield_ttm']

        data.append({
            'YF_Ticker': ticker_id,
            'Return': total_return,
            'Volatility': volatility,
            'Dividend Yield': dividend_yield
        })

    return pd.DataFrame(data)


def calculate_yearly_evolution(ticker_id: str, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a DataFrame containing Returns, Volatility, and Dividend Sums
    for the last 5 years.
    """
    current_year = datetime.datetime.now().year
    years = list(range(current_year - 4, current_year + 1))

    evol_data = {'Year': [], 'Return': [], 'Volatility': [], 'Dividends': []}

    # Fetch Dividends from DB
    div_df = get_ticker_dividends(ticker_id)

    for year in years:
        evol_data['Year'].append(year)

        # 1. Dividends
        if not div_df.empty:
            year_divs = div_df[div_df.index.year == year]
            evol_data['Dividends'].append(year_divs['amount'].sum() if not year_divs.empty else 0.0)
        else:
            evol_data['Dividends'].append(0.0)

        # 2. Price Metrics
        year_prices = historical_df[historical_df.index.year == year]

        if year == current_year:
            # Current Year: YTD Return, TTM Volatility
            ytd = calculate_ytd_return(historical_df)
            evol_data['Return'].append(ytd)

            # Volatility (Last 252 days)
            vol = _calculate_annualized_volatility(historical_df, 252)
            evol_data['Volatility'].append(vol)
        else:
            # Past Years
            if not year_prices.empty:
                start_p = year_prices['close'].iloc[0]
                end_p = year_prices['close'].iloc[-1]
                ret = (end_p - start_p) / start_p
                evol_data['Return'].append(ret)

                # Volatility for that specific year
                log_ret = np.log(year_prices['close'] / year_prices['close'].shift(1))
                vol = log_ret.std() * (252 ** 0.5)
                evol_data['Volatility'].append(vol)
            else:
                evol_data['Return'].append(0.0)
                evol_data['Volatility'].append(0.0)

    return pd.DataFrame(evol_data)


def _get_price_at_lag(df: pd.DataFrame, target_date: pd.Timestamp, window: int = 5) -> Optional[float]:
    """Helper to find price closest to a target date within a window."""
    df = df.sort_index()
    mask = (df.index >= target_date - pd.Timedelta(days=window)) & \
           (df.index <= target_date + pd.Timedelta(days=window))
    subset = df.loc[mask]

    if subset.empty:
        return None

    df_nearest = subset.copy()
    df_nearest['diff'] = abs(df_nearest.index - target_date)
    return df_nearest.sort_values('diff')['close'].iloc[0]


def _calculate_ticker_momentum_raw(ticker_id: str, ref_date: pd.Timestamp) -> Dict[str, float]:
    """
    Calculates the raw 6M and 12M risk-adjusted price momentum components.
    """
    # 1. Fetch 3+ Years of Data
    start_fetch = ref_date - pd.DateOffset(years=3, days=50)
    hist_mi = get_historical_prices_data(ticker_id, start_date=start_fetch.strftime('%Y-%m-%d'))

    if hist_mi.empty:
        return {}

    df = hist_mi.droplevel('ticker_id')
    df.index = pd.to_datetime(df.index)

    # 2. Calculate Volatility (Sigma_i) over 3 years prior to ref_date
    vol_start = ref_date - pd.DateOffset(years=3)
    vol_df = df.loc[df.index >= vol_start]['close']

    # Use weekly returns, annualized (52 weeks)
    weekly_df = vol_df.resample('W-FRI').last()

    if len(weekly_df) < 52:
        return {}

    weekly_ret = weekly_df.pct_change().dropna()
    sigma_i = weekly_ret.std() * np.sqrt(52)

    if sigma_i == 0:
        return {}

    # 3. Get Lagged Prices
    # T = Rebalancing Date (Ref Date)
    t_minus_1 = ref_date - pd.DateOffset(months=1)
    t_minus_7 = ref_date - pd.DateOffset(months=7)
    t_minus_13 = ref_date - pd.DateOffset(months=13)

    p_t1 = _get_price_at_lag(df, t_minus_1)
    p_t7 = _get_price_at_lag(df, t_minus_7)
    p_t13 = _get_price_at_lag(df, t_minus_13)

    # 4. Calculate Momenta (Assuming Rf = 0)
    rf = 0.0
    res = {}

    # 6-Month Momentum: (PT-1 / PT-7) - 1 - Rf
    if p_t1 and p_t7 and p_t7 > 0:
        mom_6 = (p_t1 / p_t7) - 1 - rf
        res['adj_mom_6'] = mom_6 / sigma_i
    else:
        res['adj_mom_6'] = np.nan

    # 12-Month Momentum: (PT-1 / PT-13) - 1 - Rf
    if p_t1 and p_t13 and p_t13 > 0:
        mom_12 = (p_t1 / p_t13) - 1 - rf
        res['adj_mom_12'] = mom_12 / sigma_i
    else:
        res['adj_mom_12'] = np.nan

    return res


def calculate_sector_rankings(target_ticker: str, sector: str) -> Dict[str, Any]:
    """
    Compares target vs peers. Calculates:
    1. Performance Rank (1Y Return)
    2. Yield Rank
    3. Momentum Score (Z-Score based) & Rank
    """
    # 1. Get Sector Peers
    all_tickers = get_ticker_details()
    sector_tickers = all_tickers[all_tickers['icb_sector'] == sector]['ticker_id'].tolist()

    if target_ticker not in sector_tickers:
        sector_tickers.append(target_ticker)

    # Optimization: limit to top 50 peers if sector is huge
    if len(sector_tickers) > 50 and target_ticker in sector_tickers:
        sector_tickers = [target_ticker] + [t for t in sector_tickers if t != target_ticker][:49]

    # 2. Bulk Metrics Collection
    today = pd.Timestamp.now()
    metrics_list = []
    start_date_1y = (today - datetime.timedelta(days=380)).strftime('%Y-%m-%d')

    for t in sector_tickers:
        # Initialize all keys to ensure DataFrame creation is successful
        row = {'ticker': t, 'return_1y': np.nan, 'yield': np.nan, 'adj_mom_6': np.nan, 'adj_mom_12': np.nan}

        try:
            # A. Basic Metrics (1Y Ret, Yield)
            hist_mi = get_historical_prices_data(t, start_date=start_date_1y)
            if not hist_mi.empty:
                hist = hist_mi.droplevel('ticker_id')
                current_price = hist['close'].iloc[-1]
                start_price = hist['close'].iloc[0]
                row['return_1y'] = (current_price / start_price) - 1.0
                div_m = calculate_dividend_indicators(t, current_price)
                row['yield'] = div_m['yield_ttm']

            # B. Momentum Metrics
            mom_raw = _calculate_ticker_momentum_raw(t, today)
            row.update(mom_raw)

            metrics_list.append(row)
        except Exception:
            metrics_list.append(row)
            continue

    if not metrics_list:
        return {"perf_rank": "N/A", "yield_rank": "N/A", "momentum_score": np.nan, "momentum_rank": "N/A"}

    df_ranks = pd.DataFrame(metrics_list)

    # 3. Calculate Momentum Z-Score

    # Calculate Z-Scores
    mom_6_mean = df_ranks['adj_mom_6'].mean()
    mom_6_std = df_ranks['adj_mom_6'].std()

    mom_12_mean = df_ranks['adj_mom_12'].mean()
    mom_12_std = df_ranks['adj_mom_12'].std()

    # Z-Score standardization: Z = (X - mean) / std.
    df_ranks['z_score_6'] = (df_ranks['adj_mom_6'] - mom_6_mean) / mom_6_std if mom_6_std else 0
    df_ranks['z_score_12'] = (df_ranks['adj_mom_12'] - mom_12_mean) / mom_12_std if mom_12_std else 0

    # Combined Z-score (0.5*Z_6 + 0.5*Z_12), with fallback for missing components
    df_ranks['Momentum_Z'] = df_ranks.apply(
        lambda row: row['z_score_6'] if pd.isna(row['z_score_12']) else (
            row['z_score_12'] if pd.isna(row['z_score_6']) else
            (row['z_score_6'] * 0.5 + row['z_score_12'] * 0.5)
        ), axis=1
    )

    # Remove rows where Momentum_Z is NaN (i.e., ticker had no data for both)
    df_ranks.dropna(subset=['Momentum_Z'], inplace=True)

    # Winsorization (+/- 3)
    df_ranks['Momentum_Z'] = df_ranks['Momentum_Z'].clip(lower=-3, upper=3)

    # 4. Calculate Final Ranks
    total_peers = len(df_ranks)

    if total_peers == 0:
        return {"perf_rank": "N/A", "yield_rank": "N/A", "momentum_score": np.nan, "momentum_rank": "N/A"}

    # Rank descending (Higher is better)
    df_ranks['rank_perf'] = df_ranks['return_1y'].rank(ascending=False, method='min')
    df_ranks['rank_yield'] = df_ranks['yield'].rank(ascending=False, method='min')
    df_ranks['rank_momentum'] = df_ranks['Momentum_Z'].rank(ascending=False, method='min')

    target_stats = df_ranks[df_ranks['ticker'] == target_ticker]

    if target_stats.empty:
        return {"perf_rank": "N/A", "yield_rank": "N/A", "momentum_score": np.nan, "momentum_rank": "N/A"}

    # Safely convert to int if not NaN, otherwise use 'N/A'
    perf_rank = int(target_stats['rank_perf'].iloc[0]) if not pd.isna(target_stats['rank_perf'].iloc[0]) else 'N/A'
    yield_rank = int(target_stats['rank_yield'].iloc[0]) if not pd.isna(target_stats['rank_yield'].iloc[0]) else 'N/A'
    momentum_rank = int(target_stats['rank_momentum'].iloc[0]) if not pd.isna(
        target_stats['rank_momentum'].iloc[0]) else 'N/A'
    momentum_score = target_stats['Momentum_Z'].iloc[0]

    return {
        "perf_rank": f"{perf_rank} / {total_peers}",
        "yield_rank": f"{yield_rank} / {total_peers}",
        "momentum_score": momentum_score,
        "momentum_rank": f"{momentum_rank} / {total_peers}"
    }