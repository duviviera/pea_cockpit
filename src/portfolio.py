import difflib

import numpy as np
import pandas as pd
import io
import re
from typing import Dict
from datetime import datetime

from database import get_historical_prices_data


def parse_bourse_direct_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.xlsx'):
            # Use calamine engine to bypass openpyxl's styling engine entirely
            # This solves the PatternFill / Styles crash
            df = pd.read_excel(uploaded_file, engine='calamine')
        else:
            # CSV handling
            content = uploaded_file.read()
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(io.BytesIO(content), sep=';')
            except:
                df = pd.read_csv(io.BytesIO(content), sep=',')

        # --- Dynamic Header Detection (Improved) ---
        # Look for the row containing 'ISIN' if it's not the first row
        if 'ISIN' not in df.columns:
            # Search all cells for 'ISIN' (case-insensitive)
            mask = df.astype(str).apply(lambda row: row.str.contains('ISIN', case=False).any(), axis=1)
            if mask.any():
                idx = mask.idxmax()
                df.columns = [str(c).strip() for c in df.iloc[idx]]
                df = df.iloc[idx + 1:].reset_index(drop=True)

        # Standardize columns for mapping
        df.columns = [str(c).strip() for c in df.columns]

        col_map = {'ISIN': 'isin', 'Nom': 'bd_name', 'Quantité': 'quantity', 'PRU (EUR)': 'pru'}
        # Filter only columns we can find
        df = df[[c for c in col_map.keys() if c in df.columns]].rename(columns=col_map)

        # --- Numeric Cleaning (Handling non-breaking spaces) ---
        def clean_num(val):
            if pd.isna(val) or val == "": return 0.0
            if isinstance(val, (int, float)): return float(val)
            # Remove \s (whitespace) and \xa0 (non-breaking space common in FR files)
            s = str(val).replace(',', '.').replace('\xa0', '')
            s = re.sub(r'[^\d\.\-]', '', s)
            try:
                return float(s)
            except ValueError:
                return 0.0

        df['quantity'] = df['quantity'].apply(clean_num)
        df['pru'] = df['pru'].apply(clean_num)

        return df

    except Exception as e:
        raise Exception(f"Portfolio Parser Error: {str(e)}")


def suggest_tickers(portfolio_df: pd.DataFrame, db_tickers_map: Dict[str, str]) -> Dict[str, str]:
    """
    Improved similitude logic for matching names.
    """
    suggestions = {}
    # Create a lowercase map for case-insensitive exact matching
    # { "bic": "BIC", "air liquide": "AIR LIQUIDE" }
    db_names_lower = {name.lower(): name for name in db_tickers_map.keys()}
    db_names_list = list(db_tickers_map.keys())

    for _, row in portfolio_df.iterrows():
        original_name = str(row['name']).strip()
        search_name = original_name.lower()
        isin = row['isin']

        # 1. Try Exact Case-Insensitive Match (Fixes Bic vs BIC)
        if search_name in db_names_lower:
            actual_db_name = db_names_lower[search_name]
            suggestions[isin] = db_tickers_map[actual_db_name]
            continue

        # 2. Try matching the ISIN directly if it exists in your DB keys
        # (Optional: only if your db_tickers_map uses ISINs as keys elsewhere)

        # 3. Fuzzy Match with Pre-processing
        # We increase the cutoff slightly but match against lowercase for better results
        matches = difflib.get_close_matches(search_name, [n.lower() for n in db_names_list], n=1, cutoff=0.5)

        if matches:
            # Map the lowercase match back to the original DB casing to get the Ticker ID
            matched_lower = matches[0]
            original_db_name = db_names_lower[matched_lower]
            suggestions[isin] = db_tickers_map[original_db_name]
        else:
            suggestions[isin] = ""

    return suggestions


def calculate_marginal_risk(port_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Percentage Contribution to Risk for each asset.
    """
    if port_df.empty:
        return port_df

    tickers = port_df['ticker_id'].unique().tolist()

    # 1. Fetch historical returns (last 1 year)
    start_fetch = (datetime.today() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    hist_df = get_historical_prices_data(tickers, start_date=start_fetch)

    if hist_df.empty:
        port_df['marginal_risk'] = 0.0
        return port_df

    hist_df = hist_df.reset_index()

    # 2. Pivot and calculate returns
    # ffill() handles non-trading days/missing data points
    prices = hist_df.pivot(index='date', columns='ticker_id', values='close').ffill()
    returns = prices.pct_change().dropna()

    if returns.empty:
        port_df['marginal_risk'] = 0.0
        return port_df

    # 3. Covariance Matrix (Annualized)
    cov_matrix = returns.cov() * 252

    # 4. Weights
    # We filter only for tickers that actually had historical price data
    valid_tickers = cov_matrix.columns.tolist()
    total_val = port_df[port_df['ticker_id'].isin(valid_tickers)]['current_value'].sum()

    # Map weights to the exact order of the covariance matrix columns
    weight_map = port_df.set_index('ticker_id')['current_value'].to_dict()
    weights = np.array([weight_map[t] / total_val for t in valid_tickers])

    # 5. Risk Calculations
    port_variance = weights.T @ cov_matrix.values @ weights
    port_vol = np.sqrt(port_variance)

    if port_vol == 0:
        port_df['marginal_risk'] = 0.0
        return port_df

    # Component Risk Contribution calculation
    marginal_risk = (cov_matrix.values @ weights) / port_vol
    component_risk = weights * marginal_risk
    risk_pct = (component_risk / port_vol) * 100

    # 6. Map back to original dataframe
    risk_results = dict(zip(valid_tickers, risk_pct))
    port_df['marginal_risk'] = port_df['ticker_id'].map(lambda x: risk_results.get(x, 0.0))

    return port_df