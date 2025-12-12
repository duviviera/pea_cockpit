import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot_candlestick_chart(df: pd.DataFrame, ticker_id: str):
    """Generates an interactive Plotly Candlestick chart with EMAs and Volume."""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.8, 0.2])

    df.index = df.index.strftime('%Y-%m-%d')

    # Candlestick Trace (Row 1)
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Price'), row=1, col=1)

    # Moving Average Traces (Row 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=1), name='EMA 200'), row=1,
                  col=1)

    # Volume Bar Trace (Row 2)
    # Color volume bars based on price movement (green if close >= open, red otherwise)
    colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' for i in range(len(df))]
    # Note: marker_color accepts a list of colors for per-bar styling.
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title=f"**{ticker_id.split('.')[0]}** Candlestick Chart with EMAs",
        height=600,
        # Remove empty space from volume plot
        xaxis2=dict(type='category'),
        yaxis2=dict(title='Volume'),
    )

    return fig


def plot_comparison_chart(df: pd.DataFrame):
    """
    Generates a Plotly line chart comparing the normalized performance of multiple tickers.

    The input DataFrame 'df' is expected to have a DatetimeIndex and columns corresponding
    to ticker IDs, with values normalized to a starting base of 100.
    """

    fig = go.Figure()

    # Add a trace for each ticker (column in the DataFrame)
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))

    # Add a horizontal line at the 100% baseline
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Start (100%)")

    fig.update_layout(
        title="Comparative Performance (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Index (Start = 100)",
        legend_title="Ticker",
        hovermode="x unified",
        yaxis=dict(tickformat='.2f', separatethousands=True),
        height=500
    )

    return fig