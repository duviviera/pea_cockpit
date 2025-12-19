import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd


def plot_candlestick_chart(df: pd.DataFrame, ticker_id: str):
    """
    Plots a Candlestick chart with Volume and technical indicators (EMAs, RSI).
    Assumes df is indexed by Date and contains 'open', 'high', 'low', 'close',
    'volume', 'EMA_20', 'EMA_50', 'EMA_200', and 'RSI'.
    """
    if df.empty:
        return go.Figure().update_layout(title="No data to display.")

    # Create subplots with 3 rows: Candlestick/EMAs, Volume, and RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00CC96',
            decreasing_line_color='#EF553B'
        ),
        row=1, col=1
    )

    # 2. EMAs (Exponential Moving Averages)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line={'color': 'blue', 'width': 1}), row=1,
        col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line={'color': 'orange', 'width': 1}),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200', line={'color': 'red', 'width': 1}), row=1,
        col=1)

    # 3. Volume Trace (Bar chart)
    # Use per-bar coloring based on price movement
    colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker={'color': colors, 'opacity': 0.7}
        ),
        row=2, col=1
    )

    # 4. RSI (Relative Strength Index) Trace
    if 'RSI' in df.columns and not df['RSI'].isnull().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line={'color': 'purple', 'width': 1.5}),
            row=3, col=1)
        # Overbought/Oversold lines
        fig.add_trace(go.Scatter(x=df.index, y=[70] * len(df), mode='lines', line={'dash': 'dash', 'color': 'red'},
                                 name='Overbought (70)', opacity=0.5, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[30] * len(df), mode='lines', line={'dash': 'dash', 'color': 'green'},
                                 name='Oversold (30)', opacity=0.5, showlegend=False), row=3, col=1)

        # Update RSI axis
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)

    # Update layout and axis settings
    fig.update_layout(
        title=f'[{ticker_id}]',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        height=700,
        template='plotly_white'
    )

    # Hide X-axis tick labels for all but the last subplot
    fig.update_xaxes(showticklabels=False, row=1)
    fig.update_xaxes(showticklabels=False, row=2)
    fig.update_xaxes(showticklabels=True, row=3)

    # Clean up Y-axes
    fig.update_yaxes(title_text="Price / EMAs", row=1, col=1)
    fig.update_yaxes(title_text="Volume", showticklabels=False, row=2, col=1)

    return fig


def plot_comparison_chart(df: pd.DataFrame):
    """
    Generates a Plotly line chart comparing the normalized performance of multiple tickers.
    (Existing function logic preserved)
    """

    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))

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


def plot_force_comparison_chart(df: pd.DataFrame):
    """
    Plots a scatter chart comparing Return (X), Volatility (Y), and Dividend Yield (Color/Size).
    Assumes df contains 'YF_Ticker', 'Return', 'Volatility', 'Dividend Yield'.
    """
    if df.empty:
        return go.Figure().update_layout(title="No data to display.")

    df['Yield_Label'] = df['Dividend Yield'].apply(lambda x: f"{x:.2%}")

    fig = px.scatter(
        df,
        x='Volatility',
        y='Return',
        size='Dividend Yield',
        color='Dividend Yield',
        text='YF_Ticker',
        hover_data={
            'YF_Ticker': True,
            'Return': ':.2%',
            'Volatility': ':.2%',
            'Dividend Yield': False,
            'Yield_Label': True
        },
        labels={
            'Return': 'Total Return (X)',
            'Volatility': 'Annualized Volatility (Y)',
            'color': 'Dividend Yield',
            'Yield_Label': 'Dividend Yield'
        },
        color_continuous_scale=px.colors.sequential.Viridis,
        height=650
    )

    # Add annotations for context
    fig.add_annotation(
        text="Best: High Return, Low Risk",
        xref="paper", yref="paper", x=0.1, y=0.95, showarrow=False, font={'color': 'green', 'size': 14}
    )
    fig.add_annotation(
        text="Worst: Low Return, High Risk",
        xref="paper", yref="paper", x=0.9, y=0.05, showarrow=False, font={'color': 'red', 'size': 14}
    )

    fig.update_traces(textposition='top center',
                      marker={'opacity': 0.8, 'line': {'width': 1, 'color': 'DarkSlateGray'}})
    fig.update_layout(
        title="Risk vs. Return with Dividend Yield Bubble Size",
        xaxis={'tickformat': '.0%', 'zeroline': True, 'zerolinecolor': 'gray', 'zerolinewidth': 1},
        yaxis={'tickformat': '.0%', 'zeroline': True, 'zerolinecolor': 'gray', 'zerolinewidth': 1},
        template='plotly_white'
    )

    return fig

def plot_evolution_bar_chart(df: pd.DataFrame, ticker_id: str) -> go.Figure:
    """
    Plots the yearly evolution of Return, Volatility, and Dividends in three
    VERTICALLY aligned bar charts, sharing the same figure.
    """
    if df.empty:
        return go.Figure().update_layout(
            title=f"Yearly Evolution for {ticker_id}",
            annotations=[
                dict(text="No data to display for Yearly Evolution.",
                     xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            ]
        )

    # Create a figure with 3 rows and 1 column
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,  # Reduce spacing for vertical alignment
        subplot_titles=("Yearly Return", "Annualized Volatility", "Total Dividends (€)")
    )

    # 1. Yearly Return (Row 1)
    fig.add_trace(
        go.Bar(
            x=df['Year'],
            y=df['Return'],
            name='Return',
            marker_color='#1f77b4',
            hovertemplate='Year %{x}: %{y:.2%}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Return", tickformat=".0%", row=1, col=1)

    # 2. Annualized Volatility (Row 2)
    fig.add_trace(
        go.Bar(
            x=df['Year'],
            y=df['Volatility'],
            name='Volatility',
            marker_color='#ff7f0e',
            hovertemplate='Year %{x}: %{y:.2%}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Volatility", tickformat=".0%", row=2, col=1)

    # 3. Total Dividends (Row 3)
    fig.add_trace(
        go.Bar(
            x=df['Year'],
            y=df['Dividends'],
            name='Dividends',
            marker_color='#2ca02c',
            hovertemplate='Year %{x}: €%{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )
    fig.update_yaxes(title_text="Dividends (€)", tickprefix="€", row=3, col=1)

    # 4. Update X-axes (only the bottom one should have labels)
    for i in range(1, 4):
        fig.update_xaxes(
            tickvals=df['Year'].tolist(),
            ticktext=[str(y) for y in df['Year'].tolist()],
            row=i, col=1,
            showticklabels=(i == 3)  # Only show labels for the last subplot (Row 3)
        )

    # 5. Update overall layout
    fig.update_layout(
        title={
            'text': f"Yearly Evolution for {ticker_id}",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=650,  # Set a height that aligns with the candlestick chart
        margin=dict(t=50, b=50, l=60, r=20),
        template='plotly_white',
    )

    # Adjust subplot title size
    fig.for_each_annotation(lambda a: a.update(font=dict(size=14)))

    return fig


def plot_sector_risk_pie(df: pd.DataFrame):
    """Visualizes risk by sector. Robust against missing 'icb_sector' column."""

    # Safety: If column is missing, create a dummy one so the app doesn't crash
    if 'icb_sector' not in df.columns:
        df['icb_sector'] = 'Unknown'

    # Fill NaN sectors
    df['icb_sector'] = df['icb_sector'].fillna('Unknown')

    # Use marginal_risk if available, otherwise use current_value for weights
    weight_col = 'marginal_risk' if 'marginal_risk' in df.columns else 'current_value'

    sector_data = df.groupby('icb_sector')[weight_col].sum().reset_index()

    # Remove rows where weight is 0 to keep chart clean
    sector_data = sector_data[sector_data[weight_col] > 0]

    fig = px.pie(
        sector_data,
        values=weight_col,
        names='icb_sector',
        hole=0.4,
        title=f"Portfolio Distribution by Sector ({'Risk' if weight_col == 'marginal_risk' else 'Value'})",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig