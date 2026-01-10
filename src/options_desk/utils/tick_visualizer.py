"""
Tick Data Visualization Utilities

Functions for visualizing high-frequency trade data with multiple views:
- Price trajectory with buy/sell markers
- Volume profile
- Trade size distribution
- Inter-arrival time histogram

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple, Union
from datetime import datetime, timedelta


def visualize_tick_data(
    df: pd.DataFrame,
    start_time: Optional[Union[str, datetime, int]] = None,
    end_time: Optional[Union[str, datetime, int]] = None,
    duration_seconds: Optional[int] = None,
    timestamp_col: str = 'timestamp',
    price_col: str = 'price',
    amount_col: str = 'amount',
    type_col: str = 'type',
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize tick-level trade data with multiple panels.

    Args:
        df: DataFrame with tick data
        start_time: Start time (datetime, string 'YYYY-MM-DD HH:MM:SS', or unix timestamp)
        end_time: End time (datetime, string, or unix timestamp)
        duration_seconds: Alternative to end_time - duration from start_time
        timestamp_col: Column name for timestamp (unix seconds)
        price_col: Column name for price
        amount_col: Column name for trade amount
        type_col: Column name for trade type ('buy' or 'sell')
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        title: Optional custom title

    Returns:
        matplotlib Figure object

    Example:
        # Visualize first minute
        fig = visualize_tick_data(
            df,
            start_time=df['timestamp'].min(),
            duration_seconds=60,
        )

        # Visualize specific time range
        fig = visualize_tick_data(
            df,
            start_time='2025-12-14 22:34:00',
            end_time='2025-12-14 22:35:00',
            save_path='tick_analysis.png',
        )
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Convert timestamp to numeric and datetime
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
    df['datetime'] = pd.to_datetime(df[timestamp_col], unit='s')

    # Convert price and amount to numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')

    # Handle time range
    if start_time is not None:
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time).timestamp()
        elif isinstance(start_time, datetime):
            start_time = start_time.timestamp()

        df = df[df[timestamp_col] >= start_time]

    if end_time is not None:
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time).timestamp()
        elif isinstance(end_time, datetime):
            end_time = end_time.timestamp()

        df = df[df[timestamp_col] <= end_time]
    elif duration_seconds is not None and start_time is not None:
        end_time = start_time + duration_seconds
        df = df[df[timestamp_col] <= end_time]

    if len(df) == 0:
        raise ValueError("No data in specified time range")

    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Separate buy and sell trades
    buys = df[df[type_col] == 'buy']
    sells = df[df[type_col] == 'sell']

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Main title
    if title is None:
        time_range = f"{df['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')} to {df['datetime'].iloc[-1].strftime('%H:%M:%S')}"
        title = f"Tick Data Analysis: {len(df):,} trades\n{time_range}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ========================================================================
    # Panel 1: Price trajectory with buy/sell markers
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Plot price line
    ax1.plot(df['datetime'], df[price_col], color='gray', alpha=0.3, linewidth=0.5, label='Price')

    # Scatter buys and sells
    ax1.scatter(buys['datetime'], buys[price_col], c='green', alpha=0.6, s=20, marker='^', label='Buy')
    ax1.scatter(sells['datetime'], sells[price_col], c='red', alpha=0.6, s=20, marker='v', label='Sell')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Trajectory with Trade Flow')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ========================================================================
    # Panel 2: Cumulative volume
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    df['cumulative_volume'] = df[amount_col].cumsum()
    buys['cumulative_volume'] = buys[amount_col].cumsum()
    sells['cumulative_volume'] = sells[amount_col].cumsum()

    ax2.plot(df['datetime'], df['cumulative_volume'], label='Total', linewidth=2)
    ax2.plot(buys['datetime'], buys['cumulative_volume'], label='Buy', alpha=0.7, color='green')
    ax2.plot(sells['datetime'], sells['cumulative_volume'], label='Sell', alpha=0.7, color='red')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Volume (BTC)')
    ax2.set_title('Cumulative Volume Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ========================================================================
    # Panel 3: Trade size distribution
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    # Remove extreme outliers for better visualization
    amount_99 = df[amount_col].quantile(0.99)
    amounts_clean = df[df[amount_col] <= amount_99][amount_col]

    ax3.hist(amounts_clean, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.axvline(df[amount_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df[amount_col].mean():.4f}')
    ax3.axvline(df[amount_col].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df[amount_col].median():.4f}')

    ax3.set_xlabel('Trade Size (BTC)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Trade Size Distribution (99th percentile: {amount_99:.4f} BTC)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # Panel 4: Inter-arrival time distribution
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    inter_arrivals = np.diff(df[timestamp_col].values)
    # Remove negative and very large outliers
    inter_arrivals_clean = inter_arrivals[(inter_arrivals > 0) & (inter_arrivals < 10)]

    if len(inter_arrivals_clean) > 0:
        ax4.hist(inter_arrivals_clean, bins=50, alpha=0.7, edgecolor='black', color='coral')
        ax4.axvline(np.mean(inter_arrivals_clean), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(inter_arrivals_clean):.3f}s')

        ax4.set_xlabel('Inter-arrival Time (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Inter-arrival Time Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # Panel 5: Buy/Sell imbalance over time
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])

    # Resample to 1-second bins for smoother visualization
    df['second'] = df['datetime'].dt.floor('1s')
    buy_volume = buys.groupby(buys['datetime'].dt.floor('1s'))[amount_col].sum()
    sell_volume = sells.groupby(sells['datetime'].dt.floor('1s'))[amount_col].sum()

    # Align indices
    all_seconds = pd.date_range(df['second'].min(), df['second'].max(), freq='1s')
    buy_volume = buy_volume.reindex(all_seconds, fill_value=0)
    sell_volume = sell_volume.reindex(all_seconds, fill_value=0)

    imbalance = buy_volume - sell_volume

    ax5.bar(all_seconds, imbalance, width=1/86400, color=['green' if x > 0 else 'red' for x in imbalance], alpha=0.7)
    ax5.axhline(0, color='black', linewidth=0.8)

    ax5.set_xlabel('Time')
    ax5.set_ylabel('Buy - Sell Volume (BTC/s)')
    ax5.set_title('Order Flow Imbalance (1-second bins)')
    ax5.grid(True, alpha=0.3, axis='y')

    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ========================================================================
    # Add statistics text box
    # ========================================================================
    stats_text = (
        f"Trades: {len(df):,}\n"
        f"Buys: {len(buys):,} ({len(buys)/len(df)*100:.1f}%)\n"
        f"Sells: {len(sells):,} ({len(sells)/len(df)*100:.1f}%)\n"
        f"Total Volume: {df[amount_col].sum():.2f} BTC\n"
        f"Price Range: ${df[price_col].min():.2f} - ${df[price_col].max():.2f}\n"
        f"Avg Trade Size: {df[amount_col].mean():.4f} BTC"
    )

    fig.text(0.98, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='bottom', horizontalalignment='right')

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")

    return fig


def plot_price_microstructure(
    df: pd.DataFrame,
    start_time: Optional[Union[str, datetime, int]] = None,
    duration_seconds: int = 60,
    timestamp_col: str = 'timestamp',
    price_col: str = 'price',
    amount_col: str = 'amount',
    type_col: str = 'type',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Detailed microstructure view: price, spread, and volume bars.

    Useful for analyzing market making dynamics at sub-second level.

    Args:
        df: DataFrame with tick data
        start_time: Start time
        duration_seconds: Duration to plot (default 60s)
        Other args: Same as visualize_tick_data()

    Returns:
        matplotlib Figure

    Example:
        fig = plot_price_microstructure(df, duration_seconds=30)
    """
    # Prepare data
    df = df.copy()
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
    df['datetime'] = pd.to_datetime(df[timestamp_col], unit='s')
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')

    # Filter time range
    if start_time is None:
        start_time = df[timestamp_col].min()
    elif isinstance(start_time, str):
        start_time = pd.to_datetime(start_time).timestamp()
    elif isinstance(start_time, datetime):
        start_time = start_time.timestamp()

    end_time = start_time + duration_seconds
    df = df[(df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)]
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No data in specified time range")

    buys = df[df[type_col] == 'buy']
    sells = df[df[type_col] == 'sell']

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])
    fig.suptitle(f"Microstructure View: {len(df):,} trades over {duration_seconds}s", fontweight='bold')

    # Top panel: Price with bid/ask markers
    ax1 = axes[0]

    ax1.plot(df['datetime'], df[price_col], color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    ax1.scatter(buys['datetime'], buys[price_col], c='green', alpha=0.7, s=30, marker='^', label='Buy', zorder=2)
    ax1.scatter(sells['datetime'], sells[price_col], c='red', alpha=0.7, s=30, marker='v', label='Sell', zorder=2)

    # Add rolling mid price
    df['mid_price'] = df[price_col].rolling(window=10, min_periods=1).mean()
    ax1.plot(df['datetime'], df['mid_price'], color='blue', linewidth=1.5, label='Rolling Mid', alpha=0.8, zorder=3)

    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Dynamics')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Volume bars
    ax2 = axes[1]

    # Create volume bars
    bar_width = pd.Timedelta(seconds=0.5)  # Half-second bars
    ax2.bar(buys['datetime'], buys[amount_col], width=bar_width, color='green', alpha=0.6, label='Buy Volume')
    ax2.bar(sells['datetime'], -sells[amount_col], width=bar_width, color='red', alpha=0.6, label='Sell Volume')

    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volume (BTC)')
    ax2.set_title('Order Flow')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved microstructure plot: {save_path}")

    return fig
