"""
Visualization utilities for quantitative finance.

Professional, publication-quality plots for:
- Historical price data with regime overlays
- Volatility surfaces and smiles
- Simulated paths and distributions
- P&L and hedging performance
- Model calibration results

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple, Union

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'danger': '#D00000',       # Red
    'gray': '#6C757D',         # Gray
    'light_gray': '#E9ECEF',   # Light gray
}

# Regime colors
REGIME_COLORS = {
    0: '#FFE5E5',  # Light red (bear/high vol)
    1: '#E5F5E5',  # Light green (bull/low vol)
    2: '#E5E5FF',  # Light blue
    3: '#FFF5E5',  # Light orange
}

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def plot_historical_prices(
    data: pd.DataFrame,
    price_col: str = 'Close',
    date_col: str = 'Date',
    regime_labels: Optional[np.ndarray] = None,
    regime_names: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    show_volume: bool = True,
    volume_col: str = 'Volume',
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot historical price data with optional regime overlays.

    Creates a professional, publication-quality plot showing:
    - Price evolution over time
    - Volume bars (optional)
    - Regime periods highlighted with colored backgrounds
    - Clean formatting with proper date labels

    Args:
        data: DataFrame with price data
        price_col: Column name for prices (default 'Close')
        date_col: Column name for dates (default 'Date')
        regime_labels: Optional array of regime labels (same length as data)
        regime_names: Optional mapping from regime_id to regime_name
        title: Plot title (auto-generated if None)
        show_volume: Whether to show volume bars
        volume_col: Column name for volume
        figsize: Figure size (width, height)
        save_path: Path to save figure (if provided)
        show: Whether to display the plot

    Returns:
        Figure object

    Example:
        >>> # Simple price plot
        >>> fig = plot_historical_prices(btc_data)
        >>>
        >>> # With regime overlays
        >>> fig = plot_historical_prices(
        ...     btc_data,
        ...     regime_labels=regime_result.regime_labels,
        ...     regime_names={0: 'Bear', 1: 'Bull'},
        ... )
    """
    # Create figure with subplots
    if show_volume and volume_col in data.columns:
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=figsize,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # Extract data
    dates = pd.to_datetime(data[date_col])
    prices = data[price_col].values

    # Plot regimes as background shading
    if regime_labels is not None:
        _add_regime_backgrounds(ax1, dates, regime_labels, regime_names)
        if ax2 is not None:
            _add_regime_backgrounds(ax2, dates, regime_labels, regime_names)

    # Plot price line
    ax1.plot(
        dates,
        prices,
        color=COLORS['primary'],
        linewidth=1.5,
        label=price_col,
        zorder=10,  # Ensure line is on top of shading
    )

    # Formatting for price axis
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='y', labelsize=10)

    # Add legend if regimes exist
    if regime_labels is not None and regime_names is not None:
        # Create custom legend handles
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=REGIME_COLORS.get(r, '#E5E5E5'),
                  edgecolor='black', linewidth=0.5,
                  label=regime_names.get(r, f'Regime {r}'))
            for r in sorted(set(regime_labels))
        ]
        ax1.legend(
            handles=handles,
            loc='upper left',
            framealpha=0.9,
            edgecolor='black',
        )

    # Title
    if title is None:
        date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
        title = f"Historical Price Data: {date_range}"
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Volume subplot
    if ax2 is not None and volume_col in data.columns:
        volumes = data[volume_col].values

        # Color bars based on price change
        colors = []
        for i in range(len(prices)):
            if i == 0:
                colors.append(COLORS['gray'])
            else:
                colors.append(COLORS['success'] if prices[i] >= prices[i-1] else COLORS['danger'])

        ax2.bar(
            dates,
            volumes,
            color=colors,
            alpha=0.6,
            width=0.8,
            edgecolor='none',
        )

        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.tick_params(axis='y', labelsize=10)

        # Format volume numbers
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: _format_large_number(x))
        )

    # X-axis formatting (bottom subplot or single plot)
    bottom_ax = ax2 if ax2 is not None else ax1
    bottom_ax.set_xlabel('Date', fontsize=12, fontweight='bold')

    # Smart date formatting based on time range
    date_range_days = (dates.max() - dates.min()).days

    if date_range_days <= 7:
        # Hourly or daily
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        bottom_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    elif date_range_days <= 60:
        # Daily
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        bottom_ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    elif date_range_days <= 365:
        # Weekly
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        bottom_ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    else:
        # Monthly
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Rotate date labels
    plt.setp(bottom_ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    # Show if requested
    if show:
        plt.show()
        plt.close(fig)  # Close to prevent Jupyter from auto-displaying

    return fig


def _add_regime_backgrounds(
    ax: plt.Axes,
    dates: pd.Series,
    regime_labels: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
):
    """
    Add colored background shading for regime periods.

    Args:
        ax: Matplotlib axis
        dates: Date series
        regime_labels: Array of regime labels
        regime_names: Optional regime name mapping
    """
    # Find regime change points
    regime_changes = np.where(np.diff(regime_labels) != 0)[0] + 1
    regime_starts = np.concatenate([[0], regime_changes])
    regime_ends = np.concatenate([regime_changes, [len(regime_labels)]])

    # Add shaded rectangles for each regime period
    y_min, y_max = ax.get_ylim()
    if y_min == y_max:
        y_min, y_max = 0, 1  # Default range

    for start_idx, end_idx in zip(regime_starts, regime_ends):
        regime = regime_labels[start_idx]
        color = REGIME_COLORS.get(regime, '#E5E5E5')

        # Get date range for this regime
        x_start = mdates.date2num(dates.iloc[start_idx])
        x_end = mdates.date2num(dates.iloc[end_idx - 1])

        # Add rectangle
        rect = Rectangle(
            (x_start, y_min),
            x_end - x_start,
            y_max - y_min,
            facecolor=color,
            alpha=0.3,
            edgecolor='none',
            zorder=0,  # Send to back
        )
        ax.add_patch(rect)

    # Reset y-limits (rectangles may have changed them)
    ax.relim()
    ax.autoscale_view()


def _format_large_number(x: float) -> str:
    """
    Format large numbers with K, M, B suffixes.

    Args:
        x: Number to format

    Returns:
        Formatted string

    Example:
        >>> _format_large_number(1500)
        '1.5K'
        >>> _format_large_number(2500000)
        '2.5M'
    """
    if abs(x) >= 1e9:
        return f'{x/1e9:.1f}B'
    elif abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.1f}K'
    else:
        return f'{x:.0f}'
