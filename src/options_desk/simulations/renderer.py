"""
Dynamic Matplotlib Renderer for HestonEnv

Provides real-time animated visualization with:
- Left side: Spot/volatility evolution and positions
- Right side: 3D Implied volatility surface

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from typing import Optional, List, Dict, Any
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class HestonEnvRenderer:
    """
    Dynamic matplotlib renderer for Heston environment.

    Layout:
    - Left column (2 rows):
      - Top: Spot price and volatility evolution (dual y-axis)
      - Bottom: Trading positions over time
    - Right column (2 rows):
      - Top: 3D Implied Volatility surface
      - Bottom: Volatility smile/skew by maturity

    Updates in real-time as environment steps forward.
    """

    def __init__(
        self,
        max_steps: int = 246,
        n_instruments: int = 1,
        figsize: tuple = (18, 11),
        update_interval: int = 1,  # Update every N steps
    ):
        """
        Initialize renderer.

        Args:
            max_steps: Maximum episode length
            n_instruments: Number of tradable instruments (1 + n_options)
            figsize: Figure size (width, height)
            update_interval: Update plots every N steps (1 = every step)
        """
        self.max_steps = max_steps
        self.n_instruments = n_instruments
        self.figsize = figsize
        self.update_interval = update_interval

        # History buffers
        self.t_history = []
        self.S_history = []
        self.v_history = []
        self.vol_history = []  # sqrt(v) in %
        self.positions_history = []  # List of position arrays
        self.portfolio_value_history = []  # Portfolio value over time

        # IV surface data
        self.iv_surface_data = None
        self.moneyness_grid = None
        self.ttm_grid = None

        # Matplotlib objects
        self.fig = None
        self.ax_price_vol = None  # Combined spot + vol
        self.ax_vol_twin = None  # Twin axis for volatility
        self.ax_positions = None
        self.ax_wealth_twin = None  # Twin axis for wealth
        self.ax_iv_surface = None
        self.ax_smile = None  # Volatility smile/skew
        self.lines = {}
        self.position_lines = []
        self.wealth_line = None
        self.smile_lines = []
        self.surface = None

        # Interactive mode flag
        self.interactive_mode = False

        # Style settings
        self.colors = {
            'spot': '#2E86AB',  # Blue
            'vol': '#A23B72',   # Purple
            'underlying': '#06A77D',  # Green
            'options': '#F18F01',  # Orange
            'grid': '#CCCCCC',
        }

    def initialize(self, interactive: bool = True):
        """
        Initialize matplotlib figure and axes.

        Args:
            interactive: Use plt.ion() for interactive updating
        """
        self.interactive_mode = interactive

        if self.interactive_mode:
            plt.ion()  # Interactive mode for real-time updates

        # Create figure with custom layout
        self.fig = plt.figure(figsize=self.figsize, facecolor='white')

        # Use GridSpec: 2 rows x 2 columns
        # Left column: price/vol (top), positions/wealth (bottom)
        # Right column: IV surface (top), smile/skew (bottom)
        gs = GridSpec(2, 2, figure=self.fig,
                      width_ratios=[1.2, 1],
                      height_ratios=[1, 1],
                      hspace=0.35, wspace=0.3,
                      left=0.08, right=0.96, top=0.94, bottom=0.08)

        # Left top: Spot price and volatility (dual y-axis)
        self.ax_price_vol = self.fig.add_subplot(gs[0, 0])
        self.ax_vol_twin = self.ax_price_vol.twinx()

        self.ax_price_vol.set_title('Market Dynamics', fontsize=13, fontweight='bold', pad=10)
        self.ax_price_vol.set_xlabel('Time Step', fontsize=10)
        self.ax_price_vol.set_ylabel('Spot Price', fontsize=10, color=self.colors['spot'])
        self.ax_vol_twin.set_ylabel('Volatility (%)', fontsize=10, color=self.colors['vol'])

        self.ax_price_vol.tick_params(axis='y', labelcolor=self.colors['spot'])
        self.ax_vol_twin.tick_params(axis='y', labelcolor=self.colors['vol'])

        self.ax_price_vol.grid(True, alpha=0.2, color=self.colors['grid'], linestyle='--')

        # Left bottom: Positions and Wealth
        self.ax_positions = self.fig.add_subplot(gs[1, 0])
        self.ax_wealth_twin = self.ax_positions.twinx()

        self.ax_positions.set_title('Trading Positions & Wealth', fontsize=13, fontweight='bold', pad=10)
        self.ax_positions.set_xlabel('Time Step', fontsize=10)
        self.ax_positions.set_ylabel('Position Size', fontsize=10, color='black')
        self.ax_wealth_twin.set_ylabel('Portfolio Value ($)', fontsize=10, color='#E63946')

        self.ax_wealth_twin.tick_params(axis='y', labelcolor='#E63946')

        self.ax_positions.grid(True, alpha=0.2, color=self.colors['grid'], linestyle='--')

        # Right top: IV Surface (3D)
        self.ax_iv_surface = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax_iv_surface.set_title('Implied Volatility Surface', fontsize=13, fontweight='bold', pad=10)
        self.ax_iv_surface.set_xlabel('Moneyness (K/S)', fontsize=9, labelpad=8)
        self.ax_iv_surface.set_ylabel('TTM (days)', fontsize=9, labelpad=8)
        self.ax_iv_surface.set_zlabel('IV (%)', fontsize=9, labelpad=8)
        self.ax_iv_surface.tick_params(labelsize=8)

        # Right bottom: Volatility smile/skew
        self.ax_smile = self.fig.add_subplot(gs[1, 1])
        self.ax_smile.set_title('Volatility Smile/Skew by Maturity', fontsize=13, fontweight='bold', pad=10)
        self.ax_smile.set_xlabel('Moneyness (K/S)', fontsize=10)
        self.ax_smile.set_ylabel('Implied Vol (%)', fontsize=10)
        self.ax_smile.grid(True, alpha=0.2, color=self.colors['grid'], linestyle='--')
        self.ax_smile.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='ATM')

        # Initialize empty plots
        self.lines['spot'], = self.ax_price_vol.plot(
            [], [], color=self.colors['spot'], linewidth=2.5, label='Spot Price', alpha=0.9
        )
        self.lines['vol'], = self.ax_vol_twin.plot(
            [], [], color=self.colors['vol'], linewidth=2.5, label='Volatility', alpha=0.9, linestyle='--'
        )

        # Add legends
        lines1, labels1 = self.ax_price_vol.get_legend_handles_labels()
        lines2, labels2 = self.ax_vol_twin.get_legend_handles_labels()
        self.ax_price_vol.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                                  framealpha=0.9, fontsize=9)

        if self.interactive_mode:
            plt.show(block=False)
            plt.pause(0.001)

    def update(
        self,
        t: int,
        S: float,
        v: float,
        positions: Optional[np.ndarray] = None,
        portfolio_value: Optional[float] = None,
        option_chain: Optional[Any] = None,
    ):
        """
        Update plots with new data point.

        Args:
            t: Current time step
            S: Current spot price
            v: Current variance
            positions: Current positions array [underlying, opt1, opt2, ...]
            portfolio_value: Current portfolio value
            option_chain: Current option chain (OptionChain object)
        """
        # Update history
        self.t_history.append(t)
        self.S_history.append(S)
        self.v_history.append(v)
        self.vol_history.append(np.sqrt(v) * 100)  # Convert to annualized %

        if positions is not None:
            self.positions_history.append(positions.copy())

        if portfolio_value is not None:
            self.portfolio_value_history.append(portfolio_value)

        # Only update plots at specified interval
        if t % self.update_interval != 0:
            return

        # Update spot price and volatility (dual axis)
        self.lines['spot'].set_data(self.t_history, self.S_history)
        self.lines['vol'].set_data(self.t_history, self.vol_history)

        self.ax_price_vol.relim()
        self.ax_price_vol.autoscale_view()
        self.ax_vol_twin.relim()
        self.ax_vol_twin.autoscale_view()

        # Update positions plot
        if len(self.positions_history) > 0:
            self._update_positions()

        # Update IV surface and smile if option chain is provided
        if option_chain is not None:
            self._update_iv_surface(option_chain, S)
            self._update_smile(option_chain, S)

        # Redraw
        if self.interactive_mode:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def _update_positions(self):
        """Update positions plot with current history."""
        if len(self.positions_history) == 0:
            return

        # Clear previous position lines
        for line in self.position_lines:
            line.remove()
        self.position_lines.clear()

        # Convert positions history to array
        positions_array = np.array(self.positions_history)  # shape: (n_steps, n_instruments)

        # Plot underlying position
        line, = self.ax_positions.plot(
            self.t_history[:len(self.positions_history)],
            positions_array[:, 0],
            color=self.colors['underlying'],
            linewidth=2.5,
            label='Underlying',
            alpha=0.9
        )
        self.position_lines.append(line)

        # Plot total options position (sum of all option positions)
        if positions_array.shape[1] > 1:
            total_options = np.sum(np.abs(positions_array[:, 1:]), axis=1)
            line, = self.ax_positions.plot(
                self.t_history[:len(self.positions_history)],
                total_options,
                color=self.colors['options'],
                linewidth=2.5,
                label='Total Options (abs)',
                alpha=0.9,
                linestyle='--'
            )
            self.position_lines.append(line)

        # Add zero line
        self.ax_positions.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        self.ax_positions.relim()
        self.ax_positions.autoscale_view()

        # Plot wealth on twin axis
        if len(self.portfolio_value_history) > 0:
            if self.wealth_line is not None:
                self.wealth_line.remove()

            self.wealth_line, = self.ax_wealth_twin.plot(
                self.t_history[:len(self.portfolio_value_history)],
                self.portfolio_value_history,
                color='#E63946',
                linewidth=2.5,
                label='Portfolio Value',
                alpha=0.9,
                linestyle='-'
            )

            self.ax_wealth_twin.relim()
            self.ax_wealth_twin.autoscale_view()

        # Combined legend
        lines_pos = self.position_lines
        labels_pos = [l.get_label() for l in lines_pos]
        if self.wealth_line is not None:
            lines_pos = lines_pos + [self.wealth_line]
            labels_pos = labels_pos + [self.wealth_line.get_label()]

        self.ax_positions.legend(lines_pos, labels_pos, loc='upper left', framealpha=0.9, fontsize=9)

    def _update_iv_surface(self, option_chain, S: float):
        """
        Update 3D IV surface from option chain.

        Args:
            option_chain: OptionChain object with options
            S: Current spot price
        """
        if not hasattr(option_chain, 'options') or len(option_chain.options) == 0:
            return

        # Extract data from option chain, separating calls and puts
        calls_moneyness = []
        calls_ttm = []
        calls_iv = []

        puts_moneyness = []
        puts_ttm = []
        puts_iv = []

        moneyness_list = []
        ttm_list = []
        iv_list = []

        for opt in option_chain.options:
            # Compute moneyness
            moneyness = opt.strike / S

            # Compute time to maturity in days
            ttm = (opt.expiry - option_chain.reference_date).days

            # Get implied volatility
            iv = opt.implied_volatility * 100  # Convert to %

            moneyness_list.append(moneyness)
            ttm_list.append(ttm)
            iv_list.append(iv)

            # Separate calls and puts
            if opt.option_type == 'call':
                calls_moneyness.append(moneyness)
                calls_ttm.append(ttm)
                calls_iv.append(iv)
            else:
                puts_moneyness.append(moneyness)
                puts_ttm.append(ttm)
                puts_iv.append(iv)

        if len(moneyness_list) == 0:
            return

        # Create grid for surface plot with interpolation
        try:
            # Convert to numpy arrays for interpolation
            points = np.column_stack([moneyness_list, ttm_list])
            values = np.array(iv_list)

            # Create a finer grid for smooth surface
            moneyness_min, moneyness_max = min(moneyness_list), max(moneyness_list)
            ttm_min, ttm_max = min(ttm_list), max(ttm_list)

            # Extend grid slightly for better visualization
            moneyness_range = moneyness_max - moneyness_min
            ttm_range = ttm_max - ttm_min

            grid_moneyness = np.linspace(
                moneyness_min - 0.02 * moneyness_range,
                moneyness_max + 0.02 * moneyness_range,
                50
            )
            grid_ttm = np.linspace(
                ttm_min - 0.02 * ttm_range,
                ttm_max + 0.02 * ttm_range,
                50
            )

            M_grid, T_grid = np.meshgrid(grid_moneyness, grid_ttm)

            # Interpolate using griddata (cubic for smoothness)
            IV_grid = griddata(
                points,
                values,
                (M_grid, T_grid),
                method='cubic',  # Smooth interpolation
                fill_value=np.nan
            )

            # Clear previous surface
            self.ax_iv_surface.clear()

            # Plot new surface with prettier styling (more transparent to see scatter points)
            self.surface = self.ax_iv_surface.plot_surface(
                M_grid, T_grid, IV_grid,
                cmap=cm.plasma,
                alpha=0.6,  # More transparent to see scatter points clearly
                edgecolor='none',
                antialiased=True,
                shade=True,
                vmin=np.nanmin(IV_grid) if not np.all(np.isnan(IV_grid)) else 0,
                vmax=np.nanmax(IV_grid) if not np.all(np.isnan(IV_grid)) else 100,
            )

            # Scatter calls and puts with different colors - LARGER and more visible
            if len(calls_moneyness) > 0:
                self.ax_iv_surface.scatter(
                    calls_moneyness,
                    calls_ttm,
                    calls_iv,
                    c='#00FF41',  # Bright green for calls
                    s=60,  # Much larger size
                    alpha=1.0,  # Fully opaque
                    edgecolors='darkgreen',
                    linewidths=1.5,
                    marker='o',
                    label='Calls',
                    depthshade=False  # Disable depth shading for better visibility
                )

            if len(puts_moneyness) > 0:
                self.ax_iv_surface.scatter(
                    puts_moneyness,
                    puts_ttm,
                    puts_iv,
                    c='#FF1744',  # Bright red for puts
                    s=60,  # Much larger size
                    alpha=1.0,  # Fully opaque
                    edgecolors='darkred',
                    linewidths=1.5,
                    marker='s',  # Square marker
                    label='Puts',
                    depthshade=False  # Disable depth shading for better visibility
                )

            # Update labels and title
            self.ax_iv_surface.set_title('Implied Volatility Surface', fontsize=13, fontweight='bold', pad=10)
            self.ax_iv_surface.set_xlabel('Moneyness (K/S)', fontsize=9, labelpad=8)
            self.ax_iv_surface.set_ylabel('TTM (days)', fontsize=9, labelpad=8)
            self.ax_iv_surface.set_zlabel('IV (%)', fontsize=9, labelpad=8)
            self.ax_iv_surface.tick_params(labelsize=8)

            # Set reasonable view angle
            self.ax_iv_surface.view_init(elev=20, azim=-60)

            # Set background color
            self.ax_iv_surface.xaxis.pane.fill = False
            self.ax_iv_surface.yaxis.pane.fill = False
            self.ax_iv_surface.zaxis.pane.fill = False

            # Add legend for call/put distinction
            self.ax_iv_surface.legend(
                loc='upper left',
                framealpha=0.95,
                fontsize=9,
                markerscale=0.8
            )

        except Exception as e:
            # If gridding fails, just skip this update
            print(f"Warning: IV surface update failed: {e}")
            pass

    def _update_smile(self, option_chain, S: float):
        """
        Update volatility smile/skew plot.

        Args:
            option_chain: OptionChain object with options
            S: Current spot price
        """
        if not hasattr(option_chain, 'options') or len(option_chain.options) == 0:
            return

        try:
            # Clear previous smile lines
            for line in self.smile_lines:
                line.remove()
            self.smile_lines.clear()

            # Group options by maturity
            from collections import defaultdict
            maturity_groups = defaultdict(lambda: {'moneyness': [], 'iv': []})

            for opt in option_chain.options:
                moneyness = opt.strike / S
                ttm = (opt.expiry - option_chain.reference_date).days
                iv = opt.implied_volatility * 100

                maturity_groups[ttm]['moneyness'].append(moneyness)
                maturity_groups[ttm]['iv'].append(iv)

            # Get unique maturities and sort
            maturities = sorted(maturity_groups.keys())

            # Color map for different maturities
            colors_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            # Plot smile for each maturity
            for idx, ttm in enumerate(maturities):
                moneyness_data = np.array(maturity_groups[ttm]['moneyness'])
                iv_data = np.array(maturity_groups[ttm]['iv'])

                # Sort by moneyness for clean line plot
                sort_idx = np.argsort(moneyness_data)
                moneyness_sorted = moneyness_data[sort_idx]
                iv_sorted = iv_data[sort_idx]

                # Plot
                color = colors_cycle[idx % len(colors_cycle)]
                line, = self.ax_smile.plot(
                    moneyness_sorted,
                    iv_sorted,
                    color=color,
                    linewidth=2.0,
                    marker='o',
                    markersize=4,
                    label=f'{ttm}d',
                    alpha=0.8
                )
                self.smile_lines.append(line)

            # Update axes
            self.ax_smile.relim()
            self.ax_smile.autoscale_view()

            # Update legend
            self.ax_smile.legend(
                loc='upper right',
                framealpha=0.9,
                fontsize=9,
                title='Maturity',
                title_fontsize=9
            )

            # Add ATM line
            self.ax_smile.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        except Exception as e:
            print(f"Warning: Smile update failed: {e}")
            pass

    def reset(self):
        """Reset renderer for new episode."""
        self.t_history.clear()
        self.S_history.clear()
        self.v_history.clear()
        self.vol_history.clear()
        self.positions_history.clear()
        self.portfolio_value_history.clear()

        if self.fig is not None:
            # Clear all plots
            self.lines['spot'].set_data([], [])
            self.lines['vol'].set_data([], [])

            # Clear position lines
            for line in self.position_lines:
                line.remove()
            self.position_lines.clear()

            # Clear wealth line
            if self.wealth_line is not None:
                self.wealth_line.remove()
                self.wealth_line = None

            # Clear surface
            if self.surface is not None:
                self.surface.remove()
                self.surface = None

            self.ax_iv_surface.clear()
            self.ax_iv_surface.set_title('Implied Volatility Surface', fontsize=13, fontweight='bold', pad=10)
            self.ax_iv_surface.set_xlabel('Moneyness (K/S)', fontsize=9, labelpad=8)
            self.ax_iv_surface.set_ylabel('TTM (days)', fontsize=9, labelpad=8)
            self.ax_iv_surface.set_zlabel('IV (%)', fontsize=9, labelpad=8)
            self.ax_iv_surface.tick_params(labelsize=8)

            # Clear smile lines
            for line in self.smile_lines:
                line.remove()
            self.smile_lines.clear()

            self.ax_smile.clear()
            self.ax_smile.set_title('Volatility Smile/Skew by Maturity', fontsize=13, fontweight='bold', pad=10)
            self.ax_smile.set_xlabel('Moneyness (K/S)', fontsize=10)
            self.ax_smile.set_ylabel('Implied Vol (%)', fontsize=10)
            self.ax_smile.grid(True, alpha=0.2, color=self.colors['grid'], linestyle='--')
            self.ax_smile.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

            if self.interactive_mode:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def close(self):
        """Close renderer and cleanup."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

        if self.interactive_mode:
            plt.ioff()

    def save_figure(self, filepath: str, dpi: int = 150):
        """
        Save current figure to file.

        Args:
            filepath: Output file path (e.g., 'output.png')
            dpi: Resolution in dots per inch
        """
        if self.fig is not None:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to: {filepath}")

    def get_figure(self):
        """Return the matplotlib figure object."""
        return self.fig
