"""
Implied Volatility Surface Tools

Standalone module for constructing and visualizing IV surfaces from option chains.
Completely independent from environment code - works with any OptionChain object.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Dict
from datetime import date
from scipy.interpolate import griddata

# Import option chain types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration.data.data_provider import OptionChain


class IVSurface:
    """
    Implied Volatility Surface constructor and visualizer.

    Extracts IV surface from an option chain and provides multiple
    visualization methods.

    Example:
        >>> # From synthetic chain
        >>> chain = generator.generate_single_chain(...)
        >>> iv_surface = IVSurface(chain)
        >>>
        >>> # Plot smile by maturity
        >>> iv_surface.plot_smile_by_maturity()
        >>>
        >>> # Plot 3D surface
        >>> iv_surface.plot_3d_surface()
        >>>
        >>> # Get skew metrics
        >>> metrics = iv_surface.get_skew_metrics()
    """

    def __init__(self, option_chain: OptionChain):
        """
        Initialize IV surface from option chain.

        Args:
            option_chain: OptionChain object containing option quotes
        """
        self.chain = option_chain
        self.spot = option_chain.spot_price
        self.reference_date = option_chain.reference_date

        # Extract data
        self._extract_surface_data()

    def _extract_surface_data(self):
        """Extract moneyness, TTM, and IV from option chain."""
        data = []

        for opt in self.chain.options:
            ttm_days = (opt.expiry - self.reference_date).days
            ttm_years = ttm_days / 365.0
            moneyness = opt.strike / self.spot

            data.append({
                'strike': opt.strike,
                'moneyness': moneyness,
                'ttm_days': ttm_days,
                'ttm_years': ttm_years,
                'option_type': opt.option_type,
                'iv': opt.implied_volatility,
                'mid': opt.mid,
                'bid': opt.bid,
                'ask': opt.ask,
            })

        self.df = pd.DataFrame(data)

        # Separate calls and puts
        self.calls = self.df[self.df['option_type'] == 'call'].copy()
        self.puts = self.df[self.df['option_type'] == 'put'].copy()

        # Get unique maturities
        self.maturities = sorted(self.df['ttm_days'].unique())

    def plot_smile_by_maturity(
        self,
        maturities: Optional[List[int]] = None,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot volatility smiles for each maturity.

        Args:
            maturities: List of maturities to plot (None = all)
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Figure object
        """
        if maturities is None:
            maturities = self.maturities

        # Determine grid layout
        n_maturities = len(maturities)
        n_cols = min(3, n_maturities)
        n_rows = (n_maturities + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_maturities == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, ttm in enumerate(maturities):
            ax = axes[idx]

            # Get options for this maturity
            ttm_calls = self.calls[self.calls['ttm_days'] == ttm].sort_values('moneyness')
            ttm_puts = self.puts[self.puts['ttm_days'] == ttm].sort_values('moneyness')

            # Plot calls
            if len(ttm_calls) > 0:
                ax.plot(
                    ttm_calls['moneyness'],
                    ttm_calls['iv'],
                    'o-',
                    color='blue',
                    linewidth=2,
                    markersize=8,
                    label='Calls',
                    alpha=0.7,
                )

            # Plot puts
            if len(ttm_puts) > 0:
                ax.plot(
                    ttm_puts['moneyness'],
                    ttm_puts['iv'],
                    's--',
                    color='red',
                    linewidth=2,
                    markersize=6,
                    label='Puts',
                    alpha=0.7,
                )

            # ATM line
            ax.axvline(x=1.0, color='black', linestyle=':', alpha=0.5, linewidth=1.5)

            # Formatting
            ax.set_xlabel('Moneyness (K/S)', fontsize=11)
            ax.set_ylabel('Implied Volatility', fontsize=11)
            ax.set_title(f'{ttm}-day Options', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_ylim(0, max(1.0, self.df['iv'].max() * 1.1))

        # Hide unused subplots
        for idx in range(n_maturities, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f'Volatility Smile by Maturity\\nSpot={self.spot:.2f}, Date={self.reference_date}',
            fontsize=14,
            fontweight='bold',
            y=1.00,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_3d_surface(
        self,
        use_calls_only: bool = True,
        grid_resolution: int = 50,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 3D implied volatility surface.

        Args:
            use_calls_only: Use only call options (True) or both (False)
            grid_resolution: Number of points in interpolation grid
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Select data
        if use_calls_only:
            data = self.calls.copy()
            title_suffix = "(Calls)"
        else:
            data = self.df.copy()
            title_suffix = "(Calls + Puts)"

        # Extract coordinates
        moneyness = data['moneyness'].values
        ttm = data['ttm_days'].values
        iv = data['iv'].values

        # Create interpolation grid
        m_min, m_max = moneyness.min(), moneyness.max()
        t_min, t_max = ttm.min(), ttm.max()

        m_grid = np.linspace(m_min, m_max, grid_resolution)
        t_grid = np.linspace(t_min, t_max, grid_resolution)
        M, T = np.meshgrid(m_grid, t_grid)

        # Interpolate IV surface
        IV = griddata(
            (moneyness, ttm),
            iv,
            (M, T),
            method='cubic',
            fill_value=np.nan,
        )

        # Plot surface
        surf = ax.plot_surface(
            M, T, IV,
            cmap='viridis',
            alpha=0.7,
            edgecolor='gray',
            linewidth=0.2,
            antialiased=True,
        )

        # Plot original points on top
        ax.scatter(
            moneyness, ttm, iv,
            c=iv,
            cmap='viridis',
            s=50,
            alpha=1.0,
            edgecolors='black',
            linewidths=0.5,
            depthshade=True,
        )

        # Plot vertical lines from points to z=0 plane
        for m, t, v in zip(moneyness, ttm, iv):
            ax.plot(
                [m, m], [t, t], [0, v],
                color='gray',
                linestyle='--',
                linewidth=0.5,
                alpha=0.4,
            )

        # Labels and formatting
        ax.set_xlabel('\\nMoneyness (K/S)', fontsize=11, labelpad=10)
        ax.set_ylabel('\\nTime to Maturity (days)', fontsize=11, labelpad=10)
        ax.set_zlabel('\\nImplied Volatility', fontsize=11, labelpad=10)
        ax.set_title(
            f'Implied Volatility Surface {title_suffix}\\nSpot={self.spot:.2f}, Date={self.reference_date}',
            fontsize=13,
            fontweight='bold',
            pad=20,
        )

        # Set z-axis to start at 0
        ax.set_zlim(bottom=0)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='IV')

        # Viewing angle
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def get_skew_metrics(
        self,
        ttm_days: Optional[int] = None,
    ) -> Dict:
        """
        Calculate skew metrics for a specific maturity.

        Args:
            ttm_days: Time to maturity in days (None = shortest maturity)

        Returns:
            Dictionary with skew metrics
        """
        if ttm_days is None:
            ttm_days = min(self.maturities)

        # Get calls for this maturity
        ttm_calls = self.calls[self.calls['ttm_days'] == ttm_days].sort_values('moneyness')

        if len(ttm_calls) == 0:
            return {'error': f'No options found for {ttm_days}-day maturity'}

        # Find ATM IV
        atm = ttm_calls.iloc[(ttm_calls['moneyness'] - 1.0).abs().argmin()]
        atm_iv = atm['iv']
        atm_moneyness = atm['moneyness']

        # Find OTM put (low strike)
        otm_puts = ttm_calls[ttm_calls['moneyness'] < 0.98]
        if len(otm_puts) > 0:
            otm_put = otm_puts.iloc[0]
            otm_put_iv = otm_put['iv']
            otm_put_moneyness = otm_put['moneyness']
            put_skew = otm_put_iv - atm_iv
        else:
            otm_put_iv = None
            otm_put_moneyness = None
            put_skew = None

        # Find OTM call (high strike)
        otm_calls = ttm_calls[ttm_calls['moneyness'] > 1.02]
        if len(otm_calls) > 0:
            otm_call = otm_calls.iloc[-1]
            otm_call_iv = otm_call['iv']
            otm_call_moneyness = otm_call['moneyness']
            call_skew = otm_call_iv - atm_iv
        else:
            otm_call_iv = None
            otm_call_moneyness = None
            call_skew = None

        # Determine smile type
        has_smile = False
        has_left_skew = False
        has_right_skew = False

        if put_skew is not None and call_skew is not None:
            has_smile = put_skew > 0 and call_skew > 0
            has_left_skew = put_skew > call_skew
            has_right_skew = call_skew > put_skew

        return {
            'ttm_days': ttm_days,
            'atm_iv': atm_iv,
            'atm_moneyness': atm_moneyness,
            'otm_put_iv': otm_put_iv,
            'otm_put_moneyness': otm_put_moneyness,
            'otm_call_iv': otm_call_iv,
            'otm_call_moneyness': otm_call_moneyness,
            'put_skew': put_skew,
            'call_skew': call_skew,
            'has_smile': has_smile,
            'has_left_skew': has_left_skew,
            'has_right_skew': has_right_skew,
        }

    def print_summary(self):
        """Print summary of IV surface characteristics."""
        print("="*80)
        print("IMPLIED VOLATILITY SURFACE SUMMARY")
        print("="*80)
        print(f"Spot Price: {self.spot:.2f}")
        print(f"Reference Date: {self.reference_date}")
        print(f"Number of Options: {len(self.df)}")
        print(f"  Calls: {len(self.calls)}")
        print(f"  Puts: {len(self.puts)}")
        print(f"\nMaturities (days): {self.maturities}")
        print(f"Moneyness Range: {self.df['moneyness'].min():.3f} - {self.df['moneyness'].max():.3f}")
        print(f"IV Range: {self.df['iv'].min():.2%} - {self.df['iv'].max():.2%}")

        print(f"\n{'='*80}")
        print("SKEW METRICS BY MATURITY")
        print("="*80)

        for ttm in self.maturities:
            metrics = self.get_skew_metrics(ttm)

            if 'error' in metrics:
                print(f"\n{ttm}-day: {metrics['error']}")
                continue

            print(f"\n{ttm}-day Options:")
            print(f"  ATM (K/S={metrics['atm_moneyness']:.3f}): IV={metrics['atm_iv']:.2%}")

            if metrics['otm_put_iv'] is not None:
                print(f"  OTM Put (K/S={metrics['otm_put_moneyness']:.3f}): "
                      f"IV={metrics['otm_put_iv']:.2%}, Skew={metrics['put_skew']:+.2%}")

            if metrics['otm_call_iv'] is not None:
                print(f"  OTM Call (K/S={metrics['otm_call_moneyness']:.3f}): "
                      f"IV={metrics['otm_call_iv']:.2%}, Skew={metrics['call_skew']:+.2%}")

            # Classify pattern
            if metrics['has_smile']:
                pattern = "Symmetric Smile (both wings up)"
            elif metrics['has_left_skew']:
                pattern = "Left Skew (equity-style)"
            elif metrics['has_right_skew']:
                pattern = "Right Skew (rare)"
            else:
                pattern = "Flat / Indeterminate"

            print(f"  Pattern: {pattern}")


def compare_iv_surfaces(
    chains: List[OptionChain],
    labels: List[str],
    figsize: Tuple[float, float] = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare multiple IV smiles side-by-side (2D plots).

    Note: For 3D surface comparison, chains must have multiple maturities.
    This function creates 2D smile plots for single-maturity comparison.

    Args:
        chains: List of OptionChain objects
        labels: Labels for each chain
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    n_chains = len(chains)
    fig, axes = plt.subplots(1, n_chains, figsize=figsize)

    if n_chains == 1:
        axes = [axes]

    for idx, (chain, label) in enumerate(zip(chains, labels)):
        ax = axes[idx]

        # Create IV surface
        iv_surf = IVSurface(chain)

        # Get unique maturities
        maturities = sorted(iv_surf.calls['ttm_days'].unique())

        # Plot each maturity
        for ttm in maturities:
            ttm_calls = iv_surf.calls[iv_surf.calls['ttm_days'] == ttm].sort_values('moneyness')

            if len(ttm_calls) > 0:
                ax.plot(
                    ttm_calls['moneyness'],
                    ttm_calls['iv'],
                    'o-',
                    linewidth=2.5,
                    markersize=8,
                    label=f'{ttm}d',
                    alpha=0.8,
                )

        # ATM line
        ax.axvline(x=1.0, color='black', linestyle=':', alpha=0.5, linewidth=1.5)

        # Labels
        ax.set_xlabel('Moneyness (K/S)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Implied Volatility', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(0.5, iv_surf.df['iv'].max() * 1.1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
