"""
Environment Inspector for Hedging Visualization

Records snapshots during environment runs and provides comprehensive
visualization tools for Jupyter notebooks.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import date, timedelta
import warnings


@dataclass
class Snapshot:
    """
    Snapshot of environment state at a given timestep.

    Stores complete state including option chain for IV surface reconstruction.
    """
    step: int
    S: float
    v: float
    t: float
    cash: float
    positions: np.ndarray
    portfolio_value: float
    hedge_portfolio_value: float
    liability_mtm: float
    hedge_error: float
    reward: float
    transaction_cost: float

    # Option chain snapshot
    option_chain: Any = None
    option_grid_prices: np.ndarray = None
    atm_iv: float = None

    # Agent-specific
    agent_state: Dict[str, Any] = field(default_factory=dict)


class EnvironmentInspector:
    """
    Inspector for recording and visualizing hedging environment runs.

    Usage:
        inspector = EnvironmentInspector(env)

        # Record episode
        obs, info = env.reset()
        inspector.start_recording(obs, info)

        for step in range(max_steps):
            action = agent.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            inspector.record_step(obs, reward, info, action, agent)

            if terminated or truncated:
                break

        inspector.end_recording()

        # Visualize
        inspector.plot_overview()
        inspector.plot_iv_surface(step=30)
        inspector.plot_positions()
    """

    def __init__(
        self,
        env,
        snapshot_interval: int = 1,  # Record every N steps
        record_option_chains: bool = True,
    ):
        """
        Initialize inspector.

        Args:
            env: Heston environment instance
            snapshot_interval: Record snapshot every N steps (1 = every step)
            record_option_chains: Store option chains (needed for IV surface)
        """
        self.env = env
        self.snapshot_interval = snapshot_interval
        self.record_option_chains = record_option_chains

        # Recording state
        self.is_recording = False
        self.current_episode = 0

        # Data storage
        self.episodes: List[Dict[str, Any]] = []
        self.current_snapshots: List[Snapshot] = []
        self.current_actions: List[np.ndarray] = []

    def start_recording(self, observation: Dict, info: Dict):
        """Start recording a new episode."""
        self.is_recording = True
        self.current_snapshots = []
        self.current_actions = []

        # Record initial state
        snapshot = self._create_snapshot(
            step=0,
            observation=observation,
            info=info,
            reward=0.0,
            transaction_cost=0.0,
            action=None,
            agent=None,
        )
        self.current_snapshots.append(snapshot)

    def record_step(
        self,
        observation: Dict,
        reward: float,
        info: Dict,
        action: np.ndarray,
        agent: Any = None,
    ):
        """Record a step."""
        if not self.is_recording:
            warnings.warn("Inspector not recording. Call start_recording() first.")
            return

        step = int(observation['time_step'][0])

        # Only record at snapshot intervals
        if step % self.snapshot_interval == 0 or step == 0:
            # Get transaction cost from history
            transaction_cost = self.env.history['transaction_costs'][-1]

            snapshot = self._create_snapshot(
                step=step,
                observation=observation,
                info=info,
                reward=reward,
                transaction_cost=transaction_cost,
                action=action,
                agent=agent,
            )
            self.current_snapshots.append(snapshot)

        # Always record actions
        self.current_actions.append(action.copy())

    def end_recording(self):
        """End recording and store episode."""
        if not self.is_recording:
            return

        # Store episode data
        episode_data = {
            'episode': self.current_episode,
            'snapshots': self.current_snapshots,
            'actions': self.current_actions,
            'history': {k: v.copy() if isinstance(v, list) else v
                       for k, v in self.env.history.items()},
        }

        self.episodes.append(episode_data)
        self.current_episode += 1
        self.is_recording = False

    def _create_snapshot(
        self,
        step: int,
        observation: Dict,
        info: Dict,
        reward: float,
        transaction_cost: float,
        action: Optional[np.ndarray],
        agent: Any,
    ) -> Snapshot:
        """Create snapshot from current state."""
        # Extract state
        S = float(observation['spot_price'][0])
        t = float(observation['time_step'][0])

        # Get from environment
        v = self.env.v
        cash = self.env.cash
        positions = self.env.positions.copy()
        portfolio_value = self.env.portfolio_value
        hedge_portfolio_value = self.env.hedge_portfolio_value
        liability_mtm = self.env.liability_mtm if self.env.liability_mtm is not None else 0.0
        hedge_error = abs(hedge_portfolio_value)

        # Option chain
        option_chain = None
        option_grid_prices = None
        atm_iv = None

        if self.record_option_chains and self.env.include_options:
            option_chain = self.env.current_option_chain
            option_grid_prices = self.env.option_grid_prices.copy() if self.env.option_grid_prices is not None else None
            atm_iv = info.get('atm_iv', None)

        # Agent state
        agent_state = {}
        if agent is not None and hasattr(agent, 'get_state'):
            agent_state = agent.get_state()

        return Snapshot(
            step=step,
            S=S,
            v=v,
            t=t,
            cash=cash,
            positions=positions,
            portfolio_value=portfolio_value,
            hedge_portfolio_value=hedge_portfolio_value,
            liability_mtm=liability_mtm,
            hedge_error=hedge_error,
            reward=reward,
            transaction_cost=transaction_cost,
            option_chain=option_chain,
            option_grid_prices=option_grid_prices,
            atm_iv=atm_iv,
            agent_state=agent_state,
        )

    def get_episode(self, episode: int = -1) -> Dict[str, Any]:
        """Get episode data (default: last episode)."""
        if len(self.episodes) == 0:
            raise ValueError("No episodes recorded")
        return self.episodes[episode]

    def get_snapshot(self, step: int, episode: int = -1) -> Snapshot:
        """Get snapshot at specific step."""
        ep = self.get_episode(episode)
        snapshots = ep['snapshots']

        # Find closest snapshot
        closest = min(snapshots, key=lambda s: abs(s.step - step))
        return closest

    def get_snapshots(self, episode: int = -1) -> List[Snapshot]:
        """Get all snapshots for episode."""
        ep = self.get_episode(episode)
        return ep['snapshots']

    def get_history(self, episode: int = -1) -> Dict[str, List]:
        """Get full history for episode."""
        ep = self.get_episode(episode)
        return ep['history']

    # ========================================================================
    # Plotting Methods
    # ========================================================================

    def plot_overview(
        self,
        episode: int = -1,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
    ):
        """
        Plot comprehensive overview of episode.

        Shows:
        - Environmental path (S_t, v_t, σ_t)
        - Wealth evolution (portfolio, hedge portfolio, liability)
        - Hedge error over time
        - Transaction costs
        - Positions (underlying + top options)
        """
        ep = self.get_episode(episode)
        snapshots = ep['snapshots']
        history = ep['history']

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Extract data
        steps = [s.step for s in snapshots]
        S_vals = [s.S for s in snapshots]
        v_vals = [s.v for s in snapshots]
        sigma_vals = [np.sqrt(s.v) for s in snapshots]
        pv_vals = [s.portfolio_value for s in snapshots]
        hpv_vals = [s.hedge_portfolio_value for s in snapshots]
        liability_vals = [s.liability_mtm for s in snapshots]
        hedge_errors = [s.hedge_error for s in snapshots]
        rewards = [s.reward for s in snapshots]
        costs = [s.transaction_cost for s in snapshots]

        # 1. Spot price path
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(steps, S_vals, linewidth=2, color='blue', label='S_t')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Spot Price')
        ax1.set_title('Spot Price Path')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Variance and volatility
        ax2 = fig.add_subplot(gs[0, 1])
        ax2_twin = ax2.twinx()
        ax2.plot(steps, v_vals, linewidth=2, color='red', label='v_t (variance)', alpha=0.7)
        ax2_twin.plot(steps, sigma_vals, linewidth=2, color='orange', label='σ_t (vol)', linestyle='--')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Variance', color='red')
        ax2_twin.set_ylabel('Volatility', color='orange')
        ax2.set_title('Variance & Volatility')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')

        # 3. Wealth evolution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(steps, pv_vals, linewidth=2, label='Portfolio Value', color='green')
        ax3.plot(steps, hpv_vals, linewidth=2, label='Hedge Portfolio', color='purple')
        ax3.plot(steps, liability_vals, linewidth=2, label='Liability MTM', color='red', linestyle='--')
        ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Value ($)')
        ax3.set_title('Wealth Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Hedge error
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(steps, hedge_errors, linewidth=2, color='red')
        ax4.fill_between(steps, 0, hedge_errors, alpha=0.3, color='red')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Hedge Error ($)')
        ax4.set_title('Hedge Error Over Time')
        ax4.grid(True, alpha=0.3)

        # 5. Rewards
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(steps, rewards, linewidth=2, color='green', alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Reward')
        ax5.set_title(f'Rewards (Total: {sum(rewards):.2f})')
        ax5.grid(True, alpha=0.3)

        # 6. Transaction costs
        ax6 = fig.add_subplot(gs[1, 2])
        cumulative_costs = np.cumsum(costs)
        ax6.plot(steps, cumulative_costs, linewidth=2, color='orange')
        ax6.fill_between(steps, 0, cumulative_costs, alpha=0.3, color='orange')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Cumulative Cost ($)')
        ax6.set_title(f'Transaction Costs (Total: ${sum(costs):.4f})')
        ax6.grid(True, alpha=0.3)

        # 7. Underlying position
        ax7 = fig.add_subplot(gs[2, 0])
        underlying_positions = [s.positions[0] for s in snapshots]
        ax7.plot(steps, underlying_positions, linewidth=2, color='blue')
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Position')
        ax7.set_title('Underlying Position')
        ax7.grid(True, alpha=0.3)

        # 8. Top 3 option positions
        ax8 = fig.add_subplot(gs[2, 1])
        positions_array = np.array([s.positions[1:] for s in snapshots])  # Exclude underlying
        if positions_array.shape[1] > 0:
            # Find top 3 most traded options
            total_abs_positions = np.sum(np.abs(positions_array), axis=0)
            top_3_indices = np.argsort(total_abs_positions)[-3:][::-1]

            for idx in top_3_indices:
                ax8.plot(steps, positions_array[:, idx], linewidth=2, label=f'Option {idx+1}')
            ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax8.set_xlabel('Step')
            ax8.set_ylabel('Position')
            ax8.set_title('Top 3 Option Positions')
            ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Summary stats
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"""
        Episode Summary
        {'='*30}

        Final Values:
          Portfolio: ${pv_vals[-1]:.2f}
          Hedge Error: ${hedge_errors[-1]:.4f}
          Liability MTM: ${liability_vals[-1]:.2f}

        Performance:
          Total Reward: {sum(rewards):.4f}
          Mean Hedge Error: ${np.mean(hedge_errors):.4f}
          Std Hedge Error: ${np.std(hedge_errors):.4f}

        Costs:
          Total Tx Costs: ${sum(costs):.4f}
          Avg Cost/Step: ${np.mean(costs):.6f}

        Path Stats:
          Initial S: {S_vals[0]:.4f}
          Final S: {S_vals[-1]:.4f}
          Return: {(S_vals[-1]/S_vals[0] - 1)*100:.2f}%
          Realized Vol: {np.std(np.diff(np.log(S_vals))) * np.sqrt(252):.2%}
        """

        ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle(f'Episode {ep["episode"]} Overview', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

        return fig

    def plot_iv_smile(
        self,
        step: int,
        episode: int = -1,
        maturity_days: Optional[int] = None,
        otm_only: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ):
        """
        Plot implied volatility smile/skew for a single maturity.

        Shows calls (blue circles) and puts (red squares) on same plot.

        Args:
            step: Timestep to plot
            episode: Episode index (-1 = last)
            maturity_days: Maturity to plot (None = shortest maturity)
            otm_only: Show only OTM options (True = cleaner, False = all options)
            figsize: Figure size
            save_path: Path to save figure
        """
        snapshot = self.get_snapshot(step, episode)

        if snapshot.option_chain is None:
            print("No option chain data available. Set record_option_chains=True")
            return

        option_chain = snapshot.option_chain
        S = snapshot.S
        v = snapshot.v

        # Extract option data
        options_data = []
        for opt in option_chain.options:
            moneyness = opt.strike / S
            ttm_days = (opt.expiry - option_chain.reference_date).days
            iv = opt.implied_volatility
            option_type = opt.option_type

            options_data.append({
                'moneyness': moneyness,
                'ttm_days': ttm_days,
                'iv': iv,
                'type': option_type,
            })

        unique_ttms = sorted(set([opt['ttm_days'] for opt in options_data]))

        # Select maturity
        if maturity_days is None:
            maturity_days = unique_ttms[0]  # Default to shortest
        elif maturity_days not in unique_ttms:
            # Find closest maturity
            maturity_days = min(unique_ttms, key=lambda x: abs(x - maturity_days))
            print(f"Requested maturity not available. Using closest: {maturity_days}d")

        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get options for this maturity
        ttm_options = [opt for opt in options_data if opt['ttm_days'] == maturity_days]

        if otm_only:
            # Show only OTM: puts with K < S, calls with K > S, ATM for both
            calls = sorted([opt for opt in ttm_options
                          if opt['type'] == 'call' and opt['moneyness'] >= 0.995],
                         key=lambda x: x['moneyness'])
            puts = sorted([opt for opt in ttm_options
                         if opt['type'] == 'put' and opt['moneyness'] <= 1.005],
                        key=lambda x: x['moneyness'])
        else:
            # Show all options
            calls = sorted([opt for opt in ttm_options if opt['type'] == 'call'],
                          key=lambda x: x['moneyness'])
            puts = sorted([opt for opt in ttm_options if opt['type'] == 'put'],
                         key=lambda x: x['moneyness'])

        # Plot puts (left side, red squares)
        if puts:
            put_m = [opt['moneyness'] for opt in puts]
            put_iv = [opt['iv'] for opt in puts]
            ax.plot(put_m, put_iv, 's-', color='red',
                   linewidth=2.5, markersize=8, label='Puts (OTM)' if otm_only else 'Puts',
                   alpha=0.8, markeredgecolor='white', markeredgewidth=1.5)

        # Plot calls (right side, blue circles)
        if calls:
            call_m = [opt['moneyness'] for opt in calls]
            call_iv = [opt['iv'] for opt in calls]
            ax.plot(call_m, call_iv, 'o-', color='blue',
                   linewidth=2.5, markersize=8, label='Calls (OTM)' if otm_only else 'Calls',
                   alpha=0.8, markeredgecolor='white', markeredgewidth=1.5)

        # ATM line
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='ATM')

        # Formatting
        ax.set_xlabel('Moneyness (K/S)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Implied Volatility', fontsize=12, fontweight='bold')
        title_suffix = ' (OTM Options)' if otm_only else ''
        ax.set_title(
            f'Volatility Smile/Skew ({maturity_days}-day, Step {step}){title_suffix}\nS={S:.4f}, σ={np.sqrt(v):.1%}',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, frameon=True, shadow=True)
        ax.set_ylim(0, max(0.5, max([opt['iv'] for opt in ttm_options]) * 1.1))

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

        return fig

    def plot_iv_surface_3d(
        self,
        step: int,
        episode: int = -1,
        otm_only: bool = True,
        grid_resolution: int = 50,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ):
        """
        Plot 3D implied volatility surface with both calls and puts.

        Matches the style of utils.iv_surface.plot_3d_surface().

        Args:
            step: Timestep to plot
            episode: Episode index (-1 = last)
            otm_only: Show only OTM options (True = cleaner, False = all options)
            grid_resolution: Number of points in interpolation grid
            figsize: Figure size
            save_path: Path to save figure
        """
        snapshot = self.get_snapshot(step, episode)

        if snapshot.option_chain is None:
            print("No option chain data available. Set record_option_chains=True")
            return

        option_chain = snapshot.option_chain
        S = snapshot.S
        v = snapshot.v

        # Extract option data
        options_data = []
        for opt in option_chain.options:
            moneyness = opt.strike / S
            ttm_days = (opt.expiry - option_chain.reference_date).days
            iv = opt.implied_volatility
            option_type = opt.option_type

            # Filter to OTM only if requested
            if otm_only:
                if option_type == 'call' and moneyness < 0.995:
                    continue  # Skip ITM/ATM calls
                if option_type == 'put' and moneyness > 1.005:
                    continue  # Skip ITM/ATM puts

            options_data.append({
                'moneyness': moneyness,
                'ttm_days': ttm_days,
                'iv': iv,
                'type': option_type,
            })

        if len(options_data) == 0:
            print("No data to plot")
            return

        # Extract coordinates
        moneyness = np.array([opt['moneyness'] for opt in options_data])
        ttm = np.array([opt['ttm_days'] for opt in options_data])
        iv = np.array([opt['iv'] for opt in options_data])
        opt_types = np.array([opt['type'] for opt in options_data])

        # Create interpolation grid
        from scipy.interpolate import griddata

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

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(
            M, T, IV,
            cmap='viridis',
            alpha=0.7,
            edgecolor='gray',
            linewidth=0.2,
            antialiased=True,
        )

        # Plot original points on top (differentiate calls and puts)
        calls_mask = opt_types == 'call'
        puts_mask = opt_types == 'put'

        call_label = 'Calls (OTM)' if otm_only else 'Calls'
        put_label = 'Puts (OTM)' if otm_only else 'Puts'

        # Calls: blue circles
        if np.any(calls_mask):
            ax.scatter(
                moneyness[calls_mask], ttm[calls_mask], iv[calls_mask],
                c='blue',
                s=60,
                alpha=1.0,
                edgecolors='black',
                linewidths=0.8,
                depthshade=True,
                marker='o',
                label=call_label
            )

        # Puts: red squares
        if np.any(puts_mask):
            ax.scatter(
                moneyness[puts_mask], ttm[puts_mask], iv[puts_mask],
                c='red',
                s=50,
                alpha=1.0,
                edgecolors='black',
                linewidths=0.8,
                depthshade=True,
                marker='s',
                label=put_label
            )

        # Plot vertical lines from points to z=0 plane
        for m, t, v_val in zip(moneyness, ttm, iv):
            ax.plot(
                [m, m], [t, t], [0, v_val],
                color='gray',
                linestyle='--',
                linewidth=0.5,
                alpha=0.4,
            )

        # Labels and formatting
        ax.set_xlabel('\nMoneyness (K/S)', fontsize=11, labelpad=10)
        ax.set_ylabel('\nTime to Maturity (days)', fontsize=11, labelpad=10)
        ax.set_zlabel('\nImplied Volatility', fontsize=11, labelpad=10)
        title_suffix = ' (OTM Options)' if otm_only else ''
        ax.set_title(
            f'Implied Volatility Surface (Step {step}){title_suffix}\nS={S:.4f}, σ={np.sqrt(v):.1%}',
            fontsize=13,
            fontweight='bold',
            pad=20,
        )

        # Set z-axis to start at 0
        ax.set_zlim(bottom=0)

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('IV', fontsize=10, fontweight='bold')

        # Add legend for calls/puts
        ax.legend(fontsize=10, loc='upper left')

        # Viewing angle
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

        return fig

    def plot_pnl_distribution(
        self,
        episodes: Optional[List[int]] = None,
        metric: str = 'total_reward',
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
    ):
        """
        Plot PnL distribution across multiple episodes.

        Args:
            episodes: List of episode indices (None = all episodes)
            metric: 'total_reward', 'final_pv', 'final_hedge_error'
            figsize: Figure size
            save_path: Path to save figure
        """
        if episodes is None:
            episodes = list(range(len(self.episodes)))

        # Extract metric
        values = []
        for ep_idx in episodes:
            ep = self.episodes[ep_idx]
            snapshots = ep['snapshots']

            if metric == 'total_reward':
                val = sum([s.reward for s in snapshots])
            elif metric == 'final_pv':
                val = snapshots[-1].portfolio_value
            elif metric == 'final_hedge_error':
                val = snapshots[-1].hedge_error
            else:
                raise ValueError(f"Unknown metric: {metric}")

            values.append(val)

        values = np.array(values)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Histogram
        ax1.hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(values), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(values):.4f}')
        ax1.axvline(np.median(values), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(values):.4f}')
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{metric.replace("_", " ").title()} Distribution\n({len(values)} episodes)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Box plot + summary stats
        ax2.boxplot([values], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title('Distribution Summary')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add text with stats
        stats_text = f"""
        Statistics:
          Mean: {np.mean(values):.4f}
          Std: {np.std(values):.4f}
          Min: {np.min(values):.4f}
          25%: {np.percentile(values, 25):.4f}
          50%: {np.percentile(values, 50):.4f}
          75%: {np.percentile(values, 75):.4f}
          Max: {np.max(values):.4f}
        """

        ax2.text(1.3, np.median(values), stats_text, fontsize=9,
                family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

        return fig
