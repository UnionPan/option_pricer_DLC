"""
Standalone Plotting Functions for Hedging Analysis

Provides individual plotting functions that can be used independently
or as part of EnvironmentInspector.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Any, Optional, Tuple
import warnings


def plot_greeks_evolution(
    snapshots: List,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
):
    """
    Plot Greeks evolution over time (for delta hedging agents).

    Args:
        snapshots: List of Snapshot objects with agent_state['current_delta']
        figsize: Figure size
        save_path: Path to save figure
    """
    # Extract Greeks from agent state
    steps = [s.step for s in snapshots]
    deltas = []
    gammas = []

    for s in snapshots:
        if 'current_delta' in s.agent_state:
            deltas.append(s.agent_state['current_delta'])
        else:
            deltas.append(np.nan)

        if 'current_gamma' in s.agent_state:
            gammas.append(s.agent_state['current_gamma'])
        else:
            gammas.append(np.nan)

    # Check if we have any data
    if all(np.isnan(deltas)):
        print("No delta data available in snapshots")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Delta over time
    ax = axes[0, 0]
    ax.plot(steps, deltas, linewidth=2, color='blue', label='Liability Delta')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Delta=0.5 (ATM)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Delta')
    ax.set_title('Delta Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Gamma over time
    ax = axes[0, 1]
    if not all(np.isnan(gammas)):
        ax.plot(steps, gammas, linewidth=2, color='red', label='Liability Gamma')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gamma')
        ax.set_title('Gamma Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No gamma data', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.axis('off')

    # 3. Delta vs Spot
    ax = axes[1, 0]
    spot_prices = [s.S for s in snapshots]
    ax.scatter(spot_prices, deltas, c=steps, cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Delta')
    ax.set_title('Delta vs Spot Price')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='Step')

    # 4. Delta distribution
    ax = axes[1, 1]
    valid_deltas = [d for d in deltas if not np.isnan(d)]
    ax.hist(valid_deltas, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(valid_deltas), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(valid_deltas):.4f}')
    ax.set_xlabel('Delta')
    ax.set_ylabel('Frequency')
    ax.set_title('Delta Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return fig


def plot_action_heatmap(
    actions: List[np.ndarray],
    instrument_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Plot heatmap of actions (which instruments traded when).

    Args:
        actions: List of action arrays
        instrument_names: Names for each instrument
        figsize: Figure size
        save_path: Path to save figure
    """
    if len(actions) == 0:
        print("No actions to plot")
        return None

    # Stack actions into matrix
    actions_matrix = np.array(actions).T  # Shape: (n_instruments, n_steps)
    n_instruments, n_steps = actions_matrix.shape

    if instrument_names is None:
        instrument_names = ['Underlying'] + [f'Opt_{i}' for i in range(1, n_instruments)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Heatmap of positions over time
    im = ax1.imshow(actions_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Instrument')
    ax1.set_title('Position Heatmap Over Time')

    # Y-axis labels (show every 5th instrument if too many)
    if n_instruments <= 20:
        ax1.set_yticks(range(n_instruments))
        ax1.set_yticklabels(instrument_names, fontsize=8)
    else:
        tick_indices = range(0, n_instruments, 5)
        ax1.set_yticks(tick_indices)
        ax1.set_yticklabels([instrument_names[i] for i in tick_indices], fontsize=8)

    plt.colorbar(im, ax=ax1, label='Position')

    # 2. Trading frequency (how often each instrument was traded)
    position_changes = np.abs(np.diff(actions_matrix, axis=1))
    trading_frequency = np.sum(position_changes > 1e-6, axis=1)

    ax2.barh(range(n_instruments), trading_frequency, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Trades')
    ax2.set_ylabel('Instrument')
    ax2.set_title('Trading Frequency by Instrument')

    if n_instruments <= 20:
        ax2.set_yticks(range(n_instruments))
        ax2.set_yticklabels(instrument_names, fontsize=8)
    else:
        tick_indices = range(0, n_instruments, 5)
        ax2.set_yticks(tick_indices)
        ax2.set_yticklabels([instrument_names[i] for i in tick_indices], fontsize=8)

    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return fig


def plot_terminal_payoff(
    snapshots: List,
    liability_type: str,
    liability_strike: float,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot terminal payoff diagram.

    Shows:
    - Liability payoff at maturity
    - Hedge portfolio value
    - Net P&L

    Args:
        snapshots: List of Snapshot objects
        liability_type: 'call' or 'put'
        liability_strike: Strike price
        figsize: Figure size
        save_path: Path to save figure
    """
    final_snapshot = snapshots[-1]
    S_final = final_snapshot.S
    K = liability_strike

    # Compute payoff for range of spot prices
    S_range = np.linspace(max(0.5 * S_final, 0.1), 1.5 * S_final, 100)

    if liability_type == 'call':
        liability_payoff = -np.maximum(S_range - K, 0.0)  # Short call
        payoff_label = f'Short Call (K={K:.2f})'
    else:
        liability_payoff = -np.maximum(K - S_range, 0.0)  # Short put
        payoff_label = f'Short Put (K={K:.2f})'

    fig, ax = plt.subplots(figsize=figsize)

    # Plot liability payoff
    ax.plot(S_range, liability_payoff, linewidth=2, color='red',
           label=payoff_label, linestyle='--')

    # Actual outcome
    if liability_type == 'call':
        actual_payoff = -max(S_final - K, 0.0)
    else:
        actual_payoff = -max(K - S_final, 0.0)

    hedge_pv = final_snapshot.hedge_portfolio_value
    net_pnl = hedge_pv + actual_payoff

    # Mark actual point
    ax.scatter([S_final], [actual_payoff], s=200, color='red',
              marker='o', zorder=5, label=f'Actual Payoff: ${actual_payoff:.2f}')
    ax.scatter([S_final], [hedge_pv], s=200, color='green',
              marker='s', zorder=5, label=f'Hedge Portfolio: ${hedge_pv:.2f}')
    ax.scatter([S_final], [net_pnl], s=200, color='blue',
              marker='^', zorder=5, label=f'Net P&L: ${net_pnl:.2f}')

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=K, color='gray', linestyle=':', alpha=0.5, label=f'Strike K={K:.2f}')
    ax.axvline(x=S_final, color='blue', linestyle=':', alpha=0.5, label=f'Final S={S_final:.2f}')

    ax.set_xlabel('Spot Price at Maturity')
    ax.set_ylabel('Payoff / P&L ($)')
    ax.set_title('Terminal Payoff Diagram')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return fig


def plot_risk_metrics(
    episodes_data: List[Dict],
    confidence_level: float = 0.95,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
):
    """
    Plot risk metrics across episodes.

    Computes:
    - VaR (Value at Risk)
    - CVaR (Conditional Value at Risk)
    - Worst-case hedge error
    - Hedge error quantiles

    Args:
        episodes_data: List of episode dicts
        confidence_level: Confidence level for VaR/CVaR
        figsize: Figure size
        save_path: Path to save figure
    """
    if len(episodes_data) == 0:
        print("No episodes to analyze")
        return None

    # Extract final hedge errors
    final_hedge_errors = []
    mean_hedge_errors = []
    max_hedge_errors = []

    for ep in episodes_data:
        snapshots = ep['snapshots']
        hedge_errors = [s.hedge_error for s in snapshots]

        final_hedge_errors.append(hedge_errors[-1])
        mean_hedge_errors.append(np.mean(hedge_errors))
        max_hedge_errors.append(np.max(hedge_errors))

    final_hedge_errors = np.array(final_hedge_errors)
    mean_hedge_errors = np.array(mean_hedge_errors)
    max_hedge_errors = np.array(max_hedge_errors)

    # Calculate VaR and CVaR
    alpha = 1 - confidence_level
    var_final = np.quantile(final_hedge_errors, 1 - alpha)
    cvar_final = np.mean(final_hedge_errors[final_hedge_errors >= var_final])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Hedge error distribution with VaR/CVaR
    ax = axes[0]
    ax.hist(final_hedge_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(var_final, color='red', linestyle='--', linewidth=2,
              label=f'VaR ({confidence_level:.0%}): ${var_final:.4f}')
    ax.axvline(cvar_final, color='darkred', linestyle='--', linewidth=2,
              label=f'CVaR ({confidence_level:.0%}): ${cvar_final:.4f}')
    ax.set_xlabel('Final Hedge Error ($)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Final Hedge Error Distribution\n({len(episodes_data)} episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Risk metrics comparison
    ax = axes[1]
    metrics = ['Mean', 'Final', 'Max']
    values = [
        np.mean(mean_hedge_errors),
        np.mean(final_hedge_errors),
        np.mean(max_hedge_errors)
    ]
    errors = [
        np.std(mean_hedge_errors),
        np.std(final_hedge_errors),
        np.std(max_hedge_errors)
    ]

    x_pos = np.arange(len(metrics))
    ax.bar(x_pos, values, yerr=errors, alpha=0.7, color=['blue', 'green', 'red'],
          edgecolor='black', capsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Hedge Error ($)')
    ax.set_title('Hedge Error Metrics')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Quantile analysis
    ax = axes[2]
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    quantile_values = [np.quantile(final_hedge_errors, q) for q in quantiles]

    ax.plot(quantiles, quantile_values, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Final Hedge Error ($)')
    ax.set_title('Hedge Error Quantiles')
    ax.grid(True, alpha=0.3)

    # Add annotations
    for q, v in zip(quantiles, quantile_values):
        ax.annotate(f'${v:.4f}', xy=(q, v), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    # Print summary
    print(f"\nRisk Metrics Summary ({len(episodes_data)} episodes):")
    print(f"  VaR ({confidence_level:.0%}): ${var_final:.4f}")
    print(f"  CVaR ({confidence_level:.0%}): ${cvar_final:.4f}")
    print(f"  Mean hedge error: ${np.mean(mean_hedge_errors):.4f} ± ${np.std(mean_hedge_errors):.4f}")
    print(f"  Worst-case: ${np.max(max_hedge_errors):.4f}")

    return fig


def plot_rebalancing_analysis(
    snapshots: List,
    actions: List[np.ndarray],
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
):
    """
    Plot rebalancing analysis.

    Shows:
    - When rebalancing occurs
    - Size of rebalances
    - Relationship between hedge error and rebalancing

    Args:
        snapshots: List of Snapshot objects
        actions: List of action arrays
        figsize: Figure size
        save_path: Path to save figure
    """
    if len(actions) <= 1:
        print("Need at least 2 steps to analyze rebalancing")
        return None

    # Calculate position changes
    actions_array = np.array(actions)
    position_changes = np.diff(actions_array, axis=0)
    total_abs_change = np.sum(np.abs(position_changes), axis=1)

    # Align snapshots with position_changes
    # position_changes[i] represents change from action[i] to action[i+1]
    # which corresponds to snapshots[i+2]
    steps = [s.step for s in snapshots[2:]]  # Skip first two
    hedge_errors = [s.hedge_error for s in snapshots[2:]]
    tx_costs = [s.transaction_cost for s in snapshots[2:]]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Rebalancing over time
    ax = axes[0, 0]
    ax.plot(steps, total_abs_change, linewidth=2, color='blue', alpha=0.7)
    ax.fill_between(steps, 0, total_abs_change, alpha=0.3, color='blue')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Position Change')
    ax.set_title('Rebalancing Intensity Over Time')
    ax.grid(True, alpha=0.3)

    # 2. Hedge error vs rebalancing
    ax = axes[0, 1]
    ax.scatter(total_abs_change, hedge_errors, alpha=0.6, s=50)
    ax.set_xlabel('Total Position Change')
    ax.set_ylabel('Hedge Error ($)')
    ax.set_title('Hedge Error vs Rebalancing Size')
    ax.grid(True, alpha=0.3)

    # 3. Transaction costs vs rebalancing
    ax = axes[1, 0]
    ax.scatter(total_abs_change, tx_costs, alpha=0.6, s=50, color='orange')
    ax.set_xlabel('Total Position Change')
    ax.set_ylabel('Transaction Cost ($)')
    ax.set_title('Transaction Cost vs Rebalancing Size')
    ax.grid(True, alpha=0.3)

    # 4. Rebalancing frequency distribution
    ax = axes[1, 1]
    ax.hist(total_abs_change, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(total_abs_change), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(total_abs_change):.4f}')
    ax.set_xlabel('Total Position Change')
    ax.set_ylabel('Frequency')
    ax.set_title('Rebalancing Size Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return fig
