"""
Demo: Heston Environment with Matplotlib Rendering

This script demonstrates the dynamic matplotlib rendering for the Heston environment,
showing:
1. Spot price evolution over time
2. Volatility evolution over time
3. Implied volatility surface (3D) that updates at each step

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from options_desk.simulations import HestonEnv, Liability
from options_desk.deep_hedging.agents import DeltaHedgingAgent


def run_demo_with_rendering(n_steps: int = 100, update_interval: int = 1, use_delta_agent: bool = True):
    """
    Run Heston environment simulation with live matplotlib rendering.

    Args:
        n_steps: Number of steps to simulate
        update_interval: Update plot every N steps (1 = every step)
        use_delta_agent: Use Delta hedging agent (True) or random actions (False)
    """
    print("="*70)
    print("Heston Environment - Matplotlib Rendering Demo")
    if use_delta_agent:
        print("Using Delta Hedging Agent")
    else:
        print("Using Random Actions")
    print("="*70)

    # Create liability to hedge (short 1 ATM call, 90 days)
    liability = Liability(
        option_type='call',
        strike=1.0,
        maturity_days=90,
        quantity=-1.0,
    )

    # Create environment with matplotlib rendering
    print("\n[1/5] Creating Heston environment with matplotlib rendering...")
    env = HestonEnv(
        task='hedging',
        render_mode='matplotlib',
        liability=liability,
        max_steps=246,
        initial_cash=100.0,  # Higher initial cash to prevent early bankruptcy
    )

    # Modify renderer update interval if needed
    if env.renderer is not None:
        env.renderer.update_interval = update_interval

    print(f"   ✓ Environment created")
    print(f"   ✓ Action space: {env.action_space.shape[0]} instruments "
          f"(1 underlying + {env.n_options} options)")
    print(f"   ✓ Observation space: {env.observation_space}")

    # Reset environment (initializes matplotlib figure)
    print("\n[2/5] Resetting environment and initializing plots...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Initial spot price: {obs['spot_price'][0]:.4f}")
    print(f"   ✓ Initial volatility: {np.sqrt(info['v']):.2%}")
    print(f"   ✓ Matplotlib figure initialized")

    # Create agent
    print("\n[3/5] Creating hedging agent...")
    if use_delta_agent:
        agent = DeltaHedgingAgent(
            n_instruments=env.n_instruments,
            liability_option_type=liability.option_type,
            liability_strike=liability.strike,
            liability_maturity_days=liability.maturity_days,
            liability_quantity=liability.quantity,
            position_limits=env.position_limits,
        )
        agent.reset(obs, info)
        print("   ✓ Delta Hedging Agent initialized")
    else:
        agent = None
        print("   ✓ Using random actions")

    # Run simulation
    print(f"\n[4/5] Running simulation for {n_steps} steps...")
    print("   (Matplotlib window will update in real-time)")
    print()

    for step in range(n_steps):
        # Get action from agent or random
        if use_delta_agent:
            action = agent.select_action(obs, info)
        else:
            action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Update agent if using delta hedging
        if use_delta_agent:
            agent.update(obs, action, reward, terminated or truncated, info)

        # Render (updates matplotlib plots)
        env.render()

        # Print progress
        if step % 10 == 0:
            print(f"   Step {step:3d}: "
                  f"S={info.get('S', 0):.4f}, "
                  f"σ={np.sqrt(info.get('v', 0)):.2%}, "
                  f"Reward={reward:+.4f}, "
                  f"Cash=${env.cash:.2f}")

        # Check termination
        if terminated or truncated:
            print(f"\n   Episode terminated at step {step}")
            break

    # Save final figure
    print("\n[5/5] Saving final figure...")
    if env.renderer is not None:
        output_file = 'heston_rendering_demo.png'
        env.renderer.save_figure(output_file, dpi=150)
        print(f"   ✓ Figure saved to: {output_file}")

    # Summary statistics
    print("\n" + "="*70)
    print("Simulation Summary")
    print("="*70)
    history = env.get_history()
    print(f"Total steps:          {len(history['S'])}")
    print(f"Initial spot price:   {history['S'][0]:.4f}")
    print(f"Final spot price:     {history['S'][-1]:.4f}")
    print(f"Spot price range:     [{min(history['S']):.4f}, {max(history['S']):.4f}]")
    print(f"Initial volatility:   {np.sqrt(history['v'][0]):.2%}")
    print(f"Final volatility:     {np.sqrt(history['v'][-1]):.2%}")
    print(f"Volatility range:     [{np.sqrt(min(history['v'])):.2%}, {np.sqrt(max(history['v'])):.2%}]")
    print(f"Total reward:         {sum(history['reward']):.4f}")
    print(f"Total transaction costs: ${sum(history['transaction_costs']):.4f}")
    print(f"Final portfolio value: ${history['portfolio_value'][-1]:.2f}")
    print(f"Final P&L:            ${history['portfolio_value'][-1] - env.initial_cash:.2f}")
    print("="*70)

    # Keep window open until user closes it
    print("\nClose the matplotlib window to exit.")
    env.close()


if __name__ == "__main__":
    # Run demo with 100 steps, updating plot every step, using Delta hedging
    run_demo_with_rendering(n_steps=100, update_interval=1, use_delta_agent=True)
