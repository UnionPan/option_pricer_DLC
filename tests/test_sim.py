"""
Tests for option pricing models and simulation environments.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from options_desk.simulations import HestonEnv, Liability
from options_desk.processes import Heston


def test_heston_env_basic():
    """Test basic Heston environment functionality."""
    print("\n" + "="*60)
    print("Testing Heston Environment - Basic Functionality")
    print("="*60)

    # Create Heston environment
    liability = Liability(
        option_type='call',
        strike=1.0,
        maturity_days=90,
        quantity=-1.0,
    )
    env = HestonEnv(
        task='hedging',
        render_mode=None,
        liability=liability,
    )

    print("\nEnvironment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    obs, info = env.reset(seed=42)
    print("\nInitial observation keys:", list(obs.keys()))
    print(f"Initial spot price: {obs['spot_price'][0]:.2f}")

    # Run a few steps with random actions
    print("\nRunning 5 random steps...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step+1}: Spot={info.get('S', 'N/A'):.2f}, "
              f"Reward={reward:.4f}, Done={terminated or truncated}")

    env.close()
    print("\nTest completed successfully!")


def test_heston_process_simulation():
    """Test Heston process path generation."""
    print("\n" + "="*60)
    print("Testing Heston Process - Path Simulation")
    print("="*60)

    # Heston parameters
    S0 = 100.0
    v0 = 0.04  # initial variance (20% vol)
    kappa = 2.0  # mean reversion speed
    theta = 0.04  # long-term variance
    sigma_v = 0.3  # vol of vol
    rho = -0.7  # correlation
    mu = 0.05  # drift

    print("\nHeston parameters:")
    print(f"  S0 = {S0}, v0 = {v0} (vol = {np.sqrt(v0):.2%})")
    print(f"  kappa = {kappa}, theta = {theta}, sigma = {sigma_v}")
    print(f"  rho = {rho}, mu = {mu}")

    # Create Heston process
    heston = Heston(
        mu=mu,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho
    )

    # Simulate paths
    T = 1.0  # 1 year
    n_steps = 252  # daily steps
    n_paths = 1000

    print(f"\nSimulating {n_paths} paths over {T} years ({n_steps} steps)...")

    from options_desk.processes import SimulationConfig

    config = SimulationConfig(
        n_paths=n_paths,
        n_steps=n_steps,
        random_seed=42,
    )
    _, paths = heston.simulate(
        X0=np.array([S0, v0]),
        T=T,
        config=config,
        scheme='milstein',
    )

    S_paths = paths[:, :, 0]  # spot price paths
    v_paths = paths[:, :, 1]  # variance paths

    print("\nSimulation completed!")
    print(f"Paths shape: {paths.shape}")
    print(f"\nSpot price statistics at T={T}:")
    print(f"  Mean: {S_paths[-1].mean():.2f}")
    print(f"  Std:  {S_paths[-1].std():.2f}")
    print(f"  Min:  {S_paths[-1].min():.2f}")
    print(f"  Max:  {S_paths[-1].max():.2f}")

    print(f"\nVariance statistics at T={T}:")
    print(f"  Mean: {v_paths[-1].mean():.4f} (vol = {np.sqrt(v_paths[-1].mean()):.2%})")
    print(f"  Std:  {v_paths[-1].std():.4f}")
    print(f"  Min:  {v_paths[-1].min():.4f}")
    print(f"  Max:  {v_paths[-1].max():.4f}")

    # Check if any variance went negative (shouldn't happen with good scheme)
    negative_var = (v_paths < 0).any()
    print(f"\nNegative variance encountered: {negative_var}")

    print("\nTest completed successfully!")


def test_heston_env_matplotlib_rendering():
    """Test Heston environment with matplotlib rendering."""
    print("\n" + "="*60)
    print("Testing Heston Environment - Matplotlib Rendering")
    print("="*60)

    # Create Heston environment with matplotlib rendering
    liability = Liability(
        option_type='call',
        strike=1.0,
        maturity_days=90,
        quantity=-1.0,
    )

    env = HestonEnv(
        task='hedging',
        render_mode='matplotlib',  # Enable matplotlib rendering
        liability=liability,
    )

    print("\nEnvironment created with matplotlib rendering!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment (this will initialize the matplotlib figure)
    obs, info = env.reset(seed=42)
    print(f"\nInitial spot price: {obs['spot_price'][0]:.2f}")

    # Run a simulation for 50 steps with random actions
    print("\nRunning simulation for 50 steps with matplotlib rendering...")
    print("Close the matplotlib window to continue, or wait for completion.")

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Render at each step
        env.render()

        if step % 10 == 0:
            print(f"Step {step}: Spot={info.get('S', 'N/A'):.2f}, "
                  f"Vol={np.sqrt(info.get('v', 0)):.2%}, "
                  f"Reward={reward:.4f}")

        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            break

    # Save final figure
    if env.renderer is not None:
        env.renderer.save_figure('heston_env_rendering.png')

    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    # Run basic tests first
    test_heston_process_simulation()
    test_heston_env_basic()

    # Uncomment to test matplotlib rendering (will open interactive window)
    # test_heston_env_matplotlib_rendering()
