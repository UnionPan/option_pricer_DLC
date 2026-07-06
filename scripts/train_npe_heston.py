#!/usr/bin/env python3
"""
Train Heston NPE model for amortized calibration.

Simulates Heston paths from the prior, computes summary features,
trains a conditional MDN, saves the checkpoint, and evaluates
held-out recovery (Pearson correlations per parameter).

Memory-efficient: simulates in chunks of ≤20000 paths to avoid OOM.
GPU-friendly: no forced JAX_PLATFORMS=cpu in the script (set via env if needed).

Example:
    python scripts/train_npe_heston.py --n-sims 200000 --epochs 200 --out data/npe/heston_mdn.pkl
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from options_desk.calibration.physical.batched.npe.simulate import (
    sample_prior,
    simulate_heston_paths,
    summary_features,
    to_unconstrained,
)
from options_desk.calibration.physical.batched.npe.train import train_mdn, save_npe


def simulate_in_chunks(key, n_total, T, chunk_size=20000):
    """
    Simulate Heston paths in chunks to avoid memory overflow.

    Args:
        key: JAX PRNGKey
        n_total: total number of simulations
        T: number of time steps
        chunk_size: maximum paths per chunk

    Returns:
        thetas_natural: (n_total, 6) sampled parameters
        features: (n_total, 16) summary features
    """
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    thetas_list = []
    features_list = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_total)
        n_chunk = end - start

        print(f"Simulating chunk {i+1}/{n_chunks}: {n_chunk} paths...")

        # Sample parameters
        key, subkey = jax.random.split(key)
        thetas = sample_prior(subkey, n_chunk)

        # Simulate paths
        key, subkey = jax.random.split(key)
        returns = simulate_heston_paths(subkey, thetas, T)

        # Compute features
        mask = jnp.ones_like(returns)
        feats = summary_features(returns, mask)

        thetas_list.append(thetas)
        features_list.append(feats)

    # Concatenate
    thetas_natural = jnp.concatenate(thetas_list, axis=0)
    features = jnp.concatenate(features_list, axis=0)

    return thetas_natural, features


def evaluate_recovery(trained_npe, key, n_test=5000, T=1260):
    """
    Evaluate held-out recovery: Pearson r(posterior mean, truth) per parameter.

    Args:
        trained_npe: TrainedNPE instance
        key: JAX PRNGKey
        n_test: number of held-out simulations
        T: number of time steps

    Returns:
        dict of parameter name -> Pearson correlation
    """
    from options_desk.calibration.physical.batched.npe.train import sample_posterior

    print(f"\nEvaluating held-out recovery on {n_test} simulations...")

    # Simulate held-out data (in chunks if needed)
    key, subkey = jax.random.split(key)
    true_thetas, test_features = simulate_in_chunks(subkey, n_test, T, chunk_size=20000)

    # Sample posterior
    key, subkey = jax.random.split(key)
    posterior_samples = sample_posterior(
        trained_npe,
        apply_fn_or_none=None,
        s_raw=test_features,
        key=subkey,
        n_samples=4096,
    )

    # Compute posterior means
    posterior_means = jnp.mean(posterior_samples, axis=1)  # (n_test, 6)

    # Compute Pearson correlations
    param_names = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
    correlations = {}

    for i, name in enumerate(param_names):
        true_vals = np.array(true_thetas[:, i])
        pred_vals = np.array(posterior_means[:, i])
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
        correlations[name] = corr

    return correlations


def main():
    parser = argparse.ArgumentParser(description="Train Heston NPE model")
    parser.add_argument("--n-sims", type=int, default=200000,
                        help="Number of training simulations (default: 200000)")
    parser.add_argument("--T", type=int, default=1260,
                        help="Number of time steps per path (default: 1260)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer sizes (default: 128 128)")
    parser.add_argument("--components", type=int, default=8,
                        help="Number of mixture components (default: 8)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--out", type=str, default="data/npe/heston_mdn.pkl",
                        help="Output checkpoint path (default: data/npe/heston_mdn.pkl)")

    args = parser.parse_args()

    print("=" * 80)
    print("Training Heston NPE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  n_sims:      {args.n_sims}")
    print(f"  T:           {args.T}")
    print(f"  epochs:      {args.epochs}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  lr:          {args.lr}")
    print(f"  hidden:      {args.hidden}")
    print(f"  components:  {args.components}")
    print(f"  seed:        {args.seed}")
    print(f"  output:      {args.out}")
    print("=" * 80)

    # Initialize random key
    key = jax.random.PRNGKey(args.seed)

    # Simulate training data in chunks
    print(f"\nGenerating {args.n_sims} training simulations...")
    key, subkey = jax.random.split(key)
    thetas_natural, features = simulate_in_chunks(subkey, args.n_sims, args.T)

    # Convert to unconstrained space
    thetas_unconstrained = to_unconstrained(thetas_natural)

    print(f"Features shape: {features.shape}")
    print(f"Parameters shape: {thetas_unconstrained.shape}")

    # Train MDN
    print(f"\nTraining MDN ({args.epochs} epochs)...")
    key, subkey = jax.random.split(key)
    trained_npe = train_mdn(
        subkey,
        features,
        thetas_unconstrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=0.1,
        hidden=tuple(args.hidden),
        n_components=args.components,
    )

    print("Training complete.")

    # Save checkpoint
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npe(trained_npe, str(out_path))
    print(f"\nCheckpoint saved to: {out_path}")

    # Evaluate held-out recovery
    key, subkey = jax.random.split(key)
    correlations = evaluate_recovery(trained_npe, subkey, n_test=5000, T=args.T)

    print("\nHeld-out recovery (Pearson r, posterior mean vs truth):")
    print("-" * 40)
    for name, corr in correlations.items():
        print(f"  {name:10s}: {corr:6.3f}")
    print("-" * 40)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
