#!/usr/bin/env python3
"""
Phase 4 Task 5: Compare NPE vs QMLE calibrations

Computes Spearman rank correlations for parameters estimated by NPE vs QMLE
on tickers converged in both methods.
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description="Compare NPE vs QMLE rank correlations")
    parser.add_argument("--npe-path", type=str,
                        default="runs/calibration/sp500-npe/heston_npe.parquet",
                        help="Path to NPE calibration results")
    parser.add_argument("--qmle-path", type=str,
                        default="runs/calibration/sp500-cross/heston_qmle.parquet",
                        help="Path to QMLE calibration results")

    args = parser.parse_args()

    # Load data
    npe_df = pd.read_parquet(args.npe_path)
    qmle_df = pd.read_parquet(args.qmle_path)

    # Filter to converged only
    npe_converged = npe_df[npe_df['converged']].copy()
    qmle_converged = qmle_df[qmle_df['converged']].copy()

    print(f"NPE converged: {len(npe_converged)}")
    print(f"QMLE converged: {len(qmle_converged)}")

    # Merge on ticker
    merged = pd.merge(
        npe_converged[['ticker', 'kappa', 'theta', 'sigma_v', 'rho', 'mu', 'v0']],
        qmle_converged[['ticker', 'kappa', 'theta', 'sigma_v', 'rho', 'mu', 'v0']],
        on='ticker',
        suffixes=('_npe', '_qmle')
    )

    print(f"Both converged: {len(merged)}")

    # Compute Spearman rank correlations
    params = ['kappa', 'theta', 'sigma_v', 'rho', 'mu', 'v0']
    correlations = {}

    print("\n" + "=" * 60)
    print("Spearman Rank Correlations (NPE vs QMLE)")
    print("=" * 60)
    for param in params:
        npe_vals = merged[f'{param}_npe'].values
        qmle_vals = merged[f'{param}_qmle'].values

        # Remove NaNs
        mask = ~(np.isnan(npe_vals) | np.isnan(qmle_vals))
        if mask.sum() < 10:
            print(f"{param:10s}: INSUFFICIENT DATA (n={mask.sum()})")
            correlations[param] = np.nan
            continue

        rho, pval = spearmanr(npe_vals[mask], qmle_vals[mask])
        correlations[param] = rho

        print(f"{param:10s}: rho={rho:6.3f}  (p={pval:.2e}, n={mask.sum()})")

    print("=" * 60)

    # Summary statistics
    print("\nSummary:")
    print(f"  Mean correlation: {np.nanmean(list(correlations.values())):.3f}")
    print(f"  Median correlation: {np.nanmedian(list(correlations.values())):.3f}")

    # High correlations (>0.7)
    high_corr = [p for p, r in correlations.items() if r > 0.7]
    print(f"  High correlation (>0.7): {high_corr}")

    # Moderate correlations (0.4-0.7)
    mod_corr = [p for p, r in correlations.items() if 0.4 <= r <= 0.7]
    print(f"  Moderate correlation (0.4-0.7): {mod_corr}")

    # Low correlations (<0.4)
    low_corr = [p for p, r in correlations.items() if r < 0.4]
    print(f"  Low correlation (<0.4): {low_corr}")


if __name__ == "__main__":
    main()
