#!/usr/bin/env python3
"""
Phase 4 Task 5: Particle Filter Cross-Validation for NPE vs QMLE

Compares NPE and QMLE calibrations using particle filter log-likelihood
on held-out returns data.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from options_desk.calibration.physical.heston_particle_filter import HestonParticleFilter


def load_price_series(ticker: str, data_dir: Path = Path("data/price_lake/prices")) -> pd.Series:
    """Load adj_close series for a ticker."""
    parquet_path = data_dir / f"{ticker}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Price data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # Index is already 'Date'
    return df['adj_close']


def compute_pf_loglik(
    prices: np.ndarray,
    params: dict,
    n_particles: int = 2000,
    dt: float = 1/252,
    seed: int = 42
) -> float:
    """
    Compute particle filter log-likelihood for given Heston parameters.

    Args:
        prices: array of prices
        params: dict with keys {mu, kappa, theta, xi, v0}
        n_particles: number of particles
        dt: time step (1/252 for daily data)
        seed: random seed

    Returns:
        log-likelihood
    """
    pf = HestonParticleFilter(n_particles=n_particles)
    result = pf.filter(
        prices=prices,
        dt=dt,
        mu=params['mu'],
        kappa=params['kappa'],
        theta=params['theta'],
        xi=params['xi'],  # Note: QMLE uses 'sigma_v', PF uses 'xi'
        v0=params.get('v0', params['theta']),
        random_seed=seed,
    )
    return result.log_likelihood


def main():
    parser = argparse.ArgumentParser(description="PF cross-validation: NPE vs QMLE")
    parser.add_argument("--npe-path", type=str,
                        default="runs/calibration/sp500-npe/heston_npe.parquet",
                        help="Path to NPE calibration results")
    parser.add_argument("--qmle-path", type=str,
                        default="runs/calibration/sp500-cross/heston_qmle.parquet",
                        help="Path to QMLE calibration results")
    parser.add_argument("--tickers", type=str, nargs='+',
                        help="Tickers to evaluate (if not provided, auto-select 8)")
    parser.add_argument("--n-particles", type=int, default=2000,
                        help="Number of particles for PF")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for PF")
    parser.add_argument("--out", type=str, default="data/npe/pf_crossval.csv",
                        help="Output CSV path")

    args = parser.parse_args()

    # Load calibration results
    npe_df = pd.read_parquet(args.npe_path)
    qmle_df = pd.read_parquet(args.qmle_path)

    # Find tickers converged in both
    npe_converged = set(npe_df[npe_df['converged']]['ticker'])
    qmle_converged = set(qmle_df[qmle_df['converged']]['ticker'])
    both_converged = npe_converged & qmle_converged

    print(f"NPE converged: {len(npe_converged)}")
    print(f"QMLE converged: {len(qmle_converged)}")
    print(f"Both converged: {len(both_converged)}")

    # Select tickers
    if args.tickers:
        tickers = args.tickers
    else:
        # Auto-select 8 tickers across different sectors
        merged = pd.merge(
            npe_df[npe_df['converged']][['ticker', 'sector']],
            qmle_df[qmle_df['converged']][['ticker']],
            on='ticker'
        )
        # Sample 2 per sector (if available)
        selected = []
        for sector in merged['sector'].unique():
            sector_tickers = merged[merged['sector'] == sector]['ticker'].tolist()
            selected.extend(sector_tickers[:2])
            if len(selected) >= 8:
                break
        tickers = selected[:8]

    print(f"\nEvaluating {len(tickers)} tickers:")
    for ticker in tickers:
        print(f"  - {ticker}")

    # Run PF for each ticker
    results = []
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        # Load price data (last 5 years = ~1260 trading days)
        try:
            prices_series = load_price_series(ticker)
            prices = prices_series.tail(1260).values
        except Exception as e:
            print(f"  ERROR loading prices: {e}")
            continue

        if len(prices) < 100:
            print(f"  SKIP: insufficient data ({len(prices)} days)")
            continue

        # Get NPE params
        npe_row = npe_df[npe_df['ticker'] == ticker].iloc[0]
        npe_params = {
            'mu': npe_row['mu'],
            'kappa': npe_row['kappa'],
            'theta': npe_row['theta'],
            'xi': npe_row['sigma_v'],  # NPE uses sigma_v, PF expects xi
            'v0': npe_row['v0'],
        }

        # Get QMLE params
        qmle_row = qmle_df[qmle_df['ticker'] == ticker].iloc[0]
        qmle_params = {
            'mu': qmle_row['mu'],
            'kappa': qmle_row['kappa'],
            'theta': qmle_row['theta'],
            'xi': qmle_row['sigma_v'],
            'v0': qmle_row['v0'],
        }

        # Compute PF log-likelihoods
        try:
            npe_loglik = compute_pf_loglik(prices, npe_params, args.n_particles, seed=args.seed)
            qmle_loglik = compute_pf_loglik(prices, qmle_params, args.n_particles, seed=args.seed)

            print(f"  NPE logL:  {npe_loglik:8.2f}")
            print(f"  QMLE logL: {qmle_loglik:8.2f}")
            print(f"  Winner: {'NPE' if npe_loglik > qmle_loglik else 'QMLE'}")

            results.append({
                'ticker': ticker,
                'sector': npe_row['sector'],
                'n_obs': len(prices) - 1,
                'npe_loglik': npe_loglik,
                'qmle_loglik': qmle_loglik,
                'winner': 'NPE' if npe_loglik > qmle_loglik else 'QMLE',
            })
        except Exception as e:
            print(f"  ERROR computing PF: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results_df[['ticker', 'sector', 'npe_loglik', 'qmle_loglik', 'winner']])
    print("\nWin count:")
    print(results_df['winner'].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
