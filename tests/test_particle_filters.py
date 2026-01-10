"""
Particle filter smoke tests using S&P 500 data from yfinance.
"""

from datetime import date, timedelta
import os

import numpy as np
import pytest

from options_desk.calibration.data import YFinanceFetcher
from options_desk.calibration.physical import (
    RoughBergomiCalibrator,
    HestonParticleFilter,
    RoughBergomiParticleFilter,
)


@pytest.mark.slow
def test_particle_filters_sp500():
    fetcher = YFinanceFetcher()
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 5)

    try:
        history = fetcher.get_history("^GSPC", start=start_date, end=end_date)
    except Exception as exc:
        pytest.skip(f"yfinance fetch failed: {exc}")

    prices = history["Close"].values
    dt = 1.0 / 252.0

    # Rough Bergomi parameter initialization
    rb_cal = RoughBergomiCalibrator(window=20, max_lag=10)
    rb_result = rb_cal.fit(prices, dt=dt)

    # Rough Bergomi particle filter
    rb_pf = RoughBergomiParticleFilter(n_particles=500, resample_threshold=0.5)
    rb_out = rb_pf.filter(
        prices=prices,
        dt=dt,
        mu=rb_result.mu,
        xi0=rb_result.xi0,
        eta=rb_result.eta,
        H=rb_result.H,
        rho=rb_result.rho,
        random_seed=42,
    )

    assert len(rb_out.filtered_variance) == len(prices) - 1
    assert np.isfinite(rb_out.log_likelihood)
    assert np.all(rb_out.filtered_variance > 0)

    # Heston particle filter (use rough init for variance level)
    h_pf = HestonParticleFilter(n_particles=500, resample_threshold=0.5)
    h_out = h_pf.filter(
        prices=prices,
        dt=dt,
        mu=rb_result.mu,
        kappa=2.0,
        theta=rb_result.xi0,
        xi=0.5,
        v0=rb_result.xi0,
        random_seed=42,
    )

    assert len(h_out.filtered_variance) == len(prices) - 1
    assert np.isfinite(h_out.log_likelihood)
    assert np.all(h_out.filtered_variance > 0)

    if os.getenv("PF_VERBOSE") == "1":
        print("Rough Bergomi PF log-likelihood:", rb_out.log_likelihood)
        print("Heston PF log-likelihood:", h_out.log_likelihood)
        print("RB variance (mean/std):",
              float(np.mean(rb_out.filtered_variance)),
              float(np.std(rb_out.filtered_variance)))
        print("Heston variance (mean/std):",
              float(np.mean(h_out.filtered_variance)),
              float(np.std(h_out.filtered_variance)))

    if os.getenv("PF_PLOT") == "1":
        import matplotlib.pyplot as plt

        realized_var = np.square(np.diff(np.log(prices))) / dt
        t = np.arange(len(realized_var))

        plt.figure(figsize=(10, 5))
        plt.plot(t, realized_var, label="Realized var", alpha=0.5)
        plt.plot(t, rb_out.filtered_variance, label="RB PF variance", linewidth=1.5)
        plt.plot(t, h_out.filtered_variance, label="Heston PF variance", linewidth=1.5)
        plt.title("Particle Filter Variance vs Realized Variance")
        plt.xlabel("Time step")
        plt.ylabel("Variance")
        plt.legend()
        plt.tight_layout()
        plt.show()
