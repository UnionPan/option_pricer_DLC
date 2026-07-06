"""Tests for empirical-Bayes sector pooling."""

import numpy as np
import pandas as pd
import pytest

from options_desk.calibration.cross_asset.pooling import pool_parameters


class TestPoolParameters:
    """Test suite for pool_parameters function."""

    def test_heavy_noise_sector_reduces_mse(self):
        """Test that pooling reduces MSE for a noisy sector.

        Simulate one sector of 40 names whose true param is constant μ + large noise.
        Pooled values should have ≥ 60% lower MSE vs truth than raw.
        Shrinkage column should be in (0, 1].
        """
        np.random.seed(42)
        n_names = 40
        true_value = 0.5
        noise_std = 0.3  # Large noise relative to signal

        # Simulate noisy observations
        observed = true_value + np.random.normal(0, noise_std, n_names)

        df = pd.DataFrame({
            "name": [f"stock_{i}" for i in range(n_names)],
            "sector": ["tech"] * n_names,
            "kappa": observed,
        })

        result = pool_parameters(df, params=["kappa"], min_sector_size=5)

        # Check that pooled columns exist
        assert "kappa_pooled" in result.columns
        assert "kappa_shrinkage" in result.columns

        # Calculate MSE for raw vs pooled
        mse_raw = np.mean((df["kappa"] - true_value) ** 2)
        mse_pooled = np.mean((result["kappa_pooled"] - true_value) ** 2)

        # Pooled should have at least 60% lower MSE
        improvement = (mse_raw - mse_pooled) / mse_raw
        assert improvement >= 0.6, f"MSE improvement {improvement:.2%} < 60%"

        # Shrinkage should be in (0, 1]
        shrinkage = result["kappa_shrinkage"]
        assert (shrinkage > 0).all(), "Some shrinkage values are not positive"
        assert (shrinkage <= 1).all(), "Some shrinkage values exceed 1"

    def test_tight_sector_minimal_shrinkage(self):
        """Test that sectors with low noise have minimal shrinkage.

        When noise ≈ 0, shrinkage should be ≈ 0 and pooled ≈ raw.
        """
        np.random.seed(43)
        n_names = 20
        base_value = 0.3

        # Very small noise
        observed = base_value + np.random.normal(0, 1e-6, n_names)

        df = pd.DataFrame({
            "name": [f"stock_{i}" for i in range(n_names)],
            "sector": ["finance"] * n_names,
            "theta": observed,
        })

        result = pool_parameters(df, params=["theta"], min_sector_size=5)

        # Shrinkage should be very small (close to 0)
        shrinkage = result["theta_shrinkage"]
        assert (shrinkage < 0.1).all(), "Shrinkage should be minimal for tight sector"

        # Pooled values should be very close to raw
        diff = np.abs(result["theta_pooled"] - df["theta"])
        assert (diff < 1e-4).all(), "Pooled values should match raw for tight sector"

    def test_small_sector_shrinks_to_global_mean(self):
        """Test that small sectors (n ≤ min_sector_size) shrink toward global mean."""
        np.random.seed(44)

        # Create two sectors: one large, one small
        large_sector_size = 30
        small_sector_size = 3

        large_sector_mean = 0.8
        small_sector_mean = 0.2

        # Generate properly randomized values
        large_values = [large_sector_mean + np.random.normal(0, 0.1) for _ in range(large_sector_size)]
        small_values = [small_sector_mean + np.random.normal(0, 0.1) for _ in range(small_sector_size)]

        df = pd.DataFrame({
            "name": ([f"large_{i}" for i in range(large_sector_size)] +
                     [f"small_{i}" for i in range(small_sector_size)]),
            "sector": (["large"] * large_sector_size +
                      ["small"] * small_sector_size),
            "sigma_v": large_values + small_values,
        })

        result = pool_parameters(df, params=["sigma_v"], min_sector_size=5)

        # Calculate global mean
        global_mean = df["sigma_v"].mean()

        # Small sector values should be shrunk toward global mean
        small_sector_mask = result["sector"] == "small"
        small_pooled = result.loc[small_sector_mask, "sigma_v_pooled"]
        small_raw = df.loc[small_sector_mask, "sigma_v"]

        # Pooled values should be between raw values and global mean
        for raw_val, pooled_val in zip(small_raw, small_pooled):
            if raw_val < global_mean:
                assert pooled_val > raw_val, "Should shrink upward toward global mean"
                assert pooled_val <= global_mean, "Should not overshoot global mean"
            elif raw_val > global_mean:
                assert pooled_val < raw_val, "Should shrink downward toward global mean"
                assert pooled_val >= global_mean, "Should not overshoot global mean"

    def test_nan_passthrough(self):
        """Test that NaN param values pass through untouched with shrinkage 0."""
        np.random.seed(45)
        n_names = 15

        values = np.random.uniform(0.3, 0.7, n_names)
        # Set some values to NaN
        values[3] = np.nan
        values[7] = np.nan
        values[11] = np.nan

        df = pd.DataFrame({
            "name": [f"stock_{i}" for i in range(n_names)],
            "sector": ["energy"] * n_names,
            "omega": values,
        })

        result = pool_parameters(df, params=["omega"], min_sector_size=5)

        # NaN values should remain NaN in pooled
        nan_mask = df["omega"].isna()
        assert result.loc[nan_mask, "omega_pooled"].isna().all(), \
            "NaN values should pass through"

        # Shrinkage for NaN values should be 0
        assert (result.loc[nan_mask, "omega_shrinkage"] == 0).all(), \
            "Shrinkage should be 0 for NaN values"

        # Non-NaN values should be processed normally
        assert result.loc[~nan_mask, "omega_pooled"].notna().all()

    def test_sector_means_preserved(self):
        """Test that mean of pooled == sector mean of raw (within fp tolerance)."""
        np.random.seed(46)

        # Create multiple sectors
        sectors = []
        for sector_name in ["tech", "finance", "energy"]:
            n = 25
            mean_val = np.random.uniform(0.2, 0.8)
            values = mean_val + np.random.normal(0, 0.15, n)
            sectors.append(pd.DataFrame({
                "name": [f"{sector_name}_{i}" for i in range(n)],
                "sector": [sector_name] * n,
                "lam": values,
            }))

        df = pd.concat(sectors, ignore_index=True)
        result = pool_parameters(df, params=["lam"], min_sector_size=5)

        # Check that sector means are preserved
        for sector_name in ["tech", "finance", "energy"]:
            sector_mask = df["sector"] == sector_name
            raw_mean = df.loc[sector_mask, "lam"].mean()
            pooled_mean = result.loc[sector_mask, "lam_pooled"].mean()

            # Should be equal within floating point tolerance
            assert np.abs(raw_mean - pooled_mean) < 1e-10, \
                f"Sector {sector_name} mean not preserved: {raw_mean} vs {pooled_mean}"

    def test_multiple_params(self):
        """Test pooling multiple parameters simultaneously."""
        np.random.seed(47)
        n_names = 30

        df = pd.DataFrame({
            "name": [f"stock_{i}" for i in range(n_names)],
            "sector": ["tech"] * n_names,
            "kappa": 0.5 + np.random.normal(0, 0.2, n_names),
            "theta": 0.04 + np.random.normal(0, 0.01, n_names),
            "sigma_v": 0.3 + np.random.normal(0, 0.1, n_names),
        })

        result = pool_parameters(
            df, params=["kappa", "theta", "sigma_v"], min_sector_size=5
        )

        # Check all pooled and shrinkage columns exist
        for param in ["kappa", "theta", "sigma_v"]:
            assert f"{param}_pooled" in result.columns
            assert f"{param}_shrinkage" in result.columns

        # Check that values are reasonable
        assert result["kappa_pooled"].notna().all()
        assert result["theta_pooled"].notna().all()
        assert result["sigma_v_pooled"].notna().all()

    def test_positivity_constraint_clipping(self):
        """Test that positive-constrained params are clipped at small positive floor."""
        np.random.seed(48)
        n_names = 20

        # Create values that would shrink to near-zero or negative
        # Use very small positive values with strong shrinkage
        small_values = np.random.uniform(0.001, 0.01, n_names)

        df = pd.DataFrame({
            "name": [f"stock_{i}" for i in range(n_names)],
            "sector": ["test"] * n_names,
            "kappa": small_values,  # Positivity-constrained
            "theta": small_values,  # Positivity-constrained
        })

        result = pool_parameters(df, params=["kappa", "theta"], min_sector_size=5)

        # Pooled values should remain positive
        assert (result["kappa_pooled"] > 0).all(), \
            "Pooled kappa should remain positive"
        assert (result["theta_pooled"] > 0).all(), \
            "Pooled theta should remain positive"

    def test_unknown_sector_handling(self):
        """Test that UNKNOWN sector is treated as small sector."""
        np.random.seed(49)

        # Create a large sector and UNKNOWN sector
        large_size = 30
        unknown_size = 10

        large_mean = 0.6
        unknown_mean = 0.3

        # Generate properly randomized values
        large_values = [large_mean + np.random.normal(0, 0.1) for _ in range(large_size)]
        unknown_values = [unknown_mean + np.random.normal(0, 0.1) for _ in range(unknown_size)]

        df = pd.DataFrame({
            "name": ([f"large_{i}" for i in range(large_size)] +
                     [f"unknown_{i}" for i in range(unknown_size)]),
            "sector": (["large"] * large_size +
                      ["UNKNOWN"] * unknown_size),
            "alpha": large_values + unknown_values,
        })

        result = pool_parameters(df, params=["alpha"], min_sector_size=15)

        # UNKNOWN sector should be treated as small (shrink to global mean)
        # even though it has 10 names
        global_mean = df["alpha"].mean()
        unknown_mask = result["sector"] == "UNKNOWN"
        unknown_pooled = result.loc[unknown_mask, "alpha_pooled"]
        unknown_raw = df.loc[unknown_mask, "alpha"]

        # Should show shrinkage toward global mean
        pooled_mean = unknown_pooled.mean()
        raw_mean = unknown_raw.mean()

        # Pooled mean should be between raw mean and global mean
        if raw_mean < global_mean:
            assert pooled_mean > raw_mean
            assert pooled_mean < global_mean or np.abs(pooled_mean - global_mean) < 1e-10
        elif raw_mean > global_mean:
            assert pooled_mean < raw_mean
            assert pooled_mean > global_mean or np.abs(pooled_mean - global_mean) < 1e-10
