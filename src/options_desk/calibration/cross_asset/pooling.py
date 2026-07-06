"""Empirical-Bayes sector pooling of calibrated parameters.

This module implements positive-part James-Stein shrinkage estimators
for pooling calibrated option pricing parameters within sectors.
"""

import numpy as np
import pandas as pd


# Parameters that must remain positive
POSITIVITY_CONSTRAINED = {
    "kappa", "theta", "sigma_v", "omega", "alpha", "beta", "lam", "sigma_j"
}

# Small positive floor for clipping
EPSILON = 1e-8


def pool_parameters(
    df: pd.DataFrame,
    params: list[str],
    sector_col: str = "sector",
    min_sector_size: int = 5,
) -> pd.DataFrame:
    """Pool parameters across sectors using empirical-Bayes shrinkage.

    Uses positive-part James-Stein estimator per (param, sector).
    Small sectors (n < min_sector_size) and UNKNOWN sector shrink toward
    global mean. Larger sectors shrink toward sector mean.

    Shrinkage weight resolution:
    Standard positive-part James-Stein formula:
        SS = Σ(x_i - m_s)²  (sum of squared deviations from sector mean)
        c = max(0, 1 - (n_s - 3)·σ̂²/SS)  (shrinkage factor)
        x_pooled_i = m_s + c·(x_i - m_s)  (shrunk value)
        reported shrinkage = 1 - c  (fraction shrunk toward mean)

    Degenerate cases:
        - SS ≈ 0: full shrink to mean (shrinkage=1, harmless since x_i ≈ m_s)
        - n_s ≤ 3: treat as small sector (use global-mean rule)

    Args:
        df: DataFrame with calibrated parameters.
        params: List of parameter column names to pool.
        sector_col: Name of sector column (default "sector").
        min_sector_size: Minimum sector size for sector-specific shrinkage.
            Smaller sectors shrink toward global mean.

    Returns:
        DataFrame copy with added columns per parameter p:
            - {p}_pooled: Pooled parameter value
            - {p}_shrinkage: Shrinkage weight ∈ [0, 1]
              (0 = no shrinkage, 1 = full shrinkage to mean)

    Notes:
        - NaN values pass through unchanged with shrinkage=0
        - Positivity-constrained params (kappa, theta, etc.) are clipped
          at EPSILON when raw value is positive
        - Sector means are preserved (mean of pooled == mean of raw)
    """
    result = df.copy()

    for param in params:
        # Initialize output columns
        pooled_col = f"{param}_pooled"
        shrinkage_col = f"{param}_shrinkage"

        result[pooled_col] = result[param].copy()
        result[shrinkage_col] = 0.0

        # Get valid (non-NaN) values
        valid_mask = result[param].notna()

        if not valid_mask.any():
            continue

        # Calculate global mean for small sectors
        global_mean = result.loc[valid_mask, param].mean()

        # Group by sector
        for sector, group_idx in result.groupby(sector_col).groups.items():
            # Filter to valid values in this sector
            sector_mask = result.index.isin(group_idx) & valid_mask
            if not sector_mask.any():
                continue

            sector_data = result.loc[sector_mask, param].values
            n_sector = len(sector_data)

            # Determine if this is a small sector or UNKNOWN
            is_small_sector = (n_sector < min_sector_size) or (sector == "UNKNOWN")

            if is_small_sector:
                # Shrink toward global mean
                mean_target = global_mean
            else:
                # Shrink toward sector mean
                mean_target = sector_data.mean()

            # Calculate shrinkage using positive-part James-Stein
            deviations = sector_data - mean_target
            SS = np.sum(deviations ** 2)

            # Handle degenerate cases early
            if n_sector <= 3:
                # Small sample: use full shrinkage
                c = 0.0
                shrinkage_weight = 1.0
            elif SS < EPSILON:
                # All values essentially identical → no shrinkage needed
                # (they're already at the mean)
                c = 1.0
                shrinkage_weight = 0.0
            else:
                # Estimate noise variance using MAD (median absolute deviation)
                # σ̂² = (MAD · 1.4826)²
                mad = np.median(np.abs(deviations - np.median(deviations)))
                sigma_sq = (mad * 1.4826) ** 2

                # If MAD is also near zero, no shrinkage needed
                if sigma_sq < EPSILON:
                    c = 1.0
                    shrinkage_weight = 0.0
                else:
                    # Standard James-Stein formula
                    shrinkage_factor = (n_sector - 3) * sigma_sq / SS
                    c = max(0.0, 1.0 - shrinkage_factor)
                    shrinkage_weight = 1.0 - c

            # Apply shrinkage: x_pooled = m + c·(x - m)
            pooled_values = mean_target + c * deviations

            # Apply positivity constraint if needed
            if param in POSITIVITY_CONSTRAINED:
                # Only clip if original values were positive
                positive_original = sector_data > 0
                pooled_values = np.where(
                    positive_original,
                    np.maximum(pooled_values, EPSILON),
                    pooled_values
                )

            # Store results
            result.loc[sector_mask, pooled_col] = pooled_values
            result.loc[sector_mask, shrinkage_col] = shrinkage_weight

    return result
