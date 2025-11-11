"""
Tests for option pricing models.
"""

import pytest
import numpy as np

from options_desk.core.option import OptionType
from options_desk.pricing.black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
)


class TestBlackScholes:
    """Test Black-Scholes pricing and greeks."""

    def test_call_price(self):
        """Test call option pricing."""
        spot = 100
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05
        volatility = 0.2

        price = black_scholes_price(spot, strike, time_to_expiry, rate, volatility, OptionType.CALL)

        # ATM call should have positive value
        assert price > 0
        # Should be less than spot price
        assert price < spot

    def test_put_price(self):
        """Test put option pricing."""
        spot = 100
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05
        volatility = 0.2

        price = black_scholes_price(spot, strike, time_to_expiry, rate, volatility, OptionType.PUT)

        # ATM put should have positive value
        assert price > 0
        # Should be less than strike price
        assert price < strike

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        spot = 100
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05
        volatility = 0.2
        dividend = 0.0

        call_price = black_scholes_price(
            spot, strike, time_to_expiry, rate, volatility, OptionType.CALL, dividend
        )
        put_price = black_scholes_price(
            spot, strike, time_to_expiry, rate, volatility, OptionType.PUT, dividend
        )

        # Put-call parity: C - P = S - K*e^(-rT)
        lhs = call_price - put_price
        rhs = spot - strike * np.exp(-rate * time_to_expiry)

        assert np.isclose(lhs, rhs, rtol=1e-6)

    def test_delta_bounds(self):
        """Test that delta is within valid bounds."""
        spot = 100
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05
        volatility = 0.2

        call_delta = black_scholes_delta(
            spot, strike, time_to_expiry, rate, volatility, OptionType.CALL
        )
        put_delta = black_scholes_delta(spot, strike, time_to_expiry, rate, volatility, OptionType.PUT)

        # Call delta should be between 0 and 1
        assert 0 <= call_delta <= 1
        # Put delta should be between -1 and 0
        assert -1 <= put_delta <= 0

    def test_gamma_positive(self):
        """Test that gamma is always positive."""
        spot = 100
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05
        volatility = 0.2

        gamma = black_scholes_gamma(spot, strike, time_to_expiry, rate, volatility)

        # Gamma should be positive
        assert gamma > 0

    def test_vega_positive(self):
        """Test that vega is always positive."""
        spot = 100
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05
        volatility = 0.2

        vega = black_scholes_vega(spot, strike, time_to_expiry, rate, volatility)

        # Vega should be positive
        assert vega > 0

    def test_theta_sign(self):
        """Test theta sign for different option types."""
        spot = 100
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05
        volatility = 0.2

        call_theta = black_scholes_theta(
            spot, strike, time_to_expiry, rate, volatility, OptionType.CALL
        )
        put_theta = black_scholes_theta(spot, strike, time_to_expiry, rate, volatility, OptionType.PUT)

        # For ATM options, theta is typically negative
        # (time decay reduces option value)
        assert call_theta < 0
        assert put_theta < 0
