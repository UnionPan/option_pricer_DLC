"""Advanced pricing models (Heston, SABR, Merton)."""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
from typing import Literal


class HestonModel:
    """Heston stochastic volatility model implementation."""

    @staticmethod
    def price(
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        v0: float,  # initial variance
        theta: float,  # long-term variance
        kappa: float,  # mean reversion speed
        sigma_v: float,  # volatility of variance
        rho: float,  # correlation
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Calculate option price using Heston model via characteristic function.

        Parameters:
        - v0: initial variance (volatility^2)
        - theta: long-term variance
        - kappa: mean reversion speed
        - sigma_v: volatility of volatility
        - rho: correlation between asset and volatility
        """
        if time_to_expiry <= 0:
            return max(0, spot - strike) if option_type == "call" else max(0, strike - spot)

        # Use semi-analytical formula via Fourier inversion
        P1 = HestonModel._heston_probability(
            spot, strike, time_to_expiry, rate, v0, theta, kappa, sigma_v, rho, 1
        )
        P2 = HestonModel._heston_probability(
            spot, strike, time_to_expiry, rate, v0, theta, kappa, sigma_v, rho, 2
        )

        if option_type == "call":
            price = spot * P1 - strike * np.exp(-rate * time_to_expiry) * P2
        else:
            price = strike * np.exp(-rate * time_to_expiry) * (1 - P2) - spot * (1 - P1)

        return max(0, price)

    @staticmethod
    def _heston_probability(spot, strike, T, r, v0, theta, kappa, sigma_v, rho, j):
        """Calculate probability Pj using Fourier inversion."""

        def integrand(phi):
            return np.real(
                np.exp(-1j * phi * np.log(strike)) *
                HestonModel._characteristic_function(phi, spot, T, r, v0, theta, kappa, sigma_v, rho, j) /
                (1j * phi)
            )

        try:
            integral, _ = quad(integrand, 0, 100, limit=100)
            return 0.5 + (1 / np.pi) * integral
        except:
            return 0.5  # Fallback

    @staticmethod
    def _characteristic_function(phi, S, T, r, v0, theta, kappa, sigma_v, rho, j):
        """Heston characteristic function."""
        if j == 1:
            u, b = 0.5, kappa - rho * sigma_v
        else:
            u, b = -0.5, kappa

        a = kappa * theta
        x = np.log(S)

        d = np.sqrt((rho * sigma_v * 1j * phi - b)**2 - sigma_v**2 * (2 * u * 1j * phi - phi**2))
        g = (b - rho * sigma_v * 1j * phi + d) / (b - rho * sigma_v * 1j * phi - d)

        C = r * 1j * phi * T + (a / sigma_v**2) * (
            (b - rho * sigma_v * 1j * phi + d) * T -
            2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )

        D = ((b - rho * sigma_v * 1j * phi + d) / sigma_v**2) * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        )

        return np.exp(C + D * v0 + 1j * phi * x)


class SABRModel:
    """SABR (Stochastic Alpha Beta Rho) model."""

    @staticmethod
    def implied_volatility(
        forward: float,
        strike: float,
        time_to_expiry: float,
        alpha: float,  # volatility level
        beta: float,  # CEV parameter (0-1)
        rho: float,  # correlation
        nu: float,  # volatility of volatility
    ) -> float:
        """
        Calculate implied volatility using SABR formula.

        Parameters:
        - alpha: initial volatility
        - beta: CEV exponent (0=normal, 1=lognormal)
        - rho: correlation between forward and volatility
        - nu: volatility of volatility
        """
        if time_to_expiry <= 0 or alpha <= 0:
            return 0.0

        if abs(forward - strike) < 1e-10:  # ATM case
            fk_mid = (forward * strike) ** ((1 - beta) / 2)
            term1 = alpha / fk_mid
            term2 = 1 + ((((1 - beta)**2 / 24) * (alpha**2 / fk_mid**2) +
                          (rho * beta * nu * alpha / (4 * fk_mid)) +
                          ((2 - 3 * rho**2) / 24) * nu**2) * time_to_expiry)
            return term1 * term2

        # General case
        fk_mid = (forward * strike) ** ((1 - beta) / 2)
        log_fk = np.log(forward / strike)

        z = (nu / alpha) * fk_mid * log_fk
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        if abs(z) < 1e-5:
            x_z = 1
        else:
            x_z = z / x_z

        term1 = alpha / (fk_mid * (1 + ((1 - beta)**2 / 24) * log_fk**2 +
                                    ((1 - beta)**4 / 1920) * log_fk**4))

        term2 = 1 + ((((1 - beta)**2 / 24) * (alpha**2 / fk_mid**2) +
                      (rho * beta * nu * alpha / (4 * fk_mid)) +
                      ((2 - 3 * rho**2) / 24) * nu**2) * time_to_expiry)

        return term1 * x_z * term2


class MertonJumpDiffusion:
    """Merton Jump Diffusion model."""

    @staticmethod
    def price(
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        sigma: float,  # diffusion volatility
        lambda_j: float,  # jump intensity (jumps per year)
        mu_j: float,  # mean jump size
        sigma_j: float,  # jump volatility
        option_type: Literal["call", "put"] = "call",
        max_jumps: int = 50,
    ) -> float:
        """
        Calculate option price using Merton Jump Diffusion model.

        Uses series expansion summing over different numbers of jumps.

        Parameters:
        - sigma: diffusion volatility
        - lambda_j: jump intensity
        - mu_j: mean jump size (in log returns)
        - sigma_j: standard deviation of jump size
        """
        if time_to_expiry <= 0:
            return max(0, spot - strike) if option_type == "call" else max(0, strike - spot)

        price = 0.0
        lambda_prime = lambda_j * (1 + mu_j)

        for n in range(max_jumps):
            # Probability of n jumps
            poisson_prob = np.exp(-lambda_prime * time_to_expiry) * \
                           (lambda_prime * time_to_expiry)**n / np.math.factorial(n)

            if poisson_prob < 1e-10:
                break

            # Adjusted parameters for n jumps
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / time_to_expiry)
            r_n = rate - lambda_j * mu_j + n * np.log(1 + mu_j) / time_to_expiry

            # Black-Scholes price with adjusted parameters
            d1 = (np.log(spot / strike) + (r_n + 0.5 * sigma_n**2) * time_to_expiry) / \
                 (sigma_n * np.sqrt(time_to_expiry))
            d2 = d1 - sigma_n * np.sqrt(time_to_expiry)

            if option_type == "call":
                bs_price = spot * np.exp((r_n - rate) * time_to_expiry) * norm.cdf(d1) - \
                           strike * np.exp(-rate * time_to_expiry) * norm.cdf(d2)
            else:
                bs_price = strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d2) - \
                           spot * np.exp((r_n - rate) * time_to_expiry) * norm.cdf(-d1)

            price += poisson_prob * bs_price

        return max(0, price)


def implied_volatility_from_price(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: Literal["call", "put"],
    model: Literal["black_scholes", "heston", "merton"] = "black_scholes",
    model_params: dict = None,
) -> float:
    """
    Calculate implied volatility by inverting the pricing model.

    For advanced models, uses Black-Scholes IV as approximation.
    """
    from options_desk.pricing.black_scholes import black_scholes_price
    from options_desk.core.option import OptionType

    if time_to_expiry <= 0 or market_price <= 0:
        return 0.0

    opt_type = OptionType.CALL if option_type == "call" else OptionType.PUT

    def objective(vol):
        try:
            if model == "black_scholes":
                model_price = black_scholes_price(
                    spot, strike, time_to_expiry, rate, vol, opt_type
                )
            else:
                # For advanced models, use Black-Scholes as approximation
                model_price = black_scholes_price(
                    spot, strike, time_to_expiry, rate, vol, opt_type
                )
            return model_price - market_price
        except:
            return 1e10

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
        return max(0.001, min(5.0, iv))
    except:
        return 0.20  # Default fallback
