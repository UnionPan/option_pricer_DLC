"""
Analytical Black-Scholes delta hedger for the new ObservationBatch contract.

This is the canonical *non-learned* baseline for the BuehlerTrainer eval
pipeline. It plugs directly into ``collect_agent_rollout_from_market``
alongside ``DoNothingAgent`` and trained ``TorchPolicyAgent`` so you can
compare trained policies against an analytical reference on the same paths.

Strategy at each step:

1. Read current spot from ``observation.spot[0]``.
2. Read current variance and time-ratio from
   ``observation.context_features`` (which carries v_t and t/horizon).
3. Compute Black-Scholes delta of the liability using sigma = sqrt(v_t).
4. Set the underlying target position to ``-liability_quantity * delta``.
5. Optionally add a single-option gamma-neutralizing position picked from
   the floating grid (closest call to (ATM, liability-maturity)).

Emits trade actions (delta from previous position) — matches the
``action_mode='trades'`` convention used throughout the deep_hedging
package.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import norm

from ..utils.contracts import LiabilitySpec, ObservationBatch
from .base import BaseHedgingAgent


def _bs_delta_gamma(
    S: float, K: float, tau: float, sigma: float, r: float, kind: str,
) -> tuple[float, float]:
    """Black-Scholes delta and gamma for a vanilla European call/put."""
    if tau <= 0:
        if kind == "call":
            delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        return delta, 0.0

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau + 1e-12)
    if kind == "call":
        delta = float(norm.cdf(d1))
    else:
        delta = float(norm.cdf(d1) - 1.0)
    gamma = float(norm.pdf(d1) / (S * sigma * sqrt_tau + 1e-12))
    return delta, gamma


class AnalyticalDeltaHedger(BaseHedgingAgent):
    """
    Analytical Black-Scholes delta-hedging baseline.

    Plug-compatible with the BuehlerTrainer eval pipeline. Computes the
    liability's BS delta from the observed spot + variance and trades the
    underlying to delta-neutralize. Optionally adds a single-option
    gamma-neutralizing position in the closest-to-(ATM, liability-maturity)
    call from the floating grid.

    Args:
        n_instruments: total instruments in the action vector
            (1 stock + N_options).
        liability: ``LiabilitySpec`` (must be ``kind='call'`` or
            ``'put'`` — cliquet/barrier/etc. are not supported).
        env_horizon_steps: total horizon of the env in steps.
        dt: time per env step (typically ``1/250`` for daily).
        risk_free_rate: r used in BS (default 0).
        gamma_hedge: if True, also hedge gamma using one option from the grid.
            Requires ``option_grid_spec``.
        option_grid_spec: list of dicts (from
            :func:`build_option_grid_spec`) describing each option slot.
        position_limits: clamp bound for trades.
        name: agent name.
    """

    def __init__(
        self,
        n_instruments: int,
        liability: LiabilitySpec,
        env_horizon_steps: int,
        dt: float = 1 / 250.0,
        risk_free_rate: float = 0.0,
        gamma_hedge: bool = False,
        option_grid_spec: list | None = None,
        position_limits: float = 100.0,
        name: str = "AnalyticalDeltaHedger",
    ) -> None:
        if liability.kind not in ("call", "put"):
            raise ValueError(
                f"AnalyticalDeltaHedger supports only call/put liabilities, "
                f"got kind={liability.kind!r}"
            )
        super().__init__(n_instruments, position_limits, name)
        self.liability = liability
        self.env_horizon_steps = env_horizon_steps
        self.dt = float(dt)
        self.r = float(risk_free_rate)
        self.gamma_hedge = bool(gamma_hedge)
        self.option_grid_spec = option_grid_spec

        if self.gamma_hedge and not option_grid_spec:
            raise ValueError(
                "gamma_hedge=True requires option_grid_spec to identify "
                "tradable options on the floating grid."
            )

        self._gamma_idx = self._select_gamma_instrument() if self.gamma_hedge else None
        self._gamma_tau_steps = (
            option_grid_spec[self._gamma_idx_in_spec]["tau_steps"]
            if self._gamma_idx is not None else None
        )
        self._gamma_moneyness = (
            option_grid_spec[self._gamma_idx_in_spec]["moneyness"]
            if self._gamma_idx is not None else None
        )

        self._target_positions: np.ndarray | None = None

    def _select_gamma_instrument(self) -> int:
        """Pick the call closest to (ATM, liability_maturity) — return its env index."""
        best_spec_pos, best_idx, best_score = None, None, float("inf")
        for spec_pos, spec in enumerate(self.option_grid_spec):
            if spec["kind"] != "call":
                continue
            score = (
                abs(spec["tau_steps"] - self.liability.maturity)
                + 100 * abs(spec["moneyness"] - 1.0)
            )
            if score < best_score:
                best_score = score
                best_spec_pos = spec_pos
                best_idx = spec["idx"]
        if best_idx is None:
            raise ValueError("No call options in option_grid_spec — cannot gamma-hedge")
        self._gamma_idx_in_spec = best_spec_pos
        return best_idx

    def reset(self, observation: Any, info: Dict[str, Any]) -> None:
        super().reset(observation, info)
        self._target_positions = np.zeros(self.n_instruments, dtype=np.float32)
        if self.current_positions is None:
            self.current_positions = np.zeros(self.n_instruments, dtype=np.float32)

    def act(
        self, observation: ObservationBatch, info: Dict[str, Any],
    ) -> np.ndarray:
        S = float(np.asarray(observation.spot).ravel()[0])
        ctx = np.asarray(observation.context_features).ravel()
        variance = max(float(ctx[0]), 1e-8)
        time_ratio = float(ctx[1])
        sigma = np.sqrt(variance)

        steps_elapsed = time_ratio * self.env_horizon_steps
        steps_remaining = max(self.liability.maturity - steps_elapsed, 0.0)
        tau = max(steps_remaining * self.dt, 1e-8)

        delta, gamma = _bs_delta_gamma(
            S=S, K=self.liability.strike, tau=tau, sigma=sigma,
            r=self.r, kind=self.liability.kind,
        )

        # Buehler convention: minimize Var(portfolio − liability_payoff).
        # Portfolio must TRACK the (signed) liability — not offset it.
        # d(liability)/dS = quantity * delta; we want d(portfolio)/dS to
        # equal d(liability)/dS, so target_underlying = quantity * delta.
        liability_delta = self.liability.quantity * delta

        new_target = np.zeros(self.n_instruments, dtype=np.float32)
        new_target[0] = liability_delta

        if self.gamma_hedge and self._gamma_idx is not None:
            opt_tau = max(self._gamma_tau_steps * self.dt, 1e-8)
            opt_K = self._gamma_moneyness * S
            opt_delta_d, opt_gamma = _bs_delta_gamma(
                S=S, K=opt_K, tau=opt_tau, sigma=sigma, r=self.r, kind="call",
            )
            liability_gamma = self.liability.quantity * gamma
            if abs(opt_gamma) > 1e-8:
                # Want d²(portfolio)/dS² = liability_gamma → opt_pos * opt_gamma = liability_gamma
                opt_pos = liability_gamma / opt_gamma
                new_target[self._gamma_idx] = float(opt_pos)
                # Adjust underlying so total delta still matches liability_delta
                new_target[0] = liability_delta - opt_pos * opt_delta_d

        if observation.action_mask is not None:
            mask = np.asarray(observation.action_mask, dtype=bool).ravel()
            new_target = new_target * mask.astype(np.float32)

        if self.current_positions is None:
            self.current_positions = np.zeros(self.n_instruments, dtype=np.float32)
        trade = new_target - self.current_positions
        trade = np.clip(trade, -self.position_limits, self.position_limits)

        self.current_positions = self.current_positions + trade
        self._target_positions = new_target
        return trade.astype(np.float32)


def build_option_grid_spec(option_grid) -> list[dict]:
    """Build option_grid_spec list from a FloatingOptionGrid.

    Returns a list with one dict per option slot::

        {'idx': int, 'kind': 'call'|'put', 'tau_steps': int, 'moneyness': float}

    Slot 0 (the underlying) is omitted. Order matches the trainer's
    instrument-vector layout.
    """
    spec = []
    idx = 1
    for m in option_grid.maturities:
        for k in option_grid.moneyness_by_maturity[m]:
            spec.append({"idx": idx,     "kind": "call", "tau_steps": int(m), "moneyness": float(k)})
            spec.append({"idx": idx + 1, "kind": "put",  "tau_steps": int(m), "moneyness": float(k)})
            idx += 2
    return spec
