"""
KFAC (Kronecker-Factored Approximate Curvature) optimizer for the deep
hedging trainer.

Implements a practical KFAC variant for ``nn.Linear`` layers (which is
all the LSTM is built from when using ``ManualLSTMCell``):

    Block-diagonal Fisher F_W ≈ A ⊗ S
        A = E[a · a^T]      input activation covariance,  shape (in,  in)
        S = E[g · g^T]      pre-activation grad cov,      shape (out, out)

    Preconditioned gradient:
        G_pre = (S + λ I)^(-1) · ∇W · (A + λ I)^(-1)

We apply Tikhonov damping λ for numerical stability and scale the final
update with a trust-region rule. The default ``natural`` metric follows the
DH-KFAC notes: η = min(sqrt(ρ / <preconditioned_grad, raw_grad>), η_max).
The older ``frobenius`` metric remains available as a conservative fallback.

Implementation notes:

* Forward + backward hooks on each ``nn.Linear`` capture activations ``a``
  and pre-activation gradients ``g``. We aggregate over the batch and
  over the time dimension (LSTMCells are reused across the rollout's
  60+ steps).
* Kronecker factors A, S maintained as exponential moving averages with
  decay ``ema_decay``. Eigendecomposed periodically (every
  ``n_eigen_decomp`` steps) to amortize the cost of the inverse.
* Inverse via eigendecomp:
      eA, UA = eigh(A + λI);  eS, US = eigh(S + λI)
      G_pre  = US · ((US^T · ∇W · UA) / (eS_outer · eA_outer)) · UA^T
* Trust-region damping shrinks the update if the configured metric exceeds
  the configured cap.

References:
    Martens & Grosse (2015), "Optimizing Neural Networks with
        Kronecker-factored Approximate Curvature."
    notes/fasthedgingwith2nd/kfac.tex (paper this implementation targets)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import Dict, List

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _list_linear_layers(module: "nn.Module") -> List["nn.Linear"]:
    """Return all nn.Linear submodules in module-traversal order."""
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


class KFACOptimizer:
    """KFAC for ``nn.Linear`` layers inside an arbitrary module.

    Usage::

        optim = KFACOptimizer(policy, lr=1e-2, damping=1e-2,
                              ema_decay=0.95, n_eigen_decomp=25)

        for batch in data:
            optim.zero_grad()
            loss = compute_loss(policy(batch))
            loss.backward()
            optim.step()
    """

    def __init__(
        self,
        module: "nn.Module",
        lr: float = 1e-2,
        damping: float = 1e-2,
        ema_decay: float = 0.95,
        n_eigen_decomp: int = 25,
        trust_region: float | None = 1e-2,
        trust_region_decay: float | None = None,
        min_trust_region: float | None = None,
        trust_region_metric: str = "natural",
        weight_decay: float = 0.0,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("KFACOptimizer requires PyTorch")
        self.module = module
        self.lr = float(lr)
        self.damping = float(damping)
        self.ema_decay = float(ema_decay)
        self.n_eigen_decomp = int(n_eigen_decomp)
        self.trust_region = (
            None if trust_region is None else float(trust_region)
        )
        self.trust_region_decay = (
            None if trust_region_decay is None else float(trust_region_decay)
        )
        self.min_trust_region = (
            None if min_trust_region is None else float(min_trust_region)
        )
        self.trust_region_metric = str(trust_region_metric).lower()
        self.weight_decay = float(weight_decay)
        self._validate_hyperparameters()
        self.last_step_stats: Dict[str, float] = {}

        self._linear_layers: List["nn.Linear"] = _list_linear_layers(module)
        # Per-layer state
        self._A: Dict[int, torch.Tensor] = {}
        self._S: Dict[int, torch.Tensor] = {}
        self._UA: Dict[int, torch.Tensor] = {}
        self._SA: Dict[int, torch.Tensor] = {}
        self._US: Dict[int, torch.Tensor] = {}
        self._SS: Dict[int, torch.Tensor] = {}
        # Per-step buffers (cleared each zero_grad)
        self._a_buf: Dict[int, List[torch.Tensor]] = {}
        self._g_buf: Dict[int, List[torch.Tensor]] = {}

        self._handles = []
        self._step_count = 0
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        for idx, layer in enumerate(self._linear_layers):
            self._a_buf[idx] = []
            self._g_buf[idx] = []
            h_fwd = layer.register_forward_pre_hook(self._make_fwd_hook(idx))
            h_bwd = layer.register_full_backward_hook(self._make_bwd_hook(idx))
            self._handles.append(h_fwd)
            self._handles.append(h_bwd)

    def _make_fwd_hook(self, idx: int):
        def hook(module, inputs):
            x = inputs[0]
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            self._a_buf[idx].append(x.detach())
        return hook

    def _make_bwd_hook(self, idx: int):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return
            if g.dim() > 2:
                g = g.reshape(-1, g.shape[-1])
            self._g_buf[idx].append(g.detach())
        return hook

    def remove_hooks(self) -> None:
        """Remove all hooks (call before pickling the module)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Optimizer-API methods
    # ------------------------------------------------------------------
    def zero_grad(self) -> None:
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        for idx in self._a_buf:
            self._a_buf[idx].clear()
            self._g_buf[idx].clear()

    def step(self) -> None:
        """One preconditioned update. Call after ``loss.backward()``."""
        self._step_count += 1
        with torch.no_grad():
            self._update_factors()
            if self._step_count == 1 or (self._step_count % self.n_eigen_decomp) == 0:
                self._refresh_eigendecomps()
            self._apply_preconditioned_step()
            self._advance_schedules()

    def _validate_hyperparameters(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr!r}")
        if self.damping <= 0.0:
            raise ValueError(f"damping must be positive, got {self.damping!r}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be in [0, 1), got {self.ema_decay!r}"
            )
        if self.n_eigen_decomp <= 0:
            raise ValueError(
                f"n_eigen_decomp must be positive, got {self.n_eigen_decomp!r}"
            )
        if self.trust_region is not None and self.trust_region <= 0.0:
            raise ValueError(
                f"trust_region must be positive or None, got {self.trust_region!r}"
            )
        if self.trust_region_decay is not None and not (
            0.0 < self.trust_region_decay <= 1.0
        ):
            raise ValueError(
                "trust_region_decay must be in (0, 1] or None, "
                f"got {self.trust_region_decay!r}"
            )
        if self.min_trust_region is not None and self.min_trust_region < 0.0:
            raise ValueError(
                "min_trust_region must be non-negative or None, "
                f"got {self.min_trust_region!r}"
            )
        if self.trust_region_metric not in {"natural", "frobenius"}:
            raise ValueError(
                "trust_region_metric must be 'natural' or 'frobenius', "
                f"got {self.trust_region_metric!r}"
            )

    def _advance_schedules(self) -> None:
        if self.trust_region is None or self.trust_region_decay is None:
            return
        next_trust_region = self.trust_region * self.trust_region_decay
        if self.min_trust_region is not None:
            next_trust_region = max(next_trust_region, self.min_trust_region)
        self.trust_region = next_trust_region

    # ------------------------------------------------------------------
    # Factor maintenance
    # ------------------------------------------------------------------
    def _update_factors(self) -> None:
        beta = self.ema_decay
        for idx, layer in enumerate(self._linear_layers):
            a_chunks = self._a_buf[idx]
            g_chunks = self._g_buf[idx]
            if not a_chunks or not g_chunks:
                continue
            a_all = torch.cat(a_chunks, dim=0)
            g_all = torch.cat(g_chunks, dim=0)
            n = a_all.shape[0]
            if n == 0:
                continue
            # KFAC factor scaling that handles both feedforward (T=1) and
            # recurrent (T>1) reuse of a layer:
            #
            #   B = a_chunks[0].shape[0]      per-call batch size
            #   T = len(a_chunks)             number of times the layer was hit
            #   N = B * T                     total rows captured
            #
            # For batch-mean loss, PyTorch's captured g already carries a
            # 1/B factor. The Fisher we want is per-data-sample:
            #     F = (1/B) Σ_b grad_b grad_b^T
            #
            # Treating each (sample, time) pair as approximately independent:
            #     A = (1/N) Σ_(b,t) a_(b,t) a_(b,t)^T
            #     S = (B/T) · Σ_(b,t) g_(b,t) g_(b,t)^T
            #
            # For T=1 this reduces to A = (1/B)·sum, S = B·sum (= * N) —
            # i.e. matches the standard feedforward KFAC formula.
            # For T>>1, S avoids the spurious T² inflation that came from
            # the original `S * n = S * (B*T)` formulation.
            B_per_call = a_chunks[0].shape[0]
            T_per_step = len(a_chunks)
            A_new = (a_all.t() @ a_all) / float(n)
            S_new = (g_all.t() @ g_all) * float(B_per_call) / float(T_per_step)

            if idx in self._A:
                self._A[idx].mul_(beta).add_(A_new, alpha=1.0 - beta)
                self._S[idx].mul_(beta).add_(S_new, alpha=1.0 - beta)
            else:
                self._A[idx] = A_new.clone()
                self._S[idx] = S_new.clone()

    def _refresh_eigendecomps(self) -> None:
        for idx in self._A:
            A = self._A[idx]
            S = self._S[idx]
            d = self.damping
            try:
                eA, UA = torch.linalg.eigh(
                    A + d * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                )
                eS, US = torch.linalg.eigh(
                    S + d * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
                )
            except torch.linalg.LinAlgError:
                continue
            eA = torch.clamp(eA, min=d)
            eS = torch.clamp(eS, min=d)
            self._UA[idx] = UA
            self._SA[idx] = eA
            self._US[idx] = US
            self._SS[idx] = eS

    # ------------------------------------------------------------------
    # Preconditioned update
    # ------------------------------------------------------------------
    def _apply_preconditioned_step(self) -> None:
        preconditioned_norm_sq = 0.0
        natural_metric = 0.0
        layer_updates = []
        kfac_params = set()
        for idx, layer in enumerate(self._linear_layers):
            kfac_params.add(id(layer.weight))
            if layer.bias is not None:
                kfac_params.add(id(layer.bias))
            if layer.weight.grad is None:
                continue
            g_w = layer.weight.grad
            if self.weight_decay > 0:
                g_w = g_w + self.weight_decay * layer.weight

            if idx in self._UA and idx in self._US:
                UA, eA = self._UA[idx], self._SA[idx]
                US, eS = self._US[idx], self._SS[idx]
                tmp = US.t() @ g_w @ UA
                denom = eS.unsqueeze(1) * eA.unsqueeze(0)
                tmp = tmp / denom
                g_pre = US @ tmp @ UA.t()
            else:
                g_pre = g_w

            preconditioned_norm_sq += float(g_pre.pow(2).sum().item())
            natural_metric += float((g_pre * g_w).sum().item())
            g_b = layer.bias.grad if layer.bias is not None else None
            if g_b is not None and self.weight_decay > 0:
                g_b = g_b + self.weight_decay * layer.bias
            if g_b is not None:
                preconditioned_norm_sq += float(g_b.pow(2).sum().item())
                natural_metric += float((g_b * g_b).sum().item())
            layer_updates.append((layer, g_pre, g_b))

        plain_updates = []
        for p in self.module.parameters():
            if id(p) in kfac_params or p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p
            preconditioned_norm_sq += float(grad.pow(2).sum().item())
            natural_metric += float((grad * grad).sum().item())
            plain_updates.append((p, grad))

        step_size = self.lr
        trust_metric = 0.0
        preconditioned_norm = preconditioned_norm_sq ** 0.5
        if self.trust_region is not None:
            if self.trust_region_metric == "natural":
                trust_metric = max(natural_metric, 0.0)
                if trust_metric > 0.0:
                    step_size = min(
                        self.lr,
                        (self.trust_region / (trust_metric + 1e-12)) ** 0.5,
                    )
            else:
                trust_metric = preconditioned_norm
                full_step_norm = self.lr * preconditioned_norm
                if full_step_norm > self.trust_region and preconditioned_norm > 0.0:
                    step_size = self.trust_region / (preconditioned_norm + 1e-12)

        for layer, g_pre, g_b in layer_updates:
            layer.weight.add_(g_pre, alpha=-step_size)
            if g_b is not None:
                layer.bias.add_(g_b, alpha=-step_size)

        # Plain SGD on params not inside a Linear (RMSNorm weights, etc.)
        for p, grad in plain_updates:
            p.add_(grad, alpha=-step_size)

        update_norm = step_size * (preconditioned_norm_sq ** 0.5)
        self.last_step_stats = {
            "step_size": float(step_size),
            "preconditioned_norm": float(preconditioned_norm_sq ** 0.5),
            "update_norm": float(update_norm),
            "trust_metric": float(trust_metric),
            "natural_metric": float(natural_metric),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        return {
            "step_count": self._step_count,
            "lr": self.lr,
            "damping": self.damping,
            "ema_decay": self.ema_decay,
            "trust_region": self.trust_region,
            "trust_region_decay": self.trust_region_decay,
            "min_trust_region": self.min_trust_region,
            "trust_region_metric": self.trust_region_metric,
            "n_eigen_decomp": self.n_eigen_decomp,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self._step_count = int(state.get("step_count", 0))
        self.lr = float(state.get("lr", self.lr))
        self.damping = float(state.get("damping", self.damping))
        self.ema_decay = float(state.get("ema_decay", self.ema_decay))
        trust_region = state.get("trust_region", self.trust_region)
        self.trust_region = (
            None if trust_region is None else float(trust_region)
        )
        trust_region_decay = state.get(
            "trust_region_decay", self.trust_region_decay
        )
        self.trust_region_decay = (
            None if trust_region_decay is None else float(trust_region_decay)
        )
        min_trust_region = state.get("min_trust_region", self.min_trust_region)
        self.min_trust_region = (
            None if min_trust_region is None else float(min_trust_region)
        )
        self.trust_region_metric = str(
            state.get("trust_region_metric", self.trust_region_metric)
        ).lower()
        self.n_eigen_decomp = int(state.get("n_eigen_decomp", self.n_eigen_decomp))
        self._validate_hyperparameters()

    @property
    def param_groups(self) -> list:
        return [{"lr": self.lr, "params": list(self.module.parameters())}]
