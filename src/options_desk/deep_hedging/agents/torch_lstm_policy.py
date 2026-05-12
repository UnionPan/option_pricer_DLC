"""
Recurrent LSTM hedging policy with action recurrence and symexp output.

Architecture mirrors the deep-hedging-with-2nd-order paper
(notes/fasthedgingwith2nd):

    obs_t (which already concatenates u_{t-1}) ──► Linear ──┐
                                                            ▼
                            Block_1 ──► Block_2 ──► ... ──► Block_L
                                                            │
                                                            ▼
                                          Linear ──► symexp ──► * mask ──► u_t

Each "block" applies RMSNorm followed by an LSTMCell with a residual skip:

    h_in = RMSNorm(x)
    (h_out, c_out) = LSTMCell(h_in, (h_prev, c_prev))
    x = x + h_out

The previous action ``u_{t-1}`` is fed in via the obs tensor (already done by
``_build_batch_obs_tensor`` in the trainer), and the LSTM hidden states
``{h^l, c^l}_l`` are carried across rollout steps as the network's recurrent
memory.

The last linear layer is initialized with zero bias and He-normal weights
scaled by 1e-3 (Andrychowicz-style small-init), so the network outputs are
near-zero at the start of training and grow as needed.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import List, Tuple

from .torch_policy import _validate_positive

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional dep
    _TORCH_AVAILABLE = False


def symexp(x: "torch.Tensor", max_abs: float = 20.0) -> "torch.Tensor":
    """Symmetric exponential activation: sign(x) * (exp(|x|) - 1).

    Approximately identity near 0 and exponential away from 0, giving the
    network access to large outputs without saturating. Reference:
    Hafner et al. 2023, "Mastering Diverse Domains through World Models".
    ``max_abs`` keeps extreme pre-activations finite before the downstream
    position clamp.
    """
    max_abs = _validate_positive(max_abs, "max_abs")
    return torch.sign(x) * torch.expm1(torch.clamp(torch.abs(x), max=max_abs))


class RMSNorm(nn.Module if _TORCH_AVAILABLE else object):
    """Root-mean-square layer normalization.

    Equivalent to ``nn.RMSNorm`` (PyTorch >= 2.4) but written out so this
    module works on older torch versions too.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("RMSNorm requires PyTorch")
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x * self.weight / rms


class ManualLSTMCell(nn.Module if _TORCH_AVAILABLE else object):
    """Hand-written LSTM cell built from two ``nn.Linear`` layers.

    Numerically equivalent to ``nn.LSTMCell`` (within FP precision) but
    exposes the input-to-hidden and hidden-to-hidden weight matrices as
    standalone ``nn.Linear`` layers (``W_ih``, ``W_hh``). This is required
    for KFAC: the optimizer needs forward/backward hooks on the gate
    pre-activations, which the fused ``nn.LSTMCell`` does not expose.

    Forward signature ``(x, (h, c)) -> (h_new, c_new)`` matches
    ``nn.LSTMCell`` so the rest of ``HedgingLSTMPolicy`` is agnostic to
    which cell type is used.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("ManualLSTMCell requires PyTorch")
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Single bias on W_ih (matches sum of bias_ih + bias_hh in nn.LSTMCell).
        self.W_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        # Match nn.LSTMCell's default uniform init: U(-1/√H, 1/√H)
        bound = 1.0 / (hidden_size ** 0.5)
        for w in (self.W_ih.weight, self.W_ih.bias, self.W_hh.weight):
            nn.init.uniform_(w, -bound, bound)

    @property
    def bias_ih(self) -> "torch.Tensor":
        # Compatibility shim for code that pokes the forget-gate bias the
        # way it would on nn.LSTMCell (e.g. _init_weights in HedgingLSTMPolicy).
        return self.W_ih.bias

    @property
    def bias_hh(self) -> "torch.Tensor":
        # Hidden-bias is absorbed into bias_ih, so this is always-zero.
        # Returning a synthetic zero tensor keeps the LSTMCell-style
        # `cell.bias_hh[forget_slice].fill_(0.0)` no-op safe.
        return torch.zeros_like(self.W_ih.bias)

    def forward(
        self,
        x: "torch.Tensor",
        hc: tuple["torch.Tensor", "torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        h_prev, c_prev = hc
        gates = self.W_ih(x) + self.W_hh(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# Type alias: per-layer (h, c) tuples carried across rollout steps.
LSTMState = List[Tuple["torch.Tensor", "torch.Tensor"]]


class HedgingLSTMPolicy(nn.Module if _TORCH_AVAILABLE else object):
    """
    Recurrent LSTM policy for deep hedging.

    Maps a flat observation (which already contains the previous action via
    ``_build_batch_obs_tensor``) and the carried LSTM hidden states to bounded
    trade actions::

        obs (B, obs_dim)  +  hidden_state (L blocks of (h, c))
            ──► trades (B, n_instruments), new_hidden_state

    Args:
        obs_dim: dimension of the input observation vector
        n_instruments: dimension of the action vector
        hidden_size: width of each LSTM block (paper: 32)
        n_blocks: number of residual LSTM blocks (paper: 4)
        position_limit: soft clamp on output magnitude via tanh, applied AFTER
            symexp. Set to ``None`` to disable (paper variant) — beware: symexp
            can blow up if the network learns large pre-activations.
        last_layer_scale: down-scaling factor for the last linear layer's
            weights at initialization (paper: 1e-3).
    """

    def __init__(
        self,
        obs_dim: int,
        n_instruments: int,
        hidden_size: int = 32,
        n_blocks: int = 4,
        position_limit: float | None = 10.0,
        last_layer_scale: float = 1e-3,
        price_scale: float = 100.0,
        position_scale: float = 100.0,
        price_clip: float = 10.0,
        use_manual_cells: bool = False,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("HedgingLSTMPolicy requires PyTorch")
        super().__init__()
        self.obs_dim = obs_dim
        self.n_instruments = n_instruments
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.position_limit = position_limit
        self.last_layer_scale = last_layer_scale
        self.price_scale = _validate_positive(price_scale, "price_scale")
        self.position_scale = _validate_positive(position_scale, "position_scale")
        self.price_clip = _validate_positive(price_clip, "price_clip")
        self.use_manual_cells = bool(use_manual_cells)

        # Input projection: obs (which already concatenates u_{t-1} via
        # _build_batch_obs_tensor) → hidden_size
        self.input_proj = nn.Linear(obs_dim, hidden_size)

        # Residual blocks: RMSNorm + LSTMCell.
        # ManualLSTMCell exposes W_ih/W_hh as nn.Linear so KFAC can hook in.
        # nn.LSTMCell is faster (fused) but opaque to the optimizer.
        cell_cls = ManualLSTMCell if self.use_manual_cells else nn.LSTMCell
        self.norms = nn.ModuleList(
            [RMSNorm(hidden_size) for _ in range(n_blocks)]
        )
        self.lstm_cells = nn.ModuleList(
            [cell_cls(hidden_size, hidden_size) for _ in range(n_blocks)]
        )

        # Output projection: hidden_size → n_instruments (raw, pre-symexp)
        self.output = nn.Linear(hidden_size, n_instruments)

        self._init_weights()

    def _init_weights(self) -> None:
        # He-normal on input projection
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_proj.bias)

        # LSTM cells use PyTorch defaults (Xavier-uniform). Bump forget-gate
        # bias to 1.0 for a small stability gain (Jozefowicz et al. 2015).
        for cell in self.lstm_cells:
            n = cell.bias_ih.shape[0]
            forget_slice = slice(n // 4, n // 2)
            with torch.no_grad():
                cell.bias_ih[forget_slice].fill_(1.0)
                cell.bias_hh[forget_slice].fill_(0.0)

        # Last layer: zero bias, He-normal weights scaled by last_layer_scale
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        with torch.no_grad():
            self.output.weight.mul_(self.last_layer_scale)
            self.output.bias.zero_()

    def init_hidden_state(
        self,
        batch_size: int,
        device: "torch.device | str" = "cpu",
        dtype: "torch.dtype | None" = None,
    ) -> LSTMState:
        """Construct zero-initialized (h, c) tuples for each block."""
        if dtype is None:
            dtype = torch.float32
        return [
            (
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            )
            for _ in range(self.n_blocks)
        ]

    def step(
        self,
        obs: "torch.Tensor",
        action_mask: "torch.Tensor",
        hidden_state: LSTMState,
    ) -> tuple["torch.Tensor", LSTMState]:
        """Run one recurrent step.

        Args:
            obs: ``(B, obs_dim)`` flat observation tensor.
            action_mask: ``(B, N)`` 1/0 mask -- 1 for tradable instruments.
            hidden_state: list of ``n_blocks`` ``(h, c)`` tuples, each
                ``(B, hidden_size)``.

        Returns:
            trades: ``(B, N)`` action tensor (post-symexp, post-mask).
            new_hidden_state: updated list of ``(h, c)`` tuples.
        """
        x = self.input_proj(obs)
        new_state: LSTMState = []
        for norm, cell, (h_prev, c_prev) in zip(
            self.norms, self.lstm_cells, hidden_state
        ):
            normed = norm(x)
            h_out, c_out = cell(normed, (h_prev, c_prev))
            x = x + h_out  # residual
            new_state.append((h_out, c_out))

        raw = self.output(x)
        trades = symexp(raw)
        if self.position_limit is not None:
            # Soft clamp via tanh — keeps gradients flowing past the limit.
            trades = torch.tanh(trades / self.position_limit) * self.position_limit
        return trades * action_mask, new_state

    def forward(
        self,
        obs: "torch.Tensor",
        action_mask: "torch.Tensor",
        hidden_state: LSTMState | None = None,
    ) -> tuple["torch.Tensor", LSTMState]:
        """Compatibility wrapper: auto-init hidden state if not supplied.

        Always returns ``(trades, new_state)`` so callers can choose to
        propagate state or discard it.
        """
        if hidden_state is None:
            hidden_state = self.init_hidden_state(
                obs.shape[0], device=obs.device, dtype=obs.dtype
            )
        return self.step(obs, action_mask, hidden_state)
