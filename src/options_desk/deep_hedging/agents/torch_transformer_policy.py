"""
Bounded-window causal Transformer policy for deep hedging.

Architecture (one block at each step):

    obs_t ──► Linear ──► append to history buffer
                              │
                              ▼
              past obs embeddings (B, T_history, H) + positional encoding
                              │
                              ▼
              N × TransformerEncoderLayer (causal mask, batch_first=True)
                              │
                              ▼
              take last position output ──► Linear ──► symexp ──► * mask

State carried across rollout steps = the rolling window of past obs
embeddings (capped at ``max_history``). When the history reaches the cap
the window slides forward one step, dropping the oldest embedding. This
gives the network bounded memory (vs. LSTM's compressed unbounded
memory) with explicit attention weights that an interpretability tool
could later inspect.

Same input/output contract as :class:`HedgingLSTMPolicy` so it drops
into the existing :class:`BuehlerTrainer` with no rollout-loop changes.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import List

from .torch_lstm_policy import symexp
from .torch_policy import _validate_positive

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional dep
    _TORCH_AVAILABLE = False


# Type alias: state is the rolling history buffer (B, T_history, H).
TransformerState = "torch.Tensor"


class HedgingTransformerPolicy(nn.Module if _TORCH_AVAILABLE else object):
    """Bounded-window causal Transformer policy for deep hedging.

    Maps a flat per-step observation (which already concatenates the
    previous action via ``_build_batch_obs_tensor`` upstream) plus a
    rolling buffer of past observation embeddings to bounded trade
    actions::

        obs (B, obs_dim)  +  history_state (B, T_h, H)
            ──► trades (B, n_instruments), new_history_state

    Args:
        obs_dim: dimension of the input observation vector.
        n_instruments: dimension of the action vector.
        hidden_size: embedding dim shared by all transformer layers.
            Keep small (16-64) for tractability with 252-step BPTT.
        n_layers: number of stacked TransformerEncoderLayers.
        n_heads: number of attention heads (must divide hidden_size).
        max_history: bound on the attention window (number of past
            steps the model can attend over). Linear memory in this.
        ffn_mult: feedforward dim multiplier inside each transformer
            layer (transformer default = 4; 2-4 is fine for our scale).
        position_limit: soft clamp on output magnitude via tanh, applied
            AFTER symexp. Set to None to disable.
        last_layer_scale: down-scaling factor for the output linear
            layer's weights at initialization (small-init for stable
            early training, same as the LSTM policy default).
    """

    def __init__(
        self,
        obs_dim: int,
        n_instruments: int,
        hidden_size: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_history: int = 64,
        ffn_mult: int = 4,
        position_limit: float | None = 10.0,
        last_layer_scale: float = 1e-3,
        price_scale: float = 100.0,
        position_scale: float = 100.0,
        price_clip: float = 10.0,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("HedgingTransformerPolicy requires PyTorch")
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )
        self.obs_dim = obs_dim
        self.n_instruments = n_instruments
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_history = int(max_history)
        self.position_limit = position_limit
        self.last_layer_scale = last_layer_scale
        self.price_scale = _validate_positive(price_scale, "price_scale")
        self.position_scale = _validate_positive(position_scale, "position_scale")
        self.price_clip = _validate_positive(price_clip, "price_clip")

        self.input_proj = nn.Linear(obs_dim, hidden_size)
        # Learnable positional embedding indexed by 0..max_history-1
        self.pos_emb = nn.Embedding(self.max_history, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size * ffn_mult,
                batch_first=True,
                dropout=0.0,
                norm_first=True,  # pre-norm — more stable for small models
            )
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(hidden_size, n_instruments)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        with torch.no_grad():
            self.output.weight.mul_(self.last_layer_scale)
            self.output.bias.zero_()

    def init_hidden_state(
        self,
        batch_size: int,
        device: "torch.device | str" = "cpu",
        dtype: "torch.dtype | None" = None,
    ) -> TransformerState:
        """Construct an empty (B, 0, H) history buffer."""
        if dtype is None:
            dtype = torch.float32
        return torch.empty(batch_size, 0, self.hidden_size, device=device, dtype=dtype)

    def step(
        self,
        obs: "torch.Tensor",
        action_mask: "torch.Tensor",
        hidden_state: TransformerState,
    ) -> tuple["torch.Tensor", TransformerState]:
        """Run one bounded-window attention step.

        Args:
            obs: (B, obs_dim) flat observation.
            action_mask: (B, N) tradability mask.
            hidden_state: (B, T_h, H) rolling buffer of past obs embeddings
                with T_h in [0, max_history-1]. Returned from previous step.

        Returns:
            trades: (B, N) action tensor (post-symexp, post-mask).
            new_hidden_state: (B, T_h_new, H) updated rolling buffer.
        """
        # Embed current observation, shape (B, 1, H)
        x_cur = self.input_proj(obs).unsqueeze(1)
        history = torch.cat([hidden_state, x_cur], dim=1)  # (B, T_h+1, H)
        if history.size(1) > self.max_history:
            history = history[:, -self.max_history:, :]
        seq_len = history.size(1)

        # Positional embeddings: position 0 = oldest in window, seq_len-1 = current
        positions = torch.arange(seq_len, device=obs.device)
        x = history + self.pos_emb(positions).unsqueeze(0)  # (B, T, H)

        # Causal mask: position i can attend to positions [0, i].
        # nn.TransformerEncoderLayer expects True = MASKED.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=obs.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)

        # Decode action from the last-position output
        out_last = x[:, -1, :]
        raw = self.output(out_last)
        trades = symexp(raw)
        if self.position_limit is not None:
            trades = torch.tanh(trades / self.position_limit) * self.position_limit
        return trades * action_mask, history

    def forward(
        self,
        obs: "torch.Tensor",
        action_mask: "torch.Tensor",
        hidden_state: TransformerState | None = None,
    ) -> tuple["torch.Tensor", TransformerState]:
        """Compatibility wrapper: auto-init history if not supplied."""
        if hidden_state is None:
            hidden_state = self.init_hidden_state(
                obs.shape[0], device=obs.device, dtype=obs.dtype
            )
        return self.step(obs, action_mask, hidden_state)
