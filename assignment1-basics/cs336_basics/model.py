"""Transformer Language Model implementation following PyTorch conventions.

PyTorch Convention:
  - Input activations are row tensors: X has shape (..., in_features) or (batch_size, seq_len, d_model).
  - Linear layers store weights as (out_features, in_features) via nn.Linear(in_features, out_features).
  - Forward pass computes X @ weight.T, performing:
      (..., in_features) @ (in_features, out_features) -> (..., out_features).
"""

import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


class Linear(nn.Module):
    """Linear transformation layer: y = x @ weight.T

    Dimensions & Parameter Shapes:
      - x:      (..., in_features)
      - weight: (out_features, in_features)
      - y:      (..., out_features)
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # Truncated normal initialization: N(0, sigma^2) truncated to [-3*sigma, 3*sigma]
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation: y = x @ weight.T

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        # (..., in_features) @ (in_features, out_features) -> (..., out_features)
        y = x @ self.weight.T
        return y


class Embedding(nn.Module):
    """Embedding lookup table mapping token IDs to embedding vectors.

    Dimensions & Parameter Shapes:
      - token_ids: (...) integer tensor of token indices
      - weight:    (num_embeddings, embedding_dim)
      - output:    (..., embedding_dim)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Weight shape: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # Truncated normal initialization: N(0, 1.0) truncated to [-3.0, 3.0]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for token indices: output = weight[token_ids]

        Args:
            token_ids: Integer tensor of shape (...), typically (batch_size, seq_len).

        Returns:
            torch.Tensor: Output embeddings of shape (..., embedding_dim).
        """
        # (...) -> (..., embedding_dim)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Normalizes activations across the last dimension without mean centering:
      y = (x / RMS(x)) * weight
      where RMS(x) = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)

    Dimensions & Parameter Shapes:
      - x:      (..., d_model)
      - weight: (d_model,)
      - y:      (..., d_model)
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Weight shape: (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input activations across the feature dimension.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor of shape (..., d_model), with original dtype.
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        # mean_square: (..., 1)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # rms: (..., 1)
        rms = torch.rsqrt(mean_square + self.eps)
        # (..., d_model) * (..., 1) * (d_model,) -> (..., d_model)
        result = x * rms * self.weight
        return result.to(in_type)


def SiLU(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU) activation: SiLU(x) = x * sigmoid(x).

    Args:
        x: Input tensor of arbitrary shape (...).

    Returns:
        torch.Tensor: Output tensor with the same shape as x.
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network.

    Computes:
      FFN(x) = (SiLU(x @ w1.weight.T) * (x @ w3.weight.T)) @ w2.weight.T

    Dimensions & Parameter Shapes:
      - x:                  (..., d_model)
      - w1.weight (gate):   (d_ff, d_model)
      - w3.weight (up):     (d_ff, d_model)
      - w2.weight (down):   (d_model, d_ff)
      - output:             (..., d_model)

    Intermediate Runtime Activations:
      - x @ w1.weight.T:    (..., d_model) @ (d_model, d_ff) -> (..., d_ff)
      - x @ w3.weight.T:    (..., d_model) @ (d_model, d_ff) -> (..., d_ff)
      - hidden @ w2.weight.T: (..., d_ff) @ (d_ff, d_model) -> (..., d_model)
    """

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int(round(8 * d_model / 3 / 64)) * 64
        self.d_ff = d_ff
        # Projections: weights stored as (out_features, in_features)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)  # gate
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)  # down
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwiGLU: w2(SiLU(w1(x)) * w3(x)).

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        # x @ w1.weight.T -> (..., d_ff)
        gate = SiLU(self.w1(x))
        # x @ w3.weight.T -> (..., d_ff)
        up = self.w3(x)
        # (gate * up) @ w2.weight.T: (..., d_ff) @ (d_ff, d_model) -> (..., d_model)
        return self.w2(gate * up)


def rotate(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive 2D pairs of the feature dimension for RoPE.

    Swaps and negates adjacent coordinate pairs along the last dimension:
      (x_0, x_1), (x_2, x_3), ... -> (-x_1, x_0), (-x_3, x_2), ...

    Args:
        x: Input tensor of shape (..., d_k), where d_k must be even.

    Returns:
        torch.Tensor: Rotated tensor of shape (..., d_k).
    """
    x_rotated = torch.empty_like(x)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    x_rotated[..., 0::2] = -x1
    x_rotated[..., 1::2] = x0
    return x_rotated


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Precomputes sinusoidal frequency buffers for 2D rotational position encoding:
      RoPE(x) = x * cos(angles) + rotate(x) * sin(angles)

    Dimensions & Buffer Shapes:
      - positions:   (max_seq_len,)
      - freqs:       (d_k // 2,) = 1.0 / (theta^(2*i / d_k))
      - angles:      (max_seq_len, d_k // 2) = positions outer freqs
      - cos_cache:   (max_seq_len, d_k)
      - sin_cache:   (max_seq_len, d_k)
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, f"d_k must be an even number to form 2D rotation pairs, got {d_k}"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # positions: (max_seq_len,), dim_indices: (d_k // 2,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        # freqs: (d_k // 2,) -> 1.0 / (theta^(2*i / d_k))
        freqs = 1.0 / (theta ** (dim_indices / d_k))
        # angles: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, freqs)
        # Repeat each angle twice to match 2D rotation pairs along d_k: (max_seq_len, d_k)
        cos_cache = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
        sin_cache = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to queries or keys.

        Args:
            x: Input tensor of shape (..., seq_len, d_k).
            token_positions: Position indices of shape (..., seq_len).

        Returns:
            torch.Tensor: Rotary-embedded tensor of shape (..., seq_len, d_k).
        """
        # Lookup cached frequencies: (..., seq_len, d_k)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        # Insert broadcast dimensions for leading axes (e.g. num_heads)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        # Elementwise rotation: (..., seq_len, d_k)
        result = x * cos + rotate(x) * sin
        return result


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax along a specified dimension.

    Computes:
      softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

    Args:
        x: Input tensor of arbitrary shape (...).
        dim: Dimension along which softmax will be computed.

    Returns:
        torch.Tensor: Output tensor with probabilities summing to 1 along dim.
    """
    # Subtract maximum along dim for numerical stability
    max_value = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_value
    # Exponentiate and normalize along dim
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    M: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V.

    Dimensions & Tensor Shapes:
      - Q: (..., seq_q, d_k)
      - K: (..., seq_k, d_k)
      - V: (..., seq_k, d_v)
      - M: Optional boolean mask broadcastable to (..., seq_q, seq_k),
           where False indicates masked positions.
      - Output: (..., seq_q, d_v)

    Intermediate Runtime Activations:
      - Q @ K^T / sqrt(d_k): (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k)
      - weights @ V:         (..., seq_q, seq_k) @ (..., seq_k, d_v) -> (..., seq_q, d_v)

    Args:
        Q: Query tensor of shape (..., seq_q, d_k).
        K: Key tensor of shape (..., seq_k, d_k).
        V: Value tensor of shape (..., seq_k, d_v).
        M: Optional mask of shape (..., seq_q, seq_k).

    Returns:
        torch.Tensor: Attention output of shape (..., seq_q, d_v).
    """
    d_k = Q.size(-1)
    # Q @ K^T / sqrt(d_k): (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k)
    pre_softmax = einsum(Q, K, " ... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if M is not None:
        pre_softmax.masked_fill_(~M, float("-inf"))
    # weights @ V: (..., seq_q, seq_k) @ (..., seq_k, d_v) -> (..., seq_q, d_v)
    attention = softmax(pre_softmax, dim=-1) @ V
    return attention


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with causal masking and optional RoPE.

    Dimensions & Parameter Shapes:
      - x:                                       (..., seq_len, d_model)
      - q_proj, k_proj, v_proj, output_proj:     (d_model, d_model)
      - output:                                  (..., seq_len, d_model)

    Intermediate Runtime Activations:
      - Q, K, V = x @ W_{q,k,v}.T:               (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
      - Q_h @ K_h.T / sqrt(d_k):                 (..., num_heads, seq_len, d_k) @ (..., num_heads, d_k, seq_len)
                                                 -> (..., num_heads, seq_len, seq_len)
      - Attn_h @ V_h:                            (..., num_heads, seq_len, seq_len) @ (..., num_heads, seq_len, d_k)
                                                 -> (..., num_heads, seq_len, d_k)
      - attn_concat:                             (..., seq_len, d_model)
      - attn_concat @ W_o.T:                     (..., seq_len, d_model) @ (d_model, d_model) -> (..., seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divided by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_k = d_model // num_heads
        # Projections: weights stored as (d_model, d_model)
        self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        if self.theta is not None and self.max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=self.theta,
                d_k=self.d_k,
                max_seq_len=self.max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Compute causal multi-head self-attention.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).
            token_positions: Optional position indices of shape (..., seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        seq = x.size(-2)
        # Project and reshape: (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
        q = rearrange(self.q_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        # Apply RoPE to queries and keys
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        # Causal mask: lower-triangular (seq, seq)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        # Scaled dot-product attention: (..., num_heads, seq, d_k)
        attention = scaled_dot_product_attention(q, k, v, M=mask)
        # Concatenate heads: (..., num_heads, seq, d_k) -> (..., seq, d_model)
        attention_concat = rearrange(attention, "... h seq d -> ... seq (h d)")
        # Output projection: (..., seq, d_model) @ (d_model, d_model) -> (..., seq, d_model)
        return self.output_proj(attention_concat)


class TransformerBlock(nn.Module):
    """Transformer block with Pre-LayerNorm architecture.

    Sublayers:
      1. Pre-norm Self-Attention: y = x + MultiHeadSelfAttention(ln1(x))
      2. Pre-norm SwiGLU FFN:     output = y + SwiGLU(ln2(y))

    Dimensions & Parameter Shapes:
      - x:                    (..., seq_len, d_model)
      - ln1, ln2:             weights (d_model,)
      - attn (MHSA):          q, k, v, output proj weights (d_model, d_model)
      - ffn (SwiGLU):         w1, w3 weights (d_ff, d_model), w2 weight (d_model, d_ff)
      - output:               (..., seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for TransformerBlock.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).
            token_positions: Optional position indices of shape (..., seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        # Sublayer 1: Pre-norm causal self-attention with residual
        y = x + self.attn(self.ln1(x), token_positions=token_positions)
        # Sublayer 2: Pre-norm SwiGLU feed-forward with residual
        result = y + self.ffn(self.ln2(y))
        return result


class TransformerLM(nn.Module):
    """Transformer Autoregressive Language Model.

    Pipeline:
      token_ids -> token_embeddings -> num_layers x TransformerBlock -> ln_final -> lm_head -> logits

    Dimensions & Parameter Shapes:
      - token_ids:                  (batch_size, seq_len)
      - token_embeddings.weight:    (vocab_size, d_model)
      - ln_final.weight:            (d_model,)
      - lm_head.weight:             (vocab_size, d_model)
      - logits:                     (batch_size, seq_len, vocab_size)

    Intermediate Runtime Activations:
      - token_embeddings(token_ids): (batch_size, seq_len, d_model)
      - block(x):                    (batch_size, seq_len, d_model)
      - ln_final(x):                 (batch_size, seq_len, d_model)
      - x @ lm_head.weight.T:        (batch_size, seq_len, d_model) @ (d_model, vocab_size)
                                     -> (batch_size, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        # Token embeddings: weight shape (vocab_size, d_model)
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        # Final layer norm: weight shape (d_model,)
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # Unembedding language model head: weight shape (vocab_size, d_model)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for TransformerLM.

        Args:
            x: Integer tensor of token IDs with shape (batch_size, seq_len).
            token_positions: Optional token position indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embeddings(x)
        # Pass through each Transformer block: (batch_size, seq_len, d_model)
        for block in self.layers:
            x = block(x, token_positions=token_positions)
        # Final layer normalization: (batch_size, seq_len, d_model)
        x = self.ln_final(x)
        # LM head projection: (batch_size, seq_len, d_model) @ (d_model, vocab_size) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        return logits
