import torch
import math
import numpy as np
import torch.nn as nn
from einops import einsum, rearrange
from collections.abc import Callable, Iterable
from typing import Optional


def cross_entropy(logits, x):
    """Average cross-entropy between logits (..., vocab_size) and targets (...) -> ().

    l(theta; D) = 1/(|D| * m) * sum_{x in D} sum_{i=1..m}
                      -log p_theta(x[i+1] | x[1:i])
    p_theta(x[i+1] | x[1:i]) = softmax(o_i)[x[i+1]] = exp(o_i[x[i+1]]) / sum_{a in vocab_size}exp(o_i[a])
    log p_theta(x[i+1] | x[1:i]) = z_i[x[i+1]] -  logsum(exp(z_i))

    `...` is any batch dims: (batch, vocab) or (batch, seq, vocab). Inputs are
    already aligned -- the next-token shift lives in the training loop, not here.
    """
    # Subtract the row max for stability (softmax is shift-invariant): (..., 1) broadcasts
    z = logits - torch.max(logits, dim=-1, keepdim=True).values  # (..., vocab)
    logsum = torch.log(torch.exp(z).sum(dim=-1, keepdim=True))  # (..., 1)
    log_probs = z - logsum  # (..., vocab)
    # gather the target's log-prob from each row: (..., 1) -> (...)
    target_log_probs = log_probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()  # ()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class ADAMW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # learning rate
            dr = group["weight_decay"]  # decay rate
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data  # Get the gradient of loss with respect to p
                t = state.get("t", 0)
                if t == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["t"] = t + 1
                t = state["t"]
                m = state["m"]
                v = state["v"]
                # appy weight decay
                p.data -= lr * dr * p.data
                # update first moment
                m = beta1 * m + (1 - beta1) * grad
                state["m"] = m
                # update second moment
                v = beta2 * v + (1 - beta2) * grad**2
                state["v"] = v
                # update alpha
                alpha = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha * m / (torch.sqrt(v) + eps)
        return loss


def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        alpha = (t / T_w) * alpha_max
    elif t > T_c:
        alpha = alpha_min
    else:
        factor = (t - T_w) * math.pi / (T_c - T_w)
        alpha = alpha_min + 0.5 * (1 + math.cos(factor)) * (alpha_max - alpha_min)
    return alpha


def gradient_clipping(params, M, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    l2 = torch.sqrt(sum(torch.sum(g**2) for g in grads))
    if l2 > M:
        scale = M / (l2 + eps)
        for g in grads:
            # inplace multiplication
            g.mul_(scale)
    return params


def data_loading(x, batch_size, context_length, device):
    """
    x: integer numpy array of token IDs
    sampled input sequences and corresponding next-token targets
    (batch_size, context_length)
    """
    high = len(x) - context_length
    tokens = torch.empty((batch_size, context_length), dtype=torch.long)
    next_tokens = torch.empty((batch_size, context_length), dtype=torch.long)
    batch = torch.randint(0, high, (batch_size,), dtype=torch.long)
    for i in range(batch_size):
        idx = batch[i]
        tokens[i] = torch.from_numpy(x[idx : idx + context_length].astype(np.int64))
        next_tokens[i] = torch.from_numpy(x[idx + 1 : idx + 1 + context_length].astype(np.int64))
    return (tokens.to(device), next_tokens.to(device))


def save_checkpoint(model, optimizer, iteration, out):
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.Pathlike | typing.BinaryIO | typing.IO[bytes]
    """
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    src: str | os.Pathlike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
