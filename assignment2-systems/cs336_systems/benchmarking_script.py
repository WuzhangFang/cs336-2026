import argparse
import timeit
import torch
import numpy as np
from dataclasses import dataclass
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy 
from cs336_basics.optimizer import AdamW


@dataclass
class BenchmarkConfig:
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 10_000
    context_length: int = 512
    batch_size: int = 4
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: str = "forward"  # "forward" | "forward_backward" | "train"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> BenchmarkConfig:
    """
    Store the parameters in a dataclass
    """
    parser = argparse.ArgumentParser(description="End-to-End Benchmarking of the basics Transformer")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return BenchmarkConfig(**vars(parser.parse_args()))


def initialize_model(cfg: BenchmarkConfig) -> BasicsTransformerLM:
    """
    initialize a model based on the dataclass cfg
    """
    model = BasicsTransformerLM(cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff)
    return model.to(cfg.device)


def generate_random_data(cfg: BenchmarkConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    generate a random batch of tokens and targets based on the dataclass cfg
    """
    shape = (cfg.batch_size, cfg.context_length)
    tokens = torch.randint(0, cfg.vocab_size, shape, device=cfg.device)
    targets = torch.randint(0, cfg.vocab_size, shape, device=cfg.device)
    return (tokens, targets)


def sync(cfg: BenchmarkConfig):
    if cfg.device == "cuda":
        torch.cuda.synchronize()
    elif cfg.device == "mps":
        torch.mps.synchronize()


def run_step(cfg, model, tokens, targets, optimizer=None):
    if cfg.mode == "forward":
        logits = model(tokens)
    elif cfg.mode == "forward_backward":
        model.zero_grad()
        logits = model(tokens)
        loss = cross_entropy(logits, targets)
        loss.backward()
    elif cfg.mode == "train":
        optimizer.zero_grad()
        logits = model(tokens)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()


def run_benchmark(cfg: BenchmarkConfig) -> list[float]:
    """
    run the benchmark based on the dataclass cfg
    """
    model = initialize_model(cfg)
    if cfg.mode == "train":
        optimizer = AdamW(model.parameters(), lr=1e-3)
    else:
        optimizer = None
    tokens, targets = generate_random_data(cfg)
    # warmup steps
    for _ in range(cfg.warmup_steps):
        run_step(cfg, model, tokens, targets, optimizer)
    sync(cfg)
    # Start the timer
    times = []
    for _ in range(cfg.measure_steps):
        start = timeit.default_timer()
        run_step(cfg, model, tokens, targets, optimizer)
        sync(cfg)
        elapsed = timeit.default_timer() - start
        times.append(elapsed)
    
    return times


if __name__ == "__main__":
    cfg = parse_args()
    times = run_benchmark(cfg)
    print(f"{cfg.mode} mode averaged in {cfg.measure_steps} steps: {np.mean(times):.6f} ± {np.std(times):.6f} s")
