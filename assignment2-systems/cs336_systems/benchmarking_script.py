import argparse
import timeit
import torch
import numpy as np
from dataclasses import dataclass
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    device: torch.device = DEFAULT_DEVICE
    mixed_precision: bool = False
    memory_profiling: bool = False


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
    # type=torch.device turns the CLI string ("cuda", "cuda:0", "cpu") into a torch.device
    parser.add_argument("--device", type=torch.device, default=DEFAULT_DEVICE)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--memory_profiling", action="store_true")

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
    # .type is "cuda" for both "cuda" and "cuda:0", so the check works for any index
    if cfg.device.type == "cuda":
        torch.cuda.synchronize(cfg.device)
    elif cfg.device.type == "mps":
        torch.mps.synchronize()


def run_step(cfg, model, tokens, targets, optimizer=None):
    # autocast is a no-op when enabled=False, so the same code path covers both cases
    ctx = torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16, enabled=cfg.mixed_precision)

    if cfg.mode == "forward":
        with ctx:
            logits = model(tokens)
    elif cfg.mode == "forward_backward":
        model.zero_grad()
        with ctx:
            logits = model(tokens)
            loss = cross_entropy(logits, targets)
        loss.backward()
    elif cfg.mode == "train":
        optimizer.zero_grad()
        with ctx:
            logits = model(tokens)
            loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()


def run_benchmark(cfg: BenchmarkConfig) -> list[float]:
    """
    run the benchmark based on the dataclass cfg
    """
    model = initialize_model(cfg)
    print(f"Running on {next(model.parameters()).device}")
    if cfg.mode == "train":
        optimizer = AdamW(model.parameters(), lr=1e-3)
    else:
        optimizer = None
    tokens, targets = generate_random_data(cfg)
    # warmup steps
    for _ in range(cfg.warmup_steps):
        run_step(cfg, model, tokens, targets, optimizer)
    sync(cfg)
    # memory profiling
    if cfg.memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        run_step(cfg, model, tokens, targets, optimizer)
        sync(cfg)
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")
        tag = f"mode-{cfg.mode}_ctx-{cfg.context_length}_mp-{int(cfg.mixed_precision)}"
        torch.cuda.memory._dump_snapshot(f"memory_snapshot_{tag}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return []

    # start the timer
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
    if times:
        print(f"{cfg.mode} mode (mixed_precision={cfg.mixed_precision}) averaged in {cfg.measure_steps} steps: {np.mean(times):.6f} ± {np.std(times):.6f} s")
