"""Generate text from a trained TinyStories checkpoint (handout section 6, Decoding).

Example:
    python -m cs336_basics.generate \
        --checkpoint checkpoints/ts_run1/final.pt \
        --prompt "Once upon a time" \
        --max_new_tokens 256 --temperature 0.8 --top_p 0.9
"""

import argparse

import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenization import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"


# --------------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()

    # files
    p.add_argument("--checkpoint", default="checkpoints/ts_run1/final.pt")
    p.add_argument("--vocab", default="data/ts_vocab.pkl")
    p.add_argument("--merges", default="data/ts_merges.pkl")

    # model -- must be the SAME sizes used in training, or load_state_dict fails
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # decoding
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy (always pick the top token)")
    p.add_argument("--top_p", type=float, default=0.9, help="1.0 = no nucleus filtering")
    p.add_argument("--num_samples", type=int, default=1)

    # system
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    p.add_argument("--device", default=default_device)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# 2. Load the trained model
# --------------------------------------------------------------------------- #
def load_model(args):
    """Rebuild the TransformerLM with the training sizes, then load the checkpoint's weights."""
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# 3. Sampling one token
# --------------------------------------------------------------------------- #
def top_p_filter(probs, p):
    """Nucleus filtering (Eq. 24): keep the smallest set of top tokens whose total prob >= p.

    probs: (vocab_size,) probabilities that sum to 1
    returns: (vocab_size,) same shape, zeros outside the kept set, renormalized to sum to 1

    Example: probs = [0.15, 0.5, 0.05, 0.3], p = 0.7
        sorted high -> low: 0.5, 0.3, 0.15, 0.05
        0.5 alone < 0.7, 0.5 + 0.3 = 0.8 >= 0.7  -> keep {0.5, 0.3}
        result: [0, 0.625, 0, 0.375]   (0.5 / 0.8, 0.3 / 0.8, in their ORIGINAL positions)
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    idx = (cum_probs < p).sum() + 1
    filter_probs = torch.zeros_like(probs)
    filter_probs[sorted_indices[:idx]] = sorted_probs[:idx]
    return filter_probs / filter_probs.sum(dim=-1)


def sample_next_token(logits, temperature, top_p):
    """One decoding step: next-token logits (vocab_size,) -> one token id (int)."""
    if temperature == 0:
        return torch.argmax(logits).item()
    probs  = softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
    return int(torch.multinomial(probs, num_samples=1))


# --------------------------------------------------------------------------- #
# 4. The decoding loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature, top_p, eot_id, context_length, device):
    """Keep sampling the next token until <|endoftext|> or max_new_tokens.

    prompt_ids: list[int] from tokenizer.encode(prompt)
    returns: list[int] of the NEW tokens only (<|endoftext|> is not included)
    """
    ids = list(prompt_ids)  # the prompt, never changes
    new_ids = []            # tokens generated so far
    for _ in range(max_new_tokens):
        # full text so far, cropped to the last context_length tokens (RoPE only has that many positions)
        inputs = (ids + new_ids)[-context_length:]
        # wrap in [...] to add the batch dim, like training's (batch_size, seq_len): (1, seq_len)
        inputs = torch.tensor([inputs], dtype=torch.long, device=device)
        # (1, seq_len, vocab_size) -> story 0, last position = scores for the next token: (vocab_size,)
        logits = model(inputs)[0, -1]
        next_id = sample_next_token(logits, temperature, top_p)
        # the model says the story is finished: stop, and leave <|endoftext|> out of the output
        if next_id == eot_id:
            break
        new_ids.append(next_id)
    return new_ids


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[SPECIAL_TOKEN])
    eot_id = tokenizer.token_to_id[SPECIAL_TOKEN.encode("utf-8")]
    model = load_model(args)
    prompt_ids = tokenizer.encode(args.prompt)

    for i in range(args.num_samples):
        new_ids = generate(
            model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p,
            eot_id, args.context_length, args.device,
        )
        print(f"=== sample {i + 1}: {len(new_ids)} new tokens "
              f"(temperature {args.temperature}, top_p {args.top_p}) ===")
        print(args.prompt + tokenizer.decode(new_ids))


if __name__ == "__main__":
    main()
