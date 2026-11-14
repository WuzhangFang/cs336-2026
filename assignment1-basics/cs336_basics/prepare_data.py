"""Train a BPE tokenizer on a text corpus and encode train/valid into token ids.

Example (TinyStories, handout section 2.5 / 2.7):
    python -m cs336_basics.prepare_data \
        --train_text data/TinyStoriesV2-GPT4-train.txt \
        --valid_text data/TinyStoriesV2-GPT4-valid.txt \
        --vocab_size 10000 --out_prefix data/ts

Writes:
    data/ts_vocab.pkl, data/ts_merges.pkl   tokenizer, load with Tokenizer.from_files
    data/ts_valid.bin, data/ts_train.bin    raw uint16 token ids, read by training.py
"""

import argparse
import multiprocessing as mp
import os
import pickle
import resource
import sys
import time

import numpy as np
import regex as re

from cs336_basics.pretokenization import PAT, find_chunk_boundaries
from cs336_basics.tokenization import Tokenizer, train_bpe

SPECIAL_TOKEN = "<|endoftext|>"
ENCODE_CHUNK_BYTES = 4 * 1024 * 1024  # each encode job reads ~4 MB of text
PAT_RE = re.compile(PAT)


# --------------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_text", default="data/TinyStoriesV2-GPT4-train.txt")
    p.add_argument("--valid_text", default="data/TinyStoriesV2-GPT4-valid.txt")
    p.add_argument("--out_prefix", default="data/ts")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--num_processes", type=int, default=os.cpu_count())
    p.add_argument("--skip_bpe", action="store_true", help="reuse existing <out_prefix>_vocab/merges.pkl")
    return p.parse_args()


def peak_rss_gb(who):
    """Peak resident memory of this process (RUSAGE_SELF) or its finished workers (RUSAGE_CHILDREN)."""
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, KB on Linux
    return resource.getrusage(who).ru_maxrss * scale / 1e9


# --------------------------------------------------------------------------- #
# 2. Tokenizer: train BPE + save
# --------------------------------------------------------------------------- #
def train_and_save_tokenizer(text_path, vocab_size, vocab_path, merges_path):
    t0 = time.time()
    vocab, merges = train_bpe(text_path, vocab_size, [SPECIAL_TOKEN])
    elapsed = time.time() - t0

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    longest = max((t for t in vocab.values() if t != SPECIAL_TOKEN.encode("utf-8")), key=len)
    print(f"BPE: {len(vocab)} tokens, {len(merges)} merges in {elapsed / 60:.1f} min")
    print(f"BPE: peak memory {peak_rss_gb(resource.RUSAGE_SELF):.2f} GB main, "
          f"{peak_rss_gb(resource.RUSAGE_CHILDREN):.2f} GB largest worker")
    print(f"BPE: longest token {longest!r} ({len(longest)} bytes)")
    print(f"BPE: saved {vocab_path}, {merges_path}")


# --------------------------------------------------------------------------- #
# 3. Encoding (parallel, streamed to disk)
# --------------------------------------------------------------------------- #
_tokenizer = None  # per-worker globals, set once by _init_worker
_cache = None


def _init_worker(vocab_path, merges_path):
    global _tokenizer, _cache
    _tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[SPECIAL_TOKEN])
    _cache = {}


def encode_chunk(job):
    """Encode bytes [start, end) of a text file into a uint16 array of token ids.

    Gives the same ids as _tokenizer.encode(text), but memoizes each pretoken:
    the corpus reuses a small set of words, so nearly every pretoken is a cache hit
    and we skip the slow merge loop.
    """
    path, start, end = job
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    eot_id = _tokenizer.token_to_id[SPECIAL_TOKEN.encode("utf-8")]
    ids = []
    for i, doc in enumerate(text.split(SPECIAL_TOKEN)):
        if i > 0:
            ids.append(eot_id)  # keep the document separators in the data
        for pretoken in PAT_RE.findall(doc):
            pretoken_ids = _cache.get(pretoken)
            if pretoken_ids is None:
                pretoken_ids = _cache[pretoken] = _tokenizer.encode(pretoken)
            ids.extend(pretoken_ids)
    return np.array(ids, dtype=np.uint16)


def encode_file(text_path, out_path, vocab_path, merges_path, num_processes):
    """Encode `text_path` into raw uint16 ids at `out_path` (no header; read with np.memmap)."""
    t0 = time.time()
    size = os.path.getsize(text_path)
    with open(text_path, "rb") as f:
        # boundaries land on SPECIAL_TOKEN, so chunks encode independently
        boundaries = find_chunk_boundaries(f, max(1, size // ENCODE_CHUNK_BYTES), SPECIAL_TOKEN.encode("utf-8"))
    jobs = [(text_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    n_tokens = 0
    with (
        mp.Pool(num_processes, initializer=_init_worker, initargs=(vocab_path, merges_path)) as pool,
        open(out_path, "wb") as out,
    ):
        # imap (not imap_unordered) returns chunks in file order
        for k, ids in enumerate(pool.imap(encode_chunk, jobs), start=1):
            ids.tofile(out)
            n_tokens += len(ids)
            if k % 50 == 0 or k == len(jobs):
                print(f"  {out_path}: {k}/{len(jobs)} chunks, {n_tokens:,} tokens, {time.time() - t0:.0f}s")

    elapsed = time.time() - t0
    print(f"encoded {text_path} -> {out_path}: {n_tokens:,} tokens, "
          f"{size / n_tokens:.2f} bytes/token, {size / elapsed / 1e6:.1f} MB/s")
    return n_tokens


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    assert args.vocab_size <= 2**16, "token ids are stored as uint16"
    vocab_path = f"{args.out_prefix}_vocab.pkl"
    merges_path = f"{args.out_prefix}_merges.pkl"

    # --- tokenizer ---------------------------------------------------------
    if args.skip_bpe and os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"BPE: reusing {vocab_path}, {merges_path}")
    else:
        train_and_save_tokenizer(args.train_text, args.vocab_size, vocab_path, merges_path)

    # --- encode (valid first: it's small, so problems show up fast) --------
    for split, text_path in [("valid", args.valid_text), ("train", args.train_text)]:
        encode_file(text_path, f"{args.out_prefix}_{split}.bin", vocab_path, merges_path, args.num_processes)

    # --- sanity check: decode the start of valid ---------------------------
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[SPECIAL_TOKEN])
    head = np.memmap(f"{args.out_prefix}_valid.bin", dtype=np.uint16, mode="r")[:60]
    print(f"decoded start of valid: {tokenizer.decode(head.tolist())!r}")


if __name__ == "__main__":
    main()
