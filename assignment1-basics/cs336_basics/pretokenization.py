import regex as re
import os
from typing import BinaryIO
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_tuple_bytes(word):
    """
    convert a word into a tuple of its individual UTF-8 bytes
    e.g. "low" -> (b'l', b'o', b'w')
    @Input: word: str
    @Return: tuple[bytes, ...]
    """
    word_bytes = word.encode("utf-8")
    return tuple(bytes([b]) for b in word_bytes)


def get_freq(segments):
    """
    build a frequency table
    @Input: segments: list[str]
    @Return: freq: dict[tuple[bytes, ...], int]
    """
    freq = {}
    for segment in segments:
        for match in re.finditer(PAT, segment):
            tuple_bytes = get_tuple_bytes(match.group())
            # inquiry the key, if nothing, default to 0
            freq[tuple_bytes] = freq.get(tuple_bytes, 0) + 1
    return freq


def process_chunk(args):
    filepath, start, end, special_tokens = args
    assert special_tokens, "special_tokens must be non-empty"
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    special_token_pat = "|".join(re.escape(special_token) for special_token in special_tokens)
    # remove the special tokens before pretokenization
    text = re.split(special_token_pat, chunk)
    freq = get_freq(text)
    return freq


def parallel_pretokenize(filepath, special_tokens, num_processes=4):
    """
    use num_processes to do the parallel pretokenization
    @Input: filepath: str
            special_tokens: list[str]
            num_processes: int
    @Return: word_freq: dict[tuple[bytes, ...], int]
    """
    # many ~32 MB chunks instead of one giant chunk per process: a decoded chunk can take
    # 4 bytes/char in RAM, so 4 x 550 MB chunks of TinyStories would need ~12 GB
    num_chunks = max(num_processes, os.path.getsize(filepath) // (32 * 1024 * 1024))
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    chunk_args = [(filepath, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    # combine chunks as they finish, so only a few freq dicts are alive at a time
    word_freq = {}
    with mp.Pool(processes=num_processes) as pool:
        for chunk in pool.imap_unordered(process_chunk, chunk_args):
            for k, v in chunk.items():
                word_freq[k] = word_freq.get(k, 0) + v
    return word_freq
