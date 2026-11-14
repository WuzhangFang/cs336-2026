import pickle
import regex as re
from typing import Iterable, Iterator
from cs336_basics.pretokenization import parallel_pretokenize, PAT


def get_pair_index(freq):
    """
    index the pair of bytes
    @Input: freq: dict[tuple[bytes, ...], int]
    @Return: pair_freq: dict[tuple[bytes, bytes], int]
             words: list[list[bytes]]
             counts: list[int]
             pair_index: dict[tuple[bytes, bytes], set[int]]
    """
    pair_freq = {}
    words = []
    counts = []
    pair_index = {}
    for i, (tuple_bytes, num) in enumerate(freq.items()):
        words.append(list(tuple_bytes))
        counts.append(num)
        for pair in zip(tuple_bytes, tuple_bytes[1:]):
            pair_freq[pair] = pair_freq.get(pair, 0) + num
            if pair not in pair_index:
                pair_index[pair] = set()
            pair_index[pair].add(i)
    return pair_freq, words, counts, pair_index


def merge_pair(word, max_pair):
    """
    merging (b'l', b'o') in [b'l', b'o', b'w'] -> [b'lo', b'w']
    @Input: word: list[bytes]
            max_pair: tuple[bytes, bytes]
    @Return: new_word: list[bytes]
    """
    new_word = []
    i = 0
    while i < len(word):
        if (i + 1) < len(word) and (word[i], word[i + 1]) == max_pair:
            new_word.append(word[i] + word[i + 1])
            i = i + 2
        else:
            new_word.append(word[i])
            i = i + 1
    return new_word


def merge(max_pair, pair_freq, words, counts, pair_index):
    """
    merge the max_pair and update pair_freq, words, counts, pair_index
    @Input: max_pair: tuple[bytes, bytes]
            pair_freq: dict[tuple[bytes, bytes], int]
            words: list[list[bytes]]
            counts: list[int]
            pair_index: dict[tuple[bytes, bytes], set[int]]
    @Return: new_token: bytes
    """
    new_token = max_pair[0] + max_pair[1]
    for index in list(pair_index[max_pair]):
        word = words[index]
        count = counts[index]
        new_word = merge_pair(word, max_pair)
        # update words
        words[index] = new_word
        # update the pair_freq and pair_index
        old_pairs = {}
        for pair in zip(word, word[1:]):
            old_pairs[pair] = old_pairs.get(pair, 0) + 1
        new_pairs = {}
        for pair in zip(new_word, new_word[1:]):
            new_pairs[pair] = new_pairs.get(pair, 0) + 1
        # remove old pairs from old word
        for pair, num in old_pairs.items():
            pair_freq[pair] -= num * count
            if pair_freq[pair] <= 0:
                del pair_freq[pair]
            pair_index[pair].discard(index)
            if not pair_index[pair]:
                del pair_index[pair]
        # add new pairs from new word
        for pair, num in new_pairs.items():
            pair_freq[pair] = pair_freq.get(pair, 0) + num * count
            if pair not in pair_index:
                pair_index[pair] = set()
            pair_index[pair].add(index)
    return new_token


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a byte-level BPE tokenizer.

    Returns:
        vocab: mapping from token id to bytes
        merges: ordered list of merge rules as byte-pairs
    """
    # 1. initial vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    # 2. parallel pretokenization
    word_freq = parallel_pretokenize(input_path, special_tokens, num_processes=4)
    # 3. index
    pair_freq, words, counts, pair_index = get_pair_index(word_freq)
    # 4. merge
    merges = []
    while len(vocab) < vocab_size and pair_freq:
        max_pair = max(pair_freq, key=lambda p: (pair_freq[p], p))
        merges.append(max_pair)
        new_token = merge(max_pair, pair_freq, words, counts, pair_index)
        vocab[len(vocab)] = new_token

    return vocab, merges


class Tokenizer:
    """
    give a vocabulary and a list of merges, encodes text into integer IDs,
    decodes integer IDs into text
    """

    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None=None
        """
        self.vocab = vocab.copy()
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merges = merges.copy()
        self.merges_priority = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens or []
        # add special_tokens into the vocab
        for stoken in self.special_tokens:
            encoded_stoken = stoken.encode("utf-8")
            if encoded_stoken not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = encoded_stoken
                self.token_to_id[encoded_stoken] = new_id
        # special_pattern
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(re.escape(t) for t in sorted_tokens) + ")")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str   (pickled dict[int, bytes], written by prepare_data.py)
        merges_filepath: str  (pickled list[tuple[bytes, bytes]])
        special_tokens: list[str] | None=None
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        ids = []
        # 1. split text based on special_pattern
        if self.special_pattern:
            chunks = self.special_pattern.split(text)
        else:
            chunks = [text]
        # 2. pre-tokenize
        for chunk in chunks:
            # if chunk is a special token, add the id
            if chunk in self.special_tokens:
                encoded = chunk.encode("utf-8")
                ids.append(self.token_to_id[encoded])
                continue
            # if not a special token, pre-tokenize
            for match in re.finditer(PAT, chunk):
                pretoken = match.group()
                byte_tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
                # repeatedly merge based on merges_priority
                # need at least two bytes to form a pair
                while len(byte_tokens) >= 2:
                    best_pair = None
                    best_rank = float("inf")
                    for i in range(len(byte_tokens) - 1):
                        pair = (byte_tokens[i], byte_tokens[i + 1])
                        rank = self.merges_priority.get(pair)
                        if rank is not None and rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                    if best_pair is None:
                        break
                    byte_tokens = merge_pair(byte_tokens, best_pair)
                for token in byte_tokens:
                    ids.append(self.token_to_id[token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Give an iterable of strings, return a generator that lazily yields token IDs.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        token_list = []
        for token_id in ids:
            token = self.vocab.get(token_id)
            if token is None:
                token = "\ufffd".encode("utf-8")
            token_list.append(token)
        return b"".join(token_list).decode("utf-8", errors="replace")
