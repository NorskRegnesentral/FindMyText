"""Winnower module for document fingerprinting using the winnowing algorithm.

This module provides a fast implementation of the winnowing algorithm for generating
document fingerprints, which can be used for tasks like plagiarism detection and
document similarity. The implementation is based on the original paper by Schleimer et
al. (2003) and includes optimizations using numba for JIT compilation and NumPy for
efficient array operations.
"""

import re
from typing import Dict, List, Set, Tuple

import numpy as np
from numba import njit

HASH_METHOD = "polynomial64-fmix64"
LEGACY_HASH = "legacy31"
HASH_METHODS = (HASH_METHOD, LEGACY_HASH)
ROLLING_BASE64 = 0x9E3779B185EBCA87
_HASH_MOD = (1 << 31) - 1


class Winnower:
    """Winnower class for computing document fingerprints using the winnowing
    algorithm."""

    def __init__(
        self, length: int = 5, window_size: int = 6,
        base: int | None = None, punctuation=False,
        hash_method: str = HASH_METHOD,
    ):
        """Initialize the Winnower with specified parameters.

        Args:
            length: The length of k-grams for fingerprinting.
            window_size: The size of the window for winnowing.
            base: Rolling base; defaults to an odd 64-bit base, or 256 for legacy31.
            punctuation: Whether to include punctuation in the tokenization.
            hash_method: polynomial64-fmix64 (default) or paper-compatible legacy31.

        """
        if hash_method not in HASH_METHODS:
            raise ValueError(f"Unsupported hash_method {hash_method!r}; choose from {HASH_METHODS}")
        if length < 1 or window_size < 1:
            raise ValueError("length and window_size must be positive")
        if base is None:
            base = 256 if hash_method == LEGACY_HASH else ROLLING_BASE64
        if (
            not isinstance(base, (int, np.integer)) or isinstance(base, (bool, np.bool_))
        ):
            raise ValueError("The rolling base must be an integer")
        if hash_method == HASH_METHOD and (not 1 < base < 2**64 or base % 2 == 0):
            raise ValueError("The rolling hash requires an odd base between 3 and 2**64 - 1")
        if hash_method == LEGACY_HASH and not -(2**63) <= base < 2**63:
            raise ValueError("legacy31 requires a signed 64-bit rolling base")
        self.length = length
        self.window_size = window_size
        self.base = int(base)
        self.punctuation = punctuation
        self.hash_method = hash_method

    def get_winnowed_fingerprints(self, document: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute winnowed fingerprints for the given document string.

        The process includes tokenization, full-width token hashing, rolling hash
        computation for k-grams, and winnowing to select representative fingerprints.

        Arguments:
            document: The input document as a string.

        Returns:
            A tuple of two NumPy arrays: (winnowed_fingerprints, winnowed_positions) where
            winnowed_fingerprints contains the selected fingerprint values and winnowed_positions contains
            their corresponding positions in the original sequence.

        """
        tokens = self.tokenize(document)

        fingerprints = self.get_token_fingerprints(tokens)

        winnowed_fingerprints, winnowed_positions = winnow(
            fingerprints, self.window_size
        )

        return winnowed_fingerprints, winnowed_positions

    def get_token_fingerprints(self, tokens: List[str]) -> np.ndarray:
        """Hash every k-gram before winnowing with the configured method."""
        if self.hash_method == LEGACY_HASH:
            return get_fingerprints(tokens2unicode(tokens), self.base, self.length)
        return get_fingerprints64(tokens2uint64(tokens), self.length, np.uint64(self.base))

    def to_metadata(self) -> dict:
        return {
            "length": self.length, "window_size": self.window_size,
            "base": self.base, "punctuation": self.punctuation,
            "hash_method": self.hash_method,
        }

    @classmethod
    def from_metadata(cls, meta: dict) -> "Winnower":
        # Paper-era metadata had no method field; its hashes are always legacy31.
        method = meta.get("hash_method", LEGACY_HASH)
        if method == "rolling64-v1":
            method = HASH_METHOD
        return cls(
            length=meta["length"], window_size=meta["window_size"],
            base=meta["base"], punctuation=meta["punctuation"],
            hash_method=method,
        )

    def get_winnowed_fingerprints_batch(
        self, batch: Dict[str, str]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Compute winnowed fingerprints for a batch of {doc_id: text} documents.

        Arguments:
            batch: A dictionary mapping doc IDs to their corresponding text strings.

        Returns:
            A dictionary mapping doc IDs to lists of (fingerprint, position) tuples for each document in the batch.

        """
        results = {}
        for doc_id, text in batch.items():
            fps, positions = self.get_winnowed_fingerprints(text)
            results[doc_id] = list(zip(fps.tolist(), positions.tolist()))
        return results

    def tokenize(self, document: str) -> List[str]:
        """Split the document into tokens.

        Optionally remove punctuation before tokenization.
        """
        if not self.punctuation:
            document = re.sub(r"[^\w\s]", "", document, flags=re.UNICODE)
        tokens = document.split()
        return tokens


def tokens2unicode(tokens: List[str]) -> np.ndarray:
    """Original paper token hashing, including reduction modulo 2**31 - 1."""
    results = np.empty(len(tokens), dtype=np.int64)
    for i, token in enumerate(tokens):
        h = 1469598103934665603
        for b in token.encode("utf-8"):
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        results[i] = h % _HASH_MOD
    return results


@njit(cache=True)
def get_fingerprints(unicode_tokens, base, length):
    """Original paper rolling kernel; retain its signed arithmetic for reproduction."""
    mod = _HASH_MOD
    b_pow_k = 1
    for _ in range(length):
        b_pow_k = (b_pow_k * base) % mod
    n = len(unicode_tokens)
    if length <= 0 or n < length:
        return np.empty(0, dtype=np.int64)
    H = 0
    for i in range(length):
        H = (H * base + unicode_tokens[i]) % mod
    H = (H * base) % mod
    fingerprints = np.empty(n - length + 1, dtype=np.int64)
    fingerprints[0] = H
    for i in range(1, n - length + 1):
        H = (
            ((H - unicode_tokens[i - 1] * b_pow_k) + unicode_tokens[i + length - 1])
            * base
            % mod
        )
        fingerprints[i] = H
    return fingerprints


@njit(cache=True, nogil=True)
def _hash_token_bytes64(data, offsets):
    result = np.empty(len(offsets) - 1, dtype=np.uint64)
    for token in range(len(result)):
        value = np.uint64(1469598103934665603)
        for position in range(offsets[token], offsets[token + 1]):
            value ^= np.uint64(data[position])
            value *= np.uint64(1099511628211)
        result[token] = value
    return result


def tokens2uint64(tokens: List[str]) -> np.ndarray:
    """Hash UTF-8 bytes in compiled code, retaining the inherited token recurrence."""
    encoded = [token.encode("utf-8") for token in tokens]
    offsets = np.zeros(len(tokens) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(token) for token in encoded])
    return _hash_token_bytes64(
        np.frombuffer(b"".join(encoded), dtype=np.uint8), offsets,
    )


@njit(cache=True, nogil=True)
def get_fingerprints64(tokens, length, base=ROLLING_BASE64):
    """Polynomial rolling hash modulo 2**64, followed by the fmix64 finalizer.

    Unsigned wraparound is intentional. Roll the unmixed state; fmix64 is
    applied only to output values. This is not a cryptographic hash.
    """
    if length <= 0 or len(tokens) < length:
        return np.empty(0, dtype=np.uint64)
    unsigned_base = np.uint64(base)
    power = np.uint64(1)
    for _ in range(length):
        power *= unsigned_base
    state = np.uint64(0)
    for position in range(length):
        state = state * unsigned_base + tokens[position]
    state *= unsigned_base
    result = np.empty(len(tokens) - length + 1, dtype=np.uint64)
    for position in range(len(result)):
        if position:
            state = (state - tokens[position - 1] * power
                     + tokens[position + length - 1]) * unsigned_base
        value = state
        value ^= value >> np.uint64(33)
        value *= np.uint64(0xFF51AFD7ED558CCD)
        value ^= value >> np.uint64(33)
        value *= np.uint64(0xC4CEB9FE1A85EC53)
        value ^= value >> np.uint64(33)
        result[position] = value
    return result


@njit(cache=True, nogil=True)
def winnow(
    fingerprint_values: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the winnowing algorithm to select representative fingerprints from the
    sequence of fingerprint values. The algorithm slides a window of the specified size
    over the fingerprint values, selects the minimum value in each window, and emits it
    as a representative fingerprint if its position is different from the last emitted
    one.

    Args:
        fingerprint_values: A NumPy array of fingerprint values computed from the document.
        window_size: The size of the sliding window for winnowing.

    Returns:
        A tuple of two NumPy arrays: (winnowed_fingerprints, winnowed_positions), where
        winnowed_fingerprints contains the selected fingerprint values and winnowed_positions contains
        their corresponding positions in the original sequence.

    """
    n = len(fingerprint_values)
    if window_size <= 0 or n == 0 or window_size > n:
        return np.empty(0, dtype=fingerprint_values.dtype), np.empty(0, dtype=np.int32)

    # The maximum number of winnowed fingerprints is n - window_size + 1 (one per window), but we may emit
    # fewer due to duplicates. We preallocate arrays of this maximum size and keep track of how many we
    # actually emit.
    size = n - window_size + 1
    tmp_fingerprints = np.empty(size, dtype=fingerprint_values.dtype)
    tmp_positions = np.empty(size, dtype=np.int32)
    out_count = 0
    last_pos = -1

    for i in range(size):
        min_val = fingerprint_values[i]
        min_pos = i

        # Find the minimum fingerprint value and its position in the current window
        for j in range(1, window_size):
            v = fingerprint_values[i + j]
            if v <= min_val:
                min_val = v
                min_pos = i + j

        # Emit only when selected minimum position changes.
        if min_pos != last_pos:
            tmp_fingerprints[out_count] = min_val
            tmp_positions[out_count] = min_pos
            out_count += 1
            last_pos = min_pos

    return tmp_fingerprints[:out_count], tmp_positions[:out_count]
