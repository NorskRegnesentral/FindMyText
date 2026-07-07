"""Winnower module for document fingerprinting using the winnowing algorithm.

This module provides a fast implementation of the winnowing algorithm for generating
document fingerprints, which can be used for tasks like plagiarism detection and
document similarity. The implementation is based on the original paper by Schleimer et
al. (2003) and includes optimizations using numba for JIT compilation and NumPy for
efficient array operations.
"""

import re
from typing import List, Set, Tuple

import numpy as np
from numba import njit

# Modulus for the rolling hash. 2^31 - 1 is the largest Mersenne prime that
# guarantees all intermediate multiplications (token * b_pow_k and H * base)
# fit within signed int64 without overflow. The 31-bit hash space (~2 billion values)
# is sufficient for document similarity tasks where min_fingerprints > 1.
_HASH_MOD = (1 << 31) - 1


class Winnower:
    """Winnower class for computing document fingerprints using the winnowing
    algorithm."""

    def __init__(
        self, length: int = 5, window_size: int = 6, base: int = 256, punctuation=False
    ):
        """Initialize the Winnower with specified parameters.

        Args:
            length: The length of k-grams for fingerprinting.
            window_size: The size of the window for winnowing.
            base: The base for the rolling hash computation.
            punctuation: Whether to include punctuation in the tokenization.

        """
        self.length = length
        self.window_size = window_size
        self.base = base
        self.punctuation = punctuation

    def get_winnowed_fingerprints(self, document: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute winnowed fingerprints for the given document string.

        The process includes tokenization, conversion to Unicode code point sums, rolling hash
        computation for k-grams, and winnowing to select representative fingerprints.

        Arguments:
            document: The input document as a string.

        Returns:
            A tuple of two NumPy arrays: (winnowed_fingerprints, winnowed_positions) where
            winnowed_fingerprints contains the selected fingerprint values and winnowed_positions contains
            their corresponding positions in the original sequence.

        """
        tokens = self.tokenize(document)

        unicode_tokens = tokens2unicode(tokens)

        fingerprints = get_fingerprints(unicode_tokens, self.base, self.length)

        winnowed_fingerprints, winnowed_positions = winnow(
            fingerprints, self.window_size
        )

        return winnowed_fingerprints, winnowed_positions

    def tokenize(self, document: str) -> List[str]:
        """Split the document into tokens.

        Optionally remove punctuation before tokenization.
        """
        if not self.punctuation:
            document = re.sub(r"[^\w\s]", "", document, flags=re.UNICODE)
        tokens = document.split()
        return tokens


def tokens2unicode(tokens: List[str]) -> np.ndarray:
    """Convert a list of tokens to a NumPy array of deterministic integer values.

    We use a simple FNV-1a hash on the UTF-8 bytes of each token to produce a consistent
    integer representation. The top bit is cleared to ensure the values fit in signed
    int64 for compatibility with NumPy and Numba.
    """
    results = np.empty(len(tokens), dtype=np.int64)

    # Deterministic FNV-1a hash on UTF-8 bytes. Keep the top bit clear so the
    # value always fits in signed int64 for NumPy/Numba downstream.
    for i, token in enumerate(tokens):
        h = 1469598103934665603
        for b in token.encode("utf-8"):
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        results[i] = h % _HASH_MOD

    return results


@njit
def get_fingerprints(unicode_tokens, base, length):
    """Compute rolling hash fingerprints for k-grams of the token sequence using an
    improved Karp-Rabin algorithm:
    H'(c1..ck)    = c1*b^k + c2*b^(k-1) + ... + ck*b
    H'(c2..ck+1)  = ((H'(c1..ck) - c1*b^k) + c_{k+1}) * b.

    The rolling hash is computed modulo a large prime to keep values bounded and avoid overflow.

    Args:
        unicode_tokens: A NumPy array of integer values representing the tokens.
        base: The base for the rolling hash (e.g., 256).
        length: The length of k-grams to compute fingerprints for.

    Returns:
        A NumPy array of fingerprint values for each k-gram in the document.

    """

    # Precompute b^k mod the large prime to avoid repeated exponentiation in the loop.
    mod = _HASH_MOD
    b_pow_k = 1
    for _ in range(length):
        b_pow_k = (b_pow_k * base) % mod

    n = len(unicode_tokens)
    if length <= 0 or n < length:
        return np.empty(0, dtype=np.int64)

    # Initial hash: (c1*b^(k-1) + c2*b^(k-2) + ... + ck) * b
    H = 0
    for i in range(length):
        H = (H * base + unicode_tokens[i]) % mod
    H = (H * base) % mod

    # We preallocate the output array for fingerprints.
    fingerprints = np.empty(n - length + 1, dtype=np.int64)
    fingerprints[0] = H

    # Rolling hash: H'(c2..ck+1)  = ((H'(c1..ck) - c1*b^k) + c_{k+1}) * b
    for i in range(1, n - length + 1):
        H = (
            ((H - unicode_tokens[i - 1] * b_pow_k) + unicode_tokens[i + length - 1])
            * base
            % mod
        )
        fingerprints[i] = H

    return fingerprints


@njit
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
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    # The maximum number of winnowed fingerprints is n - window_size + 1 (one per window), but we may emit
    # fewer due to duplicates. We preallocate arrays of this maximum size and keep track of how many we
    # actually emit.
    size = n - window_size + 1
    tmp_fingerprints = np.empty(size, dtype=np.int64)
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
