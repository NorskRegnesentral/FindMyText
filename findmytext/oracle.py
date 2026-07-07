"""Local sequence alignment using the Smith-Waterman algorithm with custom gap and substitution costs.

The `LocalAligner` class provides methods to compute local alignments between two strings, returning
the best matching regions along with their scores and offsets. The implementation uses JIT compilation
for performance and includes options for verbose output and handling long sequences efficiently.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from numba import jit, objmode


def align(text1: str, text2: str, multiple_passes: bool = False) -> AlignmentResult:
    """Align two texts and return an AlignmentResult object.

    Arguments:
        text1: The first text to align.
        text2: The second text to align.
        multiple_passes: If True, the alignment will be performed in multiple passes to find all matching regions.
    Results:
        An AlignmentResult object containing the original texts and a list of matching regions.

    """

    aligner = LocalAligner()
    if multiple_passes:
        return aligner.run_all(text1, text2)
    else:
        return aligner.run(text1, text2)


class AlignmentResult:
    """Class representing the result of a local alignment, including the original texts
    and a list of matching regions.

    The `__repr__` method provides a human-readable representation of the alignment results.
    """

    def __init__(self, text1: str, text2: str):
        """Initialize the AlignmentResult with the original texts and an empty list of matching regions."""
        self.text1 = text1
        self.text2 = text2
        self.regions = []

    def add_matching_region(
        self, start1: int, end1: int, start2: int, end2: int, score: float
    ):
        """Add a matching region to the result, including the start and end offsets in both texts
        and the alignment score.
        """
        self.regions.append(
            {
                "start1": start1,
                "end1": end1,
                "start2": start2,
                "end2": end2,
                "score": score,
            }
        )

    def __repr__(self) -> str:
        """Return a string representation of the alignment results."""
        alignment_strs = []
        for region in self.regions:
            start1 = region["start1"]
            end1 = region["end1"]
            start2 = region["start2"]
            end2 = region["end2"]
            score = region["score"]
            aligned_seq1 = self.text1[start1:end1]
            aligned_seq2 = self.text2[start2:end2]
            alignment_strs.append(
                f"Alignment(score={score}, "
                f"[{start1}:{end1}]={aligned_seq1!r}, "
                f"[{start2}:{end2}]={aligned_seq2!r})"
            )
        return "\n".join(alignment_strs)

    def show(self, show_regions_only=True):
        """Display a colour-coded HTML visualisation of the alignment in a Jupyter notebook.

        Characters that are matched between the two texts are highlighted in green; characters
        that differ (mismatches or gaps) are highlighted in red. If `show_regions_only` is True,
        only the matching regions are shown side by side. Otherwise the full texts are shown, with
        non-aligned portions left unformatted.
        """
        from difflib import SequenceMatcher
        from html import escape

        from IPython.display import HTML, display

        GREEN = "background-color:#90ee90"
        RED = "background-color:#ff9999"

        def _span(text, color):
            return (
                f'<mark style="{color}">{escape(text).replace(chr(10), "<br>")}</mark>'
            )

        def _diff_render(s1, s2):
            """Return (html1, html2) with matched characters in green, differing in red."""
            parts1, parts2 = [], []
            for tag, i1, i2, j1, j2 in SequenceMatcher(
                None, s1, s2, autojunk=False
            ).get_opcodes():
                c1, c2 = s1[i1:i2], s2[j1:j2]
                color = GREEN if tag == "equal" else RED
                if c1:
                    parts1.append(_span(c1, color))
                if c2:
                    parts2.append(_span(c2, color))
            return "".join(parts1), "".join(parts2)

        col_style = (
            "font-family:monospace;white-space:pre-wrap;"
            "border:1px solid #ccc;padding:10px;flex:1;min-width:0"
        )
        row_style = "display:flex;gap:10px;margin:6px 0"
        header_style = "font-weight:bold;margin-bottom:6px"

        if show_regions_only:
            blocks = []
            for i, r in enumerate(
                sorted(self.regions, key=lambda r: r["score"], reverse=True), 1
            ):
                s1 = self.text1[r["start1"] : r["end1"]]
                s2 = self.text2[r["start2"] : r["end2"]]
                html1, html2 = _diff_render(s1, s2)
                blocks.append(
                    f'<div style="margin:6px 0">'
                    f'<div style="{header_style}">Region {i} &mdash; score {r["score"]:.1f}</div>'
                    f'<div style="{row_style}">'
                    f'<div style="{col_style}"><b>Text 1</b> [{r["start1"]}:{r["end1"]}]<br><br>{html1}</div>'
                    f'<div style="{col_style}"><b>Text 2</b> [{r["start2"]}:{r["end2"]}]<br><br>{html2}</div>'
                    f"</div></div>"
                )
            html = "\n".join(blocks) if blocks else "<i>No aligned regions found.</i>"
        else:
            # Build per-character colour maps for the full texts.
            # Regions are paired: region i covers [start1_i:end1_i] in text1 and
            # [start2_i:end2_i] in text2. Outside regions, characters are unstyled.
            rendered1: dict[int, str] = {}  # char_index → html span
            rendered2: dict[int, str] = {}
            for r in self.regions:
                s1 = self.text1[r["start1"] : r["end1"]]
                s2 = self.text2[r["start2"] : r["end2"]]
                for tag, i1, i2, j1, j2 in SequenceMatcher(
                    None, s1, s2, autojunk=False
                ).get_opcodes():
                    color = GREEN if tag == "equal" else RED
                    for k, ch in enumerate(s1[i1:i2]):
                        rendered1[r["start1"] + i1 + k] = _span(ch, color)
                    for k, ch in enumerate(s2[j1:j2]):
                        rendered2[r["start2"] + j1 + k] = _span(ch, color)

            def _apply(text, rendered):
                parts = []
                for i, ch in enumerate(text):
                    if i in rendered:
                        parts.append(rendered[i])
                    else:
                        parts.append(escape(ch).replace(chr(10), "<br>"))
                return "".join(parts)

            body1 = _apply(self.text1, rendered1)
            body2 = _apply(self.text2, rendered2)
            html = (
                f'<div style="{row_style}">'
                f'<div style="{col_style}"><b>Text 1</b><br><br>{body1}</div>'
                f'<div style="{col_style}"><b>Text 2</b><br><br>{body2}</div>'
                f"</div>"
            )
        display(HTML(html))


class LocalAligner:
    """Class encapsulating the Smith-Waterman local alignment algorithm with custom gap
    and substitution costs.

    The main method is `run`, which takes two strings and returns an `Alignment` object.
    """

    def __init__(
        self,
        verbose: bool = False,
        max_length_one_pass: int = 1000,
        min_region_length: int = 10,
        match_score: float = 1.0,
    ):
        """Initialize the LocalAligner.

        Parameters
        ----------
        verbose : bool
            Whether to print verbose output.
        max_length_one_pass : int
            Maximum length for single-pass computation.
        min_region_length : int
            Minimum length for a matching region.
        match_score : float
            The score for a match between two characters.

        """

        self.match_score = match_score
        self.verbose = verbose
        self.max_length_one_pass = max_length_one_pass
        self.min_region_length = min_region_length

        # We also use a cache to store results of alignments between pairs of texts, to avoid redundant computations.
        self.cache: Dict[Tuple[str, str], AlignmentResult] = {}

    def get_largest_region_score(self, text1: str, text2: str) -> float:
        """Run the alignment and return the score of the largest matching region, or 0 if there are no regions."""
        result = self.run(text1, text2)
        if result.regions:
            largest_region = max(result.regions, key=lambda r: r["score"])
            return largest_region["score"]
        else:
            return 0.0

    def run_all(self, text1: str, text2: str, max_runs: int = 3) -> AlignmentResult:
        """Run the alignment and return all matching regions above the minimum length.

        This method recursively finds the best local alignment, then splits the texts around
        the aligned region and continues searching for additional alignments in the remaining
        parts of the texts.
        """

        result = self.run(text1, text2)
        if result.regions:
            # After one run, there is at most one region in the result
            largest_region = result.regions[0]

            largest_region_start1 = largest_region["start1"]
            largest_region_end1 = largest_region["end1"]
            largest_region_start2 = largest_region["start2"]
            largest_region_end2 = largest_region["end2"]

            text_before1 = text1[:largest_region_start1]
            text_before2 = text2[:largest_region_start2]
            text_after1 = text1[largest_region_end1:]
            text_after2 = text2[largest_region_end2:]

            before_result = self.run_all(text_before1, text_before2, max_runs=max_runs)
            for before_result_region in before_result.regions:
                result.add_matching_region(
                    before_result_region["start1"],
                    before_result_region["end1"],
                    before_result_region["start2"],
                    before_result_region["end2"],
                    before_result_region["score"],
                )

            after_result = self.run_all(text_after1, text_after2)
            for after_result_region in after_result.regions:
                result.add_matching_region(
                    after_result_region["start1"] + largest_region_end1,
                    after_result_region["end1"] + largest_region_end1,
                    after_result_region["start2"] + largest_region_end2,
                    after_result_region["end2"] + largest_region_end2,
                    after_result_region["score"],
                )

        return result

    def run(self, text1: str, text2: str) -> AlignmentResult:
        """Given two texts, compute the best local alignment using the Smith-Waterman algorithm.

        The method relies on custom gap and substitution costs.
        """

        if (text1, text2) in self.cache:
            return self.cache[(text1, text2)]

        # Convert strings to codepoints
        if self.verbose:
            print("Converting strings to codepoints", end="...", flush=True)
        seq1_cp = np.frombuffer(text1.encode("utf-32le"), dtype=np.uint32)
        seq2_cp = np.frombuffer(text2.encode("utf-32le"), dtype=np.uint32)
        if self.verbose:
            print("done")

        # Build one compact alphabet for both texts and map codepoints to 0..K-1.
        if self.verbose:
            print("Building compact alphabet", end="...", flush=True)
        combined = np.concatenate((seq1_cp, seq2_cp))
        unique_codepoints, inverse = np.unique(combined, return_inverse=True)
        if self.verbose:
            print("done")

        split = len(seq1_cp)
        seq1_int = inverse[:split].astype(np.int32)
        seq2_int = inverse[split:].astype(np.int32)

        if self.verbose:
            print("Compiling gap and substitution tables", end="...", flush=True)
        gap_costs_array = compile_gap_table(unique_codepoints)
        substitution_costs_array = compile_substitution_table(unique_codepoints)
        if self.verbose:
            print("done")

        alignment_result = AlignmentResult(text1, text2)

        if len(inverse) < self.max_length_one_pass:
            best_score, seq1_start, seq1_end, seq2_start, seq2_end = (
                self._run_single_pass(
                    seq1_int, seq2_int, gap_costs_array, substitution_costs_array
                )
            )

        else:
            best_score, seq1_start, seq1_end, seq2_start, seq2_end = (
                self._run_two_passes(
                    seq1_int, seq2_int, gap_costs_array, substitution_costs_array
                )
            )

        if (
            best_score > 0
            and (seq1_end - seq1_start) >= self.min_region_length
            and (seq2_end - seq2_start) >= self.min_region_length
        ):
            alignment_result.add_matching_region(
                seq1_start, seq1_end, seq2_start, seq2_end, best_score
            )

            # Cache the result of this alignment, and evict the oldest cache entry if we exceed the cache size limit.
            while len(self.cache) >= 50:
                oldest_cache_key = next(iter(self.cache))
                del self.cache[oldest_cache_key]
            self.cache[(text1, text2)] = alignment_result

        return alignment_result

    def _run_single_pass(
        self,
        seq1_int: np.ndarray,
        seq2_int: np.ndarray,
        gap_costs_array: np.ndarray,
        substitution_costs_array: np.ndarray,
    ) -> tuple[float, int, int, int, int]:
        """Run the Smith-Waterman algorithm in a single pass, returning the best score and offsets."""

        if self.verbose:
            print("Running in a single pass")

        best_score, seq1_start, seq1_end, seq2_start, seq2_end = (
            _smith_waterman_single_pass(
                seq1_int,
                seq2_int,
                self.match_score,
                gap_costs_array,
                substitution_costs_array,
                progress_counter=self.verbose,
            )
        )
        if self.verbose:
            print("Single pass complete.")

        return best_score, seq1_start, seq1_end, seq2_start, seq2_end

    def _run_two_passes(
        self,
        seq1_int: np.ndarray,
        seq2_int: np.ndarray,
        gap_costs_array: np.ndarray,
        substitution_costs_array: np.ndarray,
    ) -> tuple[float, int, int, int, int]:
        """Run the Smith-Waterman algorithm in two passes, returning the best score and offsets.

        The first pass finds the largest score and the end positions of the best local alignment.
        The second pass then focuses on the region preceding those positions to find the start offsets.
        This split in two passes makes it possible to avoid the memory overhead of the traceback
        matrix for very long sequences.
        """

        if self.verbose:
            print("Running in two passes")

        best_score, seq1_end, seq2_end = _smith_waterman_long_seq(
            seq1_int,
            seq2_int,
            self.match_score,
            gap_costs_array,
            substitution_costs_array,
            progress_counter=self.verbose,
        )

        if self.verbose:
            print("First pass complete.")

        maximum_length_shared_region = int(best_score * 10 // self.match_score)

        start_offset1: int = max(0, seq1_end - maximum_length_shared_region)
        start_offset2: int = max(0, seq2_end - maximum_length_shared_region)
        shorter_seq1 = seq1_int[start_offset1:seq1_end]
        shorter_seq2 = seq2_int[start_offset2:seq2_end]

        best_score2, start_i, best_i2, start_j, best_j2 = _smith_waterman_single_pass(
            shorter_seq1,
            shorter_seq2,
            self.match_score,
            gap_costs_array,
            substitution_costs_array,
            progress_counter=self.verbose,
        )

        if best_score2 != best_score:
            raise ValueError(
                "Warning: Score mismatch between passes:", best_score, best_score2
            )

        seq1_start = start_offset1 + start_i
        seq1_end = start_offset1 + best_i2
        seq2_start = start_offset2 + start_j
        seq2_end = start_offset2 + best_j2

        if self.verbose:
            print("Two passes complete.")

        return best_score, seq1_start, seq1_end, seq2_start, seq2_end


@jit(nopython=True, cache=True)
def _smith_waterman_single_pass(
    seq1: np.ndarray,
    seq2: np.ndarray,
    match_score: float,
    gap_costs: np.ndarray,
    substitution_costs: np.ndarray,
    progress_counter: bool = False,
) -> tuple[float, int, int, int, int]:
    """JIT-compiled Smith-Waterman computation for short sequences, returning score and offsets for the best local alignment.

    Args:
        seq1: Input sequence as array of compact codepoint ids.
        seq2: Input sequence as array of compact codepoint ids.
        match_score: Score for a match.
        gap_costs: Array of gap costs indexed by compact codepoint id.
        substitution_costs: Matrix of substitution costs indexed by compact codepoint ids.
        progress_counter: Whether to print progress during computation.

    Returns:
        A tuple containing the best score, start and end offsets in seq1, and start and end offsets in seq2 for the best local alignment.

    """

    n = len(seq1)
    m = len(seq2)

    # Rolling rows for score matrix H: O(m) memory instead of O(n*m).
    H_prev = np.zeros(m + 1, dtype=np.float32)
    H_curr = np.zeros(m + 1, dtype=np.float32)

    # Traceback matrix to reconstruct the best local alignment.
    # 0 = stop, 1 = diagonal, 2 = up, 3 = left.
    TB = np.zeros((n + 1, m + 1), dtype=np.uint8)

    best_score = 0.0
    best_i: int = 0
    best_j: int = 0

    for i in range(1, n + 1):
        c1 = seq1[i - 1]
        gap1 = gap_costs[c1]
        H_curr[0] = 0.0
        for j in range(1, m + 1):
            # Calculate score for coming from diagonal (match/mismatch).
            c2 = seq2[j - 1]
            if c1 == c2:
                diag = H_prev[j - 1] + match_score
            else:
                diag = H_prev[j - 1] - substitution_costs[c1, c2]

            # Calculate scores for coming from left (gap in seq1) and up (gap in seq2).
            gap2 = gap_costs[c2]
            from_left = H_curr[j - 1] - gap2
            from_up = H_prev[j] - gap1

            # Choose the maximum score
            score = max(0, diag, from_left, from_up)
            H_curr[j] = score

            # 0 = stop, 1 = diagonal, 2 = up, 3 = left
            if score == 0:
                TB[i, j] = 0
            elif score == diag:
                TB[i, j] = 1
            elif score == from_up:
                TB[i, j] = 2
            else:
                TB[i, j] = 3

            if score > best_score:
                best_score = score
                best_i = i
                best_j = j

        # We only need to keep the current and previous rows
        # of the score matrix, so we can roll them.
        H_prev, H_curr = H_curr, H_prev

        # Print progress every 100 rows if enabled.
        if i > 0 and i % 100 == 0 and progress_counter:
            with objmode():
                print(
                    f"\rProgress: {i}/{n} ({100 * i // max(1, n)}%)", end="", flush=True
                )
    if progress_counter:
        print()

    # Trace back from the best score position to find the start offsets.
    start_i: int = best_i
    start_j: int = best_j
    while start_i > 0 and start_j > 0 and TB[start_i, start_j] != 0:
        direction = TB[start_i, start_j]
        if direction == 1:
            start_i -= 1
            start_j -= 1
        elif direction == 2:
            start_i -= 1
        elif direction == 3:
            start_j -= 1
        else:
            break
    return best_score, start_i, best_i, start_j, best_j


@jit(nopython=True, cache=True)
def _smith_waterman_long_seq(
    seq1: np.ndarray,
    seq2: np.ndarray,
    match_score: float,
    gap_costs: np.ndarray,
    substitution_costs: np.ndarray,
    progress_counter: bool = False,
) -> tuple[float, int, int]:
    """JIT-compiled Smith-Waterman computation for long sequences, returning the best score and end offsets for the best local alignment.

    Parameters
    ----------
    seq1 : np.ndarray
        Input sequence as array of compact codepoint ids.
    seq2 : np.ndarray
        Input sequence as array of compact codepoint ids.
    match_score : float
        Score for a match.
    gap_costs : np.ndarray
        Array of gap costs indexed by compact codepoint id.
    substitution_costs : np.ndarray
        Matrix of substitution costs indexed by compact codepoint ids.
    progress_counter : bool
        Whether to print progress during computation.

    Returns
    -------
    tuple[float, int, int]
        A tuple containing the best score, end offset in seq1, and end offset in seq2 for the best local alignment.

    """
    n = len(seq1)
    m = len(seq2)

    # Rolling rows for score matrix
    H_prev = np.zeros(m + 1, dtype=np.float32)
    H_curr = np.zeros(m + 1, dtype=np.float32)

    best_score = 0.0
    best_i: int = 0
    best_j: int = 0

    for i in range(1, n + 1):
        c1 = seq1[i - 1]
        gap1 = gap_costs[c1]
        H_curr[0] = 0.0
        for j in range(1, m + 1):
            c2 = seq2[j - 1]
            gap2 = gap_costs[c2]
            if c1 == c2:
                diag = H_prev[j - 1] + match_score
            else:
                diag = H_prev[j - 1] - substitution_costs[c1, c2]
            from_left = H_curr[j - 1] - gap2
            from_up = H_prev[j] - gap1

            score = max(0, diag, from_left, from_up)
            H_curr[j] = score

            if score > best_score:
                best_score = score
                best_i = i
                best_j = j

        H_prev, H_curr = H_curr, H_prev

        if i > 0 and i % 100 == 0 and progress_counter:
            with objmode():
                print(
                    f"\rProgress: {i}/{n} ({100 * i // max(1, n)}%)", end="", flush=True
                )
    if progress_counter:
        print()

    # We only return the end offsets and score, since we will run a second pass
    # to find the start offsets.
    return best_score, best_i, best_j


@jit(nopython=True, cache=True)
def compile_gap_table(unique_chars) -> np.ndarray:
    """Create a gap-cost array indexed by compact codepoint id."""
    n = len(unique_chars)
    gap_costs = np.empty(n, dtype=np.float32)

    for idx in range(n):
        char = chr(int(unique_chars[idx]))
        gap_costs[idx] = (
            0.1
            if char in " \t\n\r"
            else 0.2
            if char in "-—–"
            else 0.3
            if char in ".,!?;:"
            else 0.5
            if char in "@#$%^&*()_+=/\\|<>"
            else 1.0
        )

    return gap_costs


@jit(nopython=True, cache=True)
def compile_substitution_table(unique_chars) -> np.ndarray:
    """Create a substitution-cost matrix indexed by compact codepoint ids."""
    n = len(unique_chars)
    substitution_costs = np.empty((n, n), dtype=np.float32)
    chars = [chr(int(cp)) for cp in unique_chars]

    for i in range(n):
        char1 = chars[i]
        for j in range(n):
            char2 = chars[j]
            if char1 == char2:
                substitution_costs[i, j] = 0.0
            elif char1.isspace() and char2.isspace():
                substitution_costs[i, j] = 0.1
            elif char1 in "-—–" and char2 in "-—–":
                substitution_costs[i, j] = 0.2
            elif char1.lower() == char2.lower():
                substitution_costs[i, j] = 0.2
            elif char1 in ".,!?;:" and char2 in ".,!?;:":
                substitution_costs[i, j] = 0.3
            elif char1 in "@#$%^&*()_+=/\\|<>" and char2 in "@#$%^&*()_+=/\\|<>":
                substitution_costs[i, j] = 0.5
            else:
                substitution_costs[i, j] = 1.0

    return substitution_costs


if __name__ == "__main__":
    with open("pg9645.txt", "r", encoding="utf-8") as f:
        text1 = f.read()
    with open("pg12137.txt", "r", encoding="utf-8") as f:
        text2 = f.read()
    aligner = LocalAligner(verbose=True, min_region_length=15)
    alignment = aligner.run(text1, text2)
    print(alignment)
