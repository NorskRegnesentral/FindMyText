"""Utility functions for handling JSONL and JSONL.ZST files."""

import gzip
import io
import json
import random
from typing import Dict, Generator, Optional

import orjson
import zstandard as zstd


def stream_json_zst(
    file_path: str,
    min_length: int = 2000,
    skip_prob: float = 0.0,
    max_length: int = 100000,
    max_nb: Optional[int] = None,
):
    """Stream JSON objects from a .jsonl.zst file, yielding one JSON object at a
    time."""
    # Open the file in binary read mode
    with open(file_path, "rb") as f:
        # Create a decompression context
        dctx = zstd.ZstdDecompressor()
        # Create a stream reader to decompress on the fly
        with dctx.stream_reader(f) as reader:
            # Wrap the binary stream in a text wrapper to handle line-by-line reading
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            generator = json_line_reader(
                text_stream,
                min_length=min_length,
                skip_prob=skip_prob,
                max_length=max_length,
                max_nb=max_nb,
            )
            for data in generator:
                yield data


def stream_jsonl(
    file_path: str,
    min_length: int = 2000,
    skip_prob: float = 0.0,
    max_length: int = 100000,
    max_nb: Optional[int] = None,
):
    """Stream JSON objects from a .jsonl or .jsonl.gz file, yielding one JSON object at
    a time."""
    if file_path.endswith(".gz"):
        fd = gzip.open(file_path, "rt", encoding="utf-8")
    else:
        fd = open(file_path, "r", encoding="utf-8")

    generator = json_line_reader(
        fd,
        min_length=min_length,
        skip_prob=skip_prob,
        max_length=max_length,
        max_nb=max_nb,
    )
    for data in generator:
        yield data

    fd.close()


def json_line_reader(
    line_iterator,
    min_length: int = 2000,
    skip_prob: float = 0.0,
    max_length: int = 100000,
    max_nb: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """Read lines from a text stream, parse them as JSON, and yield valid JSON objects
    based on specified criteria such as minimum and maximum length, and skipping with a
    certain probability."""
    nb_read = 0
    nb_skipped = 0
    for line in line_iterator:
        if line.strip():  # Skip empty lines
            # Parse each line as a JSON object
            if random.random() < skip_prob:
                nb_skipped += 1
                continue
            data = json.loads(line)
            if len(data["text"]) < min_length or len(data["text"]) > max_length:
                nb_skipped += 1
                continue
            if any(
                [word in data["text"].split() for word in {"sex", "porn"}]
            ):  # Avoiding a certain type of websites ...
                nb_skipped += 1
                continue
            data = {
                k: v
                for k, v in data.items()
                if k not in {"seg_langs", "web-register", "doc_scores"}
            }
            yield data
            nb_read += 1
            if max_nb is not None and nb_read >= max_nb:
                break
    print("Nb documents: %i kept, %i skipped" % (nb_read, nb_skipped))


def stream_to_file(
    input_data_file: str,
    output_file: str,
    skip_prob: float = 0.0,
    min_length: int = 2000,
    max_length: int = 1000000,
    cutoff: Optional[int] = None,
):
    """Stream JSON objects from an input file and write them to an output file,
    optionally skipping some objects based on a specified probability and stopping after
    a certain number of objects.

    Args:
        input_data_file (str): Path to the input .jsonl, .jsonl.gz, or .jsonl.zst file to read from.
        output_file (str): Path to the output .jsonl file to write the streamed JSON objects to.
        skip_prob (float, optional): Probability of skipping a JSON object while streaming.
        Defaults to 0.0 (no skipping).
        min_length (int, optional): Minimum text lengths to include. Defaults to 2000.
        max_length (int, optional): Maximum text lengths to include. Defaults to 1000000.
        cutoff (Optional[int], optional): Maximum number of JSON objects to write to the output file.
        If None, all objects will be written. Defaults to None.

    """
    if input_data_file.endswith(".jsonl") or input_data_file.endswith(".jsonl.gz"):
        stream = stream_jsonl(
            input_data_file,
            skip_prob=skip_prob,
            min_length=min_length,
            max_length=max_length,
        )
    elif input_data_file.endswith(".jsonl.zst"):
        stream = stream_json_zst(
            input_data_file,
            skip_prob=skip_prob,
            min_length=min_length,
            max_length=max_length,
        )
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .jsonl, .jsonl.gz, or .jsonl.zst file."
        )
    count = 0
    with open(output_file, "w") as f:
        for input_doc in stream:
            f.write(orjson.dumps(input_doc).decode("utf-8"))
            f.write("\n")
            count += 1
            if cutoff is not None and count >= cutoff:
                break
