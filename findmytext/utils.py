"""Utility functions for handling JSONL and JSONL.ZST files."""

import gzip
import io
import json
import os
import random
import sys
import uuid
import multiprocessing as mp
from tqdm import tqdm
import zstandard
import re

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

try:
    import resource  # not available on Windows
except ImportError:
    resource = None  # type: ignore[assignment]
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import orjson

# Common field names for text and ID in JSON objects
COMMON_TEXT_FIELDS = ["text", "content", "document", "report"]
COMMON_ID_FIELDS = ["id", "doc_id", "document_id", "report_id", "text_id", "title"]


def generate_batches(
    data_stream: Any, stop_after: Optional[int], batch_size: int = 256
) -> Iterator[List[Tuple[str, str]]]:
    """Yield batches of (doc_id, text) tuples from the data_stream."""

    batch: Dict[str, str] = {}
    for i, item in enumerate(data_stream):
        if stop_after is not None and i >= stop_after:
            break
        item = normalise_json(item)
        batch[item["id"]] = item["text"]
        if len(batch) >= batch_size:
            yield batch
            batch = {}
    if batch:
        yield batch


def stream_json_zst(
    file_path: str,
    min_length: Optional[int] = None,
    skip_prob: float = 0.0,
    max_length: Optional[int] = None,
    max_nb: Optional[int] = None,
):
    """Stream JSON objects from a .jsonl.zst file, yielding one JSON object at a
    time."""
    yield from json_line_reader(
        stream_lines(file_path),
        min_length=min_length,
        skip_prob=skip_prob,
        max_length=max_length,
        max_nb=max_nb,
    )


def stream_lines(file_path: str) -> Iterator[bytes]:
    """Yield raw (bytes) lines from a .zst, .gz or uncompressed file."""
    if file_path.endswith(".zst"):
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with open(file_path, "rb") as f, dctx.stream_reader(f) as reader:
            yield from io.BufferedReader(reader, buffer_size=1 << 20)
    elif file_path.endswith(".gz"):
        with gzip.open(file_path, "rb") as f:
            yield from f
    else:
        with open(file_path, "rb") as f:
            yield from f
            
def stream_jsonl(
    file_path: str,
    min_length: Optional[int] = None,
    skip_prob: float = 0.0,
    max_length: Optional[int] = None,
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
    min_length: Optional[int] = None,
    skip_prob: float = 0.0,
    max_length: Optional[int] = None,
    max_nb: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """Read lines (str or bytes) from a stream, parse them as JSON, and yield valid
    JSON objects based on specified criteria such as minimum and maximum length, and
    skipping with a certain probability.

    Only the first document with a given ID is yielded; later repeats of the same ID
    are skipped with a warning (some crawled corpora contain the same page several
    times)."""
    nb_read = 0
    nb_skipped = 0
    nb_duplicates = 0
    processed_ids = set()
    for line_nr, line in enumerate(line_iterator, start=1):
        if line.strip():  # Skip empty lines
            # Parse each line as a JSON object
            if random.random() < skip_prob:
                nb_skipped += 1
                continue
            data = json.loads(line)
            data = normalise_json(data)

            if data["id"] in processed_ids:
                nb_duplicates += 1
                print(
                    f"Warning: duplicate document ID {data['id']} "
                    f"at line {line_nr:,}, skipping",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            processed_ids.add(data["id"])

            if (min_length is not None and len(data["text"]) < min_length) or (max_length is not None and len(data["text"]) > max_length):
                nb_skipped += 1
                continue
            words = set(data["text"].split())

            yield data
            nb_read += 1
            if max_nb is not None and nb_read >= max_nb:
                break
    print("Nb documents: %i kept, %i skipped, %i duplicates"
        % (nb_read, nb_skipped, nb_duplicates)
    )


def normalise_json(
    data: dict, text_field: Optional[str] = None, id_field: Optional[str] = None
):
    """Normalise a JSON object to ensure it has the required fields and remove any
    unnecessary fields. The function checks for the presence of specified text and ID
    fields, and if they are not provided, it attempts to find common alternatives. It also
    removes any fields that are not relevant for further processing."""

    if text_field is not None and text_field not in data:
        raise ValueError(f"Missing required field '{text_field}' in JSON object")
    if id_field is not None and id_field not in data:
        raise ValueError(f"Missing required field '{id_field}' in JSON object")
    if text_field is None:
        for field in COMMON_TEXT_FIELDS:
            if field in data:
                text_field = field
                break
        else:
            raise ValueError(
                "JSON object does not contain a recognized text field. Please specify the text_field parameter."
            )
    if id_field is None:
        for field in COMMON_ID_FIELDS:
            if field in data:
                id_field = field
                break
        else:
            print("No id or doc_id field found in JSON object; creating a random ID")
            data["id"] = str(uuid.uuid4())

    if text_field != "text":
        data["text"] = data[text_field]
        del data[text_field]
    if id_field != "id":
        data["id"] = data[id_field]
        del data[id_field]

    data = {
        k: v
        for k, v in data.items()
        if k not in {"seg_langs", "web-register", "doc_scores"}
    }

    return data


def stream_to_file(
    input_data_file: str,
    output_file: str,
    skip_prob: float = 0.0,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
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
        min_length (int, optional): Minimum text lengths to include. Defaults to None.
        max_length (int, optional): Maximum text lengths to include. Defaults to None.
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


def get_filtered_files(
    dir: str, file_prefixes: Iterable[str] = (), file_suffixes: Iterable[str] = ()
) -> List[str]:
    """Return the sorted full paths of the entries of dir (files and directories)
    whose name starts with at least one of file_prefixes and ends with at least one
    of file_suffixes.  An empty prefix or suffix collection imposes no constraint on
    that side, so get_filtered_files(dir) lists the whole directory.  Returns an
    empty list when nothing matches."""
    prefixes = tuple(file_prefixes)
    suffixes = tuple(file_suffixes)
    return sorted(
        os.path.join(dir, name)
        for name in os.listdir(dir)
        if (not prefixes or name.startswith(prefixes))
        and (not suffixes or name.endswith(suffixes))
    )

def find_doc(doc_id: str, file_or_directory: str, n_workers=None):
    """Search all .jsonl.zst files in directory in parallel and return the doc with doc_id.

    Returns None if no document has that id.
    """
    if not os.path.exists(file_or_directory):
        raise ValueError(f"File or directory '{file_or_directory}' does not exist.")
    elif os.path.isdir(file_or_directory):
        files = sorted(
          os.path.join(file_or_directory, f) for f in os.listdir(file_or_directory) if f.endswith(".jsonl.zst") or f.endswith(".jsonl.gz")
      )
    elif file_or_directory.endswith(".jsonl.zst") or file_or_directory.endswith(".jsonl.gz"):
        files = [file_or_directory]
    else:
        raise ValueError(f"File or directory '{file_or_directory}' is not a supported format.")

    n_workers = n_workers or min(len(files), len(os.sched_getaffinity(0)))
    ctx = mp.get_context("fork")
    with ctx.Manager() as manager, ProcessPoolExecutor(n_workers, mp_context=ctx) as ex:
        stop = manager.Event()
        counts = manager.dict()  # path -> nb of lines read so far
        pending = {ex.submit(_search_file, p, doc_id, stop, counts) for p in files}
        with tqdm(unit=" lines", unit_scale=True) as bar:
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                bar.n = sum(counts.values())
                bar.set_postfix(files_left=len(pending))
                for fut in done:
                    doc = fut.result()
                    if doc is not None:
                        stop.set()
                        for f in pending:
                            f.cancel()
                        return doc
    return None

def _search_file(path, doc_id: str, stop, counts):
    """Scan one .jsonl.zst/.jsonl.gz file for doc_id; give up early if `stop` is set."""
    needle = str(doc_id).encode()
    for i, line in enumerate(stream_lines(path)):
        if needle in line:
            doc = orjson.loads(line)
            if str(doc["id"]) == doc_id:
                stop.set()
                return doc
        if i % 100_000 == 0:
            counts[path] = i
            if stop.is_set():
                return None
    return None