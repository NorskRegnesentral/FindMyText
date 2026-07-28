"""Utility functions for handling JSONL and JSONL.ZST files."""

import gzip
import io
import json
import random
import sys
import uuid

try:
    import resource  # not available on Windows
except ImportError:
    resource = None  # type: ignore[assignment]
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

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
    min_length: int = 100,
    skip_prob: float = 0.0,
    max_length: int = 500_000,
    max_nb: Optional[int] = None,
):
    """Stream JSON objects from a .jsonl.zst file, yielding one JSON object at a
    time."""

    import zstandard as zstd

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
    min_length: int = 100,
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
    min_length: int = 100,
    skip_prob: float = 0.0,
    max_length: int = 100000,
    max_nb: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """Read lines from a text stream, parse them as JSON, and yield valid JSON objects
    based on specified criteria such as minimum and maximum length, and skipping with a
    certain probability."""
    nb_read = 0
    nb_skipped = 0
    processed_ids = set()
    for line in line_iterator:
        if line.strip():  # Skip empty lines
            # Parse each line as a JSON object
            if random.random() < skip_prob:
                nb_skipped += 1
                continue
            data = json.loads(line)
            data = normalise_json(data)

            if data["id"] in processed_ids:
                raise ValueError(f"Duplicate document ID found: {data['id']}")
            processed_ids.add(data["id"])

            if len(data["text"]) < min_length or len(data["text"]) > max_length:
                nb_skipped += 1
                continue
            if any(
                [word in data["text"].split() for word in {"sex", "porn"}]
            ):  # Avoiding a certain type of websites ...
                nb_skipped += 1
                continue

            yield data
            nb_read += 1
            if max_nb is not None and nb_read >= max_nb:
                break
    print("Nb documents: %i kept, %i skipped" % (nb_read, nb_skipped))


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
    min_length: int = 100,
    max_length: int = 500_000,
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
        min_length (int, optional): Minimum text lengths to include. Defaults to 100.
        max_length (int, optional): Maximum text lengths to include. Defaults to 500000.
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


def get_process_memory_gb() -> float:
    """Return current RSS of the current process in GB (Linux, macOS, and Windows)."""
    if sys.platform == "linux":
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / (1024 * 1024)  # kB → GB
        except (OSError, ValueError):
            pass
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        ctypes.windll.kernel32.K32GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize / (1024**3)
    else:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KiB → GB; macOS: bytes → GB
        return rss / (1024**2) if sys.platform != "darwin" else rss / (1024**3)
