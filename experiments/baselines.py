"""Baseline detectors to compare against the fingerprint-based ones in :mod:`eval`.

:class:`DenseRetrievalDetector` embeds the query text with a sentence-embedding model
and returns the closest documents of a database of pre-computed document embeddings
(exact cosine similarity, via a faiss ``IndexFlatIP``).

The database is a directory containing:

- ``base.fbin``: the embeddings in the DiskANN "fbin" format, i.e. a header of two
  ``int32`` (number of vectors, dimension) followed by the ``float32`` vectors.  Row ``i``
  is the embedding of the document with id ``i``;
- ``meta.json``: the repository the vectors come from, the model used to compute them,
  and the number / dimension of the vectors.

:meth:`DenseRetrievalDetector.build_database` creates one from the Hugging Face dataset
``maknee/wikipedia_qwen_4b`` (1 M Wikipedia articles embedded with
``Qwen/Qwen3-Embedding-4B``, whose integer ``id`` column equals the row index and the
``source_doc_id`` used in the evaluation data)::

    python experiments/baselines.py build wiki_dense_db

Evaluating it, from Python::

    from baselines import DenseRetrievalDetector
    from eval import Evaluator

    evaluator = Evaluator(None)  # or Evaluator("my_index") to compare with the default detectors
    evaluator.add_detector("dense", DenseRetrievalDetector("wiki_dense_db"))
    evaluator.run("eval_set.jsonl")

Requires ``faiss-cpu``, ``sentence-transformers`` and ``huggingface_hub`` (see
``requirements.txt``).  The database (1 M x 2560 float32) takes about 10 GB of RAM.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import orjson

FBIN_NAME = "base.fbin"
META_NAME = "meta.json"
DEFAULT_REPO_ID = "maknee/wikipedia_qwen_4b"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
FBIN_HEADER_BYTES = 8

# Prompt used by Qwen3-Embedding for retrieval *queries* (documents are embedded as is).
QWEN_QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"


class DenseRetrievalDetector:
    """Nearest-neighbour search of a query embedding in a database of document embeddings.

    Parameters
    ----------
    db_dir : str
        Database directory, as created by :meth:`build_database`.
    model_name : str, optional
        Sentence-embedding model (default: the one recorded in ``meta.json``, i.e. the
        model the documents were embedded with).
    top_k : int
        Number of documents returned by :meth:`search` (default: 10).
    device : str, optional
        Device of the embedding model (``"cuda"``, ``"cpu"``, ...; default: auto).
    max_seq_length : int
        Maximum number of tokens per text; longer texts are truncated (default: 8192).
    instruction : str, optional
        If given, the query text is wrapped in the Qwen3-Embedding query prompt with this
        instruction.  By default the text is embedded as a plain document, like the
        documents of the database.
    batch_size : int
        Encoding batch size (default: 8).
    chunk_size : int
        Number of vectors read from ``base.fbin`` at a time when loading the faiss index
        (default: 50 000).

    """

    def __init__(
        self,
        db_dir: str,
        model_name: Optional[str] = None,
        top_k: int = 10,
        device: Optional[str] = None,
        max_seq_length: int = 8192,
        instruction: Optional[str] = None,
        batch_size: int = 8,
        chunk_size: int = 50_000,
    ):
        self.db_dir = db_dir
        self.meta = read_meta(db_dir)
        self.model_name = model_name or self.meta["model_name"]
        self.top_k = top_k
        self.instruction = instruction
        self.batch_size = batch_size

        self.vectors = read_fbin(os.path.join(db_dir, FBIN_NAME))
        self.n_docs, self.dim = self.vectors.shape
        self.index = self._build_faiss_index(chunk_size)
        self.model = self._load_model(device, max_seq_length)

    # --------------------------------------------------------------- loading

    def _build_faiss_index(self, chunk_size: int):
        import faiss

        index = faiss.IndexFlatIP(self.dim)
        for start in range(0, self.n_docs, chunk_size):
            # Copy (the memmap is read-only) and normalise so that inner product = cosine.
            chunk = np.array(self.vectors[start : start + chunk_size], dtype=np.float32)
            faiss.normalize_L2(chunk)
            index.add(chunk)
        return index

    def _load_model(self, device: Optional[str], max_seq_length: int):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=device)
        model.max_seq_length = max_seq_length
        return model

    # --------------------------------------------------------------- querying

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised ``float32`` embeddings of ``texts``, shape ``(len(texts), dim)``."""
        if self.instruction is not None:
            texts = [
                QWEN_QUERY_TEMPLATE.format(instruction=self.instruction, text=t)
                for t in texts
            ]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def search(self, text: str) -> dict[int, float]:
        """Return the ``top_k`` closest documents as ``{doc_id: cosine_similarity}``.

        """
        query = self.embed([text])
        scores, ids = self.index.search(query, min(self.top_k, self.n_docs))
        return {int(i): float(s) for s, i in zip(scores[0], ids[0]) if i >= 0}

    # --------------------------------------------------------------- building

    @staticmethod
    def build_database(
        db_dir: str,
        repo_id: str = DEFAULT_REPO_ID,
        model_name: str = DEFAULT_MODEL_NAME,
        fbin_path_in_repo: str = "fbin/base.fbin",
        verify: bool = True,
    ) -> str:
        """Download the document embeddings of ``repo_id`` into ``db_dir``.

        The ``fbin`` file is downloaded as is (resumable) and renamed ``base.fbin``; its
        header is checked against the file size.  With ``verify``, the first rows of
        the repository's parquet split are streamed and compared with the first rows of
        the ``fbin`` file, to make sure that row ``i`` of the file is indeed the document
        with ``id == i``.  Finally ``meta.json`` is written.

        Returns the path of ``meta.json``.
        """
        from huggingface_hub import hf_hub_download

        os.makedirs(db_dir, exist_ok=True)
        fbin_path = os.path.join(db_dir, FBIN_NAME)
        if not _is_valid_fbin(fbin_path):
            downloaded = hf_hub_download(
                repo_id, fbin_path_in_repo, repo_type="dataset", local_dir=db_dir
            )
            if os.path.abspath(downloaded) != os.path.abspath(fbin_path):
                os.replace(downloaded, fbin_path)
                _remove_empty_dirs(os.path.dirname(downloaded), stop=db_dir)

        vectors = read_fbin(fbin_path)
        if verify:
            _verify_against_parquet(repo_id, vectors)
        return write_meta(
            db_dir,
            repo_id=repo_id,
            model_name=model_name,
            n_docs=int(vectors.shape[0]),
            dim=int(vectors.shape[1]),
            source=fbin_path_in_repo,
        )


# ------------------------------------------------------------------- helpers


def read_fbin(path: str) -> np.ndarray:
    """Read-only memmap of the vectors of an fbin file, shape ``(n, dim)`` ``float32``."""
    n, dim = (int(x) for x in np.fromfile(path, dtype=np.int32, count=2))
    expected = FBIN_HEADER_BYTES + n * dim * 4
    actual = os.path.getsize(path)
    if n <= 0 or dim <= 0 or actual != expected:
        raise ValueError(
            f"{path}: header announces {n} x {dim} float32 vectors ({expected} bytes), "
            f"but the file has {actual} bytes"
        )
    return np.memmap(path, dtype=np.float32, mode="r", offset=FBIN_HEADER_BYTES, shape=(n, dim))


def _is_valid_fbin(path: str) -> bool:
    """Whether ``path`` is a complete fbin file (header consistent with the file size)."""
    if not os.path.isfile(path) or os.path.getsize(path) < FBIN_HEADER_BYTES:
        return False
    try:
        read_fbin(path)
    except ValueError:
        return False
    return True


def read_meta(db_dir: str) -> dict:
    with open(os.path.join(db_dir, META_NAME), "rb") as f:
        return orjson.loads(f.read())


def write_meta(db_dir: str, **fields) -> str:
    path = os.path.join(db_dir, META_NAME)
    with open(path, "wb") as f:
        f.write(orjson.dumps(fields, option=orjson.OPT_INDENT_2))
    return path


def _verify_against_parquet(repo_id: str, vectors: np.ndarray, n_rows: int = 5) -> None:
    """Check that the first ``n_rows`` parquet rows have ``id == row`` and match ``vectors``."""
    from datasets import load_dataset

    stream = load_dataset(repo_id, split="train", streaming=True)
    for row_nr, row in enumerate(stream):
        if row_nr >= n_rows:
            break
        if int(row["id"]) != row_nr:
            raise ValueError(
                f"{repo_id}: parquet row {row_nr} has id {row['id']}; row index and "
                "document id must coincide"
            )
        if not np.allclose(np.asarray(row["embedding"], dtype=np.float32), vectors[row_nr], atol=1e-3):
            raise ValueError(f"{repo_id}: fbin row {row_nr} differs from the parquet embedding")


def _remove_empty_dirs(path: str, stop: str) -> None:
    stop = os.path.abspath(stop)
    while os.path.abspath(path) != stop and os.path.isdir(path) and not os.listdir(path):
        os.rmdir(path)
        path = os.path.dirname(path)


# ----------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="download a database of document embeddings")
    build.add_argument("db_dir", help="destination directory")
    build.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    build.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    build.add_argument(
        "--no-verify", action="store_true", help="skip the check against the parquet rows"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        path = DenseRetrievalDetector.build_database(
            args.db_dir, args.repo_id, args.model_name, verify=not args.no_verify
        )
        print(f"Database ready: {path}")


if __name__ == "__main__":
    main()
