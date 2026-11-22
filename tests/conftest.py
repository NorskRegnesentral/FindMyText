import types

import psutil
import pytest

from findmytext import index_builder


@pytest.fixture(autouse=True)
def single_cpu_by_default(monkeypatch):
    """Make the auto-detected worker count 1, so that Indexer()/Merger() with
    default arguments do not spawn one process per CPU of the test machine."""
    fake = types.SimpleNamespace(
        Process=lambda: types.SimpleNamespace(cpu_affinity=lambda: [0]),
        cpu_count=lambda: 1,
        virtual_memory=psutil.virtual_memory,
    )
    monkeypatch.setattr(index_builder, "psutil", fake)
