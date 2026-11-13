from pathlib import Path

import pytest


def _cpu_has_avx() -> bool:
    try:
        return " avx " in f" {Path('/proc/cpuinfo').read_text()} ".replace("\n", " ")
    except OSError:
        return True  # non-Linux: assume jaxlib works


def test_import_coilgen():
    import coilgen

    assert coilgen.__version__


@pytest.mark.skipif(not _cpu_has_avx(), reason="jaxlib wheels require AVX (missing on this CPU)")
def test_import_upstream():
    import coilstellaration
    from coilstellaration.benchmark import scoring

    assert coilstellaration is not None
    assert scoring is not None
