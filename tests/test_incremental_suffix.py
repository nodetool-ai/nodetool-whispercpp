import pytest

from nodetool.nodes.whispercpp.whispercpp import compute_incremental_suffix


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("", " hello", " hello"),
        (" hello", "", ""),
        (" hello", " hello world", " world"),
        (" hello world", " world again", " again"),
        (" hello", " goodbye", " goodbye"),
        (" hello world", " world", ""),
    ],
)
def test_compute_incremental_suffix(previous: str, current: str, expected: str) -> None:
    assert compute_incremental_suffix(previous, current) == expected
