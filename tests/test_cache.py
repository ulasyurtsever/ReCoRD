"""Tests for atomic cache I/O.

Verifies that the atomic write leaves a complete archive at the requested
path, including where NumPy would otherwise append its own ``.npz`` extension.
"""

import numpy as np

from record.cache import atomic_savez, is_complete


def test_atomic_savez_roundtrip(tmp_path):
    path = tmp_path / "sample.npz"
    arrays = {
        "a": np.arange(12, dtype=np.float16).reshape(3, 4),
        "b": np.array([1, 2, 3], dtype=np.int16),
    }
    atomic_savez(path, **arrays)

    assert path.exists()
    assert path.stat().st_size > 0, "archive must not be empty at the target path"
    assert is_complete(path)

    loaded = np.load(path)
    assert set(loaded.files) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"], arrays["a"])
    np.testing.assert_array_equal(loaded["b"], arrays["b"])


def test_atomic_savez_leaves_no_temp_files(tmp_path):
    path = tmp_path / "sample.npz"
    atomic_savez(path, x=np.zeros(4))
    leftovers = [p for p in tmp_path.iterdir() if p.name != "sample.npz"]
    assert leftovers == []


def test_atomic_savez_overwrites(tmp_path):
    path = tmp_path / "sample.npz"
    atomic_savez(path, x=np.zeros(4))
    atomic_savez(path, x=np.ones(8))
    loaded = np.load(path)
    np.testing.assert_array_equal(loaded["x"], np.ones(8))


def test_is_complete_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.npz"
    path.touch()
    assert not is_complete(path)
    assert not is_complete(tmp_path / "absent.npz")
