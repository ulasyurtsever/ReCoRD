"""The threshold grid and its RECORD_GRID switch (record.grid)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from record.grid import LAMBDA_GRID, N_POINTS, _build_grid, curve_on_grid

REPO = Path(__file__).resolve().parents[1]


def test_default_grid_is_uniform():
    assert LAMBDA_GRID.size == N_POINTS
    np.testing.assert_allclose(LAMBDA_GRID, np.linspace(0, 1, N_POINTS))


def test_logtail_grid_keeps_the_endpoints_and_resolves_the_top():
    g = _build_grid("logtail")
    assert g.size == N_POINTS
    assert g[0] == 0.0 and g[-1] == 1.0
    assert np.all(np.diff(g) > 0)
    # Six decades of cutoff: half the points sit within 1e-3 of lambda = 1,
    # where the uniform grid has a single point.
    assert int((1.0 - g < 1e-3).sum()) >= 400
    assert int((1.0 - LAMBDA_GRID < 1e-3).sum()) == 1
    # The ECDF machinery is grid-agnostic: monotone on either grid.
    scores = np.random.default_rng(0).uniform(0, 1, 1000)
    assert np.all(np.diff(curve_on_grid(scores)) >= 0)


def test_unknown_grid_is_refused():
    try:
        _build_grid("bogus")
    except ValueError:
        return
    raise AssertionError("unknown grid kind accepted")


def test_environment_variable_selects_the_grid_at_import():
    env = dict(os.environ, RECORD_GRID="logtail", PYTHONPATH=str(REPO / "src"))
    out = subprocess.run(
        [sys.executable, "-c",
         "from record.grid import LAMBDA_GRID, GRID_KIND; "
         "print(GRID_KIND, LAMBDA_GRID[1], LAMBDA_GRID[-2])"],
        env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == "logtail"
    assert float(out[1]) > 0.01           # first step is 1 - 10^(-6/1000)
    assert 1.0 - float(out[2]) < 2e-6     # second-to-last point is 1e-6 from 1
