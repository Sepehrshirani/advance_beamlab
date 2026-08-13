# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

"""The panel's data file is committed, so it can fall out of step with the code.

Rebuilding it takes six minutes and the ``sample`` dataset, which is too much to
ask of a test run. These checks are the cheap half: they read the committed file
and confirm it still describes the grid the build tool would produce now, and
that the numbers inside it are the ones the methods are supposed to give.
"""

import json
from pathlib import Path

import numpy as np
import pytest

DATA = (
    Path(__file__).resolve().parents[3] / "doc" / "_static" / "constraint_panel_data.js"
)
TOOL = Path(__file__).resolve().parents[3] / "tools" / "build_constraint_panel.py"

pytestmark = pytest.mark.skipif(
    not DATA.exists() or not TOOL.exists(),
    reason="documentation sources are not part of an installed package",
)


@pytest.fixture(scope="module")
def header():
    """The JSON header of the committed panel data, without the payload."""
    text = DATA.read_text()
    start = text.index("{")
    end = text.index("\n", start)
    return json.loads(text[start : end - 1])


@pytest.fixture(scope="module")
def constants():
    """The grid the build tool is currently configured to produce."""
    source = TOOL.read_text()
    out = {}
    for name in (
        "METHODS",
        "CORRELATIONS",
        "SEPARATIONS",
        "SNRS",
        "DISPLAY_SAMPLES",
        "N_SENSOR_TRACES",
    ):
        line = next(ln for ln in source.splitlines() if ln.startswith(f"{name} = "))
        out[name] = eval(line.split("=", 1)[1].strip())  # noqa: S307
    return out


def test_committed_data_matches_the_current_grid(header, constants):
    """A changed grid without a rebuild would silently mislabel every control."""
    assert tuple(header["methods"]) == constants["METHODS"]
    assert tuple(header["correlations"]) == constants["CORRELATIONS"]
    assert tuple(header["separations"]) == constants["SEPARATIONS"]
    assert tuple(header["snrs"]) == constants["SNRS"]
    assert header["n_times"] == constants["DISPLAY_SAMPLES"]
    assert header["n_sensor_traces"] == constants["N_SENSOR_TRACES"]


def test_every_combination_is_present(header):
    n_scenes = (
        len(header["correlations"]) * len(header["separations"]) * len(header["snrs"])
    )
    assert len(header["scenes"]) == n_scenes
    assert len(header["results"]) == n_scenes * len(header["methods"])
    seen = {(r["scene"], r["method"]) for r in header["results"]}
    assert len(seen) == len(header["results"])


def test_array_offsets_are_aligned(header):
    """A misaligned offset makes the browser read the wrong bytes, quietly."""
    for key in (
        "positions",
        "cortex",
        "maps",
        "true_tcs",
        "reconstructed",
        "sensor",
        "sensor_pos",
        "topography",
    ):
        entry = header[key]
        assert entry["dtype"] in ("int16", "uint8"), key
        width = 2 if entry["dtype"] == "int16" else 1
        assert entry["offset"] % width == 0, key


def test_the_topography_belongs_to_the_window_that_is_drawn(header):
    """The field map and the marker on the traces must be the same instant."""
    times = header["topography_time"]
    assert len(times) == len(header["scenes"])
    assert all(0 <= t < header["n_times"] for t in times)


def test_peaks_reproduce_the_reported_errors(header):
    """The drawn estimate has to be the one the error was measured against."""
    for result in header["results"]:
        assert len(result["peaks"]) == 2
        assert all(0 <= p < header["n_sources"] for p in result["peaks"])


def test_the_distortionless_constraint_holds_everywhere(header):
    for result in header["results"]:
        np.testing.assert_allclose(np.diag(np.array(result["gains"])), 1.0, rtol=1e-3)


def test_lcmv_cancels_where_mcmv_does_not(header):
    """The panel's whole claim, asserted against the data it will draw."""
    scenes = header["scenes"]
    highest = max(header["correlations"])
    by = {(r["scene"], r["method"]): r for r in header["results"]}
    checked = 0
    for i, scene in enumerate(scenes):
        if scene["requested"]["corr"] != highest or scene["requested"]["snr"] < 3:
            continue
        lcmv, mcmv = by[(i, "lcmv")], by[(i, "mcmv")]
        assert abs(lcmv["gains"][0][1]) > 0.5
        assert abs(mcmv["gains"][0][1]) < 1e-5
        assert lcmv["amplitude_ratio"][0] < 0.6 * mcmv["amplitude_ratio"][0]
        checked += 1
    assert checked >= 4
