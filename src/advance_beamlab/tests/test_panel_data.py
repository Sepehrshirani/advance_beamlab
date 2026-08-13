# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

"""The panel's data file is committed, so it can fall out of step with the code.

Rebuilding it takes twenty-five minutes and the ``sample`` dataset, which is too
much to ask of a test run. These checks are the cheap half: they read the
committed file and confirm it still describes the grid the build tool would
produce now, and that the numbers inside it are the ones the methods are
supposed to give.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "doc" / "_static" / "constraint_panel_data.js"
TOOL = ROOT / "tools" / "build_constraint_panel.py"

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
        "N_SOURCES",
        "MORPHOLOGIES",
        "DISPLAY_SAMPLES",
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
    assert tuple(header["source_counts"]) == constants["N_SOURCES"]
    assert tuple(header["morphologies"]) == constants["MORPHOLOGIES"]
    assert header["n_times"] == constants["DISPLAY_SAMPLES"]


def test_every_combination_is_present(header):
    expected = (
        len(header["correlations"])
        * len(header["separations"])
        * len(header["snrs"])
        * len(header["source_counts"])
        * len(header["morphologies"])
    )
    assert len(header["scenes"]) == expected
    assert len(header["results"]) == expected * len(header["methods"])
    seen = {(r["scene"], r["method"]) for r in header["results"]}
    assert len(seen) == len(header["results"])


def test_scenes_carry_the_source_count_they_claim(header):
    for scene in header["scenes"]:
        assert len(scene["sources"]) == scene["requested"]["n"]


def test_array_offsets_are_aligned(header):
    """A misaligned offset makes the browser read the wrong bytes, quietly."""
    for key in (
        "positions",
        "cortex",
        "maps",
        "true_tcs",
        "reconstructed",
        "noise",
        "mix",
        "sensor_pos",
        "topography",
    ):
        entry = header[key]
        assert entry["dtype"] in ("int16", "uint8"), key
        width = 2 if entry["dtype"] == "int16" else 1
        assert entry["offset"] % width == 0, key


def test_variable_length_blocks_have_the_right_totals(header):
    """Scenes hold one to three sources, so these are cumulative, not strided."""
    n_t = header["n_times"]
    n_ch = header["n_channels"]
    counts = [len(s["sources"]) for s in header["scenes"]]
    assert header["true_tcs"]["length"] == sum(n * n_t for n in counts)
    assert header["mix"]["length"] == sum(n_ch * n for n in counts)
    assert header["noise"]["length"] == n_ch * n_t
    assert len(header["noise_gain"]) == len(header["scenes"])
    assert header["reconstructed"]["length"] == sum(
        counts[r["scene"]] * n_t for r in header["results"]
    )


def test_the_topography_belongs_to_the_window_that_is_drawn(header):
    times = header["topography_time"]
    assert len(times) == len(header["scenes"])
    assert all(0 <= t < header["n_times"] for t in times)


def test_the_distortionless_constraint_holds_everywhere(header):
    for result in header["results"]:
        gains = np.array(result["gains"])
        assert gains.shape[0] == gains.shape[1]
        np.testing.assert_allclose(np.diag(gains), 1.0, rtol=1e-3)


def test_lcmv_cancels_where_mcmv_does_not(header):
    """The panel's whole claim, asserted against the data it will draw."""
    highest = max(header["correlations"])
    by = {(r["scene"], r["method"]): r for r in header["results"]}
    checked = 0
    for i, scene in enumerate(header["scenes"]):
        want = scene["requested"]
        if want["corr"] != highest or want["snr"] < 3 or want["n"] != 2:
            continue
        lcmv, mcmv = by[(i, "lcmv")], by[(i, "mcmv")]
        assert abs(lcmv["gains"][0][1]) > 0.5
        assert abs(mcmv["gains"][0][1]) < 1e-5
        assert lcmv["amplitude_ratio"][0] < 0.7 * mcmv["amplitude_ratio"][0]
        checked += 1
    assert checked >= 2


def test_a_single_source_has_nothing_to_cancel(header):
    """The control case: with one source LCMV should keep its amplitude."""
    by = {(r["scene"], r["method"]): r for r in header["results"]}
    checked = 0
    for i, scene in enumerate(header["scenes"]):
        if scene["requested"]["n"] != 1 or scene["requested"]["snr"] < 10:
            continue
        assert by[(i, "lcmv")]["amplitude_ratio"][0] > 0.8
        checked += 1
    assert checked > 0
