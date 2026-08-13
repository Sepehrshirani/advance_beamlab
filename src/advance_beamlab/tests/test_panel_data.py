# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

"""The panel's data file is committed, so it can fall out of step with the code.

Rebuilding it takes about an hour and the ``sample`` dataset, which is too much
to ask of a test run. These checks are the cheap half: they read the committed
file and confirm it still describes the grid the build tool would produce now,
and that the numbers inside it are the ones the methods are supposed to give.
"""

import ast
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
    """The grid the build tool is currently configured to produce.

    Read from the source rather than imported, so the test does not pull in the
    build tool's dependencies. ``LAYOUTS`` holds ``dict(...)`` calls rather than
    literals, so its keys are walked out of the syntax tree by hand.
    """
    tree = ast.parse(TOOL.read_text())
    wanted = {
        "METHODS",
        "CORRELATIONS",
        "MORPHOLOGIES",
        "TRIALS",
        "HEAD_MODELS",
        "DISPLAY_SAMPLES",
        "WAVE_SAMPLES",
        "SINGLE_TRIAL_SNR",
    }
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in wanted:
            out[name] = ast.literal_eval(node.value)
        elif name == "LAYOUTS":
            out["LAYOUT_KEYS"] = [
                next(
                    ast.literal_eval(kw.value)
                    for kw in call.keywords
                    if kw.arg == "key"
                )
                for call in node.value.elts
            ]
    return out


def test_committed_data_matches_the_current_grid(header, constants):
    """A changed grid without a rebuild would silently mislabel every control."""
    assert tuple(header["methods"]) == constants["METHODS"]
    assert tuple(header["correlations"]) == constants["CORRELATIONS"]
    assert tuple(header["morphologies"]) == constants["MORPHOLOGIES"]
    assert tuple(header["trials"]) == constants["TRIALS"]
    assert tuple(header["head_models"]) == constants["HEAD_MODELS"]
    assert header["n_times"] == constants["DISPLAY_SAMPLES"]
    assert header["wave_times"] == constants["WAVE_SAMPLES"]
    assert [lay["key"] for lay in header["layouts"]] == constants["LAYOUT_KEYS"]


def test_trials_map_to_the_stated_signal_to_noise(header):
    """Averaging N trials buys a factor of sqrt(N), which is the point of the axis."""
    for trials, snr in zip(header["trials"], header["snrs"], strict=True):
        assert snr == pytest.approx(
            header["single_trial_snr"] * np.sqrt(trials), abs=1e-3
        )


def test_every_combination_is_present(header):
    expected = (
        len(header["layouts"])
        * len(header["morphologies"])
        * len(header["correlations"])
        * len(header["trials"])
        * len(header["head_models"])
    )
    assert len(header["scenes"]) == expected
    assert len(header["results"]) == expected * len(header["methods"])
    seen = {(r["scene"], r["method"]) for r in header["results"]}
    assert len(seen) == len(header["results"])


def test_anatomical_layouts_are_bilateral_pairs(header):
    anatomical = {
        lay["key"] for lay in header["layouts"] if lay["kind"] == "anatomical"
    }
    assert anatomical
    for scene in header["scenes"]:
        if scene["requested"]["layout"] in anatomical:
            assert len(scene["sources"]) == 2
            # A bilateral pair is centimetres apart, not millimetres.
            assert scene["separation"] > 0.03


def test_each_head_model_does_its_own_job(header):
    """The toggle only earns its place if the two modes really differ.

    The matched model puts the sources on scanned grid points, which is the only
    way to see what the constraint itself does, at the price of an inverse crime:
    a single source is then localised exactly. The realistic model takes the
    truth from a finer forward, which gives a real localisation error but costs
    every method its amplitude equally, so the constraint contrast disappears.
    Both properties are asserted, because the panel claims both.
    """
    scene_head = [s["requested"]["head"] for s in header["scenes"]]
    by_head = {h: [] for h in header["head_models"]}
    for result in header["results"]:
        by_head[scene_head[result["scene"]]].extend(result["peak_errors"])

    realistic = np.array(by_head["realistic"])
    assert (realistic < 0.5).mean() < 0.1
    assert 1.0 < np.median(realistic) < 40.0

    # The contrast the panel is built on, checked where it is supposed to hold.
    highest = max(header["correlations"])
    most = max(header["trials"])
    by = {(r["scene"], r["method"]): r for r in header["results"]}
    ratios = []
    for i, scene in enumerate(header["scenes"]):
        want = scene["requested"]
        if want["head"] != "matched" or want["corr"] != highest:
            continue
        if want["trials"] != most or len(scene["sources"]) != 2:
            continue
        lcmv, mcmv = by[(i, "lcmv")], by[(i, "mcmv")]
        ratios.append(
            mcmv["amplitude_ratio"][0] / max(lcmv["amplitude_ratio"][0], 1e-6)
        )
    assert ratios
    assert max(ratios) > 1.3


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
        "true_positions",
    ):
        entry = header[key]
        assert entry["dtype"] in ("int16", "uint8"), key
        width = 2 if entry["dtype"] == "int16" else 1
        assert entry["offset"] % width == 0, key


def test_variable_length_blocks_have_the_right_totals(header):
    """Layouts hold one to three sources, so these are cumulative, not strided."""
    n_t, n_w = header["n_times"], header["wave_times"]
    n_ch = header["n_channels"]
    counts = [len(s["sources"]) for s in header["scenes"]]
    assert header["true_tcs"]["length"] == sum(n * n_t for n in counts)
    assert header["mix"]["length"] == sum(n_ch * n for n in counts)
    assert header["true_positions"]["length"] == sum(n * 3 for n in counts)
    assert header["noise"]["length"] == n_ch * n_t
    assert len(header["noise_gain"]) == len(header["scenes"])
    assert header["reconstructed"]["length"] == sum(
        counts[r["scene"]] * n_w for r in header["results"]
    )


def test_the_distortionless_constraint_holds_everywhere(header):
    for result in header["results"]:
        gains = np.array(result["gains"])
        assert gains.shape[0] == gains.shape[1]
        np.testing.assert_allclose(np.diag(gains), 1.0, rtol=1e-3)


def test_lcmv_cancels_where_mcmv_does_not(header):
    """The panel's whole claim, asserted against the data it will draw.

    Restricted to the matched head model, which is where the claim is made. With
    a mismatched model every method loses amplitude for the same reason and the
    off-diagonal is no longer the thing driving it; that regime is covered by
    ``test_each_head_model_does_its_own_job``.
    """
    highest = max(header["correlations"])
    by = {(r["scene"], r["method"]): r for r in header["results"]}
    checked = 0
    for i, scene in enumerate(header["scenes"]):
        want = scene["requested"]
        if want["head"] != "matched" or want["corr"] != highest:
            continue
        if want["trials"] < 10 or len(scene["sources"]) != 2:
            continue
        lcmv, mcmv = by[(i, "lcmv")], by[(i, "mcmv")]
        assert abs(lcmv["gains"][0][1]) > 0.4
        assert abs(mcmv["gains"][0][1]) < 1e-5
        checked += 1
    assert checked >= 4
