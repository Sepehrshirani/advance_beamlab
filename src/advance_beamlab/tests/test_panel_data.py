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
import base64
import gzip
import itertools
import json
import re
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


def test_trials_map_to_the_stated_signal_to_noise(header, constants):
    """Averaging N trials buys a factor of sqrt(N), which is the point of the axis."""
    # Asserted against the tool as well as internally, so the axis cannot drift.
    assert header["single_trial_snr"] == constants["SINGLE_TRIAL_SNR"]
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


def test_region_layouts_are_bilateral_pairs(header):
    regions = {lay["key"] for lay in header["layouts"] if lay["kind"] != "geometric"}
    assert regions
    for scene in header["scenes"]:
        if scene["requested"]["layout"] in regions:
            assert len(scene["sources"]) == 2
            # A bilateral pair is centimetres apart, not millimetres.
            assert scene["separation"] > 0.015


def test_the_hippocampus_is_really_in_the_source_space(header):
    """It is not on the cortical surface, so it has to come from the aseg.

    Without a mixed source space this layout could only ever be a cortical
    stand-in, which is what it was before. The check is that the layout exists,
    is subcortical, and its two sources are a plausible bilateral pair.
    """
    keys = {lay["key"]: lay for lay in header["layouts"]}
    assert "hippocampus" in keys
    assert keys["hippocampus"]["kind"] == "subcortical"
    scene = next(
        s for s in header["scenes"] if s["requested"]["layout"] == "hippocampus"
    )
    assert len(scene["sources"]) == 2
    assert 0.02 < scene["separation"] < 0.09


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

    # And the matched model has to show the inverse crime it is named for, which
    # was collected here and then never looked at.
    matched = np.array(by_head["matched"])
    assert (matched < 0.5).mean() > 0.4
    assert np.median(matched) < np.median(realistic)

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


BLOCKS = (
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
)

DTYPES = {"int16": np.int16, "int8": np.int8, "uint8": np.uint8}


@pytest.fixture(scope="module")
def payload():
    """The decoded binary blob, read exactly as the browser reads it.

    Nothing used to open this at all, so the several megabytes the panel
    actually draws -- every localiser map, waveform and marker position -- were
    unverified by the suite.
    """
    text = DATA.read_text()
    marker = 'window.CONSTRAINT_PANEL.blob = "'
    blob = text[text.index(marker) + len(marker) :].split('"', 1)[0]
    return gzip.decompress(base64.b64decode(blob))


def view(header, payload, key):
    """One block, as the typed array the panel builds over it."""
    entry = header[key]
    return np.frombuffer(
        payload,
        dtype=DTYPES[entry["dtype"]],
        count=entry["length"],
        offset=entry["offset"],
    )


def test_every_block_decodes_within_the_payload(header, payload):
    """A block running past the end, or overlapping its neighbour, is silent."""
    spans = []
    for key in BLOCKS:
        entry = header[key]
        assert entry["dtype"] in DTYPES, key
        width = np.dtype(DTYPES[entry["dtype"]]).itemsize
        assert entry["offset"] % width == 0, key
        end = entry["offset"] + entry["length"] * width
        assert end <= len(payload), key
        assert len(view(header, payload, key)) == entry["length"], key
        spans.append((entry["offset"], end, key))
    spans.sort()
    for (_, end, first), (start, _, second) in itertools.pairwise(spans):
        assert end <= start, f"{first} overlaps {second}"


def test_no_stored_array_is_saturated(header, payload):
    """Saturation is silent and changes what the panel draws.

    Reconstructions were scaled by the truth's own peak, so any that overshot
    were flattened to exactly the truth's amplitude and became indistinguishable
    from a perfect recovery: 53 per cent of the one-trial traces were clipped.
    The mix block saturated int16 in 67 of 816 scenes, so the recording the
    browser rebuilt was not the one that had been simulated.

    Every block is scaled by its own peak, so that peak lands exactly on the
    limit -- that is the scale working, and a rounded peak stays there for a
    sample or two. Clipping is different: it pins a long stretch of the trace to
    one value and flattens whatever shape it had.
    """
    for key, limit, width in (
        ("true_tcs", 127, header["n_times"]),
        ("reconstructed", 127, header["wave_times"]),
        ("noise", 127, header["n_times"]),
        ("mix", 32767, None),
        ("topography", 32767, None),
        ("sensor_pos", 32767, None),
    ):
        values = view(header, payload, key).astype(np.int32)
        assert np.abs(values).max() <= limit, key
        traces = values.reshape(-1, width) if width else values[None]
        pinned = (np.abs(traces) == limit).sum(axis=1)
        assert pinned.max() <= 8, f"{key}: {pinned.max()} samples pinned in one trace"
        share = pinned.sum() / values.size
        assert share < 0.01, f"{key}: {share:.1%} of samples sit at the limit"


def test_the_localiser_maps_use_their_whole_range(header, payload):
    maps = view(header, payload, "maps").reshape(
        len(header["results"]), header["n_sources"]
    )
    assert maps.min(axis=1).max() == 0
    assert maps.max(axis=1).min() == 255


def test_the_recording_rebuilds_from_its_ingredients(header, payload):
    """The panel does not store the recording; it recombines it."""
    n_ch, n_t = header["n_channels"], header["n_times"]
    counts = [len(s["sources"]) for s in header["scenes"]]
    true_tcs = view(header, payload, "true_tcs").astype(float)
    mix = view(header, payload, "mix").astype(float)
    noise = view(header, payload, "noise").astype(float).reshape(n_ch, n_t)
    b, m = header["byte_scale"], header["mix_scale"]
    t_off = m_off = 0
    peaks = []
    for i, n in enumerate(counts):
        block = true_tcs[t_off : t_off + n * n_t].reshape(n, n_t) / b
        coeff = mix[m_off : m_off + n_ch * n].reshape(n_ch, n) / m
        rebuilt = coeff @ block + header["noise_gain"][i] * noise / b
        peaks.append(float(np.abs(rebuilt).max()))
        t_off += n * n_t
        m_off += n_ch * n
    # Every scene, not the loudest one. Taking a maximum over the grid meant a
    # single intact scene satisfied the lower bound while the other 815 could
    # have rebuilt as silence and nothing would have said so.
    peaks = np.array(peaks)
    assert len(peaks) == len(header["scenes"])
    assert peaks.min() > 0.5, f"quietest scene rebuilds to {peaks.min():.3f}"
    assert peaks.max() <= 1.05, f"loudest scene rebuilds to {peaks.max():.3f}"


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


def test_the_model_describes_itself(header):
    """The panel states its own provenance, so the numbers have to be real."""
    model = header["model"]
    assert model["channels"] == header["n_channels"]
    assert model["n_scan"] == header["n_sources"]
    assert model["n_cortical"] + model["n_subcortical"] == model["n_scan"]
    # The truth is drawn from a finer forward than the one being scanned; that
    # difference is what the realistic head model rests on.
    assert model["n_truth"] > model["n_scan"]
    assert model["orientation"] == "fixed"
    assert model["sfreq"] == header["sfreq"]
    assert 2.0 < model["cortical_spacing_mm"] < 20.0
    assert any("Hippocampus" in name for name in model["subcortical_structures"])


def test_the_amplitude_readout_is_sane_over_the_whole_grid(header):
    """The defect a reader caught was invisible to this suite.

    ``amplitude_ratio`` used to be the output's root mean square over the
    truth's, which at one averaged trial ran to fifteen. Only one assertion read
    the field at all and it skipped the low-signal half of the grid entirely.
    """
    ratios = np.array([v for r in header["results"] for v in r["amplitude_ratio"]])
    assert np.isfinite(ratios).all()
    # A filter cannot deliver several times the source it is pointed at.
    assert np.abs(ratios).max() < 3.0
    # And the low-signal half has to be covered, since that is where it broke.
    low = min(header["trials"])
    quiet = np.array(
        [
            v
            for r in header["results"]
            for v in r["amplitude_ratio"]
            if header["scenes"][r["scene"]]["requested"]["trials"] == low
        ]
    )
    assert quiet.size > 0
    assert np.abs(quiet).max() < 3.0


def test_the_constraint_table_explains_the_amplitude(header):
    """The readout is derived from the gains, so the two must agree."""
    checked = 0
    for result in header["results"]:
        scene = header["scenes"][result["scene"]]
        if len(scene["sources"]) != 2 or scene["requested"]["head"] != "matched":
            continue
        gains = np.array(result["gains"])
        expected = gains[0, 0] + scene["correlation"] * gains[0, 1]
        assert result["amplitude_ratio"][0] == pytest.approx(expected, abs=0.05)
        checked += 1
    assert checked > 50


# The size placeholders the browser knows how to substitute. Anything else in a
# stored size string reaches the page as a literal "{Q}".
PLACEHOLDERS = {"C", "V", "T", "n", "n2", "K", "q", "q2"}


def test_every_method_describes_its_own_parameters(header):
    """The sizes table used to be the same for all four methods.

    It was written for LCMV, so switching to MCMV, ReciPSIICOS or ABMC changed
    the equation above it and left every term those methods introduce -- the
    inverted matrix, the projection space, the template, the iteration counts --
    with no size given anywhere.
    """
    algorithms = header["model"]["algorithms"]
    assert set(algorithms) == set(header["methods"])
    for method, entry in algorithms.items():
        assert entry["summary"].strip(), method
        assert len(entry["rows"]) >= 3, method
        for row in entry["rows"]:
            symbol, what, size, note = row
            assert symbol.strip() and what.strip() and size.strip() and note.strip()


def test_no_size_placeholder_is_left_unresolved(header):
    """A typo here ships as a literal brace in the page, silently."""
    algorithms = header["model"]["algorithms"]
    used = set()
    for entry in algorithms.values():
        for text in [entry["summary"]] + [f"{r[2]} {r[3]}" for r in entry["rows"]]:
            used.update(re.findall(r"\{(\w+)\}", text))
    assert used <= PLACEHOLDERS, f"unknown placeholders: {sorted(used - PLACEHOLDERS)}"


def test_the_projection_rank_is_reported_with_the_space_it_lives_in(header):
    """K* means nothing without q: it is a rank out of q squared, not out of q."""
    model = header["model"]
    rank, virtual = model["recipsiicos_rank"], model["recipsiicos_virtual"]
    assert 0 < rank <= virtual**2
    assert 0 < virtual <= model["channels"]


def test_the_build_takes_its_rank_from_a_real_scene():
    """The rank must be selected on the covariance the scenes actually use.

    Calling ``recipsiicos_rank_curve`` directly here is the trap: without a
    noise covariance the curve reduces the array to a different number of
    virtual sensors from the filter it feeds, so K* is picked out of one q^2 and
    spent out of another. On this forward that was 80 out of 2401 against 80 out
    of 6084. Going through ``constraint_demo`` makes the two the same object by
    construction, so the build is required to do that.
    """
    tree = ast.parse(TOOL.read_text())
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "recipsiicos_rank"
    )
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    names = {node.func.id for node in calls}
    # Scoped to the helper. Looking at the whole file passed no matter what this
    # function did, because _one_scene calls constraint_demo for every scene.
    assert "recipsiicos_rank_curve" not in names
    assert "constraint_demo" in names

    # And the scene it runs has to be representative in every argument that
    # reaches the covariance, since that is what sets q. Omitting true_forward
    # alone moved q from 75 to 78 and the rank was selected for the wrong space
    # a second time.
    demo_call = next(c for c in calls if c.func.id == "constraint_demo")
    passed = {kw.arg for kw in demo_call.keywords}
    scene_call = next(
        node
        for node in ast.walk(
            next(
                f
                for f in ast.walk(tree)
                if isinstance(f, ast.FunctionDef) and f.name == "_one_scene"
            )
        )
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "constraint_demo"
    )
    required = {kw.arg for kw in scene_call.keywords} - {"method", "recipsiicos_rank"}
    assert required <= passed, f"representative scene omits {sorted(required - passed)}"


def test_the_recorded_half_is_present_and_self_consistent(header):
    """The recorded scenes have to index the arrays that describe them."""
    scenes = header["real_scenes"]
    results = header["real_results"]
    assert scenes and results
    assert len(results) == len(scenes) * len(header["methods"])
    n_rec = header["real_n_recordings"]
    for scene in scenes:
        assert 0 <= scene["recording"] < n_rec
        assert len(scene["sources"]) == 2
        assert 0 < scene["trials"] <= scene["available"]
        assert len(scene["reference"]) == len(scene["reference_gof"])
        for gof in scene["reference_gof"]:
            assert 0 <= gof <= 100
    seen = {(r["scene"], r["method"]) for r in results}
    assert len(seen) == len(results)


def test_the_recorded_constraint_table_is_distortionless(header):
    """Whatever else a recording does, the diagonal is what the method pins."""
    checked = 0
    for result in header["real_results"]:
        for i in range(len(result["gains"])):
            # Element by element. A list compared against a scalar approx is
            # simply False, so the whole assertion fired on correct data.
            assert result["gains"][i][i] == pytest.approx(1.0, abs=1e-3), (
                f"{result['method']} filter {i}"
            )
            checked += 1
    assert checked == 2 * len(header["real_results"])


def test_the_recording_shows_the_cancellation_it_was_added_for(header):
    """A recorded half that does not reproduce the effect is not worth shipping.

    The bilateral auditory response is the correlated pair the page is about. On
    the panel's own simulated grid this recording gives an LCMV off-diagonal of
    +0.03 -- no cancellation at all -- which is why the locations are chosen at
    full resolution instead. Averaged over every epoch the effect is unmistakable
    and MCMV removes it, so assert both rather than trusting the build.
    """
    scenes = {s["index"]: s for s in header["real_scenes"]}
    worst_lcmv, matching_mcmv = 0.0, None
    for result in header["real_results"]:
        scene = scenes[result["scene"]]
        if scene["family"] != "auditory" or scene["trials"] < 50:
            continue
        off = abs(result["gains"][0][1])
        if result["method"] == "lcmv" and off > worst_lcmv:
            worst_lcmv = off
            matching_mcmv = scene["index"]
    assert worst_lcmv > 0.5, f"LCMV off-diagonal only reached {worst_lcmv:.3f}"
    mcmv = next(
        r
        for r in header["real_results"]
        if r["scene"] == matching_mcmv and r["method"] == "mcmv"
    )
    assert abs(mcmv["gains"][0][1]) < 1e-3


def test_the_recorded_blocks_are_sized_as_the_header_says(header, payload):
    """A wrong length here draws one scene's recording under another's label."""
    n_ch = header["n_channels"]
    n_t = header["real_n_times"]
    rec = view(header, payload, "real_recording")
    assert rec.size == header["real_n_recordings"] * n_ch * n_t
    maps = view(header, payload, "real_maps")
    assert maps.size == len(header["real_results"]) * header["n_real_sources"]
    pos = view(header, payload, "real_positions")
    assert pos.size == header["n_real_sources"] * 3
    recon = view(header, payload, "real_recon")
    total = sum(
        len(header["real_scenes"][r["scene"]]["sources"])
        for r in header["real_results"]
    )
    assert recon.size == total * n_t
