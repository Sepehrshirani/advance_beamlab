"""Precompute the constraint panel that ships with the documentation.

The panel lets a reader place one, two or three sources on a real cortex, choose
how correlated they are and what they look like, watch the recording they make,
and see how each beamformer localises and reconstructs them. The point of it is
the constraint table: LCMV pins its own gain to one and leaves the gain at the
*other* source free, and when the two are
correlated the value that minimises output power is a large negative one, which
cancels the target along with its partner. MCMV forbids exactly that. Reading
that off a formula is hard; watching the number move is not.

GitHub Pages is static, so every configuration is computed here and the page only
displays them. Run this whenever the engine or the grid changes:

    python tools/build_constraint_panel.py doc/_static/constraint_panel_data.js

Everything comes from :func:`advance_beamlab.constraint_demo`, so the panel and
the local :func:`advance_beamlab.constraint_explorer` cannot drift apart.
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import argparse
import base64
import gzip
import json
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np

from advance_beamlab import constraint_demo

# The axes a reader can move. Every combination is precomputed.
METHODS = ("lcmv", "mcmv", "recipsiicos", "abmc")
# ABMC is steered to whatever waveform it is told to look for, so that choice is
# the method's main control and belongs on the page rather than buried in the
# build. "truth" is the target's own time course, which no experiment has and
# which is here to show what a perfect template would be worth; "matched" is an
# independent draw from the same band, which is what someone looking for
# hippocampal theta would really supply; "mismatched" is a wrong guess.
TEMPLATES = ("truth", "matched", "mismatched")
REAL_TEMPLATES = ("estimate", "mismatched")


def templates_for(method, choices):
    """Return the template settings a method needs; one pass for all but ABMC."""
    return choices if method == "abmc" else (None,)


CORRELATIONS = (0.0, 0.9, 0.99)
MORPHOLOGIES = ("theta", "alpha", "beta", "transient")

# Whether the beamformer's model is right. "matched" puts the sources on points
# the beamformer scans, which is the only way to see what the constraint itself
# does; "realistic" takes them from a finer forward so they sit where no method
# can scan them, which is what real data looks like. The difference between the
# two is the panel's clearest statement of how much a head model is worth.
HEAD_MODELS = ("matched", "realistic")

# Subcortical structures to add to the source space. The cortical surface has no
# hippocampus, and hippocampal theta is one of the things people most want to
# ask about, so the source space is mixed: the surface plus discrete sources
# inside these aseg labels.
SUBCORTICAL = (
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Thalamus-Proper",
    "Right-Thalamus-Proper",
)

# Where the sources sit, in three groups.
#
# The geometric layouts are the abstract cases: they control separation directly
# and are the cleanest way to see what a constraint does. The bilateral ones put
# the same structure in both hemispheres. The circuits are pairs that
# computational psychiatry and clinical neurophysiology actually study, so that
# a reader can ask their own question rather than a synthetic one -- hippocampal
# to prefrontal theta, amygdala to ventromedial prefrontal coupling,
# thalamocortical drive, the frontoparietal and salience networks and the
# default mode. Each is a left-hemisphere pair unless it says otherwise, because
# a within-hemisphere circuit is the usual object of study.
LAYOUTS = (
    dict(
        key="single",
        label="one source",
        group="geometry",
        kind="geometric",
        n=1,
        sep=0.0,
    ),
    dict(
        key="near",
        label="pair, 2 cm apart",
        group="geometry",
        kind="geometric",
        n=2,
        sep=0.02,
    ),
    dict(
        key="far",
        label="pair, 6 cm apart",
        group="geometry",
        kind="geometric",
        n=2,
        sep=0.06,
    ),
    dict(
        key="triple",
        label="three sources",
        group="geometry",
        kind="geometric",
        n=3,
        sep=0.04,
    ),
    dict(
        key="hippocampus",
        label="hippocampus",
        group="bilateral",
        labels=("Left-Hippocampus", "Right-Hippocampus"),
    ),
    dict(
        key="amygdala",
        label="amygdala",
        group="bilateral",
        labels=("Left-Amygdala", "Right-Amygdala"),
    ),
    dict(
        key="thalamus",
        label="thalamus",
        group="bilateral",
        labels=("Left-Thalamus-Proper", "Right-Thalamus-Proper"),
    ),
    dict(
        key="auditory",
        label="auditory cortex",
        group="bilateral",
        labels=("transversetemporal-lh", "transversetemporal-rh"),
    ),
    dict(
        key="visual",
        label="visual cortex",
        group="bilateral",
        labels=("pericalcarine-lh", "pericalcarine-rh"),
    ),
    dict(
        key="motor",
        label="motor cortex",
        group="bilateral",
        labels=("precentral-lh", "precentral-rh"),
    ),
    dict(
        key="dlpfc",
        label="lateral PFC",
        group="bilateral",
        labels=("rostralmiddlefrontal-lh", "rostralmiddlefrontal-rh"),
    ),
    dict(
        key="hpc-mpfc",
        label="hippocampus \u2013 mPFC",
        group="circuit",
        labels=("Left-Hippocampus", "rostralanteriorcingulate-lh"),
    ),
    dict(
        key="amy-vmpfc",
        label="amygdala \u2013 vmPFC",
        group="circuit",
        labels=("Left-Amygdala", "medialorbitofrontal-lh"),
    ),
    dict(
        key="thal-motor",
        label="thalamus \u2013 motor",
        group="circuit",
        labels=("Left-Thalamus-Proper", "precentral-lh"),
    ),
    dict(
        key="fronto-parietal",
        label="lateral PFC \u2013 parietal",
        group="circuit",
        labels=("rostralmiddlefrontal-lh", "superiorparietal-lh"),
    ),
    dict(
        key="salience",
        label="ACC \u2013 insula",
        group="circuit",
        labels=("caudalanteriorcingulate-lh", "insula-lh"),
    ),
    dict(
        key="default-mode",
        label="mPFC \u2013 precuneus",
        group="circuit",
        labels=("rostralanteriorcingulate-lh", "precuneus-lh"),
    ),
)

# The signal-to-noise axis, expressed as the thing an experimenter controls.
# A single trial of an evoked MEG response sits well below unit sensor SNR;
# averaging N trials buys a factor of sqrt(N).
SINGLE_TRIAL_SNR = 0.2
TRIALS = (1, 100)
SNRS = tuple(round(SINGLE_TRIAL_SNR * t**0.5, 4) for t in TRIALS)

# Ten seconds at 200 Hz. The covariance is estimated from these samples, and at
# 203 gradiometers anything shorter is rank deficient: at 200 samples MNE reports
# both "too few samples" and a negative smallest eigenvalue, and the numbers move
# with it (ReciPSIICOS returned an off-diagonal gain of -1.19 at 1000 samples and
# -0.88 at 2000). A panel whose whole content is the numbers cannot be built on
# that, so the simulation is long even though only a slice of it is drawn.
N_TIMES = 2000
# What the sensor viewer can scroll over, and the shorter window the waveform
# panel draws. Two and a half seconds rather than one: at 250 samples the traces
# were too short to read as anything but flat lines.
DISPLAY_SAMPLES = 500
WAVE_SAMPLES = 150
SFREQ = 200.0
SEED = 0
# Every 14th source-space vertex, which leaves 586 cortical points at a median
# spacing of 7.5 mm. That keeps ReciPSIICOS, whose Gram is quadratic in the
# source count, at about two seconds per configuration while still being a real
# cortex rather than a sphere.
DECIMATION = 14
N_BACKDROP = 4000
# Subcortical sampling. The fine space is sampled densely and the scan grid
# takes every Nth of it, exactly as the cortex is treated. Keeping every
# subcortical source in the scan grid, which is what this did at first, leaves
# the realistic head model with nothing to displace a deep source *to*: the fine
# and scan grids hold the same points, so a quarter of all localisation errors
# came back as exactly zero and picking the hippocampus reproduced the inverse
# crime the realistic setting exists to avoid.
SUBCORTICAL_SPACING = 3.0
SUBCORTICAL_DECIMATION = 5

# Warnings that mean the numbers cannot be trusted. The build stops on these
# rather than printing them into a log nobody reads.
FATAL_WARNINGS = ("Too few samples", "will likely be unstable", "rank estimate")


def sensor_layout(info):
    """Two-dimensional sensor positions, centred and scaled to the unit disc.

    The panel draws a field map from these, which is the view that makes
    "what the sensors record" mean something: twenty stacked traces show that
    there is signal, a topography shows where on the helmet it is.
    """
    layout = mne.channels.find_layout(info)
    order = [layout.names.index(ch) for ch in info["ch_names"]]
    pos = layout.pos[order, :2] + layout.pos[order, 2:4] / 2
    pos = pos - pos.mean(0)
    return pos / np.abs(pos).max()


def build_forward(verbose=True):
    """Return the scan grid, the finer forward behind it, and a backdrop.

    The source space is mixed: the cortical surface plus discrete sources inside
    the subcortical labels of the aseg. Volume sources carry no normal, so a
    fixed-orientation forward cannot be made from them directly; they are
    rebuilt as a discrete source space with an orientation assigned per
    structure, the short principal axis of the label. For the hippocampus that
    approximates the somato-dendritic direction its pyramidal cells are
    organised along. It is a simplification, and the documentation says so.
    """
    data_path = mne.datasets.sample.data_path()
    subjects_dir = data_path / "subjects"
    meg = data_path / "MEG" / "sample"
    bem = str(subjects_dir / "sample" / "bem" / "sample-5120-bem-sol.fif")
    trans = str(meg / "sample_audvis_raw-trans.fif")

    info = mne.io.read_info(meg / "sample_audvis-ave.fif")
    info = mne.pick_info(
        info, mne.pick_types(info, meg="grad", eeg=False, exclude="bads")
    )

    surface = mne.setup_source_space(
        "sample",
        spacing="oct6",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    volume = mne.setup_volume_source_space(
        "sample",
        pos=SUBCORTICAL_SPACING,
        mri="aseg.mgz",
        volume_label=list(SUBCORTICAL),
        subjects_dir=subjects_dir,
        bem=bem,
        add_interpolator=False,
        verbose=False,
    )
    points, normals, fine_groups = [], [], []
    for space in volume:
        pts = space["rr"][space["vertno"]]
        axes = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2]
        points.append(pts)
        normals.append(np.tile(axes[2], (len(pts), 1)))
        fine_groups.append((space["seg_name"], len(pts)))
    discrete = mne.setup_volume_source_space(
        pos=dict(rr=np.concatenate(points), nn=np.concatenate(normals)),
        verbose=False,
    )

    fine = mne.make_forward_solution(
        info,
        trans=trans,
        src=surface + discrete,
        bem=bem,
        meg=True,
        eeg=False,
        n_jobs=-1,
        verbose=False,
    )
    fine = mne.convert_forward_solution(
        fine, force_fixed=True, use_cps=True, verbose=False
    )

    cortex = np.vstack([s["rr"][s["vertno"]] for s in fine["src"][:2]])
    cortex = cortex[:: max(1, len(cortex) // N_BACKDROP)]

    # Decimate the cortex, and decimate the subcortical sources far less, so the
    # structures the layouts name stay properly scannable. Keeping every
    # subcortical source, which is what this did at first, left the realistic
    # head model with nothing to displace a deep source to -- see the comment on
    # SUBCORTICAL_DECIMATION.
    verts = [s["vertno"][::DECIMATION] for s in fine["src"][:2]]
    verts += [s["vertno"][::SUBCORTICAL_DECIMATION] for s in fine["src"][2:]]
    n = sum(len(v) for v in verts)
    stc = mne.MixedSourceEstimate(
        np.ones((n, 1)), verts, tmin=0, tstep=1, subject="sample"
    )
    fwd = mne.forward.restrict_forward_to_stc(fine, stc)

    rr = fwd["source_rr"]
    gaps = [
        np.sort(np.linalg.norm(rr - rr[i], axis=1))[1] * 1000
        for i in range(0, sum(len(v) for v in verts[:2]), 41)
    ]
    if verbose:
        print(
            f"scan grid: {fwd['sol']['data'].shape[1]} sources "
            f"({len(verts[0]) + len(verts[1])} cortical at {np.median(gaps):.1f} mm, "
            f"{len(verts[2])} subcortical); truth from "
            f"{fine['sol']['data'].shape[1]}; {fwd['nchan']} gradiometers"
        )
    # How many of each structure survived into the scan grid, in the same order.
    kept = fwd["src"][2]["vertno"]
    groups, start = [], 0
    for name, count in fine_groups:
        stop = start + count
        groups.append((name, int(((kept >= start) & (kept < stop)).sum())))
        start = stop
    if verbose:
        print(
            "  subcortical in the scan grid: "
            + ", ".join(f"{n} {c}" for n, c in groups)
        )
    model = dict(
        subject="sample",
        bem="single-layer BEM, 5120 triangles",
        channels=int(fwd["nchan"]),
        channel_type="gradiometers",
        n_scan=int(fwd["sol"]["data"].shape[1]),
        n_cortical=int(fwd["src"][0]["nuse"] + fwd["src"][1]["nuse"]),
        n_subcortical=int(fwd["src"][2]["nuse"]),
        cortical_spacing_mm=round(float(np.median(gaps)), 1),
        n_truth=int(fine["sol"]["data"].shape[1]),
        surface="oct-6",
        subcortical_structures=[name for name, _ in groups],
        orientation="fixed",
        sfreq=SFREQ,
        n_times_simulated=N_TIMES,
    )
    return info, fwd, fine, cortex, groups, model


def resolve_layouts(fwd, groups, verbose=True):
    """Grid indices for each layout, from the parcellation or the aseg.

    A layout names its targets by string, and each is resolved wherever it
    lives: ``Left-Hippocampus`` in the subcortical segmentation, ``insula-lh``
    in the cortical parcellation. That lets a circuit pair a deep structure with
    a cortical one, which is what most of them are.
    """
    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    parc = {
        lab.name: lab
        for lab in mne.read_labels_from_annot(
            "sample", "aparc", subjects_dir=subjects_dir, verbose=False
        )
    }
    src = fwd["src"]
    rr = fwd["source_rr"]
    n_cortical = src[0]["nuse"] + src[1]["nuse"]

    subcortical, offset = {}, n_cortical
    for name, count in groups:
        subcortical[name] = np.arange(offset, offset + count)
        offset += count

    def centroid_index(indices):
        """Return the grid point nearest a set's centroid, chosen in space.

        Taking the median of positions *in* the index array instead orders by
        vertex number and lands somewhere unrelated.
        """
        pos = rr[indices]
        return int(indices[np.argmin(np.linalg.norm(pos - pos.mean(0), axis=1))])

    def resolve(name):
        if name in subcortical:
            return centroid_index(subcortical[name]), "subcortical"
        label = parc[name]
        hemi = 0 if label.hemi == "lh" else 1
        base = 0 if hemi == 0 else src[0]["nuse"]
        shared = np.intersect1d(src[hemi]["vertno"], label.vertices)
        if shared.size == 0:
            raise RuntimeError(f"{name} has no grid point after decimation")
        idx = np.array(
            [base + int(np.flatnonzero(src[hemi]["vertno"] == v)[0]) for v in shared]
        )
        pick = centroid_index(idx)
        vertex = src[hemi]["vertno"][pick - base]
        assert vertex in label.vertices, f"{name} representative escaped its label"
        return pick, "cortical"

    resolved = []
    for layout in LAYOUTS:
        entry = dict(layout)
        if layout["group"] == "geometry":
            entry["sources"] = None
        else:
            picks, kinds = zip(*(resolve(n) for n in layout["labels"]), strict=True)
            entry["sources"] = list(picks)
            entry["n"] = len(picks)
            entry["kind"] = "subcortical" if "subcortical" in kinds else "anatomical"
            if verbose:
                d = np.linalg.norm(rr[picks[0]] - rr[picks[1]]) * 100
                print(
                    f"  {layout['group']:>9} {layout['label']:>26}: "
                    f"{list(picks)}, {d:.1f} cm apart"
                )
        resolved.append(entry)
    return resolved


def _one_scene(info, fwd, fine, rank, combo, verbose):
    """All four methods on one scene, with the scene itself returned once."""
    layout, corr, trials, morph, head_model = combo
    snr = SINGLE_TRIAL_SNR * trials**0.5
    # The fine forward is always supplied: it provides the ongoing background
    # activity in both cases, so the two head models see identical interference
    # and differ only in where the sources are.
    off_grid = head_model == "realistic"
    scene, rows = None, []
    for method in METHODS:
        for template in templates_for(method, TEMPLATES):
            t0 = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                demo = constraint_demo(
                    info,
                    fwd,
                    method=method,
                    sources=layout["sources"],
                    n_sources=layout.get("n", 2),
                    separation=layout.get("sep", 0.04),
                    morphology=morph,
                    correlation=corr,
                    snr=snr,
                    n_times=N_TIMES,
                    sfreq=SFREQ,
                    seed=SEED,
                    # The rank curve is a property of the forward, not of the data,
                    # so it is computed once and handed in. Left to default it is
                    # recomputed for every ReciPSIICOS configuration, which cost
                    # about thirty seconds each and dominated the whole build.
                    recipsiicos_rank=rank,
                    true_forward=fine,
                    off_grid_sources=off_grid,
                    template=template if template is not None else "truth",
                )
            fatal = sorted(
                {
                    str(w.message)
                    for w in caught
                    if any(f in str(w.message) for f in FATAL_WARNINGS)
                }
            )
            if fatal:
                raise RuntimeError(
                    f"{method} on layout {layout['key']}, correlation {corr}, "
                    f"{trials} trial(s), {morph} produced a covariance that cannot "
                    "be trusted:\n  " + "\n  ".join(fatal)
                )
            if scene is None:
                scene = dict(
                    sources=[int(v) for v in demo.sources],
                    separation=float(demo.separation),
                    correlation=float(demo.correlation),
                    requested=dict(
                        layout=layout["key"],
                        corr=corr,
                        trials=trials,
                        morph=morph,
                        head=head_model,
                    ),
                    true_tcs=demo.true_tcs,
                    sensor=demo.sensor_data,
                    leadfield=demo.extra["leadfield"],
                    noise_scale=demo.extra["noise_scale"],
                    true_positions=demo.extra["true_positions"],
                )
            else:
                # The comparison is only fair if the data are identical.
                np.testing.assert_allclose(
                    demo.true_tcs, scene["true_tcs"], rtol=0, atol=0
                )
                np.testing.assert_allclose(
                    demo.sensor_data, scene["sensor"], rtol=0, atol=0
                )
                assert list(demo.sources) == scene["sources"]
            # The grid points the map actually peaks at. constraint_demo scores its
            # peak_errors against exactly these, so the panel can draw the estimate
            # the error refers to rather than a different peak of its own choosing.
            peaks = np.argsort(demo.power_map)[::-1][: len(demo.sources)]
            rows.append(
                dict(
                    method=method,
                    template=template,
                    gains=demo.gains.tolist(),
                    amplitude_ratio=demo.amplitude_ratio.tolist(),
                    peak_errors=(demo.peak_errors * 1000).tolist(),
                    peaks=[int(v) for v in peaks],
                    power_map=demo.power_map,
                    reconstructed=demo.reconstructed,
                )
            )
            if verbose:
                n_src = len(demo.sources)
                off = demo.gains[0, 1] if n_src > 1 else float("nan")
                print(
                    f"  {method:>11} {layout['key']:>15} {morph:>9} {head_model:>9} "
                    f"r {corr:+.2f} "
                    f"{trials:>4} trial  off-diag {off:+.3f}  "
                    f"recovered {demo.amplitude_ratio[0]:.3f}  "
                    f"error {demo.peak_errors[0] * 1000:5.1f} mm  "
                    f"[{time.time() - t0:.1f}s]"
                )
    return scene, rows


def recipsiicos_rank(info, fwd, fine, layouts, verbose=True):
    """Return the projection rank and the dimension of the space it lives in.

    The rank is computed once for the whole grid, from one representative scene,
    rather than per configuration: it costs about thirty seconds and it is
    stable across the grid, because every scene draws its interference from the
    same field and differs only in how far that field is scaled, and a scalar on
    the noise covariance cancels out of the whitened leadfield's spectrum.

    Two things about this call are load bearing, and both were got wrong once.

    It has to go through ``constraint_demo`` rather than call the rank curve
    directly. Both the curve and the filter reduce the array to q virtual
    sensors first, and q is a truncated SVD of the *whitened* leadfield, so a
    curve built without the scenes' noise covariance lands on a different q:
    that chose K* = 80 out of q^2 = 2401 and spent it out of 6084.

    And it has to be handed the same forward the scenes get. ``true_forward``
    decides which leadfield the background sources are projected through, so it
    changes the interference, the noise covariance, the whitener and therefore q
    again -- 78 without it against the 75 every scene in the grid actually uses.
    A representative scene has to be representative in every argument that
    reaches the covariance.
    """
    from advance_beamlab._explain import constraint_demo

    layout = next(entry for entry in layouts if entry.get("n", 2) > 1)
    demo = constraint_demo(
        info,
        fwd,
        method="recipsiicos",
        sources=layout["sources"],
        n_sources=layout.get("n", 2),
        separation=layout.get("sep", 0.04),
        morphology="alpha",
        correlation=CORRELATIONS[-1],
        snr=SINGLE_TRIAL_SNR * TRIALS[-1] ** 0.5,
        n_times=N_TIMES,
        sfreq=SFREQ,
        seed=SEED,
        recipsiicos_rank=None,
        true_forward=fine,
        off_grid_sources=False,
        verbose=False,
    )
    k_opt = demo.extra["recipsiicos_rank"]
    virtual = demo.extra["recipsiicos_virtual"]
    if k_opt is None or virtual is None:
        raise RuntimeError("constraint_demo did not report the rank it selected")
    if verbose:
        print(
            f"ReciPSIICOS rank K* = {k_opt} of q^2 = {virtual**2} "
            f"(q = {virtual} virtual sensors), chosen on the scenes' own "
            "covariance and reused for the whole grid"
        )
    return int(k_opt), int(virtual)


def algorithm_parameters():
    """Every quantity each method actually forms, with its size.

    The panel used to print one dimension table for all four methods, which was
    only ever right for LCMV: switching to MCMV, ReciPSIICOS or ABMC changed the
    equation above the table while the table itself stayed put, so the terms the
    chosen method introduces -- the matrix MCMV inverts, the space ReciPSIICOS
    projects in, the template ABMC matches, the iteration counts -- were never
    given a size at all.

    Sizes are written with placeholders the browser fills in from the current
    selection: ``{C}`` channels, ``{V}`` scan points, ``{T}`` samples, ``{n}``
    constrained sources, ``{K}`` projection rank, ``{q}`` virtual sensors and
    ``{q2}`` for q squared. Defaults are read off the functions themselves so
    that this cannot drift away from the code it describes.
    """
    import inspect

    from advance_beamlab._abmc import make_abmc, sbl_covariance
    from advance_beamlab._explain import constraint_demo

    def default(fn, name):
        return inspect.signature(fn).parameters[name].default

    reg = default(constraint_demo, "reg")
    template_p = default(constraint_demo, "template_p")
    return dict(
        lcmv=dict(
            summary=(
                "One scan point at a time. Every filter is solved on its own "
                "and knows only its own location, so the problem is {V} "
                "independent scalar constraints, never a system."
            ),
            rows=[
                [
                    "R<sup>-1</sup>g<sub>i</sub>",
                    "one linear solve per scan point",
                    "{C} &times; 1",
                    "done {V} times, once per scanned point",
                ],
                [
                    "g<sub>i</sub><sup>T</sup>R<sup>-1</sup>g<sub>i</sub>",
                    "the denominator, and the whole constraint",
                    "1 &times; 1",
                    "a single number: the entire equation LCMV writes for a source",
                ],
                [
                    "&lambda;",
                    "diagonal loading",
                    "1 &times; 1",
                    f"regularisation {reg:g}, as a fraction of the mean eigenvalue "
                    "of R",
                ],
            ],
        ),
        mcmv=dict(
            summary=(
                "All {n} constrained sources in one system. The locations have "
                "to be known in advance, and the {n} &times; {n} matrix below "
                "is what makes the whole constraint table an identity."
            ),
            rows=[
                [
                    "G<sub>s</sub>",
                    "leadfields of the constrained set only",
                    "{C} &times; {n}",
                    "not the whole grid: only the sources named",
                ],
                [
                    "A = R<sup>-1</sup>G<sub>s</sub>",
                    "solve, all sources at once",
                    "{C} &times; {n}",
                    "one column per constrained source",
                ],
                [
                    "B = G<sub>s</sub><sup>T</sup>R<sup>-1</sup>G<sub>s</sub>",
                    "the matrix that is inverted",
                    "{n} &times; {n}",
                    "this inverse is the method. It grows with the number of "
                    "sources, and it is what fails when two are too close to "
                    "tell apart",
                ],
                [
                    "I",
                    "right-hand side",
                    "{n} &times; {n}",
                    "the constraint table is set equal to the identity, all "
                    "{n2} entries of it at once",
                ],
                [
                    "W = AB<sup>-1</sup>",
                    "all filters together",
                    "{C} &times; {n}",
                    "solved jointly, so no filter can be built without the others",
                ],
                [
                    "MAI map",
                    "the search that finds the locations",
                    "{V} &times; 1",
                    "scan_mcmv adds one source at a time and rescans; the panel "
                    "shows the first pass",
                ],
                [
                    "&lambda;",
                    "diagonal loading",
                    "1 &times; 1",
                    f"regularisation {reg:g}",
                ],
            ],
        ),
        recipsiicos=dict(
            summary=(
                "LCMV, but run on a covariance that has been edited first. The "
                "edit happens in the space of vectorised covariances, which is "
                "why the numbers below are so much larger than anything else on "
                "this page."
            ),
            rows=[
                [
                    "vec(R)",
                    "the covariance, unrolled into one vector",
                    "{q2} &times; 1",
                    "q = {q} virtual sensors after whitening, so this space has "
                    "{q2} dimensions",
                ],
                [
                    "G<sub>pwr</sub>",
                    "the source-power subspace",
                    "{q2} &times; {V}",
                    "one column vec(g&thinsp;g<sup>T</sup>) per scan point: what a "
                    "single uncorrelated source contributes to a covariance",
                ],
                [
                    "U<sub>K</sub>",
                    "its leading singular vectors",
                    "{q2} &times; {K}",
                    "K* = {K}, the 45-degree point of the retention curves. It "
                    "depends on the noise covariance as well as the forward, "
                    "because that is what sets q, so it is computed once on a "
                    "representative scene rather than per configuration",
                ],
                [
                    "P<sub>K</sub> = U<sub>K</sub>U<sub>K</sub><sup>T</sup>",
                    "the projector",
                    "{q2} &times; {q2}",
                    "the largest array the whole page involves, and the reason "
                    "the reduction to q virtual sensors happens first: at the "
                    "full {C} channels this would be {C} squared on a side "
                    "instead",
                ],
                [
                    "R&#771;",
                    "the projected covariance",
                    "{q} &times; {q}",
                    "folded back into a matrix, symmetrised, and handed to LCMV. "
                    "Everything in the LCMV row set then applies to R&#771; "
                    "instead of R",
                ],
            ],
        ),
        abmc=dict(
            summary=(
                "LCMV plus a second term that rewards output resembling a known "
                "waveform. P sets what that resemblance is worth against staying "
                "distortionless."
            ),
            rows=[
                [
                    "u",
                    "the template waveform",
                    "{T} &times; 1",
                    "one time course, the thing the output is asked to look like",
                ],
                [
                    "c",
                    "template-constraint topography",
                    "{C} &times; {V}",
                    "one column per scan point: the sensor pattern that correlates "
                    "with u at that location",
                ],
                [
                    "lag search",
                    "template alignment per scan point",
                    "2&thinsp;&times;&thinsp;{T} &minus; 1 candidates",
                    "each column picks its own lag before P is applied, so a "
                    "sweep over P costs one solve rather than a rebuild",
                ],
                [
                    "P",
                    "constraint trade-off",
                    "1 &times; 1",
                    f"{template_p:g} here; at P = 0 this is exactly LCMV, and above "
                    "a critical value the solution stops being stable",
                ],
                [
                    "f",
                    "distortionless target",
                    "1 &times; 1",
                    "g<sup>T</sup>w = f, the same unit-gain condition LCMV imposes",
                ],
                [
                    "&beta;<sub>1</sub>",
                    "the multiplier",
                    "1 &times; 1",
                    "one per scan point; &beta;<sub>2</sub> = P&thinsp;&beta;"
                    "<sub>1</sub> is not free",
                ],
                [
                    "w*",
                    "the solution, in closed form",
                    "{C} &times; 1",
                    "f R<sup>-1</sup>(g + Pc) / g<sup>T</sup>R<sup>-1</sup>(g + Pc)",
                ],
                [
                    "SBL covariance",
                    "the R this method uses",
                    "{C} &times; {C}",
                    f"a sparse Bayesian estimate, up to "
                    f"{default(sbl_covariance, 'max_iter')} iterations, tolerance "
                    f"{default(sbl_covariance, 'tol'):g}",
                ],
                [
                    "descent",
                    "the paper's iteration",
                    "{C} &times; {V} per step",
                    f"available as method='iterative', up to "
                    f"{default(make_abmc, 'max_iter'):,} steps at tolerance "
                    f"{default(make_abmc, 'tol'):g}. The panel uses the closed form "
                    "above, which is the point the descent converges to",
                ],
                [
                    "template match",
                    "the localiser",
                    "{V} &times; 1",
                    "scores resemblance to u, not power, so it is not comparable "
                    "with the other three maps",
                ],
            ],
        ),
    )


def run_grid(info, fwd, fine, layouts, rank, verbose=True):
    """Every (scene, method) combination, as plain arrays.

    A scene is a layout, a correlation, a number of averaged trials and a
    morphology. The simulated data depend on those and on the seed but not on
    the method, so the four methods in one scene really are handed identical
    data; that is asserted rather than assumed, because the whole comparison
    rests on it.
    """
    combos = [
        (layout, corr, trials, morph, head_model)
        for head_model in HEAD_MODELS
        for layout in layouts
        for morph in MORPHOLOGIES
        for corr in CORRELATIONS
        for trials in TRIALS
    ]
    scenes, results = [], []
    t_start = time.time()
    for i, combo in enumerate(combos):
        if verbose:
            print(f"scene {i + 1}/{len(combos)}")
        scene, rows = _one_scene(info, fwd, fine, rank, combo, verbose)
        index = len(scenes)
        scenes.append(scene)
        for row in rows:
            row["scene"] = index
            results.append(row)
    if verbose:
        print(
            f"{len(results)} configurations over {len(scenes)} scenes "
            f"in {time.time() - t_start:.0f}s"
        )
    return scenes, results


# --------------------------------------------------------------------------- #
# The recorded half: MNE's sample auditory and visual responses
# --------------------------------------------------------------------------- #
# Everything above this line is simulated, which is what lets it be checked
# against a truth. None of that is available here, so this half shows only what
# needs no truth -- the constraint table, the localiser, the recording and the
# reconstruction -- and reports an independent dipole fit beside the peaks
# rather than pretending to know where the sources are.
REAL_CONDITIONS = (
    ("Auditory/Left", "auditory left", "auditory"),
    ("Auditory/Right", "auditory right", "auditory"),
    ("Auditory", "auditory both", "auditory"),
    ("Visual/Left", "visual left", "visual"),
    ("Visual/Right", "visual right", "visual"),
    ("Visual", "visual both", "visual"),
)
# The active window the covariance is built from. This is the control with the
# most to teach: a beamformer is tuned by the covariance it is given, and the
# published auditory example shows the wider window moving the selected left
# source 25 mm out of superior temporal cortex.
REAL_WINDOWS = ((0.08, 0.13), (0.05, 0.20))
REAL_TRIALS = (1, 4, 16, 0)  # 0 means every epoch of that condition
# Epoching follows the published auditory example exactly, and that is not
# cosmetic. Shortening the baseline and adding amplitude rejection moved the
# right-hemisphere peak to a different vertex, and LCMV's off-diagonal fell from
# -0.78 to -0.15: the cancellation this page exists to show is a property of the
# pair that gets selected, so the preprocessing that selects it has to be the
# one whose result is already documented.
REAL_TMIN, REAL_TMAX = -0.2, 0.25
# The scan grid is decimated from the full surface, but the *locations* are not:
# they are chosen at full resolution and forced into the grid. That separation
# matters. Decimating before choosing moves the right auditory peak into
# parietal cortex and the pair stops being correlated; decimating after changes
# nothing at all, because LCMV, MCMV and ABMC each build a filter from one
# leadfield and the covariance, and the rest of the grid never enters. Measured,
# the constraint table is identical to three decimals from 7498 sources down to
# 627. Only ReciPSIICOS spans the grid, and it is the reason for decimating: its
# projector needs a q^2-by-grid factorisation, which costs 0.6 s here and over
# five minutes on the undecimated surface.
REAL_DECIMATION = 8


def real_recording(info, verbose=True):
    """Load epochs, the full-resolution forward, and one shared noise covariance.

    The noise covariance is pooled over every epoch in the session rather than
    estimated per scene, for the reason the simulated half learned the hard way:
    it sets the whitener, the whitener sets q, and q is the space the
    ReciPSIICOS rank is drawn from. Estimating it per scene would give each
    scene its own q and a precomputed rank would mean something different in
    each one. Pooling is also the better estimate and standard practice.
    """
    meg = mne.datasets.sample.data_path() / "MEG" / "sample"
    raw = mne.io.read_raw_fif(meg / "sample_audvis_filt-0-40_raw.fif", preload=True)
    events = mne.read_events(meg / "sample_audvis_filt-0-40_raw-eve.fif")
    raw.pick(info["ch_names"])
    epochs = mne.Epochs(
        raw,
        events,
        {"Auditory/Left": 1, "Auditory/Right": 2, "Visual/Left": 3, "Visual/Right": 4},
        tmin=REAL_TMIN,
        tmax=REAL_TMAX,
        baseline=(None, 0.0),
        preload=True,
        verbose=False,
    )
    epochs.pick(info["ch_names"])

    base = mne.read_forward_solution(
        meg / "sample_audvis-meg-eeg-oct-6-fwd.fif", verbose=False
    )
    base = mne.pick_types_forward(base, meg=True, eeg=False)
    base = mne.convert_forward_solution(
        base, force_fixed=True, use_cps=True, verbose=False
    )
    base = mne.pick_channels_forward(base, info["ch_names"], ordered=True)

    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="shrunk", verbose=False
    )
    if verbose:
        counts = ", ".join(f"{k} {len(epochs[k])}" for k in epochs.event_id)
        print(
            f"real recording: {counts}; {base['sol']['data'].shape[1]} vertices at "
            f"full resolution, {epochs.info['sfreq']:.0f} Hz"
        )
    return epochs, base, noise_cov


def _dipole_reference(evoked, noise_cov, window, verbose=True):
    """Fit two dipoles, one after the other, as an independent reference.

    Not a truth. A single fit lands in whichever hemisphere is stronger, so the
    second is fitted to what the first leaves behind, which is what separates
    the pair. Their goodness of fit is reported with them because a reference
    worth drawing is one whose quality the reader can see.
    """
    data_path = mne.datasets.sample.data_path()
    bem = data_path / "subjects" / "sample" / "bem" / "sample-5120-bem-sol.fif"
    trans = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    peak = evoked.copy().crop(*window)
    first, residual = mne.fit_dipole(
        peak, noise_cov, str(bem), str(trans), verbose=False
    )
    second, _ = mne.fit_dipole(residual, noise_cov, str(bem), str(trans), verbose=False)
    out, gof = [], []
    for dip in (first, second):
        k = int(np.argmax(dip.gof))
        out.append(dip.pos[k])
        gof.append(float(dip.gof[k]))
    return np.asarray(out), np.asarray(gof)


def real_selection(info, base, epochs, noise_cov, verbose=True):
    """Choose where to constrain, once per condition and window.

    Chosen from every available epoch rather than from the subset a scene
    averages, so that the trials control is a signal-to-noise axis and nothing
    else. If the locations moved with the trial count the constraint tables
    either side of that control would describe different pairs.
    """
    from advance_beamlab._explain import evoked_sources

    picked = {}
    for condition, label, _family in REAL_CONDITIONS:
        for window in REAL_WINDOWS:
            ep = epochs[condition]
            evoked = ep.average()
            data_cov = mne.compute_covariance(
                ep, tmin=window[0], tmax=window[1], method="shrunk", verbose=False
            )
            src = evoked_sources(info, base, data_cov, noise_cov)
            reference, gof = _dipole_reference(evoked, noise_cov, window)
            vertno = [base["src"][0]["vertno"], base["src"][1]["vertno"]]
            n_lh = len(vertno[0])
            vertices = (int(vertno[0][src[0]]), int(vertno[1][src[1] - n_lh]))
            picked[(condition, window)] = dict(
                vertices=vertices, reference=reference, gof=gof, label=label
            )
            if verbose:
                sep = np.linalg.norm(
                    base["source_rr"][src[0]] - base["source_rr"][src[1]]
                )
                print(
                    f"  {label:>15} {window[0]:.2f}-{window[1]:.2f}s: "
                    f"vertices {vertices}, {sep * 100:.1f} cm apart, "
                    f"dipole gof {gof[0]:.0f}%/{gof[1]:.0f}%"
                )
    return picked


def real_scan_grid(base, picked, verbose=True):
    """Build one decimated grid holding every location any scene constrains.

    A single grid for all of them, so that every localiser map is the same
    length and drawn against the same positions. Building a grid per condition
    would make the maps incomparable and the payload a set of unrelated arrays.
    """
    wanted = [set(), set()]
    for entry in picked.values():
        wanted[0].add(entry["vertices"][0])
        wanted[1].add(entry["vertices"][1])
    keep = []
    for hemi in (0, 1):
        vertno = base["src"][hemi]["vertno"]
        kept = set(vertno[::REAL_DECIMATION].tolist()) | wanted[hemi]
        keep.append(np.array(sorted(kept)))
    total = sum(len(k) for k in keep)
    stc = mne.SourceEstimate(
        np.ones((total, 1)), keep, tmin=0.0, tstep=1.0, subject="sample"
    )
    fwd = mne.forward.restrict_forward_to_stc(base, stc)
    index = {}
    left = fwd["src"][0]["vertno"].tolist()
    right = fwd["src"][1]["vertno"].tolist()
    for hemi, table in ((0, left), (1, right)):
        for v in wanted[hemi]:
            index[(hemi, v)] = table.index(v) + (0 if hemi == 0 else len(left))
    if verbose:
        print(
            f"real scan grid: {total} sources (1 in {REAL_DECIMATION} of the "
            f"surface, plus every constrained location)"
        )
    return fwd, index


def real_rank(info, fwd, epochs, noise_cov, window, verbose=True):
    """Return the ReciPSIICOS rank, once, for every real scene.

    Same rule as the simulated half: the rank is drawn out of q^2 and q comes
    from the whitened leadfield, so it has to be chosen against the covariance
    it will be spent against. Every real scene shares one noise covariance,
    which is what makes a single rank legitimate here.
    """
    from advance_beamlab._explain import evoked_demo, evoked_sources

    ep = epochs["Auditory"]
    data_cov = mne.compute_covariance(
        ep, tmin=window[0], tmax=window[1], method="shrunk", verbose=False
    )
    src = evoked_sources(info, fwd, data_cov, noise_cov)
    demo = evoked_demo(
        info,
        fwd,
        ep.average(),
        data_cov,
        noise_cov,
        src,
        method="recipsiicos",
        recipsiicos_rank=None,
        verbose=False,
    )
    rank = demo.extra.get("recipsiicos_rank")
    virtual = demo.extra.get("recipsiicos_virtual")
    if verbose:
        print(
            f"real ReciPSIICOS rank K* = {rank} of q^2 = {virtual**2} "
            f"(q = {virtual} virtual sensors)"
        )
    return int(rank), int(virtual)


def run_real_grid(info, fwd, index, epochs, noise_cov, picked, rank, verbose=True):
    """Every real scene: condition, window and trial count, all four methods."""
    from advance_beamlab._explain import evoked_demo

    scenes, results = [], []
    combos = [
        (condition, label, family, window, trials)
        for condition, label, family in REAL_CONDITIONS
        for window in REAL_WINDOWS
        for trials in REAL_TRIALS
    ]
    for n, (condition, label, family, window, trials) in enumerate(combos):
        ep = epochs[condition]
        available = len(ep)
        # ``trials`` of 0 means every epoch; anything larger than the condition
        # holds is clamped, and the scene records what it actually averaged
        # rather than what was asked for.
        used = available if trials == 0 else min(int(trials), available)
        subset = ep[:used]
        evoked = subset.average()
        data_cov = mne.compute_covariance(
            subset, tmin=window[0], tmax=window[1], method="shrunk", verbose=False
        )
        entry = picked[(condition, window)]
        sources = [
            index[(0, entry["vertices"][0])],
            index[(1, entry["vertices"][1])],
        ]
        # The joint filter estimates the waveform overlap that turns a
        # constraint table into a delivered amplitude. Read that overlap off
        # per-source LCMV traces instead and a strongly correlated pair reads as
        # uncorrelated, through the very filter whose cancellation it explains.
        joint = evoked_demo(
            info,
            fwd,
            evoked,
            data_cov,
            noise_cov,
            sources,
            method="mcmv",
            verbose=False,
        )
        scene = dict(
            index=len(scenes),
            condition=condition,
            label=label,
            family=family,
            window=[float(window[0]), float(window[1])],
            trials=used,
            available=available,
            sources=sources,
            separation=float(joint.separation),
            correlation=float(joint.correlation),
            reference=[[float(v) for v in p] for p in entry["reference"]],
            reference_gof=[float(v) for v in entry["gof"]],
        )
        scenes.append(scene)
        for method in METHODS:
            for template in templates_for(method, REAL_TEMPLATES):
                t0 = time.time()
                demo = evoked_demo(
                    info,
                    fwd,
                    evoked,
                    data_cov,
                    noise_cov,
                    sources,
                    method=method,
                    condition=condition,
                    n_trials=used,
                    window=window,
                    reference=entry["reference"],
                    reference_gof=entry["gof"],
                    reference_tcs=joint.reconstructed,
                    recipsiicos_rank=rank,
                    template=template if template is not None else "estimate",
                    verbose=False,
                )
                results.append(
                    dict(
                        scene=scene["index"],
                        method=method,
                        template=template,
                        gains=[[float(v) for v in row] for row in demo.gains],
                        amplitude_ratio=[float(v) for v in demo.amplitude_ratio],
                        peaks=[int(v) for v in demo.peaks],
                        reference_distance=[
                            float(v) for v in demo.reference_distance * 1000
                        ],
                        power_map=np.asarray(demo.power_map, float),
                        reconstructed=np.asarray(demo.reconstructed, float),
                        sensor=np.asarray(demo.sensor_data, float),
                        times=np.asarray(demo.times, float),
                    )
                )
                if verbose:
                    print(
                        f"  {method:>11} {label:>15} {window[0]:.2f}-{window[1]:.2f}s "
                        f"{used:>3} trial(s)  off-diag {demo.gains[0, 1]:+.3f}  "
                        f"delivered {demo.amplitude_ratio[0]:.3f}  "
                        f"to dipole {demo.reference_distance.min() * 1000:5.1f} mm "
                        f"[{time.time() - t0:.1f}s]"
                    )
        if verbose:
            print(f"real scene {n + 1}/{len(combos)}")
    return scenes, results


class Packer:
    """Concatenates typed arrays into one buffer and records where they went."""

    def __init__(self):
        self.chunks = []
        self.offset = 0

    def add(self, array, dtype):
        """Append one array and return where it landed."""
        # Pad to four bytes first: a typed-array view in the browser has to start
        # on a multiple of its element size, and a uint8 block of odd length
        # would silently misalign everything after it.
        if self.offset % 4:
            pad = 4 - self.offset % 4
            self.chunks.append(b"\0" * pad)
            self.offset += pad
        a = np.ascontiguousarray(array, dtype=dtype)
        # ``np.dtype(...).name``, not ``str(dtype)``: the latter gives
        # "<class 'numpy.int16'>", which no reader is going to match against.
        entry = dict(offset=self.offset, length=int(a.size), dtype=np.dtype(dtype).name)
        self.chunks.append(a.tobytes())
        self.offset += a.nbytes
        return entry

    def buffer(self):
        """Return the concatenated bytes."""
        return b"".join(self.chunks)


def _quantise(array, scale, name="array"):
    """Signed 16-bit, with the scale recorded by the caller.

    Saturation is an error rather than a clip. The mix block used to overflow a
    fixed scale of 20000 in 67 of 816 scenes, which silently changed the sensor
    recording the browser rebuilds.
    """
    out = np.round(np.asarray(array) * scale)
    if np.abs(out).max(initial=0.0) > 32767:
        raise ValueError(
            f"{name} saturates int16 at scale {scale}: "
            f"max |value| is {np.abs(np.asarray(array)).max():.6g}"
        )
    return out


def _quantise8(array, name="array"):
    """Signed 8-bit, for data that is only ever drawn as a line.

    A waveform plotted a few hundred pixels tall cannot show more than about one
    part in 200, so 8 bits is invisible here and halves the biggest blocks in the
    payload. Saturation is an error: reconstructions were being scaled by the
    *truth's* peak, so any that overshot were flattened to exactly the truth's
    amplitude and became indistinguishable from a perfect recovery.
    """
    out = np.round(np.asarray(array) * 127)
    if np.abs(out).max(initial=0.0) > 127:
        raise ValueError(
            f"{name} saturates int8: max |value| is "
            f"{np.abs(np.asarray(array)).max():.6g}, which needs a larger scale"
        )
    return out


def pack_real(packer, header, scenes, results, positions, *, centre, verbose=True):
    """Add the recorded half to an in-progress payload.

    Kept separate from :func:`pack` rather than folded into it because almost
    nothing is shared: the recorded scenes have their own grid, their own time
    axis, and no truth to store. What they do share is the buffer, so the two
    halves are written through the same packer and the browser reads them the
    same way.

    The recording is stored rather than rebuilt. The simulated half can
    reconstruct its sensor data from a leadfield and a noise scale, which is
    what keeps it small; a real recording has no such recipe and has to be
    carried. It is deduplicated first: the covariance window changes the
    filters but not the data, so the traces vary only with condition and trial
    count.
    """
    n_real = int(positions.shape[0])
    header["real_conditions"] = [
        dict(key=key, label=label, family=family)
        for key, label, family in REAL_CONDITIONS
    ]
    header["real_windows"] = [[float(a), float(b)] for a, b in REAL_WINDOWS]
    header["real_trials"] = list(REAL_TRIALS)
    header["real_templates"] = list(REAL_TEMPLATES)
    header["n_real_sources"] = n_real
    header["real_decimation"] = REAL_DECIMATION
    header["real_positions"] = packer.add(
        _quantise((positions - centre) * 1000, 10, "real positions"), np.int16
    )

    times = results[0]["times"]
    header["real_time0"] = float(times[0])
    header["real_sfreq"] = float(1.0 / (times[1] - times[0]))
    header["real_n_times"] = int(times.size)

    # One recording per (condition, trials); the window does not touch it.
    recordings, recording_of = [], {}
    for scene in scenes:
        key = (scene["condition"], scene["trials"])
        if key not in recording_of:
            recording_of[key] = len(recordings)
            match = next(r for r in results if r["scene"] == scene["index"])
            recordings.append(match["sensor"])
        scene["recording"] = recording_of[key]
    stack = np.stack(recordings)
    # Stored as a fraction of the loudest sample anywhere in the recorded half,
    # not in tesla per metre. Two reasons. One scale for all of them means
    # averaging more trials visibly lowers the noise instead of every scene
    # being renormalised to its own peak, which would hide exactly what the
    # trials control is there to show. And unit amplitude is the convention the
    # simulated half already uses, so the drawing code does not have to know
    # which half it is looking at: in physical units the traces came out around
    # 1e-12 and every sensor panel drew a flat line.
    peak = max(float(np.abs(stack).max()), 1e-30)
    header["real_recording_peak"] = peak
    header["real_recording_scale"] = 32000.0
    header["real_recording"] = packer.add(
        _quantise(stack / peak, 32000.0, "real recording"), np.int16
    )
    header["real_n_recordings"] = len(recordings)

    recon = np.stack([r["reconstructed"] for r in results])
    recon_peak = np.abs(recon).max(axis=(1, 2), keepdims=True)
    header["real_recon"] = packer.add(
        _quantise8(recon / np.maximum(recon_peak, 1e-30), "real reconstruction"),
        np.int8,
    )
    header["real_recon_scale"] = [float(v) for v in recon_peak.ravel()]

    maps = []
    for r in results:
        m = np.asarray(r["power_map"], float)
        span = m.max() - m.min()
        norm = (m - m.min()) / span if span > 0 else np.zeros_like(m)
        maps.append(np.round(norm ** (1 / 3) * 255))
    header["real_maps"] = packer.add(np.concatenate(maps), np.uint8)

    header["real_scenes"] = [
        dict(
            index=s["index"],
            condition=s["condition"],
            label=s["label"],
            family=s["family"],
            window=s["window"],
            trials=s["trials"],
            available=s["available"],
            recording=s["recording"],
            sources=s["sources"],
            separation=s["separation"],
            correlation=s["correlation"],
            reference=[
                [round((p[k] - centre[k]) * 1000, 2) for k in range(3)]
                for p in s["reference"]
            ],
            reference_gof=[round(v, 1) for v in s["reference_gof"]],
        )
        for s in scenes
    ]
    header["real_results"] = [
        dict(
            scene=r["scene"],
            method=r["method"],
            template=r.get("template"),
            gains=[[round(v, 4) for v in row] for row in r["gains"]],
            amplitude_ratio=[round(v, 4) for v in r["amplitude_ratio"]],
            peaks=r["peaks"],
            reference_distance=[round(v, 2) for v in r["reference_distance"]],
        )
        for r in results
    ]
    if verbose:
        print(
            f"real payload: {len(results)} results over {len(scenes)} scenes, "
            f"{len(recordings)} recordings of {header['real_n_times']} samples, "
            f"{n_real} grid points"
        )
    return header


def pack(
    scenes,
    results,
    positions,
    cortex,
    sensor_pos,
    layouts,
    model,
    real=None,
    verbose=True,
):
    """Quantise everything into one gzipped buffer plus a JSON header."""
    packer = Packer()
    header = dict(
        methods=list(METHODS),
        templates=list(TEMPLATES),
        correlations=list(CORRELATIONS),
        trials=list(TRIALS),
        snrs=list(SNRS),
        single_trial_snr=SINGLE_TRIAL_SNR,
        morphologies=list(MORPHOLOGIES),
        head_models=list(HEAD_MODELS),
        layouts=[
            dict(
                key=lay["key"],
                label=lay["label"],
                group=lay["group"],
                kind=lay.get("kind", "region"),
            )
            for lay in layouts
        ],
        n_times=DISPLAY_SAMPLES,
        wave_times=WAVE_SAMPLES,
        n_times_simulated=N_TIMES,
        sfreq=SFREQ,
        n_sources=int(positions.shape[0]),
        model=model,
    )

    # Geometry, in millimetres and centred, so int16 is plenty of resolution.
    centre = cortex.mean(0)
    header["positions"] = packer.add(
        _quantise((positions - centre) * 1000, 10), np.int16
    )
    header["cortex"] = packer.add(_quantise((cortex - centre) * 1000, 10), np.int16)
    header["geometry_scale"] = 10.0

    # Localiser maps, each normalised to its own range and compressed by a cube
    # root. Display rank was the first choice and it was a mistake: a rank map is
    # a uniform ramp whatever the underlying shape, so all four methods rendered
    # identically -- measured, the fraction of the grid above half the display
    # maximum was 0.500 for every one of them. Normalising alone is too peaky to
    # show any structure: 1.3 per cent of the grid for LCMV and MCMV, which is
    # two bright dots on an empty brain. The cube root sits between, and
    # separates the methods: over the whole grid the fraction above half the
    # display maximum is 10.9 per cent for LCMV, 7.2 for MCMV and 16.1 for
    # ReciPSIICOS. ABMC depends on which template it was given -- 34.8 with the
    # target's own waveform, 87.1 with a same-band one and 96.3 with a wrong-band
    # one -- because its map scores template match rather than power.
    #
    # These are properties of the payload, so they go stale when it is rebuilt.
    # tools/panel_statistics.py reads them back off the built file, along with
    # every other number doc/panel.rst quotes about the panel.
    maps = []
    for r in results:
        m = np.asarray(r["power_map"], float)
        span = m.max() - m.min()
        norm = (m - m.min()) / span if span > 0 else np.zeros_like(m)
        maps.append(np.round(norm ** (1 / 3) * 255))
    header["maps"] = packer.add(np.concatenate(maps), np.uint8)

    cut = slice(None, DISPLAY_SAMPLES)
    wave_cut = slice(None, WAVE_SAMPLES)
    # Waveforms, variable in length because the scenes hold one, two or three
    # sources. The reader walks the same cumulative offsets from the scene list.
    #
    # Each block gets its own scale so that each uses the full 8-bit range, and
    # the panel puts them back on a common footing when it draws them.
    #
    # Two wrong ways were tried first. Scaling the reconstruction by the truth's
    # peak flattened any reconstruction that overshot the source to exactly the
    # source's amplitude: 53 per cent of the one-trial traces were clipped, and
    # a clipped trace cannot be told from a perfect recovery. Sharing one scale
    # big enough for both then destroyed the truth's precision instead, since a
    # reconstruction overshooting fifteenfold leaves the source three bits.
    true_scale = [float(np.abs(s["true_tcs"][:, cut]).max()) or 1.0 for s in scenes]
    recon_scale = [
        float(np.abs(r["reconstructed"][:, wave_cut]).max()) or 1.0 for r in results
    ]
    header["true_tcs"] = packer.add(
        np.concatenate(
            [
                _quantise8(s["true_tcs"][:, cut] / k, "true_tcs").ravel()
                for s, k in zip(scenes, true_scale, strict=True)
            ]
        ),
        np.int8,
    )
    header["reconstructed"] = packer.add(
        np.concatenate(
            [
                _quantise8(r["reconstructed"][:, wave_cut] / k, "reconstructed").ravel()
                for r, k in zip(results, recon_scale, strict=True)
            ]
        ),
        np.int8,
    )
    # Physical amplitudes, so the panel can draw truth and reconstruction on one
    # scale per scene and keep the amplitude loss visible.
    header["true_scale"] = [float(v) for v in true_scale]
    header["recon_scale"] = [float(v) for v in recon_scale]
    header["waveform_scale"] = 20000.0
    header["byte_scale"] = 127.0

    # The sensor recording is not stored. It is exactly
    # ``leadfield @ true_tcs + noise_scale * noise``, and because the noise is
    # drawn from a generator seeded independently of the waveforms it is the same
    # field in every scene. Storing that field once and the mixing coefficients
    # per scene costs about 300 kB; storing 203 channels for every scene would
    # cost several megabytes of noise, which does not compress.
    unit_noise = None
    for s in scenes:
        raw = (s["sensor"] - s["leadfield"] @ s["true_tcs"]) / s["noise_scale"]
        if unit_noise is None:
            unit_noise = raw
        else:
            np.testing.assert_allclose(raw, unit_noise, atol=1e-9)
    noise_max = float(np.abs(unit_noise[:, cut]).max()) or 1.0
    header["noise"] = packer.add(
        _quantise8(unit_noise[:, cut] / noise_max, "noise"), np.int8
    )

    mix, noise_gain = [], []
    for s, k in zip(scenes, true_scale, strict=True):
        sensor_max = float(np.abs(s["sensor"][:, cut]).max()) or 1.0
        mix.append((s["leadfield"] * k / sensor_max).ravel())
        noise_gain.append(s["noise_scale"] * noise_max / sensor_max)
    # A scale taken from the data rather than a fixed one. At 20000 the mix
    # block saturated int16 in 67 of 816 scenes, so the recording the browser
    # rebuilt was not the recording that had been simulated.
    flat_mix = np.concatenate(mix)
    mix_scale = 32000.0 / max(float(np.abs(flat_mix).max()), 1e-30)
    header["mix"] = packer.add(_quantise(flat_mix, mix_scale, "mix"), np.int16)
    header["mix_scale"] = mix_scale
    header["noise_gain"] = [round(v, 6) for v in noise_gain]

    header["sensor_pos"] = packer.add(_quantise(sensor_pos, 20000), np.int16)
    header["n_channels"] = int(sensor_pos.shape[0])

    # The field over the whole array at one instant, for the topography.
    topo, topo_time = [], []
    for s in scenes:
        data = np.asarray(s["sensor"])[:, cut]
        t_peak = int(np.argmax(data.std(axis=0)))
        field = data[:, t_peak]
        topo.append(_quantise(field / (np.abs(field).max() or 1.0), 20000))
        topo_time.append(t_peak)
    header["topography"] = packer.add(np.concatenate(topo), np.int16)
    header["topography_time"] = topo_time

    header["scenes"] = [
        dict(
            sources=s["sources"],
            separation=s["separation"],
            correlation=s["correlation"],
            requested=s["requested"],
        )
        for s in scenes
    ]
    # Where the sources really are, which is not a grid point. Kept so the panel
    # can draw the truth at its actual position rather than at the node that
    # stands in for it.
    header["true_positions"] = packer.add(
        _quantise(
            np.concatenate([(s["true_positions"] - centre) * 1000 for s in scenes]), 10
        ),
        np.int16,
    )
    header["results"] = [
        dict(
            scene=r["scene"],
            method=r["method"],
            template=r.get("template"),
            gains=[[round(v, 5) for v in row] for row in r["gains"]],
            amplitude_ratio=[round(v, 5) for v in r["amplitude_ratio"]],
            peak_errors=[round(v, 2) for v in r["peak_errors"]],
            peaks=r["peaks"],
        )
        for r in results
    ]

    if real is not None:
        pack_real(packer, header, *real, centre=centre, verbose=verbose)

    raw = packer.buffer()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    if verbose:
        print(
            f"payload: {len(raw) / 1e6:.2f} MB raw -> "
            f"{len(blob) / 1e6:.2f} MB gzipped base64"
        )
    return header, blob


def main(argv=None):
    """Build the panel data file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path of the .js data file")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="skip the recorded half, which needs the sample dataset",
    )
    args = parser.parse_args(argv)
    verbose = not args.quiet

    info, fwd, fine, cortex, groups, model = build_forward(verbose)
    layouts = resolve_layouts(fwd, groups, verbose)
    rank, virtual = recipsiicos_rank(info, fwd, fine, layouts, verbose)
    model["algorithms"] = algorithm_parameters()
    model["recipsiicos_rank"] = int(rank)
    model["recipsiicos_virtual"] = int(virtual)
    scenes, results = run_grid(info, fwd, fine, layouts, rank, verbose)

    real = None
    if not args.simulation_only:
        epochs, base, real_noise = real_recording(info, verbose)
        picked = real_selection(info, base, epochs, real_noise, verbose)
        scan, index = real_scan_grid(base, picked, verbose)
        real_k, real_q = real_rank(
            info, scan, epochs, real_noise, REAL_WINDOWS[0], verbose
        )
        model["real_recipsiicos_rank"] = real_k
        model["real_recipsiicos_virtual"] = real_q
        model["real_channels"] = int(scan["nchan"])
        model["real_n_scan"] = int(scan["sol"]["data"].shape[1])
        model["real_n_full"] = int(base["sol"]["data"].shape[1])
        real_scenes, real_results = run_real_grid(
            info, scan, index, epochs, real_noise, picked, real_k, verbose
        )
        real = (real_scenes, real_results, scan["source_rr"])

    header, blob = pack(
        scenes,
        results,
        fwd["source_rr"],
        cortex,
        sensor_layout(info),
        layouts,
        model,
        real=real,
        verbose=verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "// Generated by tools/build_constraint_panel.py. Do not edit by hand.\n"
        f"window.CONSTRAINT_PANEL = {json.dumps(header, separators=(',', ':'))};\n"
        f'window.CONSTRAINT_PANEL.blob = "{blob}";\n'
    )
    size = args.output.stat().st_size
    if verbose:
        print(f"wrote {args.output} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
