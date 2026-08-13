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
CORRELATIONS = (0.0, 0.5, 0.9, 0.99)
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
)

# Where the sources sit. The geometric layouts control separation directly; the
# region layouts put a bilateral pair in a named area, which is the shape most
# real questions take.
LAYOUTS = (
    dict(key="single", label="one source", kind="geometric", n=1, sep=0.0),
    dict(key="near", label="pair, 2 cm apart", kind="geometric", n=2, sep=0.02),
    dict(key="far", label="pair, 6 cm apart", kind="geometric", n=2, sep=0.06),
    dict(key="triple", label="three sources", kind="geometric", n=3, sep=0.04),
    dict(
        key="hippocampus",
        label="hippocampus L/R",
        kind="subcortical",
        labels=("Left-Hippocampus", "Right-Hippocampus"),
    ),
    dict(
        key="auditory",
        label="auditory L/R",
        kind="anatomical",
        labels=("transversetemporal-lh", "transversetemporal-rh"),
    ),
    dict(
        key="visual",
        label="visual L/R",
        kind="anatomical",
        labels=("pericalcarine-lh", "pericalcarine-rh"),
    ),
    dict(
        key="motor",
        label="motor L/R",
        kind="anatomical",
        labels=("precentral-lh", "precentral-rh"),
    ),
    dict(
        key="vmpfc",
        label="vmPFC L/R",
        kind="anatomical",
        labels=("medialorbitofrontal-lh", "medialorbitofrontal-rh"),
    ),
    dict(
        key="dlpfc",
        label="lateral PFC L/R",
        kind="anatomical",
        labels=("rostralmiddlefrontal-lh", "rostralmiddlefrontal-rh"),
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
# Every 8th source-space vertex. 938 sources keeps ReciPSIICOS, whose Gram is
# quadratic in the source count, at about two seconds per configuration while
# still being a real cortex rather than a sphere.
DECIMATION = 14
N_BACKDROP = 4000

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
        pos=4.0,
        mri="aseg.mgz",
        volume_label=list(SUBCORTICAL),
        subjects_dir=subjects_dir,
        bem=bem,
        add_interpolator=False,
        verbose=False,
    )
    points, normals, groups = [], [], []
    for space in volume:
        pts = space["rr"][space["vertno"]]
        axes = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2]
        points.append(pts)
        normals.append(np.tile(axes[2], (len(pts), 1)))
        groups.append((space["seg_name"], len(pts)))
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

    # Decimate the cortex but keep every subcortical source, so the structures
    # the layouts name are all actually scannable.
    verts = [s["vertno"][::DECIMATION] for s in fine["src"][:2]]
    verts += [s["vertno"] for s in fine["src"][2:]]
    n = sum(len(v) for v in verts)
    stc = mne.MixedSourceEstimate(
        np.ones((n, 1)), verts, tmin=0, tstep=1, subject="sample"
    )
    fwd = mne.forward.restrict_forward_to_stc(fine, stc)

    if verbose:
        rr = fwd["source_rr"]
        gaps = [
            np.sort(np.linalg.norm(rr - rr[i], axis=1))[1] * 1000
            for i in range(0, sum(len(v) for v in verts[:2]), 41)
        ]
        print(
            f"scan grid: {fwd['sol']['data'].shape[1]} sources "
            f"({len(verts[0]) + len(verts[1])} cortical at {np.median(gaps):.1f} mm, "
            f"{len(verts[2])} subcortical); truth from "
            f"{fine['sol']['data'].shape[1]}; {fwd['nchan']} gradiometers"
        )
    return info, fwd, fine, cortex, groups


def resolve_layouts(fwd, groups, verbose=True):
    """Grid indices for each layout, from the parcellation or the aseg."""
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

    # Where each subcortical structure's sources sit in the concatenated grid.
    subcortical = {}
    offset = n_cortical
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

    resolved = []
    for layout in LAYOUTS:
        entry = dict(layout)
        if layout["kind"] == "anatomical":
            indices = []
            for name in layout["labels"]:
                label = parc[name]
                hemi = 0 if label.hemi == "lh" else 1
                base = 0 if hemi == 0 else src[0]["nuse"]
                shared = np.intersect1d(src[hemi]["vertno"], label.vertices)
                idx = np.array(
                    [
                        base + int(np.flatnonzero(src[hemi]["vertno"] == v)[0])
                        for v in shared
                    ]
                )
                pick = centroid_index(idx)
                vertex = src[hemi]["vertno"][pick - base]
                assert vertex in label.vertices, f"{name} representative escaped"
                indices.append(pick)
            entry["sources"] = indices
        elif layout["kind"] == "subcortical":
            entry["sources"] = [
                centroid_index(subcortical[name]) for name in layout["labels"]
            ]
        else:
            entry["sources"] = None
        if entry["sources"] is not None:
            entry["n"] = len(entry["sources"])
            if verbose:
                d = np.linalg.norm(rr[entry["sources"][0]] - rr[entry["sources"][1]])
                print(f"  {layout['label']:>22}: {entry['sources']}, {d * 100:.1f} cm")
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
            np.testing.assert_allclose(demo.true_tcs, scene["true_tcs"], rtol=0, atol=0)
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


def recipsiicos_rank(info, fwd, verbose=True):
    """Return the projection rank, a property of the forward not of the data."""
    from advance_beamlab import recipsiicos_rank_curve

    _, _, _, k_opt = recipsiicos_rank_curve(
        fwd, info, method="whitened", return_optimal=True, verbose=False
    )
    if verbose:
        print(f"ReciPSIICOS rank K* = {k_opt}, computed once for the whole grid")
    return int(k_opt)


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


def _quantise(array, scale):
    """Signed 16-bit, clipped, with the scale recorded by the caller."""
    return np.clip(np.round(np.asarray(array) * scale), -32767, 32767)


def _quantise8(array):
    """Signed 8-bit, for data that is only ever drawn as a line.

    A waveform plotted a few hundred pixels tall cannot show more than about one
    part in 200, so 8 bits is invisible here and halves the biggest blocks in the
    payload.
    """
    return np.clip(np.round(np.asarray(array) * 127), -127, 127)


def pack(scenes, results, positions, cortex, sensor_pos, verbose=True):
    """Quantise everything into one gzipped buffer plus a JSON header."""
    packer = Packer()
    header = dict(
        methods=list(METHODS),
        correlations=list(CORRELATIONS),
        trials=list(TRIALS),
        snrs=list(SNRS),
        single_trial_snr=SINGLE_TRIAL_SNR,
        morphologies=list(MORPHOLOGIES),
        head_models=list(HEAD_MODELS),
        layouts=[
            dict(key=lay["key"], label=lay["label"], kind=lay["kind"])
            for lay in LAYOUTS
        ],
        n_times=DISPLAY_SAMPLES,
        wave_times=WAVE_SAMPLES,
        n_times_simulated=N_TIMES,
        sfreq=SFREQ,
        n_sources=int(positions.shape[0]),
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
    # show any structure (0.2 per cent of the grid above half). The cube root
    # sits between, and separates the methods: 1.9, 0.1, 8.7 and 5.7 per cent for
    # LCMV, MCMV, ReciPSIICOS and ABMC on the same scene.
    maps = []
    for r in results:
        m = np.asarray(r["power_map"], float)
        span = m.max() - m.min()
        norm = (m - m.min()) / span if span > 0 else np.zeros_like(m)
        maps.append(np.round(norm ** (1 / 3) * 255))
    header["maps"] = packer.add(np.concatenate(maps), np.uint8)

    cut = slice(None, DISPLAY_SAMPLES)
    # Waveforms, variable in length because the scenes hold one, two or three
    # sources. The reader walks the same cumulative offsets from the scene list.
    true_scale = [float(np.abs(s["true_tcs"]).max()) or 1.0 for s in scenes]
    header["true_tcs"] = packer.add(
        np.concatenate(
            [
                _quantise8(s["true_tcs"][:, cut] / k).ravel()
                for s, k in zip(scenes, true_scale, strict=True)
            ]
        ),
        np.int8,
    )
    wave_cut = slice(None, WAVE_SAMPLES)
    header["reconstructed"] = packer.add(
        np.concatenate(
            [
                _quantise8(
                    r["reconstructed"][:, wave_cut] / true_scale[r["scene"]]
                ).ravel()
                for r in results
            ]
        ),
        np.int8,
    )
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
    header["noise"] = packer.add(_quantise8(unit_noise[:, cut] / noise_max), np.int8)

    mix, noise_gain = [], []
    for s, k in zip(scenes, true_scale, strict=True):
        sensor_max = float(np.abs(s["sensor"][:, cut]).max()) or 1.0
        mix.append((s["leadfield"] * k / sensor_max).ravel())
        noise_gain.append(s["noise_scale"] * noise_max / sensor_max)
    header["mix"] = packer.add(_quantise(np.concatenate(mix), 20000), np.int16)
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
            gains=[[round(v, 5) for v in row] for row in r["gains"]],
            amplitude_ratio=[round(v, 5) for v in r["amplitude_ratio"]],
            peak_errors=[round(v, 2) for v in r["peak_errors"]],
            peaks=r["peaks"],
        )
        for r in results
    ]

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
    args = parser.parse_args(argv)
    verbose = not args.quiet

    info, fwd, fine, cortex, groups = build_forward(verbose)
    layouts = resolve_layouts(fwd, groups, verbose)
    rank = recipsiicos_rank(info, fwd, verbose)
    scenes, results = run_grid(info, fwd, fine, layouts, rank, verbose)
    header, blob = pack(
        scenes, results, fwd["source_rr"], cortex, sensor_layout(info), verbose
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
