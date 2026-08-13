"""Precompute the constraint panel that ships with the documentation.

The panel lets a reader place two correlated sources on a real cortex, watch what
they look like at the sensors, and see how each beamformer localises and
reconstructs them. The point of it is the constraint table: LCMV pins its own
gain to one and leaves the gain at the *other* source free, and when the two are
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
CORRELATIONS = (0.0, 0.5, 0.8, 0.95, 0.99)
SEPARATIONS = (0.02, 0.04, 0.08)
SNRS = (1.0, 3.0, 10.0)

# Ten seconds at 200 Hz. The covariance is estimated from these samples, and at
# 203 gradiometers anything shorter is rank deficient: at 200 samples MNE reports
# both "too few samples" and a negative smallest eigenvalue, and the numbers move
# with it (ReciPSIICOS returned an off-diagonal gain of -1.19 at 1000 samples and
# -0.88 at 2000). A panel whose whole content is the numbers cannot be built on
# that, so the simulation is long even though only a slice of it is drawn.
N_TIMES = 2000
DISPLAY_SAMPLES = 250
SFREQ = 200.0
SEED = 0
# Every 8th source-space vertex. 938 sources keeps ReciPSIICOS, whose Gram is
# quadratic in the source count, at about two seconds per configuration while
# still being a real cortex rather than a sphere.
DECIMATION = 8
N_SENSOR_TRACES = 20
N_BACKDROP = 4000

# Warnings that mean the numbers cannot be trusted. The build stops on these
# rather than printing them into a log nobody reads.
FATAL_WARNINGS = ("Too few samples", "will likely be unstable", "rank estimate")


def build_forward(verbose=True):
    """Return the ``sample`` subject's gradiometer forward, decimated."""
    data_path = mne.datasets.sample.data_path()
    meg = data_path / "MEG" / "sample"
    info = mne.io.read_info(meg / "sample_audvis-ave.fif")
    info = mne.pick_info(
        info, mne.pick_types(info, meg="grad", eeg=False, exclude="bads")
    )
    fwd = mne.read_forward_solution(meg / "sample_audvis-meg-oct-6-fwd.fif")
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True)
    fwd = mne.pick_channels_forward(fwd, info["ch_names"], ordered=True)

    # The anatomical backdrop the panel draws behind the source grid. Thinned to
    # a few thousand points: it only has to read as a brain, and the browser
    # redraws every point on every rotation frame.
    cortex = np.vstack([s["rr"][s["vertno"]] for s in fwd["src"]])
    cortex = cortex[:: max(1, len(cortex) // N_BACKDROP)]

    verts = [s["vertno"][::DECIMATION] for s in fwd["src"]]
    n = sum(len(v) for v in verts)
    stc = mne.SourceEstimate(np.ones((n, 1)), verts, tmin=0, tstep=1, subject="sample")
    fwd = mne.forward.restrict_forward_to_stc(fwd, stc)
    if verbose:
        print(
            f"forward: {fwd['sol']['data'].shape[1]} sources, "
            f"{fwd['nchan']} gradiometers; backdrop {len(cortex)} points"
        )
    return info, fwd, cortex


def run_grid(info, fwd, verbose=True):
    """Every (scene, method) combination, as plain arrays.

    A scene is a separation, a correlation and a signal-to-noise ratio. The
    simulated data depend on those three and on the seed, but not on the method,
    so the four methods in one scene really are being handed identical data;
    that is asserted below rather than assumed, because the whole comparison
    rests on it.
    """
    scenes, results = [], []
    t_start = time.time()
    for sep in SEPARATIONS:
        for corr in CORRELATIONS:
            for snr in SNRS:
                scene_idx = len(scenes)
                scene = None
                for method in METHODS:
                    t0 = time.time()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        demo = constraint_demo(
                            info,
                            fwd,
                            method=method,
                            separation=sep,
                            correlation=corr,
                            snr=snr,
                            n_times=N_TIMES,
                            sfreq=SFREQ,
                            seed=SEED,
                        )
                    fatal = [
                        str(w.message)
                        for w in caught
                        if any(f in str(w.message) for f in FATAL_WARNINGS)
                    ]
                    if fatal:
                        raise RuntimeError(
                            f"{method} at separation {sep}, correlation {corr}, "
                            f"SNR {snr} produced a covariance that cannot be "
                            f"trusted:\n  " + "\n  ".join(sorted(set(fatal)))
                        )
                    if scene is None:
                        scene = dict(
                            sources=[int(s) for s in demo.sources],
                            separation=float(demo.separation),
                            correlation=float(demo.correlation),
                            requested=dict(sep=sep, corr=corr, snr=snr),
                            true_tcs=demo.true_tcs,
                            sensor=demo.sensor_data,
                            times=demo.times,
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
                    results.append(
                        dict(
                            scene=scene_idx,
                            method=method,
                            gains=demo.gains.tolist(),
                            amplitude_ratio=demo.amplitude_ratio.tolist(),
                            peak_errors=(demo.peak_errors * 1000).tolist(),
                            power_map=demo.power_map,
                            reconstructed=demo.reconstructed,
                        )
                    )
                    if verbose:
                        print(
                            f"  scene {scene_idx:2d} ({method:>11}) "
                            f"sep {demo.separation * 100:.1f} cm  "
                            f"r {demo.correlation:+.2f}  snr {snr:>4}  "
                            f"off-diag {demo.gains[0, 1]:+.3f}  "
                            f"recovered {demo.amplitude_ratio[0]:.3f}  "
                            f"[{time.time() - t0:.1f}s]"
                        )
                scenes.append(scene)
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


def pack(scenes, results, positions, cortex, verbose=True):
    """Quantise everything into one gzipped buffer plus a JSON header."""
    packer = Packer()
    header = dict(
        methods=list(METHODS),
        correlations=list(CORRELATIONS),
        separations=list(SEPARATIONS),
        snrs=list(SNRS),
        n_times=DISPLAY_SAMPLES,
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

    # Localiser maps, stored as display rank so the four methods, whose value
    # distributions are wildly different, are comparable at a glance. The panel
    # says so in its colour bar; this is the same choice the local explorer makes.
    maps = []
    for r in results:
        m = np.asarray(r["power_map"], float)
        order = np.argsort(np.argsort(m))
        maps.append(np.round(order / max(len(order) - 1, 1) * 255))
    header["maps"] = packer.add(np.concatenate(maps), np.uint8)

    # Waveforms. Each scene is normalised by its own true peak and the
    # reconstruction is scaled by that *same* factor, so amplitude loss stays
    # visible; normalising them separately would hide the effect the panel is for.
    cut = slice(None, DISPLAY_SAMPLES)
    true_scale = [float(np.abs(s["true_tcs"]).max()) or 1.0 for s in scenes]
    header["true_tcs"] = packer.add(
        np.concatenate(
            [
                _quantise(s["true_tcs"][:, cut] / k, 20000).ravel()
                for s, k in zip(scenes, true_scale, strict=True)
            ]
        ),
        np.int16,
    )
    header["reconstructed"] = packer.add(
        np.concatenate(
            [
                _quantise(
                    r["reconstructed"][:, cut] / true_scale[r["scene"]], 20000
                ).ravel()
                for r in results
            ]
        ),
        np.int16,
    )
    header["waveform_scale"] = 20000.0

    # A sample of sensor traces, evenly spaced across the array.
    sensor = []
    for s in scenes:
        data = np.asarray(s["sensor"])
        pick = np.linspace(0, data.shape[0] - 1, N_SENSOR_TRACES).astype(int)
        d = data[pick][:, cut]
        sensor.append(_quantise(d / (np.abs(d).max() or 1.0), 20000).ravel())
    header["sensor"] = packer.add(np.concatenate(sensor), np.int16)
    header["n_sensor_traces"] = N_SENSOR_TRACES

    header["scenes"] = [
        dict(
            sources=s["sources"],
            separation=s["separation"],
            correlation=s["correlation"],
            requested=s["requested"],
        )
        for s in scenes
    ]
    header["results"] = [
        dict(
            scene=r["scene"],
            method=r["method"],
            gains=[[round(v, 5) for v in row] for row in r["gains"]],
            amplitude_ratio=[round(v, 5) for v in r["amplitude_ratio"]],
            peak_errors=[round(v, 2) for v in r["peak_errors"]],
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

    info, fwd, cortex = build_forward(verbose)
    scenes, results = run_grid(info, fwd, verbose)
    header, blob = pack(scenes, results, fwd["source_rr"], cortex, verbose)

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
