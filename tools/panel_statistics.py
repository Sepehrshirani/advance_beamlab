"""Recompute the statistics the documentation quotes about the panel.

``doc/panel.rst`` says how much of the grid each method's localiser lights up,
how much an ABMC map made with a template an experiment could really supply
resembles the one the target's own waveform produces, and how often each filter
delivers more of the source than there was. Every one of those is a property of
the payload the page ships, so every one of them goes stale when the panel is
rebuilt. Reading them back off the built file is cheap next to the build, and it
is the only way to know the prose still describes the thing a reader clicks on.

Read the maps off the same 8-bit array the browser draws rather than the
floating-point one behind it: the doc describes what a reader sees, and the
min-max normalisation and cube root between the two change the answer. Count
amplitudes per source rather than per result, because a scene holding three
sources says three things about a filter and not one.

Usage::

    python tools/panel_statistics.py doc/_static/constraint_panel_data.js
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# Half the display maximum. The 8-bit map runs 0-255, so a grid point counts as
# lit when it is drawn at least half as bright as the brightest point.
HALF_MAXIMUM = 127.5

# The same threshold on the map the cube root was applied to. That root is
# invertible, so what the panel would have shown without it can be read back out
# of what it ships: norm > 1/2 is u8 > 255 * (1/2) ** (1/3). It is worth
# reporting because the comment in the builder justifies the compression by how
# little the uncompressed map lights up, and that claim is otherwise unchecked.
HALF_MAXIMUM_UNCOMPRESSED = 255.0 * 0.5 ** (1 / 3)


def load(path):
    """Return the payload header and the arrays it points into."""
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    # The file assigns the header first and appends the blob afterwards, so both
    # have to be decoded where they start rather than by splitting on newlines.
    parts = [
        decoder.raw_decode(chunk.split("=", 1)[1].lstrip())[0]
        for chunk in text.split("window.CONSTRAINT_PANEL")[1:]
    ]
    header = parts[0]
    blob = next(p for p in parts if isinstance(p, str))
    raw = gzip.decompress(base64.b64decode(blob))

    def array(name):
        entry = header[name]
        return np.frombuffer(
            raw, np.dtype(entry["dtype"]), entry["length"], entry["offset"]
        )

    maps = array("maps").reshape(len(header["results"]), -1).astype(float)
    # Millimetres, and centred, which is all the distances below need.
    positions = array("real_positions").reshape(-1, 3) / header["geometry_scale"]
    return header, maps, positions


def coverage(header, maps, threshold=HALF_MAXIMUM):
    """Fraction of the grid each method draws above half the display maximum."""
    lit = (maps > threshold).mean(axis=1)
    groups = {}
    for result, value in zip(header["results"], lit, strict=True):
        method = result["method"]
        key = method if method != "abmc" else f"abmc, {result['template']} template"
        groups.setdefault(key, []).append(value)
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def template_agreement(header, maps):
    """How far an ABMC map moves when the template is not the target's own.

    Per morphology, because that is what the difference is about: naming the band
    is most of what a template for a rhythm has to say, and much less than what a
    template for a burst has to say.
    """
    scenes = header["scenes"]
    index = {
        (r["scene"], r["template"]): i
        for i, r in enumerate(header["results"])
        if r["method"] == "abmc"
    }
    out = {}
    for (scene, template), i in index.items():
        if template == "truth":
            continue
        oracle = index.get((scene, "truth"))
        # A map that is flat everywhere has no correlation to report, and would
        # otherwise contribute a nan that swallows the average.
        if oracle is None or maps[oracle].std() == 0 or maps[i].std() == 0:
            continue
        morph = scenes[scene]["requested"]["morph"]
        r = float(np.corrcoef(maps[oracle], maps[i])[0, 1])
        out.setdefault(morph, {}).setdefault(template, []).append(r)
    return {
        morph: {t: float(np.mean(v)) for t, v in sorted(d.items())}
        for morph, d in out.items()
    }


def amplitudes(header):
    """How the delivered amplitude sits against one, per method.

    Below one is cancellation and above it leakage, where a positive off-diagonal
    adds a correlated neighbour rather than subtracting it. Split by head model
    and by trial count, because which way averaging pushes the leakage depends on
    the head model and quoting the two together hides that.
    """
    scenes = header["scenes"]
    cells, overall = {}, {}
    for result in header["results"]:
        wanted = scenes[result["scene"]]["requested"]
        ratio = np.asarray(result["amplitude_ratio"], float)
        cells.setdefault(
            (result["method"], wanted["head"], wanted["trials"]), []
        ).append(ratio)
        overall.setdefault(result["method"], []).append(ratio)
    out = {}
    for method, values in overall.items():
        flat = np.concatenate(values)
        above = flat[flat > 1]
        out[method] = dict(
            leaks=float((flat > 1).mean()),
            negative=float((flat < 0).mean()),
            median_when_leaking=float(np.median(above)) if above.size else float("nan"),
            largest_when_leaking=float(above.max()) if above.size else float("nan"),
            cells={
                (head, trials): float((np.concatenate(v) > 1).mean())
                for (m, head, trials), v in sorted(cells.items())
                if m == method
            },
        )
    return out


def localisation(header):
    """Where each method puts its peak, over the slices the documentation quotes.

    Peak error is in millimetres from the source the scene placed. The matched
    head model is an inverse crime and reads exactly zero for the methods that
    localise on the covariance they were given, so it is reported apart from the
    realistic one rather than pooled with it.
    """
    scenes = header["scenes"]
    rows = []
    for result in header["results"]:
        wanted = scenes[result["scene"]]["requested"]
        for error in result["peak_errors"]:
            rows.append((result["method"], result["template"], wanted, float(error)))

    def median(pick):
        v = [e for m, t, q, e in rows if pick(m, t, q)]
        return float(np.median(v)) if v else float("nan")

    out = {"correlated": {}, "single_matched": {}, "realistic": {}}
    for head in ("matched", "realistic"):
        for trials in (1, 100):
            out["correlated"][head, trials] = median(
                lambda m, t, q, h=head, n=trials: (
                    q["head"] == h and q["trials"] == n and q["corr"] > 0
                )
            )

    # The panel's own realistic best case for ABMC is the same-band template, so
    # that is the one to count misses for; the oracle template is quoted beside
    # it precisely because it has nothing left to be wrong about.
    for method, template in (
        ("lcmv", None),
        ("mcmv", None),
        ("recipsiicos", None),
        ("abmc", "truth"),
        ("abmc", "matched"),
    ):
        v = np.array(
            [
                e
                for m, tm, q, e in rows
                if m == method
                and (template is None or tm == template)
                and q["layout"] == "single"
                and q["head"] == "matched"
            ]
        )
        one = np.array(
            [
                e
                for m, tm, q, e in rows
                if m == method
                and (template is None or tm == template)
                and q["layout"] == "single"
                and q["head"] == "matched"
                and q["trials"] == 1
            ]
        )
        name = method if template is None else f"{method}, {template} template"
        out["single_matched"][name] = dict(
            configurations=int(v.size),
            missed=int((v > 0).sum()),
            missed_at_one_trial=int((one > 0).sum()),
            largest=float(v.max()) if v.size else float("nan"),
        )

    v = np.array([e for m, t, q, e in rows if q["head"] == "realistic"])
    out["realistic"] = dict(
        exactly_zero=int((v == 0).sum()),
        median=float(np.median(v)),
        largest=float(v.max()),
    )
    return out


def delivered(header, **wanted):
    """Delivered amplitude for one slice of the grid, per method."""
    scenes = header["scenes"]
    out = {}
    for result in header["results"]:
        q = scenes[result["scene"]]["requested"]
        if any(q[k] != v for k, v in wanted.items()):
            continue
        name = result["method"]
        if name == "abmc":
            name = f"abmc, {result['template']} template"
        out.setdefault(name, []).extend(result["amplitude_ratio"])
    return {k: float(np.median(v)) for k, v in sorted(out.items())}


def recorded(header, positions):
    """Report each method against the dipoles fitted to the recorded pair.

    Both peaks of each map are reported rather than the better one: quoting the
    smaller of the two distances flatters a method that put both of its peaks on
    the same side of the head, and knowing whether it did is the point.
    """
    scenes = {s["index"]: s for s in header["real_scenes"]}
    out = {}
    for result in header["real_results"]:
        scene = scenes[result["scene"]]
        # The full average in the narrow response window: the most favourable
        # data the panel offers, so any failure there is not a shortage of trials.
        if scene["condition"] not in ("Auditory", "Visual"):
            continue
        peaks = result["peaks"]
        name = result["method"]
        if result["template"] is not None:
            name = f"{name}, {result['template']} template"
        reference = np.asarray(scene["reference"], float)
        key = (scene["condition"], tuple(scene["window"]), scene["trials"])
        out.setdefault(
            key,
            dict(
                dipole_separation=float(np.linalg.norm(reference[0] - reference[1])),
                methods={},
            ),
        )["methods"][name] = dict(
            distances=[float(d) for d in result["reference_distance"]],
            peak_separation=float(
                np.linalg.norm(positions[peaks[0]] - positions[peaks[1]])
            )
            if len(peaks) > 1
            else float("nan"),
            gains=result["gains"],
        )
    return out


def main(argv=None):
    """Print the statistics for one built panel file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="a built constraint_panel_data.js")
    args = parser.parse_args(argv)

    header, maps, positions = load(args.payload)
    print(f"{maps.shape[0]} results over {maps.shape[1]} grid points\n")

    print("Grid above half the display maximum, averaged over every scene:")
    plain = coverage(header, maps, HALF_MAXIMUM_UNCOMPRESSED)
    for name, value in coverage(header, maps).items():
        print(
            f"  {name:28s} {100 * value:5.1f} per cent"
            f"   ({100 * plain[name]:4.1f} without the cube root)"
        )

    print("\nCorrelation with the map the target's own waveform produces:")
    for morph, per_template in sorted(template_agreement(header, maps).items()):
        row = "   ".join(f"{t:12s}{v:+.2f}" for t, v in per_template.items())
        print(f"  {morph:11s} {row}")

    print("\nDelivered amplitude above one, per source:")
    for method, s in sorted(amplitudes(header).items()):
        cells = "  ".join(
            f"{head[:4]}/{trials:<3d} {100 * v:5.1f}"
            for (head, trials), v in s["cells"].items()
        )
        print(
            f"  {method:12s} {100 * s['leaks']:5.1f} per cent   "
            f"negative {100 * s['negative']:4.1f}   "
            f"median when leaking {s['median_when_leaking']:.2f}, "
            f"largest {s['largest_when_leaking']:.2f}"
        )
        print(f"  {'':12s}   {cells}")

    loc = localisation(header)
    print("\nMedian peak error over the correlated scenes, in millimetres:")
    for (head, trials), value in sorted(loc["correlated"].items()):
        print(f"  {head:10s} {trials:>4} trial(s)   {value:5.1f}")
    print("\nSingle source, matched head model:")
    for name, s in loc["single_matched"].items():
        print(
            f"  {name:28s} missed {s['missed']:2d} of {s['configurations']:2d}"
            f" ({s['missed_at_one_trial']:2d} at one trial), by up to"
            f" {s['largest']:5.1f} mm"
        )
    r = loc["realistic"]
    print(
        f"\nRealistic head model: {r['exactly_zero']} configuration(s) report zero,"
        f" median {r['median']:.1f} mm, largest {r['largest']:.1f} mm"
    )

    print("\nDelivered amplitude, matched head model at r = 0.99 (median):")
    for trials in (1, 100):
        row = delivered(header, head="matched", corr=0.99, trials=trials)
        print(
            f"  {trials:>4} trial(s)   "
            + "   ".join(f"{k} {v:.2f}" for k, v in row.items())
        )

    print("\nThe recorded dataset, both peaks of each map:")
    for (condition, window, trials), s in sorted(recorded(header, positions).items()):
        if trials < 100:  # the fullest average only; the rest is the same story
            continue
        print(
            f"  {condition} {window[0]:.2f}-{window[1]:.2f} s, {trials} trials;"
            f" the fitted dipoles are {s['dipole_separation'] / 10:.1f} cm apart"
        )
        for name, m in sorted(s["methods"].items()):
            distances = " and ".join(f"{d:.0f}" for d in m["distances"])
            # Every entry the constraint table leaves free, which is where a
            # neighbour gets subtracted from the target or added to it.
            off = ", ".join(
                f"{value:+.2f}"
                for i, row in enumerate(m["gains"])
                for j, value in enumerate(row)
                if i != j
            )
            print(
                f"    {name:26s} {distances} mm from the nearer dipole;"
                f" its peaks are {m['peak_separation']:5.1f} mm apart;"
                f" off-diagonal {off}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
