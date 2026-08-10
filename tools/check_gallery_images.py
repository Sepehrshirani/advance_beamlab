"""Fail the build if any gallery figure came out blank.

A blank figure is the failure mode that a build-succeeded check cannot see. When
the 3-D backend is misconfigured, ``stc.plot()`` does not raise -- the scraper
writes out a uniformly coloured placeholder, the example "passes", and the
gallery looks complete while showing nothing. That is exactly what happened when
the documentation was built with only the PyVista scraper and no
``mne.viz._brain._BrainScraper``: every cortical-surface figure was published as
a 300x300 single-colour PNG.

Checking file size is not enough, and was what let it through: a detailed
matplotlib figure at high DPI is larger than a genuine brain screenshot, so
"there are large images" says nothing about whether the brains rendered.
Counting distinct colours does say it.

Usage::

    python tools/check_gallery_images.py doc/_build/html [--min-3d N]
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Examples that render a cortical surface. If one of these produces no image at
# all, the scraper silently dropped it and the check must fail rather than pass
# on an empty set.
_EXAMPLES_WITH_3D = (
    "plot_mcmv_auditory",
    "plot_recipsiicos_auditory",
    "plot_fem_head_model",
)

# A real figure -- even a flat colour map -- has hundreds of distinct colours
# once it is antialiased and carries axes or a colour bar. Two or fewer means a
# blank canvas, possibly with a border.
_MIN_COLOURS = 8


def main():
    """Check every gallery figure and exit non-zero if any is blank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", type=Path)
    parser.add_argument(
        "--min-3d",
        type=int,
        default=1,
        help="minimum number of figures required from each 3-D example",
    )
    args = parser.parse_args()

    images = sorted((args.html_dir / "_images").glob("sphx_glr_*.png"))
    if not images:
        sys.exit(f"no gallery images at all under {args.html_dir / '_images'}")

    blank = []
    for path in images:
        try:
            arr = np.asarray(Image.open(path).convert("RGB"))
        except Exception as exc:  # truncated or not actually an image
            blank.append(f"{path.name}: unreadable ({exc})")
            continue
        if len(np.unique(arr.reshape(-1, 3), axis=0)) < _MIN_COLOURS:
            blank.append(f"{path.name} ({arr.shape[1]}x{arr.shape[0]}) is blank")

    print(f"gallery images: {len(images)}")
    for stem in _EXAMPLES_WITH_3D:
        n = len([p for p in images if p.name.startswith(f"sphx_glr_{stem}_")])
        print(f"  {stem}: {n} figure(s)")
        if n < args.min_3d:
            blank.append(f"{stem}: no figures were captured")

    if blank:
        sys.exit(
            "the 3-D backend did not render. Blank or missing figures:\n  "
            + "\n  ".join(blank)
        )
    print("all gallery figures carry real content")


if __name__ == "__main__":
    main()
