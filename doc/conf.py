"""Sphinx configuration for the advance_beamlab documentation."""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import os
import sys

import advance_beamlab

# -- Project information ----------------------------------------------------- #
project = "advance_beamlab"
author = "Sepehr Shirani and Muzhi Wang"
copyright = "2026, Sepehr Shirani and Muzhi Wang"
release = advance_beamlab.__version__
version = release

# -- General configuration --------------------------------------------------- #
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

# MyST: enable $...$ / $$...$$ math so the README's LaTeX renders here too.
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": False}
numpydoc_show_class_members = False

# Resolve the :footcite: references used throughout the docstrings.
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "mne": ("https://mne.tools/stable", None),
    # The connectivity metrics are delegated to mne-connectivity, so its API is
    # referenced throughout ``_connectivity.py`` and must resolve.
    "mne_connectivity": ("https://mne.tools/mne-connectivity/stable", None),
}

# ``sphinx_gallery_conf`` holds a function (the style reset), which Sphinx cannot
# pickle into its config cache. The warning is unavoidable and harmless, but it
# would fail the ``-W`` documentation build in CI.
suppress_warnings = ["config.cache"]

# Cross-references are checked in nitpicky mode; the entries below are targets
# that legitimately have no documentation page to point at.
nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "instance of mne.Info"),
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- sphinx-gallery (executed examples) -------------------------------------- #
# The gallery *runs* each ``examples/plot_*.py`` script and captures its figures.
# The examples fetch the MNE ``sample`` dataset on first run
# (:func:`mne.datasets.sample.data_path`) and the ReciPSIICOS example renders a
# cortical-surface plot, so a full build downloads the dataset and is slow (the
# whitened ReciPSIICOS example computes an O(N^2) correlation Gram). This is
# intentional: the built docs show real results on real MEG data.
#
# Figure capture uses the matplotlib scraper, plus the PyVista scraper for the
# cortical-surface plots. Those are the only images in the gallery that show a
# source estimate on an actual brain, so a build that quietly loses them is a
# materially worse build. The 3-D setup therefore reports what it did instead of
# failing silently, and setting ``BEAMLAB_REQUIRE_3D=1`` -- which CI does --
# turns a missing or broken 3-D backend into a build error rather than a gallery
# with the brain figures missing and nobody noticing.
image_scrapers = ["matplotlib"]
_require_3d = os.environ.get("BEAMLAB_REQUIRE_3D", "") not in ("", "0", "false")


def _try_enable_3d():
    """Enable the PyVista scraper only if this machine can actually render.

    Importing the backend is not enough: a headless runner will import PyVista
    and Qt happily and then fail at draw time with ``RenderWindowUnavailable``,
    which sphinx-gallery reports as a failed example and which aborts the whole
    build. So render a throwaway scene first and only enable the scraper if it
    succeeds. The alternative -- discovering this once per cortical-surface
    figure -- costs the entire documentation build.
    """
    import pyvista

    pyvista.OFF_SCREEN = True
    pyvista.BUILDING_GALLERY = True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        # ``start_xvfb`` was removed in PyVista 0.48; CI also wraps the build in
        # ``xvfb-run``, so this is best effort.
        starter = getattr(pyvista, "start_xvfb", None)
        if starter is not None:
            starter()

    import mne

    mne.viz.set_3d_backend("pyvistaqt")
    # Antialiasing is unreliable on software renderers and does not change the
    # captured images in any way that matters.
    mne.viz.set_3d_options(antialias=False, depth_peeling=False)

    plotter = pyvista.Plotter(off_screen=True)
    try:
        plotter.add_mesh(pyvista.Sphere())
        plotter.show(auto_close=False)
        plotter.screenshot(None)
    finally:
        plotter.close()


try:
    _try_enable_3d()
    image_scrapers.append("pyvista")
except Exception as exc:  # pragma: no cover - depends on the local 3-D stack
    message = (
        f"3-D rendering is unavailable ({type(exc).__name__}: {exc}); the "
        "cortical-surface figures will be missing from the gallery."
    )
    if _require_3d:
        raise RuntimeError(
            f"BEAMLAB_REQUIRE_3D is set but {message} Install the doc extras "
            "(pyvista, pyvistaqt, a Qt binding) and, on a headless machine, run "
            "the build under xvfb with a GLX-capable screen."
        ) from exc
    print(f"conf.py: {message}")


def _reset_mpl_style(gallery_conf, fname):
    """Journal-grade matplotlib defaults applied before each gallery example."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Wong colourblind-safe qualitative palette (Nature-style).
    wong = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ]
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.prop_cycle": cycler(color=wong),
            "grid.color": "#9e9e9e",
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
        }
    )


sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "image_scrapers": tuple(image_scrapers),
    "doc_module": ("advance_beamlab",),
    "backreferences_dir": "generated/backreferences",
    "reference_url": {"advance_beamlab": None},
    "remove_config_comments": True,
    "abort_on_example_error": False,
    "reset_modules": ("matplotlib", _reset_mpl_style),
}

# -- HTML output ------------------------------------------------------------- #
html_theme = "pydata_sphinx_theme"
html_title = f"advance_beamlab {release}"
