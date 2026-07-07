"""Sphinx configuration for the advance_beamlab documentation."""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

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
}

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
# 3-D brain plot when PyVista/Qt are available. If they are not, the gallery
# still builds -- the brain plot is simply not captured.
image_scrapers = ["matplotlib"]
try:
    import pyvista

    pyvista.OFF_SCREEN = True
    pyvista.BUILDING_GALLERY = True

    import mne

    mne.viz.set_3d_backend("pyvistaqt")
    image_scrapers.append("pyvista")
except Exception:
    pass


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
