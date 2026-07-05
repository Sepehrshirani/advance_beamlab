"""Sphinx configuration for the mne-beamlab documentation."""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import mne_beamlab

# -- Project information ----------------------------------------------------- #
project = "mne-beamlab"
author = "Sepehr Shirani and Muzhi Wang"
copyright = "2026, Sepehr Shirani and Muzhi Wang"
release = mne_beamlab.__version__
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

    pyvista.OFF_SCREEN = False
    pyvista.BUILDING_GALLERY = True

    import mne

    mne.viz.set_3d_backend("pyvistaqt")
    image_scrapers.append("pyvista")
except Exception:
    pass

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "image_scrapers": tuple(image_scrapers),
    "doc_module": ("mne_beamlab",),
    "backreferences_dir": "generated/backreferences",
    "reference_url": {"mne_beamlab": None},
    "remove_config_comments": True,
    "abort_on_example_error": False,
}

# -- HTML output ------------------------------------------------------------- #
html_theme = "pydata_sphinx_theme"
html_title = f"mne-beamlab {release}"
