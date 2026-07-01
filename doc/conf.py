"""Sphinx configuration for the mne-beamlab documentation."""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import mne_beamlab

# -- Project information ----------------------------------------------------- #
project = "mne-beamlab"
author = "Sepehr Shirani"
copyright = "2026, Sepehr Shirani"
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

# -- HTML output ------------------------------------------------------------- #
html_theme = "pydata_sphinx_theme"
html_title = f"mne-beamlab {release}"
