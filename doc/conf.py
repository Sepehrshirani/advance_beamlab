"""Sphinx configuration for the advance_beamlab documentation."""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
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
    "sphinx_design",
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
# cortical-surface plots -- the only images in the gallery that show a source
# estimate on an actual brain.
#
# The PyVista scraper is opt-out rather than probed. Probing is unreliable: a
# machine can import PyVista and Qt, render a plain off-screen ``Plotter``
# successfully, and still fail at draw time inside the scraper with
# ``RenderWindowUnavailable`` -- and sphinx-gallery counts each such failure as a
# failed example, which aborts the whole build. Rather than guess, set
# ``BEAMLAB_NO_3D=1`` on machines without a working OpenGL stack (CI runners do
# this); everywhere else the scraper stays on and the brain figures are captured.
image_scrapers = ["matplotlib"]
_no_3d = os.environ.get("BEAMLAB_NO_3D", "") not in ("", "0", "false")

if _no_3d:
    print(
        "conf.py: BEAMLAB_NO_3D is set, so the cortical-surface figures will be "
        "missing from this build."
    )
else:
    import pyvista

    pyvista.OFF_SCREEN = True
    pyvista.BUILDING_GALLERY = True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        # ``start_xvfb`` was removed in PyVista 0.48; builds are also wrapped in
        # ``xvfb-run``, so this is best effort.
        starter = getattr(pyvista, "start_xvfb", None)
        if starter is not None:
            starter()

    import mne

    mne.viz.set_3d_backend("pyvistaqt")
    # Antialiasing is unreliable on software renderers and does not change the
    # captured images in any way that matters.
    mne.viz.set_3d_options(antialias=False, depth_peeling=False)
    image_scrapers.append("pyvista")


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


class _PedagogicalOrder:
    """Order the gallery so it reads as a course rather than a directory listing.

    Sphinx-gallery sorts by filename by default, which interleaved the
    simulations, the real-data examples and the applications arbitrarily. The
    intended reading order is: establish the problem on a simulation, show the
    method finding sources, then show it on real data -- and only then the
    second algorithm, the connectivity application, and the separate spike
    problem. Anything not listed sorts last, alphabetically, so adding an
    example never silently breaks the build.
    """

    _order = [
        "plot_mcmv_simulation.py",
        "plot_mcmv_source_discovery.py",
        "plot_mcmv_auditory.py",
        "plot_recipsiicos_simulation.py",
        "plot_recipsiicos_auditory.py",
        "plot_apw_mcmv_connectivity.py",
        "plot_abmc_localization.py",
    ]

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        try:
            return (0, self._order.index(filename))
        except ValueError:
            return (1, filename)


sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "within_subsection_order": _PedagogicalOrder,
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
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Sepehrshirani/advance_beamlab",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "MNE-Python",
            "url": "https://mne.tools",
            "icon": "fa-solid fa-brain",
        },
    ],
    "use_edit_page_button": False,
    "navbar_align": "left",
    "show_prev_next": True,
    # The mathematical background is a long single page; four levels of
    # in-page contents keeps its right-hand sidebar navigable.
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {"default_mode": "auto"}
