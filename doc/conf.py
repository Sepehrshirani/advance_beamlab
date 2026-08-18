"""Sphinx configuration for the advance_beamlab documentation."""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import os

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
# Figure capture goes through the matplotlib scraper alone, and every 3-D figure
# is screenshotted into a matplotlib axis by the example that draws it. That is
# deliberate, and it is the third arrangement tried:
#
# * The PyVista scraper alone published every cortical surface as a blank
#   placeholder. It does not know how to capture an ``mne.viz.Brain``, and it
#   fails by writing an empty PNG rather than by raising, so the gallery looked
#   complete while showing nothing.
# * MNE's own ``_BrainScraper`` captures them correctly, but it calls
#   ``brain.close()`` afterwards. On the CI runner that Qt teardown left the
#   *next* example unable to reach the virtual display, and the build aborted
#   with ``qt.qpa.xcb: could not connect to display``.
#
# Screenshotting explicitly sidesteps both: the capture is correct because
# ``Brain.screenshot`` renders it, and no window is ever torn down mid-build.
# It also means what appears in the docs is exactly what the example chose to
# show, rather than whatever a scraper happened to find open.
image_scrapers = ["matplotlib"]  # replaced below by the themed wrapper
_no_3d = os.environ.get("BEAMLAB_NO_3D", "") not in ("", "0", "false")

if _no_3d:
    print(
        "conf.py: BEAMLAB_NO_3D is set, so the cortical-surface figures will be "
        "missing from this build."
    )
else:
    import pyvista

    # Render into a window on the display rather than off-screen.
    # ``OFF_SCREEN = True`` sends VTK down an OSMesa path that the hosted runners
    # do not provide, and the cortical-surface figures then fail with
    # ``RenderWindowUnavailable``. CI supplies a virtual display via
    # ``pyvista/setup-headless-display-action``. Do not make this conditional on
    # ``DISPLAY``: macOS sets that variable from the XQuartz launchd socket while
    # rendering goes through Cocoa, so it says nothing about which path works.
    # This is the global default, and it is what the ``stc.plot()`` figures need.
    # Examples that build a plain ``pyvista.Plotter`` pass ``off_screen=True``
    # themselves, which lets them screenshot without opening a window --
    # ``Plotter.show`` blocks even with ``BUILDING_GALLERY`` set, which hung the
    # build until it was timed out.
    pyvista.OFF_SCREEN = False

    import mne

    mne.viz.set_3d_backend("pyvistaqt")
    # Antialiasing is unreliable on software renderers and does not change the
    # captured images in any way that matters.
    mne.viz.set_3d_options(antialias=False, depth_peeling=False)


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
            # Resolution. 220 dpi over a ten-inch figure is about 2200 px, which
            # is already better than twice the width the theme displays it at, so
            # it stays crisp on a high-density screen. Going further would only
            # add weight -- and every figure is now stored twice, once per theme.
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            # Stated rather than inherited, so the light copy is the same on any
            # machine and the dark copy is produced by changing exactly these.
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            # A grotesque sans, as the journals set their figures. Helvetica and
            # Arial are named first for a local build; DejaVu Sans ships with
            # matplotlib and is what a headless CI runner will actually use, so
            # the maths is set to match it rather than to a serif that would not.
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "dejavusans",
            # Type sizes in a single progression: labels one step under titles,
            # tick labels one step under labels. Nothing is set large enough to
            # compete with the body text around it.
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10,
            "axes.labelpad": 4.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "figure.titleweight": "bold",
            # Rules thin enough to read as structure rather than as content.
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.prop_cycle": cycler(color=wong),
            "axes.formatter.use_mathtext": True,
            "grid.color": "#9e9e9e",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.7,
            "legend.borderaxespad": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.5,
            "lines.solid_capstyle": "round",
            # Perceptually uniform by default, and no smoothing: an interpolated
            # matrix invents structure between the values that were computed.
            "image.cmap": "viridis",
            "image.interpolation": "nearest",
        }
    )


# -- Figures that follow the page's colour scheme ---------------------------- #
# Each figure is saved twice, once on white and once on the dark page ground,
# and both are emitted carrying the theme's own ``only-light``/``only-dark``
# classes so that exactly one is ever displayed.
#
# The alternative is what the theme does unaided: it puts a white card behind
# every image in dark mode and dims it (``filter: brightness(.8) contrast(1.2)``),
# so a figure sits in a bright rectangle on a dark page and its colours are no
# longer the ones the colormap specified. Marking the dark copy ``only-dark``
# opts out of both.
#
# Only the *chrome* is repainted -- ground, spines, tick marks, grid, and any
# text still at its default near-black. Data colours are never touched. The Wong
# palette and the perceptually uniform colormaps below are legible on either
# ground, and repainting a curve or a colormapped pixel would change what the
# figure states. Text an example coloured deliberately -- a label tinted to match
# its curve -- is likewise left alone, which is what the near-black test is for.
#
# Cortical surfaces are rasters that PyVista renders on MNE's default black
# background, so they already suit the dark page and match MNE's own
# documentation on the light one; they pass through untouched.

_FIG_DARK = {
    # The theme's own dark page background, so a figure sits flush against the
    # page instead of announcing itself as a panel.
    "ground": "#14181e",
    "ink": "#e2e8ee",  # titles, axis labels, tick labels
    "chrome": "#96a0ab",  # spines and tick marks
    "grid": "#8b949e",
    # A marked band: separated from the page, but not brighter than the data.
    "band": "#2b313a",
}


def _is_default_chrome(colour):
    """Whether *colour* is the near-black matplotlib uses when left alone.

    Anything an example coloured on purpose is brighter than this and is kept,
    because recolouring it would break a correspondence the figure was drawing.
    """
    from matplotlib.colors import to_rgba

    try:
        red, green, blue, alpha = to_rgba(colour)
    except (ValueError, TypeError):
        return False
    return alpha > 0 and max(red, green, blue) <= 0.35


def _is_light_neutral(colour):
    """Whether *colour* is a pale grey, the conventional shade for a marked band.

    Achromatic and light: a coloured fill is carrying data and is left alone,
    and so is a dark one, which already reads against the dark page.
    """
    from matplotlib.colors import to_rgba

    try:
        red, green, blue, alpha = to_rgba(colour)
    except (ValueError, TypeError):
        return False
    return alpha > 0 and max(red, green, blue) - min(red, green, blue) < 0.04 and (
        min(red, green, blue) > 0.70
    )


def _save_dark_variant(fig, path):
    """Save *fig* against the dark ground, then restore every colour changed."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    undo = []

    def swap(setter, old, new):
        undo.append((setter, old))
        setter(new)

    swap(fig.set_facecolor, fig.get_facecolor(), _FIG_DARK["ground"])

    furniture = set()
    for ax in fig.get_axes():
        furniture.add(id(ax.patch))
        swap(ax.set_facecolor, ax.get_facecolor(), _FIG_DARK["ground"])
        for spine in ax.spines.values():
            furniture.add(id(spine))
            if _is_default_chrome(spine.get_edgecolor()):
                swap(spine.set_edgecolor, spine.get_edgecolor(), _FIG_DARK["chrome"])
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            furniture.add(id(gridline))
            swap(gridline.set_color, gridline.get_color(), _FIG_DARK["grid"])
        for axis in (ax.xaxis, ax.yaxis):
            for tickline in axis.get_ticklines():
                furniture.add(id(tickline))
                if _is_default_chrome(tickline.get_color()):
                    swap(tickline.set_color, tickline.get_color(), _FIG_DARK["chrome"])
        legend = ax.get_legend()
        if legend is not None and legend.get_frame_on():
            frame = legend.get_frame()
            swap(frame.set_facecolor, frame.get_facecolor(), _FIG_DARK["ground"])
            swap(frame.set_edgecolor, frame.get_edgecolor(), _FIG_DARK["chrome"])

    # A rule an example drew in black -- ``axvline(0)`` marking stimulus onset, a
    # dashed marker at a chosen rank -- is furniture rather than a measurement,
    # and on the dark ground it would be a black line on a black field. The same
    # near-black test that protects deliberately tinted text identifies them, so
    # a curve carrying data keeps the colour it was given. Grid and tick lines
    # are skipped: they were coloured just above, and are Line2D objects too.
    for line in fig.findobj(Line2D):
        if id(line) in furniture:
            continue
        if _is_default_chrome(line.get_color()):
            swap(line.set_color, line.get_color(), _FIG_DARK["ink"])
        if _is_default_chrome(line.get_markeredgecolor()):
            swap(
                line.set_markeredgecolor,
                line.get_markeredgecolor(),
                _FIG_DARK["ink"],
            )

    # Shading drawn to mark a region -- an excluded range behind the curves --
    # is picked in a pale neutral grey because the page is white. On the dark
    # page that same grey is the brightest thing in the figure, so the band a
    # reader is told to disregard becomes the first thing they look at. Only
    # achromatic, light fills are moved: a coloured band is carrying data.
    for patch in fig.findobj(Patch):
        if id(patch) in furniture or patch is fig.patch:
            continue
        if _is_light_neutral(patch.get_facecolor()):
            swap(patch.set_facecolor, patch.get_facecolor(), _FIG_DARK["band"])

    # Covers titles, axis labels, tick labels, legend entries and annotations in
    # one pass, including those belonging to colorbars.
    for text in fig.findobj(Text):
        if _is_default_chrome(text.get_color()):
            swap(text.set_color, text.get_color(), _FIG_DARK["ink"])

    try:
        fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor="none")
    finally:
        for setter, old in reversed(undo):
            setter(old)


def _theme_pair_rst(light_path, dark_path, src_dir, alt, css_class):
    """Emit one figure as its two themed copies."""
    import os as _os

    def relative(path):
        rel = _os.path.relpath(str(path), src_dir).replace(_os.sep, "/")
        return "/" + rel.lstrip("/")

    lines = []
    for path, theme in ((light_path, "only-light"), (dark_path, "only-dark")):
        lines += [
            f".. image:: {relative(path)}",
            f"   :alt: {alt}",
            f"   :class: {css_class} {theme}",
            "",
        ]
    return "\n".join(lines)


def _dual_theme_scraper(block, block_vars, gallery_conf):
    """Capture every open matplotlib figure once per colour scheme.

    Exactly one path is drawn from the iterator per figure, the same as the
    stock scraper, so ``sphinx_gallery_thumbnail_number`` in an example keeps
    pointing at the figure its author counted. The dark copy is written beside
    it under a derived name that the numbering never sees.
    """
    import os as _os
    import re as _re

    import matplotlib.pyplot as plt

    image_path_iterator = block_vars["image_path_iterator"]
    fignums = plt.get_fignums()
    css_class = "sphx-glr-single-img" if len(fignums) == 1 else "sphx-glr-multi-img"

    blocks = []
    for fig_num, image_path in zip(fignums, image_path_iterator):
        fig = plt.figure(fig_num)
        light_path = image_path
        stem, suffix = _os.path.splitext(str(light_path))
        dark_path = f"{stem}_dark{suffix}"

        fig.savefig(light_path, facecolor=fig.get_facecolor(), edgecolor="none")
        _save_dark_variant(fig, dark_path)

        title = fig._suptitle.get_text() if fig._suptitle is not None else ""
        if not title and fig.get_axes():
            title = fig.get_axes()[0].get_title()
        if not title:
            name = _os.path.splitext(_os.path.basename(str(light_path)))[0]
            title = _re.sub(r"[-,_]", " ", name[9:-4])
        alt = _re.sub(r"\s+", " ", title).strip() or "figure"

        blocks.append(
            _theme_pair_rst(
                light_path, dark_path, gallery_conf["src_dir"], alt, css_class
            )
        )

    plt.close("all")
    return "\n".join(blocks)


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
        "plot_abmc_auditory.py",
        "plot_fem_head_model.py",
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
    # The themed scraper stands in for the plain "matplotlib" entry above; the 3-D
    # entries, if any, are kept as they are.
    "image_scrapers": tuple(
        _dual_theme_scraper if s == "matplotlib" else s for s in image_scrapers
    ),
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
