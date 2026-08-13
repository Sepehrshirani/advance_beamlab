.. _constraint-panel:

What a beamformer constraint actually does
==========================================

The algebra of a beamformer is short enough to fit on one line and hard enough
to leave most people none the wiser. An LCMV filter for source :math:`i` is the
one that minimises output power subject to :math:`\mathbf{w}_i^{\mathsf T}
\mathbf{g}_i = 1`. That constraint fixes one number. It says nothing at all
about :math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j`, the gain the same filter has
at some *other* source, and leaving that free is the entire story of what goes
wrong with correlated sources.

This page is here so you can watch it happen rather than take it on trust. Put
two sources on a cortex, choose how correlated they are, and read the table of
:math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j` values off the filter that comes
back.

Start with LCMV and the correlation slider at zero. The off-diagonal entry sits
near zero on its own, because with nothing to cancel there is nothing to gain by
moving it. Now push the correlation up. The off-diagonal goes sharply negative,
and the recovered amplitude collapses in step with it. Nothing has gone wrong:
that large negative gain is the *best* answer to the question LCMV was asked,
because subtracting a correlated partner is an efficient way to minimise output
power. The filter is doing its job, and its job is not quite what you wanted.

Then switch to MCMV without touching anything else. The off-diagonal pins to
zero, because MCMV constrains the whole table to the identity rather than one
diagonal entry, and the amplitude stops moving with the correlation. Switching
between the two with the slider high is the clearest thing on this page.

ReciPSIICOS and ABMC get there differently, and the table shows how. ReciPSIICOS
never touches the constraint; it edits the covariance the filter is built from,
so the cancellation has less to work with. ABMC keeps a constraint but trades
the distortionless one against a match to a known waveform. Both leave a visible
signature in the table.

.. raw:: html

   <div id="constraint-panel" class="cp-root"></div>
   <link rel="stylesheet" href="_static/constraint_panel.css">
   <script src="_static/constraint_panel_data.js"></script>
   <script src="_static/constraint_panel.js"></script>

A few things worth knowing about what you are looking at.

Every configuration here was computed ahead of time by
``tools/build_constraint_panel.py``, which calls
:func:`~advance_beamlab.constraint_demo` for each one. The page only draws the
results, because the documentation is a static site with no Python behind it.
The four methods in any one scene are handed byte-identical simulated data, and
the build asserts that rather than assuming it.

The scene is a real cortex, the ``sample`` subject's source space decimated to
938 points, with its 203 gradiometers. The covariance behind every filter is
estimated from ten seconds of data, which is about ten samples per channel. That
is not incidental: at 200 samples the covariance is rank deficient, its smallest
eigenvalue goes negative, and the numbers in the table move enough to change what
the panel appears to teach.

The constraint table is measured, not read out of the stored weights. Each entry
comes from passing a scene containing only source :math:`j` through the finished
filter and reading what filter :math:`i` returns, which is
:math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j` by definition. The methods keep
their weights in different spaces, so their stored arrays are not comparable;
what comes out of the public apply path is.

The localiser map is coloured by rank rather than by value. The four methods
produce wildly different distributions, and any scaling that suits one of them
flattens another into a single shade. Rank is the honest compromise and the
colour bar says so. The markers on the two simulated sources are drawn as
unfilled rings for the same reason: a filled marker sitting on a peak hides the
one thing you are trying to check.

Finally, the amplitude. The true and recovered waveforms are drawn on one shared
scale per scene, so a reconstruction that lost half its amplitude looks like it
lost half its amplitude. Normalising each trace to its own peak would have made
every method look equally good, which is the opposite of the point.

If you would rather drive this from Python, with your own forward model and
without the precomputed grid,
:func:`~advance_beamlab.constraint_explorer` opens the same four views as a
Matplotlib window with live sliders, and
:func:`~advance_beamlab.constraint_demo` returns one scene as plain arrays.
