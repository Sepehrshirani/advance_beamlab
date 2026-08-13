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

The panel opens with the problem each method is solving, and the line that
changes between them is short. LCMV minimises :math:`\mathbf{w}_1^{\mathsf T}
\mathbf{R}\,\mathbf{w}_1` subject to one scalar equation. MCMV minimises
:math:`\operatorname{tr}(\mathbf{W}^{\mathsf T}\mathbf{R}\,\mathbf{W})` subject
to :math:`\mathbf{W}^{\mathsf T}\mathbf{G} = \mathbf{I}`, which is the same
objective with the rest of the table nailed down. ReciPSIICOS keeps LCMV's single
constraint and replaces :math:`\mathbf{R}` with a projected
:math:`\tilde{\mathbf{R}}`. ABMC keeps the distortionless constraint and adds a
reward for output that resembles a known waveform, traded off by :math:`P`.

The source view draws both the truth and the estimate. Rings are the simulated
sources; crosses are the two strongest points of the localiser, which is exactly
what :func:`~advance_beamlab.constraint_demo` measures its error against, and a
dashed line joins each estimate to the source it missed. Watching the crosses
walk away from the rings as the correlation rises is the localisation half of the
story, and the amplitude collapse in the waveform panel is the other half.

The bottom panel sweeps the whole correlation axis at the current settings, for
all four methods at once, with a marker where the sliders are. The sliders show
one point; that chart shows the curve those points lie on, which is a fairer way
to judge a method than any single setting.

Two further controls change the scene rather than the method. **Sources** puts
one, two or three of them on the cortex. One is the control case and worth
starting from: there is nothing to cancel against, so LCMV keeps its amplitude
and the off-diagonal question does not arise. Three is the case where a
two-source constraint is no longer enough, and MCMV has to be told about all of
them. **Morphology** switches between a modulated 10 Hz rhythm and a train of
short bursts. The second is the regime ABMC was designed for, since its extra
term rewards output matching a known waveform, and it is worth comparing the two
with ABMC selected.

The correlation is exact for any number of sources. Two sinusoids can be given
any correlation by a phase shift, but three cannot, so the simulation instead
mixes one shared factor and one private factor per source in proportions
:math:`\sqrt{r}` and :math:`\sqrt{1-r}`. Every pair then correlates at exactly
:math:`r` whatever the morphology or the count.

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

The localiser map is normalised to its own range and then compressed by a cube
root. That combination took two attempts. Colouring by rank was the first, on the
grounds that the four methods produce value distributions no common scale can
show at once, and it was wrong: a rank map is a uniform ramp whatever the
underlying shape, so all four rendered identically. Measured, the fraction of the
grid above half the display maximum was 0.500 for every method. Normalising
without compression goes too far the other way and leaves two bright dots on an
empty brain. The cube root sits between and separates them, and the same
measurement now reads 6.5, 0.6, 3.6 and 13.5 per cent.

The markers on the simulated sources are drawn as unfilled rings for a related
reason: a filled marker sitting on a peak hides the one thing you are trying to
check.

The sensor panel shows a field map beside the traces. The traces carry all 203
gradiometers, not a summary of them: drag to scroll through channels and time, or
use the sliders beside them. The topography says where on the helmet the signal
sits, and how little that pattern changes when you move the sources a couple of
centimetres. Both refer to the same instant, the peak of the global field power
inside the drawn window, and the vertical line on the traces marks it. The map is
interpolated in the browser from the array's own sensor positions, drawn with
iso-contours because a smooth wash of colour is hard to read a gradient from, and
it is diverging rather than sequential because the field has a sign.

That recording is rebuilt in the browser rather than stored. It is exactly
:math:`\mathbf{G}\mathbf{s} + \sigma\boldsymbol{\varepsilon}`, and the simulation
draws :math:`\boldsymbol{\varepsilon}` from a generator seeded independently of
the waveforms, so every scene shares one noise realisation and differs only in
:math:`\sigma`. Storing that field once with the leadfield columns costs about
300 kB where storing 203 channels for all 180 scenes would have been several
megabytes of noise, which does not compress. It also makes the comparisons paired:
switching morphology or correlation changes the signal and nothing else.

Finally, the amplitude. The true and recovered waveforms are drawn on one shared
scale per scene, so a reconstruction that lost half its amplitude looks like it
lost half its amplitude. Normalising each trace to its own peak would have made
every method look equally good, which is the opposite of the point.

If you would rather drive this from Python, with your own forward model and
without the precomputed grid,
:func:`~advance_beamlab.constraint_explorer` opens the same four views as a
Matplotlib window with live sliders, and
:func:`~advance_beamlab.constraint_demo` returns one scene as plain arrays.
