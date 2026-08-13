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
sources on a cortex, choose how correlated they are and what kind of activity
they carry, and read the table of :math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j`
values off the filter that comes back.

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
between the two with the correlation high is the clearest thing on this page.

ReciPSIICOS and ABMC get there differently, and the table shows how. ReciPSIICOS
never touches the constraint; it edits the covariance the filter is built from,
so the cancellation has less to work with. ABMC keeps a constraint but trades
the distortionless one against a match to a known waveform. Both leave a visible
signature in the table.

The panel opens with the problem each method is solving, and the line that
changes between them is short. LCMV minimises :math:`\mathbf{w}_1^{\mathsf T}
\mathbf{R}\,\mathbf{w}_1` subject to one scalar equation. MCMV minimises
:math:`\operatorname{tr}(\mathbf{W}^{\mathsf T}\mathbf{R}\,\mathbf{W})` subject
to :math:`\mathbf{W}^{\mathsf T}\mathbf{G} = \mathbf{I}`, which is the same
objective with the rest of the table nailed down. ReciPSIICOS keeps LCMV's
single constraint and replaces :math:`\mathbf{R}` with a projected
:math:`\tilde{\mathbf{R}}`. ABMC keeps the distortionless constraint and adds a
reward for output that resembles a known waveform, traded off by :math:`P`.

.. raw:: html

   <div id="constraint-panel" class="cp-root"></div>
   <link rel="stylesheet" href="_static/constraint_panel.css">
   <script src="_static/constraint_panel_data.js"></script>
   <script src="_static/constraint_panel.js"></script>

The controls, and what they are for
-----------------------------------

**Sources** chooses the scene. The geometric layouts control separation
directly: one source is the control case worth starting from, since there is
nothing to cancel against and every method should keep its amplitude; two at
2 cm and at 6 cm bracket the range where cancellation matters; three is the case
where a two-source constraint is no longer enough.

The anatomical layouts put a bilateral pair in a named region, which is the
shape most real questions take. One caveat is worth stating plainly rather than
burying: this is a **cortical surface source space, and it contains no
hippocampus**. The medial-temporal layout therefore uses parahippocampal cortex,
which is the nearest thing the model has. If you want hippocampal theta proper
you need a volume or mixed source space, and neither this panel nor the forward
behind it provides one.

**Activity** switches the rhythm: theta at 6 Hz, alpha at 10 Hz, beta at 20 Hz,
or a train of short bursts. The last is the regime ABMC was designed for, since
its extra term rewards output matching a known waveform, and it is worth
comparing the four methods there.

**Head model** switches between the two regimes described below, and it is worth
moving it with LCMV and MCMV in turn.

**Trials averaged** is the signal-to-noise axis, expressed as the thing an
experimenter actually controls. A single trial of an evoked MEG response sits
well below unit sensor signal-to-noise; this panel takes 0.2 as the single-trial
value, and averaging :math:`N` trials buys a factor of :math:`\sqrt{N}`, so the
three settings are 0.20, 0.63 and 2.0. Watching the localisation error fall as
you average is the most honest argument for collecting more trials that this
page can make.

The correlation is exact for any number of sources. Two sinusoids can be given
any correlation by a phase shift, but three cannot, so the simulation instead
mixes one shared factor and one private factor per source in proportions
:math:`\sqrt{r}` and :math:`\sqrt{1-r}`. Every pair then correlates at exactly
:math:`r` whatever the rhythm or the count.

How honest the simulation is
----------------------------

Two things about it are worth knowing, because they decide whether the numbers
mean anything.

**The interference is mostly brain, not sensor noise.** Three hundred sources
spread over the cortex carry 1/f activity, and that accounts for three quarters
of the interference; white sensor noise is the remaining quarter. This is closer
to a real recording, and it changes the answers: LCMV's recovered amplitude at
:math:`r = 0.99` moved from 0.23 to 0.49 when the background was added, because
the covariance the filter optimises against is a different matrix.

**The head model control decides whether the sources are where the beamformer
looks**, and the two settings exist because no single choice can show everything.

With **matched** selected, the sources sit exactly on points the beamformer
scans and the data are generated with the very leadfield being inverted. That is
an inverse crime, and its symptom is unmistakable: a single source is localised
to 0 mm at every signal-to-noise ratio, because no amount of noise moves a
matched filter off its own node. Read that zero as a property of the simulation,
not a claim about any method. What the matched setting buys is the only clean
view of what the *constraint* does, with everything else held exact: MCMV
recovers about 1.6 times the amplitude LCMV does at high correlation, and that
ratio is the whole point of the page.

With **realistic** selected, the sources are taken from the full 7498-vertex
forward while the beamformer still scans a decimated 751-point grid, so they sit
a few millimetres from anything any method can report. The localisation error
becomes what a localiser actually produces: none of it is zero, the median is
around 8 mm, and the tail runs to several centimetres where the methods
genuinely fail.

The price is worth understanding, because it is a real result rather than a
limitation of the panel. Under a mismatched model **every method loses its
amplitude, and by almost exactly the same amount**: the MCMV to LCMV ratio falls
from 1.6 to 1.01. A beamformer pointed at a slightly wrong location does not pass
a weakened copy of the source, it actively nulls it, so a few millimetres of
model error costs more amplitude than the entire cancellation effect the
constraint was protecting against. That is the single most practical thing on
this page: **the constraint only buys you anything if your head model and
coregistration are good enough to use it.**

The background activity is identical in both settings, so switching between them
changes the source placement and nothing else. Read the realistic errors as a
lower bound even so: the head model is still exact, the coregistration is still
perfect, and the noise is still stationary.

What you are looking at
-----------------------

Every configuration was computed ahead of time by
``tools/build_constraint_panel.py``, which calls
:func:`~advance_beamlab.constraint_demo` for each one. The page only draws the
results, because the documentation is a static site with no Python behind it.
The four methods in any one scene are handed byte-identical simulated data, and
the build asserts that rather than assuming it.

The constraint table is measured, not read out of the stored weights. Each entry
comes from passing a scene containing only source :math:`j` through the finished
filter and reading what filter :math:`i` returns, which is
:math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j` by definition. The methods keep
their weights in different spaces, so their stored arrays are not comparable;
what comes out of the public apply path is.

Colour means one thing throughout. A **method colour** is always that method's
estimate: the crosses on the cortex, the reconstruction trace, its line in the
sweep. **Grey** is always the truth: the rings marking the simulated sources,
the true waveforms. Sources are told apart by their row and label rather than by
colour, which would otherwise collide.

The localiser map is normalised to its own range and then compressed by a cube
root. That combination took two attempts. Colouring by rank was the first, on the
grounds that the four methods produce value distributions no common scale can
show at once, and it was wrong: a rank map is a uniform ramp whatever the
underlying shape, so all four rendered identically. Measured, the fraction of the
grid above half the display maximum was 0.500 for every method. Normalising
without compression goes too far the other way and leaves two bright dots on an
empty brain. The cube root sits between and separates them.

The markers on the simulated sources are drawn as unfilled rings for a related
reason: a filled marker sitting on a peak hides the one thing you are trying to
check. They are drawn at the sources' real positions, which are not grid points,
so a ring will not usually sit under a cross even when the method is right.

The sensor panel shows a field map beside the traces. The traces carry all 203
gradiometers over two and a half seconds: drag to move through channels and
time, scroll to zoom, or use the sliders. The topography says where on the
helmet the signal sits, and how little that pattern changes when you move the
sources a couple of centimetres. Both refer to the same instant, the peak of the
global field power inside the drawn window, and the vertical line on the traces
marks it. The map is interpolated in the browser from the array's own sensor
positions, drawn with iso-contours because a smooth wash of colour is hard to
read a gradient from, and it is diverging rather than sequential because the
field has a sign.

That recording is rebuilt in the browser rather than stored. It is exactly
:math:`\mathbf{G}\mathbf{s} + \sigma\boldsymbol{\varepsilon}`, and the simulation
draws :math:`\boldsymbol{\varepsilon}` from a generator seeded independently of
the waveforms, so every scene shares one interference realisation and differs
only in :math:`\sigma`. Storing that field once with the leadfield columns costs
a few hundred kilobytes where storing 203 channels for every scene would have
been megabytes of noise, which does not compress. It also makes the comparisons
paired: switching rhythm or correlation changes the signal and nothing else.

The true and recovered waveforms are drawn on one shared scale per scene, so a
reconstruction that lost half its amplitude looks like it lost half its
amplitude. Normalising each trace to its own peak would have made every method
look equally good, which is the opposite of the point.

The last panel sweeps the whole correlation axis at the current settings, for all
four methods at once, plotting both recovered amplitude and localisation error
with a marker where the sliders are. The sliders show one point; that chart shows
the curve those points lie on, which is a fairer way to judge a method than any
single setting.

If you would rather drive this from Python, with your own forward model and
without the precomputed grid, :func:`~advance_beamlab.constraint_explorer` opens
the same views as a Matplotlib window with live sliders, and
:func:`~advance_beamlab.constraint_demo` returns one scene as plain arrays.
