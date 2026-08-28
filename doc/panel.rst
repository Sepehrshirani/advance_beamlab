.. _constraint-panel:

Interactive panel
=================

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
so the cancellation has less to work with. ABMC does not trade the
distortionless constraint away either -- its diagonal reads exactly 1.000000 in
every configuration the panel holds. What flattens its off-diagonal is that it
solves against a sparse Bayesian estimate of the covariance rather than the
sample one; the template term is a mild steer on top, not a second constraint.
Measured over the whole grid its off-diagonal has a median magnitude of 0.03,
against 0.29 for LCMV and 0.18 for ReciPSIICOS, with MCMV exactly zero by
construction. Both leave a visible signature in the table.

The panel opens with the problem each method is solving, the sizes of every
quantity in it at the current settings, and a note on which head model is in use
and what that costs. The sizes are there because the algebra never states them
and readers reasonably ask: with a fixed-orientation model a single point's
leadfield is one column of 203 numbers, not three, and the constraint table is
only as big as the number of sources you have put down.

That table comes in two parts, and the second changes with the method. The first
holds what all four share: the recording, the covariance, the leadfields, the
filters, the reconstruction. The second holds what the selected method adds on
top, which is where the methods actually differ and where a single fixed table
said nothing at all. Choose MCMV and it names the :math:`n \times n` matrix that
is inverted, the one whose conditioning fails when two sources are too close to
tell apart. Choose ReciPSIICOS and it gives the size of the space the projection
happens in, which is neither the sensor space nor the source space but the space
of vectorised covariances: :math:`q^2` dimensions for :math:`q` virtual sensors,
with a projector :math:`q^2 \times q^2` on a side. That is the largest array
anything on this page touches, and it is why the method reduces 203 channels to
a few tens of virtual sensors before it starts. Choose ABMC and it names the
template, the per-point lag search, the trade-off :math:`P`, the multiplier, and
the iteration counts of both the sparse Bayesian covariance it uses and the
gradient descent it replaces with a closed form. Every one of those numbers is
written by the build from the functions themselves, so it cannot drift away from
the code it describes.

Two terms in that table get explained underneath it, because both are load
bearing and neither is obvious. A **constrained** source is one the method has
been told about and writes an equation for, which is exactly what separates the
first two methods: LCMV builds one filter per source, each knowing only its own
location, while MCMV builds one system covering all of them. And **fixed** against
**free** orientation decides whether a point contributes one leadfield column or
three, which changes the size of every array downstream and is worth choosing
deliberately: fixed when the cortical normal is a physiological statement you
trust, free when it is not, as for a volume source that has no normal at all.

The line that changes between the methods is short. LCMV minimises :math:`\mathbf{w}_1^{\mathsf T}
\mathbf{R}\,\mathbf{w}_1` subject to one scalar equation. MCMV minimises
:math:`\operatorname{tr}(\mathbf{W}^{\mathsf T}\mathbf{R}\,\mathbf{W})` subject
to :math:`\mathbf{W}^{\mathsf T}\mathbf{G} = \mathbf{I}`, which is the same
objective with the rest of the table nailed down. ReciPSIICOS keeps LCMV's
single constraint and replaces :math:`\mathbf{R}` with a projected
:math:`\tilde{\mathbf{R}}`. ABMC keeps the distortionless constraint and adds a
reward for output that resembles a known waveform, traded off by :math:`P`.

ReciPSIICOS carries one free parameter, the projection rank, and it is worth
knowing where the panel's came from. :math:`K^*` is chosen once by the 45-degree
criterion of Kuznetsova et al., here **84 out of** :math:`q^2 = 5625`.

The rank has to be chosen in the space it is spent in, and that is easier to get
wrong than it looks. Both the selection curve and the filter first reduce the
203 channels to :math:`q` virtual sensors by a truncated SVD of the *whitened*
leadfield, so anything that changes the whitener changes :math:`q`, and a rank
drawn from one :math:`q^2` and spent in another quietly stops meaning what it
says. This page got it wrong twice. Selecting the rank from the forward alone,
with no noise covariance, chose at :math:`q = 49` and spent at :math:`q = 78`.
Selecting it from a scene that was missing the finer truth forward the rest of
the grid uses changed which leadfield the interference was projected through,
and so chose at :math:`q = 78` and spent at :math:`q = 75` -- 168 where the
criterion wanted 84, a projector of twice the intended rank. The rank and its
:math:`q` are now reported together in the sizes table, because neither number
says anything on its own.

.. raw:: html

   <div id="advance-beamlab-panel" class="cp-root"></div>
   <link rel="stylesheet" href="_static/constraint_panel.css?v=8">
   <script src="_static/constraint_panel_data.js?v=5"></script>
   <script src="_static/constraint_panel.js?v=8"></script>

The controls
------------

**Sources** chooses the scene, and the options come in three groups.

The **geometry** group is the abstract case, where separation is set directly:
one source is the control worth starting from, since there is nothing to cancel
against and every method should keep its amplitude; two at 2 cm and at 6 cm
bracket the range where cancellation matters; three is the case where a
two-source constraint is no longer enough.

The **bilateral** group puts the same structure in both hemispheres:
hippocampus, amygdala, thalamus, auditory, visual and motor cortex, and lateral
prefrontal cortex.

The **circuits** group is the one most likely to match a real question. Each is
a within-hemisphere pair that computational psychiatry and clinical
neurophysiology actually study: hippocampus to medial prefrontal cortex, which
is the classic theta-coupling circuit in memory and anxiety; amygdala to
ventromedial prefrontal cortex, for emotion regulation; thalamus to motor
cortex, for thalamocortical drive, spindles and Parkinsonian beta; lateral
prefrontal to superior parietal, the frontoparietal executive network; anterior
cingulate to insula, the salience network; and medial prefrontal to precuneus,
the core of the default mode. Setting the correlation slider on one of these is
asking exactly the question a coupling analysis asks, and the constraint table
then shows what your beamformer will do to the answer.

Cortical targets are the grid point nearest the spatial centroid of the matching
parcellation label, and the build asserts each representative really falls inside
the label it is named for. Subcortical targets come from the segmentation.

The hippocampus needs a word, because a cortical surface has none. The source
space here is **mixed**: the surface plus discrete sources inside the
subcortical labels of the subject's ``aseg``, three millimetres apart. That
volume grid is decimated five to one for the scan, which still leaves 30 points
in each hippocampus, 15 and 13 in the amygdalae and 59 and 64 in the thalami:
211 subcortical points against 586 cortical ones. The structures are genuinely
searchable rather than cortical stand-ins, and they are decimated far less
aggressively than the surface because they are small.

That comes with a modelling choice worth stating. A volume source carries no
surface normal, so a fixed-orientation forward cannot be built from one
directly. Each subcortical source is given the short principal axis of its own
structure as an orientation, which for the hippocampus approximates the
somato-dendritic direction its pyramidal cells are organised along. It is a
simplification, and a real hippocampal study would want to think harder about
it.

**Template ABMC seeks** appears only with ABMC selected, because it is the only
method that has one, and it is that method's most consequential setting. ABMC
does not look for power; it is steered to the location whose output best matches
a waveform you give it, at the best lag. Choosing that waveform is a scientific
decision: to localise hippocampal theta you hand it theta. The control offers
the target's own time course, which no experiment has and which is there only to
show what a perfect template would be worth; an independent signal from the
*same band*, which is the realistic best case and what "looking for theta"
actually means; and one from a *different* band, which is what a wrong guess
costs. In your own analysis this is an argument to
:func:`~advance_beamlab.make_abmc`: pass an expert-annotated spike, an averaged
response, or a band-limited signal standing for the rhythm you are after.

Watch the localiser rather than the constraint table when you move it. At the
panel's :math:`P` of 0.03 the template barely changes the weights, which is the
method behaving as designed -- it is a mild steer, not a second distortionless
constraint -- so the delivered amplitude hardly moves. The map is a different
story, because the map *is* the template match: on hippocampal theta a
same-band template gives a map correlating 0.66 with the one the target's own
waveform produces, and a wrong-band template nothing at all, -0.05. The peak
often survives that, which is worth knowing too; what a bad template mostly
costs you is the rest of the picture.

One caveat that the burst morphology makes obvious. For a rhythm, naming the
band is most of what a template needs to say. For spike-like activity it is not:
an independent burst train drawn from the same generator has the same shape and
the wrong timing, and ABMC's lag search can only align a template that is
actually there. Matching the morphology matters more than matching the class it
came from.

The rhythms are chosen to go with them. **Activity** switches between theta,
alpha and beta, or a train of short bursts. The rhythms are band-limited processes rather than modulated sinusoids,
because a pure tone is not what a rhythm looks like and it makes every trace on
the page read as a textbook figure rather than a recording. The bursts are the
regime ABMC was designed for, since its extra term rewards output matching a
known waveform, and it is worth comparing the four methods there.

**Head model** switches between the two regimes described below, and it is worth
moving it with LCMV and MCMV in turn.

**Covariance window** appears on the recorded half only, and sets the interval
the data covariance is estimated over. It is the page's plainest demonstration
that a beamformer is tuned by what you show it rather than only by its formula:
the same recording and the same method give different filters, and different
distances to the fitted dipoles, depending on the window. Every recorded number
quoted on this page is for the narrow 80-130 ms response window; the wider
50-200 ms window admits more of the evoked response and more of the ongoing
activity along with it.

**Trials averaged** is the signal-to-noise axis, expressed as the thing an
experimenter actually controls. A single trial of an evoked MEG response sits
well below unit sensor signal-to-noise; this panel takes 0.2 as the single-trial
value, and averaging :math:`N` trials buys a factor of :math:`\sqrt{N}`, so the
two settings are 0.20 and 2.0.

This control moves two things in opposite directions, and that is worth sitting
with rather than explaining away. Averaging always improves **localisation**:
the median error over the correlated scenes, pooled over every method and
template the panel holds, falls from 15.6 mm to 0 mm with a matched model, and
from 20.4 mm to 9.0 mm with a realistic one. But averaging makes LCMV's
**cancellation worse**, not better: at :math:`r = 0.99` its delivered amplitude
falls from 0.48 at one trial to 0.03 at a hundred. Nothing
has broken. Cancellation is a property of the clean covariance, and at one trial
the noise is large enough that the filter cannot adapt sharply enough to perform
it. Averaging removes the noise that was accidentally protecting the source.
More data buys a better answer about *where*, and a more sharply wrong answer
about *how much*, unless the constraint is fixed.

The correlation is exact for any number of sources. Two sinusoids can be given
any correlation by a phase shift, but three cannot, so the simulation instead
mixes one shared factor and one private factor per source, orthonormalised
first, in proportions :math:`\sqrt{r}` and :math:`\sqrt{1-r}`. Every pair then
correlates at exactly :math:`r` whatever the rhythm or the count. The
orthonormalisation is what makes it exact rather than approximate: without it
the private factors carry some shared variance of their own and the realised
correlation drifts above :math:`r`.

Limitations of the simulation
-----------------------------

Two things about it are worth knowing, because they decide whether the numbers
mean anything.

**The interference is mostly brain, not sensor noise.** Three hundred sources
spread over the brain carry 1/f activity, and that accounts for three quarters
of the interference power; white sensor noise is the remaining quarter. This is
what makes the sensor traces look like resting MEG rather than a clean
simulation.

How much it changes the numbers depends entirely on how much data you have. The
comparison below is again one scene rather than the grid, because the panel does
not store a background-free variant to read off: a realistic head model, a
correlated pair 6 cm apart at :math:`r = 0.9`, alpha. Rerun with the background
removed, LCMV's delivered amplitude at a hundred trials barely moves, 0.19
against 0.16, because with that much averaging the filter is shaped by the
sources rather than by the interference. At a single trial it moves a great
deal, and changes sign: 0.16 without the background against -0.06 with it.
Structured brain interference is the harder problem, and it is hardest exactly
where real single-trial work lives.

**The head model control decides whether the sources are where the beamformer
looks**, and the two settings exist because no single choice can show everything.

With **matched** selected, the sources sit exactly on points the beamformer
scans and the data are generated with the very leadfield being inverted. That is
an inverse crime, and its symptom is unmistakable: LCMV and MCMV localise every
single matched source to exactly 0 mm, at both signal-to-noise settings. That is
not a theorem -- interference can and does move a localiser's peak off a node
the filter was built on -- but on this grid, at these separations and
interference levels, the strongest point of the map lands on the true node in
every matched configuration. Read those zeros as a property of the simulation,
not a claim about any method.

ABMC joins them only when it is handed the target's own waveform. Its localiser
scores template match rather than power, so an exact head model is not enough on
its own: hand it an independent signal from the same band, which is the
panel's own realistic best case, and it misses the source in nine of the
twenty-four single-source configurations, by up to 35 mm, eight of them at one
trial. What the oracle template buys ABMC is not a better filter but a localiser
with nothing left to be wrong about.

ReciPSIICOS misses them too, and it is instructive about something else. It
misses the same single sources by up to 15 mm, because it does not localise on
the covariance it was given: it projects that covariance first, and the
projection moves the peak even when the model is exact. A method that edits the
covariance gives up the inverse crime's free lunch along with everything else it
gives up.

What the matched setting buys is the only clean view of what the *constraint*
does, with everything else held exact. At :math:`r = 0.99` and a hundred trials
LCMV delivers 0.03 of the source amplitude while MCMV delivers 1.00, ReciPSIICOS
0.95 and ABMC 1.02. That gap is the whole point of the page, and it is a gap
rather than a ratio: LCMV's delivered amplitude passes through zero and changes
sign under strong cancellation, so dividing by it means nothing.

With **realistic** selected, the sources are taken from the full 9248-point
forward while the beamformer still scans the decimated 797-point grid, so they
sit a few millimetres from anything any method can report. The localisation
error becomes what a localiser actually produces: not one configuration in the
whole grid reports zero, the median is 11.9 mm, and the tail runs past 120 mm
where the methods genuinely fail.

The price is worth understanding, because it is a real result rather than a
limitation of the panel. Under a mismatched model a beamformer pointed at a
slightly wrong location does not pass a weakened copy of the source, it actively
nulls it, and at a hundred trials that costs more amplitude than the entire
cancellation effect the constraint was protecting against. At :math:`r = 0.99`
MCMV falls from 1.00 to 0.05, which is barely distinguishable from LCMV's 0.02.
**The joint constraint only buys you anything if your head model and
coregistration are good enough to use it.**

The methods do not all fail the same way, though, and that is the most useful
thing here. ReciPSIICOS holds 0.81 and ABMC 0.57 under the model error that
takes MCMV from 1.00 to 0.05. Neither is told where the sources are: one edits
the covariance and the other matches a waveform, and neither writes an equation
at a location that turns out to be wrong. What a wrong location destroys is the
methods that needed the location in the first place.

Read the two head models together rather than picking one. Matched says MCMV's
constraint is worth a factor of thirty over LCMV; realistic says you will not
collect it unless your coregistration earns it, and that a method which never
names a location keeps more of what it had.

The background activity is identical in both settings, so switching between them
changes the source placement and nothing else. Read the realistic errors as a
lower bound even so: the head model is still exact, the coregistration is still
perfect, and the noise is still stationary.

The four methods on a recorded dataset
--------------------------------------

Everything above is simulated, which is the only reason any of it can be checked
against a truth. The **Dataset** control switches to MNE's ``sample`` dataset,
the same 203 gradiometers and the same subject, and runs the identical four
filters on data that was recorded from a person rather than generated.

The bilateral auditory N100 is the reason this is worth doing. Left and right
auditory cortex respond together to a binaural click, which makes them exactly
the correlated pair the whole page is about, and the correlation is not a
setting here: it is measured, at :math:`r = 0.87` through the joint filter. Read
it off per-source LCMV traces instead and the same pair reads as *uncorrelated*,
through the very filter whose cancellation the correlation is meant to explain,
which is a good reason to distrust a correlation estimated with a filter that
cancels.

What the recording shows, averaged over every epoch, is the effect at full
strength. LCMV's off-diagonal reaches :math:`-1.03` and it delivers 0.56 of the
amplitude; MCMV holds the table at zero and delivers 1.00; ReciPSIICOS and ABMC
control it too, at :math:`+0.04` and :math:`+0.02`. Nothing here was tuned to
produce that; it is the published example's own preprocessing, run through the
same code path as the simulation.

The localiser peaks are a good deal messier, and the panel prints both of them
rather than the better one because of it. Each map is read at its two strongest
points, and each point is quoted as its distance to whichever of the two fitted
dipoles is nearer: LCMV 16 and 15 mm, MCMV 6 and 15, ReciPSIICOS 110 and 10,
ABMC 20 and 6. Quote the smaller of each pair and three of the four look like
they have found the recorded pair; quote both and the claim shrinks. Only ABMC
puts a peak by each dipole. LCMV's two strongest points sit 5 mm apart in the
left hemisphere and MCMV's 12 mm apart in the same neighbourhood, so neither
reports the right auditory source at all, and ReciPSIICOS's strongest point of
all is 11 cm from either dipole. The sweep at the foot of the panel plots the
worse of a method's two distances for that reason: minimising over the peaks as
well as over the dipoles would score each method on its best pairing and say
nothing about the other one.

**There is no truth, and nothing here pretends otherwise.** Three of the panel's
readouts need one and are therefore absent: no localisation error, no known
waveform beside the recovered one, no head-model switch. What survives is what
never needed a truth in the first place. The constraint table is the filters'
response to the leadfields of the constrained locations, so it is exact on any
data at all. In place of the error the panel reports the distance to an
independent **dipole fit** -- two dipoles fitted one after the other, the second
to what the first leaves behind, with their goodness of fit shown beside them.
That is a different method with different assumptions, so read the distance as
two estimates disagreeing rather than as one of them being wrong, and read the
rings on the cortex as a reference rather than an answer key.

Three things about how it is computed are worth stating, because each one was
got wrong first.

**The locations are chosen at full resolution and only then decimated.** The
simulated half scans a 797-point grid, and on that grid the right auditory peak
of this recording lands in parietal cortex: the pair stops being correlated and
LCMV's off-diagonal reads :math:`+0.03`, which is to say the effect disappears
entirely. Choosing the pair on the undecimated 7498-vertex surface and *then*
decimating the scan grid around it changes nothing at all -- the constraint
table is identical to three decimals from 7498 sources down to 627 -- because
LCMV, MCMV and ABMC each build a filter from one leadfield and the covariance,
and the rest of the grid never enters. Only ReciPSIICOS spans the grid, and it
is why the grid is decimated at all: its projector needs a factorisation over
every scan point, which costs 0.6 s at 954 sources and over five minutes at
7498.

**The preprocessing is the published example's, exactly.** Shortening the
baseline and adding amplitude rejection seems harmless and is not: it moved the
right-hemisphere peak to a different vertex and LCMV's off-diagonal fell from
:math:`-1.03` to :math:`-0.15`. The cancellation is a property of the pair that
gets selected, so the analysis that selects it has to be one whose result is
already on the record.

**One noise covariance is pooled over every epoch in the session** rather than
estimated per scene. That is the better estimate and standard practice, and it
is also what makes a single ReciPSIICOS rank legitimate: the covariance sets the
whitener, the whitener sets :math:`q`, and the rank is drawn out of
:math:`q^2`. Per-scene estimation would give every scene its own :math:`q` and a
precomputed rank would quietly mean something different in each one.

The **trials averaged** control is real here in a way it cannot be in the
simulation: it averages that many actual epochs. Watch it with LCMV selected.
At one trial the off-diagonal is :math:`-0.52` and the filter delivers 0.82; by
145 trials the off-diagonal has reached :math:`-1.03` and the delivered
amplitude has fallen to 0.56. More data makes the cancellation *worse* here as
it does in the simulation, but not by the same route. The covariance is
estimated from the very epochs being averaged, so at one trial the filter is
solving against a covariance estimated from a single epoch: it cannot adapt
because it does not yet know what to adapt to. Adding epochs sharpens the
estimate, and a filter that can see the correlation is a filter that can cancel
it. In the simulation the covariance is not the limitation -- there the noise
level itself is what falls -- so the two halves reach the same place from
different directions.

The **visual** conditions are there as a control, and they behave differently
rather than identically. LCMV's off-diagonal comes out *positive* there, around
:math:`+0.47`: the neighbour is added rather than subtracted. It is the same
failure to control the off-diagonal, in the other direction.

What decides that direction is the sign of the correlation between the two
sources, not how far apart they are. A filter that minimises output power
subtracts a neighbour whose activity runs with the target and adds one that runs
against it. Measured on the panel's own reconstructions in the 80-130 ms window,
the auditory pair correlates at :math:`+0.25` and its off-diagonal is
:math:`-1.03`, while the visual pair correlates at :math:`-0.79` and its
off-diagonal is :math:`+0.47`. The separations run the other way -- the visual
sources sit 3.2 cm apart against the auditory pair's 10.9 cm -- so proximity is
not what flips the sign.

How the panel is computed
-------------------------

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
sweep. **Grey** is always the reference the estimates are judged against: the rings
marking the simulated sources and the true waveforms on the simulated half, and
the independently fitted dipoles on the recorded half, where there is no truth
to be had and a dipole fit is the closest stand-in. Sources are told apart by
their row and label rather than by colour, which would otherwise collide.

The localiser map is normalised to its own range and then compressed by a cube
root. That combination took two attempts. Colouring by rank was the first, on the
grounds that the four methods produce value distributions no common scale can
show at once, and it was wrong: a rank map is a uniform ramp whatever the
underlying shape, so all four rendered identically. Measured, the fraction of the
grid above half the display maximum was 0.500 for every method. Normalising
without compression goes too far the other way and leaves two bright dots on an
empty brain. The cube root sits between and separates them: averaged over the
whole grid, that same fraction now reads 10.9 per cent for LCMV, 7.2 for MCMV
and 16.1 for ReciPSIICOS.

ABMC has to be quoted per template, and the numbers say something worth
knowing. Its map covers 34.8 per cent of the grid when it is handed the target's
own waveform, 87.1 with an independent signal from the same band, and 96.3 with
a wrong-band one. The oracle template is what makes ABMC's localiser look sharp;
give it the template an experiment can actually supply and the map is much more
diffuse. That is the honest cost of a method whose localiser scores template
match rather than power.

The markers on the simulated sources are drawn as unfilled rings for a related
reason: a filled marker sitting on a peak hides the one thing you are trying to
check. They are drawn at the sources' real positions. Under the realistic head
model those are not grid points, so a ring will not usually sit under a cross even
when the method is right; under the matched model they are grid points, and the
two coincide exactly whenever the method has found the source.

The sensor panel shows a field map beside the traces. The traces carry all 203
gradiometers over two and a half seconds: drag to move through channels and
time, scroll to zoom, or use the sliders. The topography says where on the
helmet the signal sits, and how little that pattern changes when you move the
sources a couple of centimetres. Both refer to the same instant: the peak of the
global field power over the whole trace, chosen once when the panel is built,
with the vertical line on the traces marking it. It does not follow the zoom, so
scrolling or zooming past that instant leaves the map showing a moment that is
no longer on screen -- the vertical line is what tells you where it was. The map
is interpolated in the browser from the array's own sensor
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

The **amplitude delivered** readout is what the filter does to the source's
amplitude: each source's gain, weighted by how much of this source's waveform it
carries, which for a pair is :math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_i +
r\,\mathbf{w}_i^{\mathsf T}\mathbf{g}_j`. There is no noise in it, so it says
what the filter does rather than how well one short recording measures it.

The :math:`\mathbf{g}` here are the leadfields of the sources *as simulated*,
not of the nodes the filter was constrained at, and under the realistic head
model those are different points. So the readout cannot be recomputed from the
constraint table displayed above it -- that table is the response at the
constrained nodes, with its diagonal pinned to one by construction. The gap
between the two is the whole cost of a mismatched head model: measured against
its scan node LCMV can report delivering the full amplitude of a source whose
reconstruction holds about seven per cent of it. With a matched model the two
coincide and the question does not arise.

Two other definitions were tried and rejected, and the reason is worth stating
because both look reasonable. The figures below are one scene rather than the
grid, since neither rejected measure is stored in the panel: a correlated pair
6 cm apart at :math:`r = 0.9`, alpha, one averaged trial.

The output's own amplitude over the truth's is not a recovery measure at all.
The output is the source plus whatever interference survives the filter, so it
reads **5.63 for LCMV and 5.79 for MCMV** on that scene, separating the two by
three per cent where the filters deliver -0.13 and 1.00 respectively. A measure
that cannot tell apart filters differing in sign, let alone by a factor of
seven, is measuring the noise and not the recovery. Regressing the output on
the true waveform is far better behaved and agrees with the delivered amplitude
in expectation, but it is still estimated from one short noisy recording: on the
same scene it returns 0.23 where the filter delivers -0.13, which is the right
size and the wrong sign.

Read a value below 100 per cent as cancellation and one above it as leakage,
where a positive off-diagonal adds a correlated neighbour instead of subtracting
it. Leakage is the minority case, from 7.0 per cent of configurations for LCMV
and 8.9 for MCMV to 31.7 for ABMC and 32.7 for ReciPSIICOS. Whether averaging
makes it more or less common depends on the head model, so the two are worth
quoting apart. With a realistic one every method leaks less as the data improve
-- LCMV in 26.5 per cent of its single-trial configurations against none at all
of its hundred-trial ones, MCMV 35.5 against none, ReciPSIICOS 43.4 against
36.5 -- because a filter with too little data to adapt degenerates towards a
non-adaptive one with poor spatial selectivity, while one given enough adapts
sharply enough to null a source it is pointed slightly to one side of, which
takes the amplitude below one rather than above it. With a matched model there
is far less to leak and the direction is no longer uniform: LCMV falls from
1.5 per cent to none and ReciPSIICOS from 28.9 to 22.1, MCMV never leaks there
at all, and ABMC alone rises, 18.7 to 49.0, in cases that mostly clear the line
by very little -- their median delivered amplitude is 1.02, five in six stay
below 1.05, and the largest of them is 1.24. In a small fraction of
configurations the delivered amplitude comes out **negative**: the filter
returns an inverted copy of the source, which taking a magnitude would have
hidden completely. It happens most to LCMV, at 1.3 per cent, and never to MCMV,
whose constraint forbids it. The readout beside the constraint table carries the
sign for exactly that reason.

MCMV delivers exactly 1.000 in every *matched* configuration, at every
correlation and every signal-to-noise ratio, because its constraint fixes the
whole table rather than one entry. Switch the head model to realistic and that
guarantee is gone: at :math:`r = 0.99` the median is 0.85 at one trial and 0.05
at a hundred, because the table it pins is the table at the wrong locations.

The last panel sweeps the whole correlation axis at the current settings, for all
four methods at once, plotting both recovered amplitude and localisation error
with a marker where the sliders are. The sliders show one point; that chart shows
the curve those points lie on, which is a fairer way to judge a method than any
single setting. Two things about that chart are worth knowing: it draws the
first source only, and its amplitude axis is floored at zero, so where LCMV's
delivered amplitude goes negative the curve flattens along the axis instead of
crossing it. Read the sign off the readout beside the constraint table, which
carries it.

If you would rather drive this from Python, with your own forward model and
without the precomputed grid, :func:`~advance_beamlab.constraint_explorer` opens
the same views as a Matplotlib window with live sliders, and
:func:`~advance_beamlab.constraint_demo` returns one scene as plain arrays.
