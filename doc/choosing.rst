Which method should I use?
==========================

Start with :func:`mne.beamformer.make_lcmv`. On most recordings it is not merely
an acceptable baseline but the right answer, and every method in this package is
an addition to it under a condition you have to check before you reach for it.
None is a general-purpose replacement. :func:`~advance_beamlab.make_mcmv` at
``n_sources = 1``, with ``weight_norm`` in ``{'unit-gain', None}`` and the same
``reg``, reproduces MNE's unit-gain LCMV to floating-point precision (measured:
7e-14 relative on the applied output, 2.9e-13 on the effective sensor-space
weights). The advanced methods are that same solve with something added; if the
thing they add is not the thing that is wrong with your data, you have spent
degrees of freedom, extra inputs and extra failure modes for nothing.

This repository's own real-data examples say so directly. On the MNE sample
auditory recording, ``examples/plot_recipsiicos_auditory.py`` concludes that
"a plain LCMV already recovers both hemispheres, so ReciPSIICOS does *not* beat
it on hemispheric balance here"; re-running it gives an LCMV
left/right balance of 0.94 against 0.84 for ReciPSIICOS, so the projection is
measurably worse on the metric the example chose. In
``examples/plot_mcmv_auditory.py``, widening the covariance window from the
N100 (80-130 ms, recovered inter-hemispheric r = +0.55) to 50-200 ms
(r = -0.13) collapses the MCMV/LCMV peak-amplitude ratios to 1.02 and 0.95.
The second is MCMV doing marginally *worse* than LCMV. The same file declines
to claim better focality: "It is tempting to add 'and the beamformer is
sharper', and on this recording that is simply not true" (dSPM is the most
compact of the three at the N100), and it recommends that with no named sources
a linear method "remains the safer default".

Note also what real data cannot tell you. Neither auditory result has a ground
truth, so a larger recovered amplitude is not evidence of a more correct one,
and the only real-MEG tests in this package assert finiteness and
well-formedness, never accuracy. Every quantitative demonstration below is
either a simulation or a comparison against a nominal truth that the
simulation itself defines.

Decision table
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 17 29 28

   * - Your data situation
     - Use this
     - Why
     - What it costs
   * - Two or more sources whose reconstructed time courses correlate at
       r ≳ 0.7 **within the exact window** ``data_cov`` is built from, and whose
       locations you know to a few mm and orientations to a few degrees
       (anatomy, atlas, independent localiser)
     - :func:`~advance_beamlab.make_mcmv`
     - LCMV lowers its own output variance by passing a sign-flipped copy of the
       correlated partner; recovered amplitude tracks :math:`\sqrt{1-r^2}`. The
       zero-gain constraint makes each filter blind to the partner. Measured
       amplitude ratio 3.19x at r = 0.95 on the simulation fixture.
     - The constraint set is the method and is unforgiving: 5 mm of partner
       error cut recovered amplitude from 0.99 to 0.44, 10 mm to 0.33 (LCMV
       0.31). One time course per constrained source, no whole-brain map; each
       source spends a degree of freedom.
   * - Same correlation, but you care about waveform shape rather than peak
       amplitude, including r as low as 0.2-0.3
     - :func:`~advance_beamlab.make_mcmv`
     - Below r ≈ 0.3 there is little amplitude to recover (peak ratio
       1.03-1.04), but the joint null still removes the partner's spatial
       leakage: measured waveform RMSE 0.201 (LCMV) against 0.007 (MCMV) at
       r = 0.3.
     - As above. Do not use a near-zero
       :func:`~advance_beamlab.apply_mcmv_cov` off-diagonal as a licence to drop
       the constraint: it says the amplitude gain is gone, not that there is
       nothing to fix.
   * - Correlated sources, MEG, you want **amplitude** at grid points you
       already have, and you cannot name the sources well enough to constrain
       them
     - :func:`~advance_beamlab.make_recipsiicos_lcmv`
     - The projector is built from the forward, so no source list is needed.
       Measured amplitude on the gradiometer sample BEM: LCMV 0.307 against
       ReciPSIICOS 1.019 at r = 0.95, with a gain already material at r = 0.7
       (1.40x).
     - The rank ``K`` is a real parameter with two failure ends and only
       warnings guarding them. Runtime O(N^2) in sources for ``'whitened'``,
       memory 99 MiB at q = 60 and 1.5 GiB at q = 120, rebuilt on every call.
       All quantified benefit here is MEG.
   * - Correlated sources and what you need is **position**, not amplitude
     - :func:`mne.beamformer.make_lcmv`
     - No localisation improvement is demonstrated anywhere in this package for
       either correlated-source method, and both were measured mislocalising
       where LCMV was exact (see the closing section).
     - Nothing. Read positions from the LCMV map, not amplitudes: those are
       the quantity the cancellation attacks.
   * - Coupling between named ROIs (envelope correlation, coherence, PLV) rather
       than localisation, at good sensor SNR
     - :func:`~advance_beamlab.pairwise_mcmv_connectivity`,
       :func:`~advance_beamlab.augmented_pairwise_mcmv_connectivity`
     - Independent per-ROI filters each carry a copy of the other region, so a
       third region coupled to one of them manufactures an edge. Measured on the
       repository's spurious edge (truth -0.151): LCMV +0.100 (wrong sign),
       PW-MCMV -0.093, APW-MCMV -0.151.
     - One MCMV solve and one estimate per ROI **pair**: 22-29 ms/pair at 94
       channels x 24,000 samples, so ~50-65 s per band per subject for a
       68-label atlas, plus ~15 s for the AR(1) screen. Below ~5 dB sensor SNR
       PW-MCMV is worse than LCMV.
   * - A low-variance transient with a reproducible morphology you can supply as
       a template (IED, evoked spike), contributing of order 1% of the segment's
       sensor variance
     - :func:`~advance_beamlab.make_abmc`
     - LCMV ranks locations by output power, which at that variance share barely
       varies over the grid; ABMC ranks them by agreement with your template at
       the best lag. Measured mean peak error 0.9 cm (ABMC) against 5.5 cm
       (LCMV) over 8 sources at 1.3x-peak noise.
     - You must have the template. A mismatched one gave 4.54 cm against LCMV's
       5.47 cm, worse than LCMV at 3 of 8 sources. One extra parameter ``P``,
       working set O(n_sources x n_times), single continuous segment only.
   * - Sustained, oscillatory or induced activity; or a transient whose shape you
       cannot write down
     - :func:`mne.beamformer.make_lcmv`
     - With no reproducible waveform there is no template, and at ordinary SNR
       the two agree anyway: a sustained 10 Hz source with its own sinusoid as
       template gave 0.00 cm for both methods at 0.2x and 1.3x noise.
     - Nothing.
   * - EEG only, no subject MRI, no digitised electrodes, a template head model
       is acceptable, and you report absolute dipole moments
     - :func:`~advance_beamlab.read_ny_head_forward`
     - A finite-element model represents CSF, the skull split and the skull
       openings that a nested three-surface BEM cannot. Against one fsaverage
       BEM baseline the FEM gains are lower by a median factor of 1.67, and a
       unit-gain amplitude scales as 1/gain.
     - 678 MB GPL-v3 download, EEG only, cortical surface only, no labels and no
       morph. The factor is not a constant (1.34 to 1.95 across plausible skull
       conductivities) and no localisation or topography comparison against a
       BEM exists in this repository.
   * - Any MEG channel present, or the subject's own MRI available
     - An ordinary MNE BEM forward
     - No template MEG lead field can exist, because the sensor array moves
       relative to the head every session. For EEG, a better conductivity model
       spent against a centimetre of template-anatomy mismatch is not a
       demonstrated win.
     - Nothing.
   * - Sources effectively uncorrelated in the covariance window, one source at a
       time, ordinary SNR
     - :func:`mne.beamformer.make_lcmv`
     - There is no cancellation to undo and no leakage partner to null.
     - Nothing.

MCMV: an exact null on sources you can name
-------------------------------------------

**The LCMV failure it repairs.** LCMV's null is estimated from the data, so a
correlated partner can be used to cancel the target: recovered amplitude falls
as :math:`\sqrt{1-r^2}`. MCMV :footcite:`Moiseev2011` replaces that estimated
null with an exact one, ``W.T @ H = I`` (pinned to 1e-10 on the raw
sensor-space leadfield). That fixes two distinct things: correlated-source
amplitude cancellation, and spatial leakage of a *constrained* partner into the
estimate, which is present even at r = 0. It fixes nothing about sources you
have not named.

**The condition to check.** Correlate your candidate sources' LCMV time courses
over the exact window ``data_cov`` is built from, and confirm you know both
locations, not only the target's. The tolerance is the method's main practical
risk.

**What this repository demonstrates.** On the simulation fixture (94-channel
EEG sphere, 726-point grid, sources 8.1 cm apart, leadfield overlap 0.69, noise
0.05 of peak) LCMV relative amplitude is 1.000 / 0.956 / 0.867 / 0.716 / 0.601 /
0.435 / 0.314 / 0.142 at r = 0 / 0.3 / 0.5 / 0.7 / 0.8 / 0.9 / 0.95 / 0.99,
within 0.003 of :math:`\sqrt{1-r^2}` throughout. MCMV, by contrast, stays at
1.000-1.005, giving ratios 1.40 / 1.67 / 2.30 / 3.19 / 7.10 from r = 0.7
upward. Judge the method on waveform error rather than on that ratio: at
r = 0.95 the RMSE against the injected source is 0.661 (LCMV) vs 0.006 (MCMV)
at 5% noise and 0.205 vs 0.048 at 100% noise, so the accuracy advantage
survives across the whole noise range even where the amplitude ratio has
collapsed to ~1.2. For discovery, on a 418-point grid with three sources
6.0 / 8.2 / 14.2 cm apart (pairwise 0.95 / 0.805 / 0.95) an LCMV power map ranks
the three true sources #4, #41 and #1 of 418 with two non-sources in its top
three, whereas :func:`~advance_beamlab.scan_mcmv` with ``'mai'`` and with
``'mpz'`` returns all three at 0.0 cm and recovers their time courses at
r = 0.999-1.000. That three-source result is shown only in an example; no test
pins it. On real mixed-sensor MEG, the sample auditory N100 gives 2.21x (left)
and 1.48x (right) the LCMV peak at ``reg=0.05``: evidence that the pipeline
runs and that the effect survives realistic regularisation, not that the
amplitude is the correct one.

**When it does not help.** When the constraints are wrong. On the sphere
fixture with an exactly matched forward and sources sitting on grid points,
displacing the *partner* by 5 mm cut recovered amplitude from 0.99 to 0.44 and
by 10 mm to 0.33, against LCMV's 0.31; 5 degrees of partner-orientation error
gave 0.54 and 10 degrees 0.36. How far that transfers to a real oct-6 source
space is untested here. The repository's own real-data example takes both
constrained vertices straight from an LCMV power-map peak with no cross-check.
The greedy search is also not a general-purpose localiser: on the 231-electrode
New York Head FEM with a 34 mm, r = 0.95 pair, :func:`~advance_beamlab.scan_mcmv`
and LCMV were both exact at 5% noise, but at 50% noise the scan missed by
10-24 mm while LCMV stayed at 0.0 mm. ``pseudo_z`` is a weak stopping rule: in
the repository's own three-source demo the sequence is
[2.9, 1.5, 10.4, 1.1, 1.0], so the genuine buried source scores 1.50 against a
1.06 floor and any threshold that excludes the two spurious picks also excludes
a real source. Treat the number of sources as a hypothesis you set. Finally,
constrained sources closer than ~5 mm warn, near-singular ``H.T @ R^-1 @ H``
raises, and an order above the rank of the whitened covariance raises. Note the
evidence asymmetry: every quantitative MCMV demonstration here is either
simulated EEG on a sphere or real MEG without ground truth; there is no
real-EEG demonstration anywhere.

**The parameters that matter.** ``sources`` (and ``orientations`` for a
free-orientation forward), and the window ``data_cov`` is estimated from. Take
orientations from :func:`~advance_beamlab.optimal_orientation`,
:func:`~advance_beamlab.scan_mcmv` or the source space itself
(``forward['src'][hemi]['nn'][vertno]``), never by hand. In particular
``forward['source_nn'][2::3]`` is *not* a shortcut to the cortical normals: on
the ``surf_ori=False`` forward MNE returns by default it is the head +z axis for
every source, and it is accepted silently.
Use ``weight_norm='unit-gain'`` when comparing amplitudes with
:func:`mne.beamformer.make_lcmv` (``weight_norm=None`` there);
``'unit-noise-gain'`` rescales both filters and hides the effect. Two
practical notes: compare *outputs*, not ``filters['weights']``, since MNE stores
its weights in the whitened space and :func:`~advance_beamlab.make_mcmv` folds
the whitener in; and although the tuning table advises an integer ``rank`` after
SSP/ICA/SSS, :func:`~advance_beamlab.make_mcmv` rejects a bare integer, because
the rank must be resolved per sensor type. One assumption is worth
discouraging explicitly, because it is the natural one to make:
``pick_ori='max-power'`` does not steer around the cancellation. Measured LCMV
RMS at r = 0.95 is 0.2192 fixed-orientation and 0.2191 with ``'max-power'``.
Re-optimising one source's orientation cannot undo a cancellation that comes
from the other source sitting in the same filter.

ReciPSIICOS: a forward-only projector when you cannot name the sources
----------------------------------------------------------------------

**The LCMV failure it repairs.** The same cancellation, attacked from the other
side: ReciPSIICOS :footcite:`KuznetsovaEtAl2021` deletes the cross-product
(coupling) part of the data covariance in a noise-whitened virtual-sensor space
and then runs MNE's own LCMV solve there, so no source list is required. What
the repository demonstrates it restoring is *amplitude at locations you already
have*, not peak location.

**The condition to check.** MEG; a correlated pair (worth trying from r ≈ 0.5
upward); light regularisation; a whole-brain, decimated forward; and a
read-out at positions you already trust.

**What this repository demonstrates.** On the sample BEM forward, gradiometers
only, two sources 12.1 cm apart, 60 epochs, sensor noise 2% of peak,
``reg=0.01``, ``weight_norm=None``, read out at the *true* source indices: LCMV
recovers 0.985 / 0.939 / 0.852 / 0.702 / 0.428 / 0.307 / 0.139 of the injected
amplitude at r = 0 / 0.3 / 0.5 / 0.7 / 0.9 / 0.95 / 0.99 against ReciPSIICOS's
1.000 / 1.000 / 0.991 / 0.986 / 1.012 / 1.019 / 1.027. The ratios are 1.02, 1.06,
1.16, 1.40, 2.4, 3.3 and 7.4. At r = 0 the two agree to within 2%, with no
measured penalty for projecting. The exact-removal guarantee (``'recipsiicos'``
at full power-subspace rank reproducing the uncorrelated covariance to 1e-8 and
raising power at the coupled source more than fivefold) is demonstrated only on
QR-generated orthogonal synthetic topographies, not on any forward model, and
this package gives you no way to check whether a real pair is close enough to
orthogonal. The ``'whitened'`` variant, which both examples and the tuning table
recommend for real data, carries no such guarantee: it removes only the top-K
correlation directions, and its test deliberately gives the coupled pair the
largest topography norms so that its cross-product is the dominant direction.
No test anywhere asserts that ReciPSIICOS beats LCMV. The contrast with MCMV,
whose test does assert superiority quantitatively, is deliberate.

**When it does not help.** On real mixed-sensor MEG at ``reg=0.05``: the sample
auditory N100 gives LCMV a hemispheric balance of 0.94 against ReciPSIICOS's
0.84. For localisation: nothing in this repository measures a localisation
error for ReciPSIICOS, and at exactly the settings the simulation uses to
advertise its amplitude gain (``method='whitened'``, ``rank`` = K* = 139,
``weight_norm=None``, ``reg=0.01``) an :func:`mne.beamformer.apply_lcmv_cov` map
put LCMV's top two peaks on both true sources at 0.0 mm while ReciPSIICOS left
one true source 63 mm from its nearest peak. Under ``'unit-noise-gain'`` or
``'nai'``, at ``reg=0.05``, or with ``method='recipsiicos'``, both localise
exactly. The miss is setting-specific rather than universal, but it occurs at
the advertised settings. For EEG there is no quantified benefit at all: EEG
appears only in well-formedness tests. The default ``reg=0.05`` is itself enough
diagonal loading to mask the cancellation the method exists to repair.

**The parameters that matter.** ``rank`` (K), which lives in the
q^2-dimensional working space, not in sensor space. Take it from
:func:`~advance_beamlab.recipsiicos_rank_curve` with ``return_optimal=True``,
pass it the *same* ``noise_cov`` you will build the filter with, and treat K* as
the only rank this repository actually demonstrates. Do not read the example's
rank sweep as evidence of robustness: on that forward 2K* = 278 and 4K* = 556
exceed the retained power-Gram rank of 269 and annihilate the cleaned covariance
to exactly zero, and K*/8 and K*/3 trip the ">20% negative-eigenvalue energy"
warning; the flat amplitude at high rank is an artefact of a unit-gain read-out
on a degenerate covariance. Both conditions are only *warned* about: the sole
hard error is ``rank`` above q^2, which is 1600 on that forward, three orders of
magnitude above the rank at which the covariance is already zero. Note too that
the automatic K* is not guaranteed to clear the warning: ``method='recipsiicos'``
at its own K* = 60 trips it (21.7%) on the simulation's forward at both
``reg=0.01`` and ``reg=0.05``, so the two criteria can conflict and you must
lower the rank below the automatic choice. Second parameter: ``method``.
``'whitened'`` spares source power and is what the examples use;
``'recipsiicos'`` is O(N) rather than O(N^2) and carries the orthogonal-case
guarantee. Budget the cost per call: the projector depends on the forward *and*
the channel set, ``noise_cov``, ``pct_var``/``n_virtual`` and (for
``'whitened'``) ``reg``, and no public API accepts or caches a prebuilt one.
EEG input is refused without an average-reference projector.

PW-/APW-MCMV connectivity: removing leakage instead of orthogonalising it
--------------------------------------------------------------------------

**The LCMV failure it repairs.** A plain LCMV reconstructs each ROI with its own
filter, so every other active region leaks into it and a region coupled to your
target manufactures an edge that is not there. PW-MCMV :footcite:`Nunes2020`
replaces those filters with a joint MCMV that nulls the partner exactly;
APW-MCMV additionally nulls, for significant edges only, up to
``max_neighbours`` (default 2) regions that are **already in your ROI list**,
lie within ``radius`` (default 4 cm, Euclidean) of one member of the pair, and
themselves carry a significant edge. Leakage from anything outside that set
(unmodelled, far away, or non-significant) is not removed.

**The condition to check**, in the order the code checks it: is the suspected
conductor one of the ROIs you pass in; is it within ``radius`` by Euclidean
distance; does it carry at least one significant edge in the PW-MCMV matrix.
Leadfield collinearity explains *why* the leakage exists but is not what the
selection rule uses. Use this route when the coupling you care about is at or
near zero lag (envelope correlation, coherence magnitude, PLV), so that
leakage-orthogonalisation would discard the coupling along with the leakage;
the comparison against an orthogonalised baseline is the paper's, not this
repository's.

**What this repository demonstrates.** One simulation: 94-channel 10-20 EEG on a
sphere, 1409-point 12 mm grid, fixed orientation, 120 s at 200 Hz, sensor SNR
22.7 dB, three 10 Hz sources, with the conductor 2.4 cm from one ROI (leadfield
overlap 0.978) and the pair itself 8.65 cm apart (0.657). On the spurious A-B
edge, truth -0.151: LCMV +0.100 (error 0.251, wrong sign), PW-MCMV -0.093
(0.058), APW-MCMV -0.151 (~0.000), with the conductor leakage into A's filter
falling from 0.171 to 1.2e-16. On the genuine edge, truth 1.000: LCMV 0.992,
PW-MCMV 0.995, APW-MCMV 1.000. The whole demonstrated advantage is on the
spurious edge. The test that pins it records the bias dropping 0.0597 to 0.0021;
the test named ``test_apw_beats_lcmv`` asserts only a magnitude inequality, not
an error bound.

**When it does not help.** When the edge is genuine and strong: LCMV was within
0.008 here. At low SNR: PW-MCMV's residual conductor leakage is data-adaptive
and grows with noise, so on the same scenario the spurious edge runs LCMV
+0.097 / +0.122 / +0.188 / +0.240 / +0.348 / +0.410 against PW-MCMV -0.100 /
-0.030 / +0.166 / +0.336 / +0.658 / +0.847 at 22.7 / 8.7 / 4.6 / 2.7 / -0.9 /
-5.3 dB. PW-MCMV becomes the worse estimator somewhere between 9 and 5 dB,
while APW-MCMV, whose null is not data-adaptive, held -0.14 to -0.17 throughout.
Do not import the "regularisation pushes LCMV back toward the truth" reasoning
from the amplitude examples: for connectivity, noise moved LCMV's spurious edge
*away* from the truth and dropped its genuine edge from 0.992 to 0.667. When
your ROI vertices are uncertain: displacing one ROI by 12 mm turned PW-MCMV's
spurious edge from -0.093 to +0.287, an error of 0.437 against LCMV's 0.251 on
the same edge, and cost APW-MCMV's genuine edge 1.000 to 0.937. The same
mechanism applies to forward-model error, which this package never tests. Every
zero-gain result here is measured under a perfectly matched fixed-orientation
forward. When no in-radius neighbour carries a significant edge, APW-MCMV
rebuilds the identical order-2 filter and returns the PW-MCMV value; conversely
a false positive in the screen can pull a non-conductor into the augmented set,
raising the order and the condition number for no bias. The screen has both
kinds of error in the example itself: it rejected the very A-B edge APW-MCMV
repairs (the example forces the mask to proceed) and retained A-C, whose
envelopes are independent. Scope limits: the real-MEG section of
:ref:`ex-apw-mcmv-connectivity` shows the pipeline running but has no ground
truth, so every quantitative claim above rests on the simulation;
:func:`~advance_beamlab.ar1_surrogate_significance` supports
``'coh'``/``'plv'``/``'imcoh'``/``'wpli'`` as well as the default
``'envelope'``, but the spectral ones need an *epoched*
``reference_time_courses`` of shape ``(n_epochs, n_sources, n_times)``, since a
single continuous surrogate segment carries no usable coherence or
phase-locking estimate; and the spectral metrics are tested only for shape,
finiteness and symmetry. For low-power transients there is no connectivity route here at
all: ABMC addresses their *localisation*, not coupling between them.

**The parameters that matter.** ``sources`` (and ``orientations``): the exact
null is placed on the leadfield column you name, so prefer a fixed-orientation
cortical-normal forward and take the vertices from anatomy or an independent
localiser. And the band: pass ``data`` and ``data_cov`` both filtered to the
same analysis band, and leave ``envelope_lowpass=0.5``, which is what keeps the
AR(1) screen correctly sized. The test asserts that the low-pass holds the
false-rejection rate at or below alpha and at least four times below the
no-low-pass rate (its docstring records 0.4% against 7.5%, which is not itself
asserted). Use identical ``orthogonalize`` / ``absolute`` / ``envelope_lowpass``
/ ``envelope_resample`` for the matrix, the significance test and the APW pass.
Secondary: ``radius`` and ``max_neighbours`` set the beamformer order; keep the
total order below about 8.

ABMC: localising a transient that carries almost none of the variance
-----------------------------------------------------------------------

**The LCMV failure it repairs.** LCMV ranks locations by output power, so a
short, low-variance transient is outranked by whatever carries the most variance
in the segment. ABMC :footcite:`Shirani2024` keeps the distortionless constraint
and swaps the localiser for the magnitude of the correlation between the filter
output and a caller-supplied target waveform at the best lag, and replaces the
empirical covariance with a full-rank sparse-Bayesian model covariance
(:func:`~advance_beamlab.sbl_covariance`).

**The condition to check.** Estimate the target's share of the segment's sensor
variance from an averaged prototype. In the example that share is 0.14-1.05%
(mean 0.50%). Judge by that share, not by whether the LCMV map looks plausible.
In the use case ABMC is for, you have no expected location to compare it with.

**What this repository demonstrates.** One head-to-head, on a 94-channel EEG
sphere with a 301-point 20 mm grid, 400 samples at 250 Hz and white sensor noise
1.3x the spike's peak sensor amplitude, against
:func:`mne.beamformer.make_lcmv` with ``weight_norm='unit-noise-gain'`` and
``reg=0.05``: mean peak error 0.9 cm (ABMC) against 5.5 cm (LCMV) over 8 spike
locations. It is not a lucky seed: over six noise realisations the means are
0.71-1.39 cm against 3.86-7.36 cm, with ABMC worse than LCMV at 1 of the 48
individual sources. The lag search works: three differently-shifted copies of
the same template return one location with lags +50, 0 and -50 samples, and a
test recovers a +50-sample lag to within 3 samples. Stage 1 alone is explicitly
not a localiser. Its own peak sat 4.9 cm from the truth while its variance
estimates ranked the true source 7 of 301. On waveform recovery the gain is
real but small: r = 0.40 (ABMC) against 0.34 (LCMV) at the least favourable
source, on a trace neither filter recovers well.

**When it does not help.** When the variance share is already large: on that
geometry both methods were exact at sensor noise 0.05x and 0.2x of peak; LCMV
was 0.60 cm off at 0.5x and 3.20 cm at 0.8x, against ABMC's 0.00 and 0.25 cm. So
the crossover is around 0.2x, not higher. When you have no reproducible
waveform: the reason is the missing template rather than the activity being
sustained. A sustained 10 Hz source, with its own sinusoid passed as the
template, was localised at 0.00 cm by both methods at 0.2x and 1.3x noise (and
better by ABMC, 0.85 vs 4.31 cm, only at 3.0x). When the template is wrong: a
10 Hz burst template on spike data moved ABMC from 0.85 cm to 4.54 cm, against
LCMV's unchanged 5.47 cm, and left ABMC worse than LCMV at 3 of the 8 sources.
When two sources of the same morphology are simultaneously active: ABMC returns
one map with one peak and places no null on a second source, and in a check with
two spikes 6 samples apart at sources 8.9 cm apart the peak landed 4.0 and
5.7 cm from the two true sources, which ranked 16th and 49th of 301. Use
:func:`~advance_beamlab.make_mcmv` with the indices instead when you know them.
Finally, read the map, not the argmax: over the example's 8 sources the winning
template-match beat the runner-up by 0.000-0.011 in absolute value (0.402 against
0.402 at the worst source), ``ABMCResult`` exposes no prominence or significance
measure, and the peak will move between adjacent grid points across noise
realisations. Coverage limits worth knowing: ABMC is never run against a
competing higher-power source, which is the regime its own rationale is about.
It is now run on a real MEG recording, in :ref:`ex-abmc-auditory`, and the
result is a boundary rather than a win. On a single trial ABMC is worse than
LCMV and highly variable, a median of 29 mm over eight disjoint trials against
11 mm for LCMV, with a range of 1 to 46 mm. From two trials upwards it is
consistently better, 9 mm at two, four and eight and 7 mm at sixteen. LCMV is
given the whole recording throughout, so the comparison runs against ABMC and it
still wins above one trial. Distance to an anatomical label remains a weak
criterion next to a simulated ground truth.

**The parameters that matter.** ``template`` is the load-bearing input rather
than a tuning knob: only its shape matters (the read-out is exactly invariant to
its amplitude) and its lag is searched, so supply an expert-marked or averaged
prototype. Bound the search with ``max_lag``: the default spans plus or minus
the whole segment with zero padding, so at large lags the correlation is taken
against a mostly empty template, and the per-column lag is seeded from an
LCMV-type distortionless output and never re-estimated, so it inherits the
statistic ABMC exists to improve on. ``P`` trades the distortionless constraint
against the template constraint; the default is 0.03 and the documented live
range is inconsistent between the tuning table (0.01-1) and the docstring
(0.01-0.1). Choose it on your own data with ``P='auto'`` or
:func:`~advance_beamlab.abmc_stability_curve`, which picks the widest run of
``P`` over which the peak does not move rather than the ``P`` that maximises the
match (the match rises with ``P`` by construction). Then read the diagnostics:
heed the "numerically inert" warning (``P`` too small, ABMC has degenerated to an
LCMV), keep ``blowup_fraction`` below 0.05, and treat a one-point plateau as a
warning that the localisation is parameter-sensitive. Costs: one ``template``
array of exactly the data length, a single continuous segment (``Epochs`` are
refused), and a working set of O(n_sources x n_times), against
O(n_sources x n_channels) for an LCMV scan. It is the source count that bites on
a full-resolution source space. Runtime is not the issue at moderate size:
0.045 s for :func:`~advance_beamlab.make_abmc` at the default ``P`` against
0.056 s for ``compute_covariance`` plus :func:`mne.beamformer.make_lcmv` and
:func:`mne.beamformer.apply_lcmv_cov`, and 0.114 s for ``P='auto'``. Keep
``method='closed-form'``; the two solvers are pinned to each other only on the
EEG sphere fixture, and the report that the iterative descent's step-size rule
stopped ~40% short of the fixed point and moved the peak by about 9 cm on a real
gradiometer covariance is documentation prose with no test or example behind it.

The New York Head FEM forward: a different forward, not a different beamformer
--------------------------------------------------------------------------------

**What it repairs.** Nothing about LCMV. It replaces the forward model that LCMV
consumes :footcite:`HuangEtAl2016`, which changes reconstructed amplitudes and,
in principle, topographies. This repository never demonstrates that it changes
where LCMV peaks.

**The condition to check.** EEG-only data (no template MEG lead field can exist,
because the sensor array moves relative to the head every session), channels
that can be matched by name, an average reference, and no subject MRI. The
alternative being weighed is a *template* BEM, not the subject's own anatomy. If
you have the subject's MRI, or digitised electrodes that differ materially from
the template, a better conductivity model is being spent against a centimetre of
anatomical mismatch, and this package contains no evidence that trade is
favourable: the montage agrees with MNE's ``standard_1005`` to a median of
9.6 mm (90th percentile 15.0 mm, max 19.6 mm) over the shared electrodes.

**What this repository demonstrates.** One measurement: median gain magnitude
binned by depth below the nearest electrode, FEM against a three-layer fsaverage
BEM built for the same electrodes. The BEM gains are higher by 1.750, 1.862,
1.646, 1.768, 1.674, 1.671, 1.616, 1.565 and 1.456 at 18 to 50 mm (median 1.671;
overall median 18.9 V/(A m) FEM against 31.8 BEM). That factor is not a property
of "a three-layer BEM": it is joint with the skull conductivity the example
hard-codes at 0.006 S/m, and re-running the identical comparison at 0.0125 S/m
gives a median of 1.95 and at 0.0033 S/m 1.34. The two forwards also use
different anatomies and different source spaces: minimum electrode-to-source
distance 11.9 mm (FEM) against 16.8 mm (BEM), and the shallowest bin compares 756
FEM sources against 47 BEM ones. The comparison therefore cannot separate
conductivity from anatomy, and nothing here establishes which model is the more
accurate. The practical consequence is confined to amplitude: a unit-gain
reconstruction scales as 1/gain. The measured ratio is depth-dependent (1.86 at
22 mm to 1.46 at 50 mm), so a ``unit-noise-gain`` or NAI map, being invariant to
per-source rescaling, cannot move on this difference alone, whereas an
unnormalised power map can. What *would* move a normalised map is a change in
the direction of each lead-field column, and that is never measured anywhere in
this repository.

**When it does not help.** For localisation. There is no FEM-versus-BEM
localisation or topography comparison in the package, and the beamforming
demonstrations that do exist generate the sensor data from the same gain matrix
they then invert, on sources sitting on grid nodes: they establish
self-consistency, not robustness to head-model error. They are still
informative about the *beamformers*: on this 231-electrode array, with two
sources 34 mm apart (the example requests a 45 mm offset; the nearest mesh node
is 34.5 mm away and the example prints the achieved value) and time courses
correlated at r = 0.999 (the printed "0.95" is a mixing coefficient hardcoded
into the print statement), LCMV's two highest-power vertices are exactly the two
true source indices, 0.0 and 0.0 mm error, and
:func:`~advance_beamlab.make_mcmv` given those indices returns 20.2 and
19.8 nA m against 20 and 19 simulated. :func:`~advance_beamlab.scan_mcmv` on the
same data, by contrast, returns vertices 8.9 and 23.6 mm off, contradicting the
example's own framing of this as "the regime that defeats a single-source LCMV".

**The parameters that matter**, neither of them a beamformer parameter. First,
the reference and the channel matching. Build the info with
:func:`~advance_beamlab.make_ny_head_info`, or otherwise attach an
average-reference projector, because the lead field is supplied in common
average reference. This is not enforced for you on the LCMV path:
:func:`mne.beamformer.make_lcmv` accepts an info with no average-reference
projector and returns a filter without complaint, and only
:func:`~advance_beamlab.make_mcmv` raises, so on that path a wrongly
referenced dataset gives a silently wrong result. Note also that of the 231
positions only 160 are extended 10-05 names; the rest are face, neck and
fiducial positions carried as
EEG channels. Every conventional scalp cap therefore uses ``picks``, always
drops roughly a third of the array, and always drops it from below the equator,
so the "a subset's columns no longer sum to zero" approximation applies to every
realistic use rather than only to small caps, and its topography error is never
quantified here. Second, ``resolution``: ``'10K'`` is the default and ``'5K'``
is ample for EEG beamforming, the meshes being strict nested subsets; at
``'75K'`` the normal-orientation lead field is 137 MB and the free-orientation
one 412 MB. Other costs: a one-off 678 MB download of a GPL-v3 model that is
never redistributed with this BSD-3 package, the ``h5py`` and ``pooch`` extra,
and a rank of exactly 230 of 231 by construction, so any covariance estimated
from these data is singular and the rank must be resolved rather than inverted.
Capability limits: the source space is cortical surface only (no volume or
subcortical sources), and it is not a FreeSurfer subject, so there are no
labels (``read_labels_from_annot`` raises), no morph to ``fsaverage`` for group
statistics, and ``dist``/``patch_inds``/``pinfo`` are all ``None``. On this grid
the only decimation knob is ``resolution``, which matters because ReciPSIICOS is
O(N^2) in sources and its one free parameter is hand-set in the only test that
runs it here. One measured convenience: the free-orientation lead field is
genuinely rank 3 per source (median third-to-first singular value 0.52), so
``reduce_rank`` is unnecessary, unlike free-orientation MEG.

When plain LCMV is the better choice
--------------------------------------

**You need position, and your sources are strongly correlated.** On
231-electrode average-referenced EEG with two sources 34 mm apart correlated at
r = 0.95, plain LCMV localised both at 0.0 mm error while
:func:`~advance_beamlab.scan_mcmv` and ReciPSIICOS each missed by 20-40 mm. The
same miss reproduced on an MNE BEM forward with a matched source space, so it is
a property of those methods and not of the head model, and
:func:`~advance_beamlab.make_mcmv` given the true indices was fine on the same
data (20.2 and 19.8 nA m against 20 and 19 simulated). The failure is in the
search, not the estimator. The shipped FEM example reproduces the pattern: LCMV
exact at both sources, the greedy scan 8.9 and 23.6 mm off. Use LCMV to find the
sources and a constrained method to read them out; do not read amplitudes off
the localiser, since those are exactly what the cancellation attacks.

**Your sources are not correlated within the covariance window.** Over the
sample auditory 50-200 ms window (r = -0.13) the MCMV/LCMV amplitude ratios are
1.02 and 0.95. One qualification: if what you need is waveform shape rather than
peak amplitude, the joint constraint still pays at low correlation by removing
the partner's spatial leakage (RMSE 0.201 against 0.007 at r = 0.3), so use the
recovered source covariance to decide about *amplitude*, not about whether MCMV
has anything to fix.

**Real mixed-sensor MEG at ordinary SNR and ``reg=0.05``.** On the sample
auditory N100, ReciPSIICOS's hemispheric balance is 0.84 against LCMV's 0.94.
The projection discards part of the covariance and costs a little on that
metric. The example's own conclusion is that the dataset shows the pipeline
runs correctly end to end "rather than that it beats LCMV here".

**You want a sharper map.** On the sample auditory data the beamformer is not
the most compact of the three methods compared (dSPM is), and with no named
sources a linear method remains the safer default.

**Your target already carries the variance, or is sustained.** ABMC and LCMV
were both exact on the transient geometry once sensor noise was at or below 0.2x
the peak, and a sustained 10 Hz source was localised at 0.00 cm by both at 0.2x
and 1.3x noise.

**Your connectivity question is whether a strong genuine edge exists, or your
sensor SNR is moderate.** On the repository's genuine edge (truth 1.000) LCMV
returned 0.992 against PW-MCMV's 0.995. Below roughly 5 dB sensor SNR PW-MCMV's
data-adaptive residual leakage makes it the worse estimator of the spurious edge
as well (+0.658 against LCMV's +0.348 at -0.9 dB, truth -0.151); only
APW-MCMV's explicit null held.

**You cannot meet a method's precondition.** Constrained locations known only to
a centimetre, ROI vertices you cannot place to better than a grid step, a
template you cannot write down, a rank curve that warns at its own K*, or a
recording whose reference you cannot make average: in each case the advanced
method degrades to LCMV at best and below it at worst, and LCMV is the honest
result to report.

References
----------

.. footbibliography::