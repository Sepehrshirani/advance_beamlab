"""
.. _ex-recipsiicos-auditory:

=======================================================
ReciPSIICOS beamforming of bilateral auditory responses
=======================================================

Correlated bilateral sources are the classic failure mode of the LCMV
beamformer: it minimises output power under a unit-gain constraint, so two
synchronous sources let it place a null that cancels them. ReciPSIICOS repairs
the *data covariance* by projecting out the cross-product (coupling) subspace,
so that an ordinary LCMV no longer cancels them.

This example runs ReciPSIICOS end to end on real, mixed-sensor MEG and contrasts
it with a standard LCMV. The recording is the auditory response of the MNE sample
dataset, whose left- and right-hemisphere sources are correlated across the N100
(r = +0.55).

A caveat stated up front, because the example reports it honestly below: on this
recording a plain LCMV already recovers both hemispheres, so ReciPSIICOS does
*not* beat it on hemispheric balance here. Signal cancellation is a high-SNR,
high-correlation effect that regularisation suppresses. What this example does
show is the whole real-data pipeline working: the rank curve and its 45-degree
criterion, the projector, the noise whitening across magnetometers and
gradiometers, and the virtual-sensor reduction. For the regime where the
cancellation is dramatic, and where this method demonstrably repairs it, see
:ref:`ex-recipsiicos-simulation`, which runs the same comparison on simulated
sources placed on this same realistic forward.

Two practical points, both important on real data. The projector is built from
the forward and must span where the data covariance's energy lives, so we use a
*whole-brain* grid, not a region. But the ``whitened`` correlation Gram is
:math:`O(N^2)`, so we decimate that grid to keep it tractable. And the
covariances are shrinkage-regularised, the correct estimator across the
magnetometer/gradiometer unit scales.
"""
# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import Label
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv
from mne.forward import restrict_forward_to_label

from advance_beamlab import make_recipsiicos_lcmv, recipsiicos_rank_curve

data_path = mne.datasets.sample.data_path()
meg = data_path / "MEG" / "sample"
subjects_dir = data_path / "subjects"

# %%
# Load the auditory epochs (left and right stimulation) and shrinkage-
# regularised active and baseline covariances.

raw = mne.io.read_raw_fif(meg / "sample_audvis_filt-0-40_raw.fif", preload=True)
events = mne.read_events(meg / "sample_audvis_filt-0-40_raw-eve.fif")
raw.pick("meg")  # magnetometers + gradiometers -> whitening is exercised
epochs = mne.Epochs(
    raw,
    events,
    {"Auditory/Left": 1, "Auditory/Right": 2},
    tmin=-0.2,
    tmax=0.25,
    baseline=(None, 0.0),
    preload=True,
)

# The active window is the N100 (80-130 ms), where the bilateral auditory
# response is genuinely correlated across hemispheres (r = +0.55 as recovered by
# a joint MCMV filter). Over the wider 50-200 ms window the correlation washes
# out to about zero, and no correlated-source method has anything to work on.
data_cov = mne.compute_covariance(epochs, tmin=0.08, tmax=0.13, method="shrunk")
noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.0, method="shrunk")

# %%
# Load the real BEM forward and decimate it to a coarse whole-brain grid, keeping
# both hemispheres so it still spans the head while the whitened Gram stays
# tractable. That Gram is quadratic in the number of sources.

fwd = mne.read_forward_solution(meg / "sample_audvis-meg-eeg-oct-6-fwd.fif")
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
labels = [
    Label(fwd["src"][h]["vertno"][::44], hemi=hemi, subject="sample")
    for h, hemi in enumerate(("lh", "rh"))
]
fwd = restrict_forward_to_label(fwd, labels)

# %%
# Choose the projection rank from the power-vs-correlation curve. The 45-degree
# point (``return_optimal=True``) is where the correlation subspace stops
# emptying faster than the power subspace.

ranks, p_pwr, p_cor, kstar = recipsiicos_rank_curve(
    fwd, epochs.info, method="whitened", noise_cov=noise_cov, return_optimal=True
)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(ranks, p_pwr, label="power retained")
ax.plot(ranks, p_cor, label="correlation retained")
ax.axvline(kstar, color="k", ls="--", label=f"K* = {kstar}")
ax.set(
    xlabel="projection rank K",
    ylabel="retained energy fraction",
    title="ReciPSIICOS rank curve",
)
ax.legend()

# Both curves are flat at zero once the rank passes the power cliff, so zoom the
# x-axis to the range where power and correlation actually change. Otherwise the
# informative part is squeezed into a small fraction of a q^2-wide axis.
informative = (p_pwr > 0.01).nonzero()[0]
xmax = int(ranks[informative[-1]]) if informative.size else int(ranks[-1])
ax.set_xlim(0, min(int(ranks[-1]), int(1.15 * xmax)))

# %%
# Build a standard LCMV and a ReciPSIICOS beamformer on the same data. Free
# orientation MEG needs ``reduce_rank=True`` (the radial-silent leadfield is
# rank-deficient).

lcmv = make_lcmv(
    epochs.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    reduce_rank=True,
)
recip = make_recipsiicos_lcmv(
    epochs.info,
    fwd,
    data_cov,
    rank=kstar,
    method="whitened",
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    reduce_rank=True,
)

stc_lcmv = apply_lcmv_cov(data_cov, lcmv)
stc_recip = apply_lcmv_cov(data_cov, recip)

# %%
# Compare the hemispheric balance: the ratio of the weaker to the stronger
# hemisphere peak. The textbook expectation is that a plain LCMV suppresses one
# side of a correlated pair while ReciPSIICOS retains both.
#
# **That is not what happens on this recording, and it is worth being explicit
# about it.** On the ``sample`` auditory data, at this SNR and with the usual
# ``reg=0.05``, a plain LCMV already recovers both hemispheres almost equally, so
# there is essentially no imbalance left for ReciPSIICOS to repair. The
# projection, which necessarily discards part of the covariance, comes at a small
# cost on this particular metric. Signal cancellation is an idealised, high-SNR,
# high-correlation effect: regularisation and sensor noise both push the
# recovered amplitudes back up. :ref:`ex-recipsiicos-simulation` dials the
# correlation directly and shows this method holding the amplitude LCMV loses.
#
# The honest summary is that this dataset demonstrates that ReciPSIICOS *runs
# correctly end to end on real, mixed-sensor MEG* (the rank curve, the projector,
# the whitening and the virtual-sensor reduction), and that it does not beat LCMV
# *on this balance metric*. It does recover a substantially larger waveform at
# the auditory peaks, which the last figure of this example shows.


def hemi_peaks(stc):
    """Peak power in the left and right hemisphere of a surface stc."""
    n_lh = len(stc.vertices[0])
    return stc.data[:n_lh].max(), stc.data[n_lh:].max()


for name, stc in [("LCMV", stc_lcmv), ("ReciPSIICOS", stc_recip)]:
    lh, rh = hemi_peaks(stc)
    print(
        f"{name:12s}  lh peak {lh:.3g}   rh peak {rh:.3g}   "
        f"balance {min(lh, rh) / max(lh, rh):.2f}"
    )

# %%
# Plot the ReciPSIICOS source estimate on the inflated cortex.

try:
    brain = stc_recip.plot(
        subject="sample",
        subjects_dir=subjects_dir,
        # "split", not "both". With hemi="both" the two inflated surfaces are
        # offset along x and a single lateral camera looks straight down that
        # axis, so the near hemisphere hides the far one completely. That matters
        # here more than most places: the left hemisphere carries this estimate's
        # global maximum, so the one figure of an example about hemispheric
        # balance was hiding the larger of the two peaks it compares.
        hemi="split",
        views="lateral",
        size=(1000, 500),
        clim=dict(kind="percent", lims=[90, 95, 99]),
        time_viewer=False,  # static image: no interactive picking (offscreen-safe)
    )
except Exception as exc:  # 3-D rendering is optional
    brain = None
    print(f"3-D brain plot skipped (no working 3-D backend): {exc}")

# The rendered surface is screenshotted into a matplotlib axis rather than left
# for a 3-D scraper to find. See doc/conf.py for why: it is what makes the
# figure appear correctly in the built documentation, and it keeps the window
# open, which the CI runner needs.
if brain is not None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.imshow(brain.screenshot())
    ax.set_axis_off()

# %%
# Finally, the reconstructed time courses at the two hemispheric peaks: the same
# comparison as the power map, now as a waveform. The ratio in each panel title
# is the ReciPSIICOS peak over the LCMV peak.
#
# These panels disagree with the balance metric above, and the disagreement is
# the interesting part rather than an inconsistency to smooth over. On
# hemispheric balance the two filters are close, because LCMV already recovers
# both hemispheres. On waveform amplitude at the auditory peaks they are not:
# ReciPSIICOS returns roughly four and a half times the LCMV peak on the left and
# about a third more on the right. That is not an artefact of the two weight
# normalisations, since the pre-stimulus baseline deviations are comparable
# between the two filters, so the peak signal-to-noise improves along with the
# amplitude.
#
# The two metrics are asking different questions. Balance asks whether each
# hemisphere shows up at all, and LCMV passes. Peak amplitude asks how much of
# the source survived the filter, and that is where cancellation leaves its mark.

evoked = epochs.average()
tc_lcmv = apply_lcmv(evoked, lcmv)
tc_recip = apply_lcmv(evoked, recip)

n_lh = len(stc_recip.vertices[0])
peaks = [
    int(np.argmax(stc_recip.data[:n_lh, 0])),
    n_lh + int(np.argmax(stc_recip.data[n_lh:, 0])),
]

times = evoked.times * 1e3
post = evoked.times >= 0.0
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5), constrained_layout=True)
for ax, idx, hemi in zip(axes, peaks, ("Left", "Right"), strict=True):
    r_peak = np.abs(tc_recip.data[idx, post]).max()
    l_peak = np.abs(tc_lcmv.data[idx, post]).max()
    ratio = r_peak / l_peak
    # ``pick_ori='max-power'`` makes MNE solve a non-Hermitian eigenproblem, so
    # the weights (and hence these time courses) come back complex with a
    # negligible imaginary part; take the real part explicitly to plot.
    ax.plot(times, tc_lcmv.data[idx].real, color="C0", label="LCMV")
    ax.plot(times, tc_recip.data[idx].real, color="C3", label="ReciPSIICOS")
    ax.axvline(0, color="k", lw=0.5)
    ax.set(
        title=f"{hemi} auditory peak  (ReciPSIICOS/LCMV = {ratio:.2f})",
        ylabel="amplitude (a.u.)",
    )
    ax.legend(loc="upper right")
    # Printed as well as drawn, because the paragraph above quotes these two
    # numbers and a reader should be able to check them against the run.
    base = evoked.times < 0.0
    print(
        f"{hemi:>5} peak: LCMV {l_peak:.3g}, ReciPSIICOS {r_peak:.3g}, "
        f"ratio {ratio:.2f}; pre-stimulus SD "
        f"{np.std(tc_lcmv.data[idx, base].real):.3g} vs "
        f"{np.std(tc_recip.data[idx, base].real):.3g}"
    )
axes[-1].set_xlabel("time (ms)")

# sphinx_gallery_thumbnail_number = 1
