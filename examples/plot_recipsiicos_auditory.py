"""
.. _ex-recipsiicos-auditory:

=======================================================
ReciPSIICOS beamforming of bilateral auditory responses
=======================================================

Correlated bilateral sources are the classic failure mode of the LCMV
beamformer: it minimises output power under a unit-gain constraint, so two
synchronous sources let it place a null that cancels them. ReciPSIICOS repairs
the *data covariance* -- projecting out the cross-product (coupling) subspace --
so that an ordinary LCMV no longer cancels them.

This example contrasts a standard LCMV with ReciPSIICOS on the auditory
response of the MNE sample dataset, whose left- and right-hemisphere auditory
sources are strongly correlated.

Two practical points, both important on real data. The projector is built from
the forward and must span where the data covariance's energy lives, so we use a
*whole-brain* grid, not a region -- but the ``whitened`` correlation Gram is
:math:`O(N^2)`, so we decimate that grid to keep it tractable. And the
covariances are shrinkage-regularised, the correct estimator across the
magnetometer/gradiometer unit scales.
"""
# Authors: the mne-beamlab contributors
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
from mne import Label
from mne.beamformer import apply_lcmv_cov, make_lcmv
from mne.forward import restrict_forward_to_label

from mne_beamlab import make_recipsiicos_lcmv, recipsiicos_rank_curve

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

data_cov = mne.compute_covariance(epochs, tmin=0.05, tmax=0.2, method="shrunk")
noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.0, method="shrunk")

# %%
# Load the real BEM forward and decimate it to a coarse whole-brain grid (every
# 16th vertex per hemisphere), keeping both hemispheres so it still spans the
# head while the whitened Gram stays small.

fwd = mne.read_forward_solution(meg / "sample_audvis-meg-eeg-oct-6-fwd.fif")
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
labels = [
    Label(fwd["src"][h]["vertno"][::16], hemi=hemi, subject="sample")
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
# Compare the hemispheric balance. A plain LCMV tends to suppress one side of a
# correlated pair; ReciPSIICOS retains both, so the ratio of the weaker to the
# stronger hemisphere peak is closer to one.


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

brain = stc_recip.plot(
    subject="sample",
    subjects_dir=subjects_dir,
    hemi="both",
    views="lateral",
    clim=dict(kind="percent", lims=[90, 95, 99]),
)