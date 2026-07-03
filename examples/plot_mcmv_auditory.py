"""
.. _ex-mcmv-auditory:

===================================================
MCMV joint reconstruction of correlated MEG sources
===================================================

A single-source LCMV reconstructs each location with its own filter. When two
sources are temporally correlated -- as the left and right auditory cortices are
during binaural stimulation -- each filter treats the other source as
interference and partially nulls it, so the recovered time courses are
attenuated and mutually contaminated.

The multi-source (MCMV) beamformer constrains all sources in a single filter
set: each filter passes its own source with unit gain while placing an exact
null on the *other* constrained sources (Moiseev et al., 2011, Eq. 5). This
example locates the two auditory sources with an ordinary LCMV, then contrasts
the per-source LCMV time courses with the MCMV joint reconstruction.

The covariances are shrinkage-regularised (the correct estimator across the
magnetometer/gradiometer unit scales), and the forward is the full whole-brain
BEM solution -- MCMV constrains only the two chosen sources, so no decimation is
needed.
"""
# Authors: Sepehr Shirani
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv

from mne_beamlab import apply_mcmv, apply_mcmv_cov, make_mcmv

data_path = mne.datasets.sample.data_path()
meg = data_path / "MEG" / "sample"
subjects_dir = data_path / "subjects"

# %%
# Load the auditory epochs (left and right stimulation), the evoked response,
# and shrinkage-regularised active and baseline covariances.

raw = mne.io.read_raw_fif(meg / "sample_audvis_filt-0-40_raw.fif", preload=True)
events = mne.read_events(meg / "sample_audvis_filt-0-40_raw-eve.fif")
raw.pick("meg")
epochs = mne.Epochs(
    raw,
    events,
    {"Auditory/Left": 1, "Auditory/Right": 2},
    tmin=-0.2,
    tmax=0.25,
    baseline=(None, 0.0),
    preload=True,
)
evoked = epochs.average()

data_cov = mne.compute_covariance(epochs, tmin=0.05, tmax=0.2, method="shrunk")
noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.0, method="shrunk")

# %%
# Use the full whole-brain BEM forward with fixed (surface-normal) orientation.

fwd = mne.read_forward_solution(meg / "sample_audvis-meg-eeg-oct-6-fwd.fif")
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True)

# %%
# Locate the two auditory sources with a standard LCMV: the strongest vertex in
# each hemisphere.

lcmv = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori=None,
    weight_norm="unit-noise-gain",
)
stc_pow = apply_lcmv_cov(data_cov, lcmv)

n_lh = len(stc_pow.vertices[0])
lh_idx = int(np.argmax(stc_pow.data[:n_lh]))
rh_idx = n_lh + int(np.argmax(stc_pow.data[n_lh:]))
sources = [lh_idx, rh_idx]
print(f"selected sources (left, right hemisphere): {sources}")

# %%
# The LCMV power map itself is the localisation figure: the two auditory sources
# selected above appear as the bilateral peaks on the inflated cortex. MCMV will
# then jointly constrain exactly these two locations.

brain = stc_pow.plot(
    subject="sample",
    subjects_dir=subjects_dir,
    hemi="both",
    views="lateral",
    clim=dict(kind="percent", lims=[90, 95, 99]),
)

# %%
# Build the MCMV joint filter on those two sources, and a per-source LCMV, both
# with *unit gain* so the recovered amplitudes are directly comparable. Unit gain
# reads out the physical source amplitude, so the two beamformers agree exactly
# when there is nothing to cancel and diverge only through correlated-source
# cancellation -- unlike unit-noise-gain, which offsets the two even when the
# sources are independent.

lcmv_ug = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori=None,
    weight_norm=None,  # unit gain: physical source amplitude
)
mcmv = make_mcmv(
    evoked.info,
    fwd,
    data_cov,
    sources=sources,
    noise_cov=noise_cov,
    weight_norm="unit-gain",
)
s_mcmv = apply_mcmv(evoked, mcmv)  # (2, n_times)
s_lcmv = apply_lcmv(evoked, lcmv_ug).data[sources]  # (2, n_times)

# %%
# The MCMV null on the opposite source removes the shared-signal cancellation, so
# the joint auditory time courses are recovered at a fuller amplitude than the
# per-source LCMV traces. The peak-amplitude ratio in each panel quantifies how
# much the per-source LCMV is attenuated (a ratio near one means the two auditory
# sources were only weakly correlated in this covariance, so there was little to
# cancel).

times = evoked.times * 1e3
post = evoked.times >= 0.0
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5), constrained_layout=True)
for ax, i, hemi in zip(axes, range(2), ("Left", "Right"), strict=True):
    ratio = np.abs(s_mcmv[i, post]).max() / np.abs(s_lcmv[i, post]).max()
    ax.plot(times, s_lcmv[i], color="C0", label="LCMV (per-source)")
    ax.plot(times, s_mcmv[i], color="C3", label="MCMV (joint)")
    ax.axvline(0, color="k", lw=0.5)
    ax.set(title=f"{hemi} auditory source  (MCMV/LCMV peak = {ratio:.2f})",
           ylabel="amplitude (Am)")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("time (ms)")

# %%
# Unlike a set of independent LCMV filters, MCMV also returns the joint source
# covariance directly (:func:`~mne_beamlab.apply_mcmv_cov`). Normalised to a
# correlation matrix, its off-diagonal is the recovered coupling between the two
# hemispheres -- a quantity the joint constraint estimates and per-source LCMV
# cannot provide at all.

src_cov = apply_mcmv_cov(data_cov, mcmv)
std = np.sqrt(np.diag(src_cov))
src_corr = src_cov / np.outer(std, std)  # scale-invariant correlation matrix

fig, ax = plt.subplots(constrained_layout=True, figsize=(4.2, 3.6))
im = ax.imshow(src_corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set(
    xticks=[0, 1], yticks=[0, 1],
    xticklabels=["Left", "Right"], yticklabels=["Left", "Right"],
    title=f"MCMV source coupling\nrecovered correlation r = {src_corr[0, 1]:.2f}",
)
for a in range(2):
    for b in range(2):
        ax.text(b, a, f"{src_corr[a, b]:.2f}", ha="center", va="center", color="k")
fig.colorbar(im, ax=ax, label="correlation")
