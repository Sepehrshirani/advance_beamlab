"""
.. _ex-mcmv-auditory:

====================================================
MCMV reconstruction of correlated bilateral sources
====================================================

The multi-source (MCMV) beamformer constrains several sources *jointly*,
placing a null at every other constrained location so that correlated sources
no longer cancel one another. This example reconstructs the two (correlated)
auditory sources of the MNE sample dataset with a single joint filter and
compares their time courses to a per-source LCMV, which attenuates them.

The forward is restricted to the superior-temporal (auditory) labels and
converted to fixed orientation, the natural setting for a small, known set of
constrained sources.
"""
# Authors: the mne-beamlab contributors
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv

from mne_beamlab import apply_mcmv, make_mcmv

data_path = mne.datasets.sample.data_path()
meg = data_path / "MEG" / "sample"
subjects_dir = data_path / "subjects"

# %%
# Load the auditory epochs and evoked response, the covariances, and the real
# BEM forward restricted to auditory cortex and set to fixed orientation.

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

data_cov = mne.compute_covariance(epochs, tmin=0.05, tmax=0.2, method="empirical")
noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.0, method="empirical")

fwd = mne.read_forward_solution(meg / "sample_audvis-meg-eeg-oct-6-fwd.fif")
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
labels = mne.read_labels_from_annot(
    "sample", "aparc", regexp="superiortemporal", subjects_dir=subjects_dir
)
fwd = mne.forward.restrict_forward_to_label(fwd, labels)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True)

# %%
# Locate the two sources to constrain: the strongest vertex in each hemisphere
# of a plain LCMV power map. (:func:`mne_beamlab.scan_mcmv` can instead discover
# them automatically with a data-driven sequential search.)

lcmv = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    weight_norm="unit-noise-gain",
)
power = apply_lcmv_cov(data_cov, lcmv)
n_lh = len(power.vertices[0])
lh_src = int(power.data[:n_lh].argmax())
rh_src = int(n_lh + power.data[n_lh:].argmax())
sources = [lh_src, rh_src]

# %%
# Build one joint MCMV filter over both sources and reconstruct their time
# courses from the evoked response.

mcmv = make_mcmv(evoked.info, fwd, data_cov, sources=sources, noise_cov=noise_cov)
tc_mcmv = apply_mcmv(evoked, mcmv)  # (n_sources, n_times)

# Per-source LCMV time courses for the same two vertices, for comparison.
stc_lcmv = apply_lcmv(evoked, lcmv)
tc_lcmv = stc_lcmv.data[sources]

# %%
# Plot the reconstructions. The joint MCMV recovers both hemispheres with their
# expected amplitude; the per-source LCMV cancels the correlated pair.

times = evoked.times * 1e3  # ms
fig, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True)
for ax, hemi, row in zip(axes, ("left", "right"), (0, 1), strict=False):
    ax.plot(times, tc_lcmv[row], color="tab:gray", label="LCMV (per source)")
    ax.plot(times, tc_mcmv[row], color="tab:red", label="MCMV (joint)")
    ax.axvline(0, color="k", lw=0.5)
    ax.set(ylabel=f"{hemi} auditory\n(a.u.)")
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("time (ms)")
fig.suptitle("Bilateral auditory reconstruction: MCMV vs LCMV")
