r"""
.. _ex-apw-mcmv-connectivity:

=========================================================
Leakage-free connectivity: pairwise and augmented MCMV
=========================================================

Functional connectivity is confounded by the spatial spread of the inverse
operator: a beamformer that leaks one region into another manufactures coupling
that is not there. This example shows, on a controlled simulation, how pairwise
MCMV (PW-MCMV) and augmented pairwise MCMV (APW-MCMV) remove that confound,
following Nunes et al. (2020).

We place three fixed-orientation sources in an EEG sphere model (no dataset
download):

- ``A`` and ``C`` sit about 2.4 cm apart, so ``A``'s leadfield overlaps ``C``'s;
- ``B`` is far from both;
- ``C`` is genuinely coupled to ``B`` (they share a slow amplitude envelope),
  while ``A`` carries an independent envelope.

So the *true* ``A``--``B`` coupling is essentially zero. Yet three things can
create a spurious ``A``--``B`` edge:

1. **Direct leakage / cancellation** -- a single-source LCMV reconstructs each
   region with an independent filter, so :math:`\hat s_A` is a mixture of every
   active source and picks up ``B`` directly.
2. **Indirect leakage** -- even after we forbid ``A`` and ``B`` from leaking into
   each other (a 2-source MCMV), :math:`\hat s_A` still contains a copy of the
   *conductor* ``C``; because ``C`` is coupled to ``B``, that copy correlates
   with :math:`\hat s_B` and a spurious edge survives.

PW-MCMV constrains :math:`\{A, B\}` jointly, so
:math:`\mathbf{w}_A^{\mathsf T}\mathbf{g}_B = 0` exactly and the *direct* path is
gone. APW-MCMV additionally adds the conductor ``C`` to the beamformer, which
places an exact null :math:`\mathbf{w}_A^{\mathsf T}\mathbf{g}_C = 0` and closes
the *indirect* path. The leakage coefficient
:math:`\alpha_C = \mathbf{w}_A^{\mathsf T}\mathbf{g}_C` is the sharpest summary:
nonzero for PW-MCMV, machine-zero for APW-MCMV.

Two details follow the paper. Connectivity is amplitude-envelope correlation
computed *plainly* (no orthogonalisation -- MCMV already removes leakage), and
the weights are built from a band-limited covariance matching the analysis band.
"""
# Authors: Sepehr Shirani and Muzhi Wang
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.beamformer import apply_lcmv, make_lcmv
from scipy.signal import hilbert

from advance_beamlab import (
    augmented_pairwise_mcmv_connectivity,
    make_mcmv,
    pairwise_mcmv_connectivity,
)

# %%
# Build a self-contained EEG forward: a standard 10-20 montage on a single-shell
# sphere, converted to fixed orientation. A fixed-orientation forward keeps the
# injected topographies and the beamformer's model of them perfectly consistent.

montage = mne.channels.make_standard_montage("standard_1020")
ch_names = list(dict.fromkeys(montage.ch_names))
info = mne.create_info(ch_names, sfreq=200.0, ch_types="eeg")
info.set_montage(montage)

sphere = mne.make_sphere_model("auto", "auto", info)
src = mne.setup_volume_source_space(sphere=sphere, pos=12.0)
fwd = mne.make_forward_solution(
    info, trans=None, src=src, bem=sphere, eeg=True, meg=False
)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False)

leadfield = fwd["sol"]["data"]  # (n_channels, n_sources)
source_rr = fwd["source_rr"]
n_channels = len(ch_names)

# %%
# Choose the three sources. ``A`` is central; ``C`` is a near neighbour ~2.4 cm
# away (so ``A`` leaks into it); ``B`` is ~9 cm away from ``A``.

idx_a = int(np.argmin(np.linalg.norm(source_rr - source_rr.mean(0), axis=1)))
dist_from_a = np.linalg.norm(source_rr - source_rr[idx_a], axis=1)
idx_c = int(np.argmin(np.abs(dist_from_a - 0.024)))
idx_b = int(np.argmin(np.abs(dist_from_a - 0.09)))
rois = [idx_a, idx_b, idx_c]  # matrix order: A=0, B=1, C=2

lead_a = leadfield[:, idx_a]
lead_b = leadfield[:, idx_b]
lead_c = leadfield[:, idx_c]
d_ac = np.linalg.norm(source_rr[idx_a] - source_rr[idx_c]) * 100
print(f"A-C distance: {d_ac:.1f} cm")
d_ab = np.linalg.norm(source_rr[idx_a] - source_rr[idx_b]) * 100
print(f"A-B distance: {d_ab:.1f} cm")

# %%
# Simulate alpha-band (10 Hz) sources with slowly varying amplitude envelopes.
# ``B`` shares ``C``'s envelope (a genuine C-B coupling); ``A``'s envelope is
# independent (no true A-B coupling). The sensor data is their leadfield mixture
# plus white sensor noise.

rng = np.random.default_rng(0)
sfreq = 200.0
n_times = int(120 * sfreq)
times = np.arange(n_times) / sfreq


def alpha_carrier(envelope, phase):
    """A 10 Hz carrier modulated by a slow amplitude envelope."""
    return envelope * np.cos(2 * np.pi * 10 * times + phase)


def slow_envelope(seed):
    """A smooth, strictly positive amplitude envelope."""
    smoother = np.exp(-0.5 * (np.arange(-200, 201) / 60) ** 2)
    smoother /= smoother.sum()
    white = np.random.default_rng(seed).standard_normal(n_times)
    raw = np.convolve(white, smoother, "same")
    return 1.0 + 0.8 * (raw - raw.mean()) / raw.std()


env_shared = slow_envelope(1)  # drives both C and B -> genuine coupling
env_a = slow_envelope(2)  # independent -> no true A-B coupling
sig_c = 2.0 * alpha_carrier(env_shared, 0.0)
sig_b = alpha_carrier(env_shared, 1.3)
sig_a = alpha_carrier(env_a, 2.1)

signal = np.outer(lead_a, sig_a) + np.outer(lead_b, sig_b) + np.outer(lead_c, sig_c)
noise = 0.1 * np.abs(lead_a).max() * rng.standard_normal((n_channels, n_times))
data = signal + noise

# %%
# Estimate the covariances. For envelope connectivity the sources are already
# narrow-band, so the broadband covariance *is* the band covariance here; on
# broadband recordings you would band-pass first and estimate the covariance in
# the band. Resting-state analyses have no baseline, so a diagonal (ad-hoc)
# noise covariance is used, as in the paper.

raw = mne.io.RawArray(data, info)
raw.set_eeg_reference("average", projection=True)
data_cov = mne.compute_covariance(
    mne.make_fixed_length_epochs(raw, duration=2.0), method="empirical"
)
noise_cov = mne.make_ad_hoc_cov(info)
evoked = mne.EvokedArray(raw.get_data(), info, tmin=0.0)
evoked.set_eeg_reference("average", projection=True)


def envelope_correlation(x, y):
    """Signed Pearson correlation of the Hilbert amplitude envelopes."""
    return float(np.corrcoef(np.abs(hilbert(x)), np.abs(hilbert(y)))[0, 1])


# ground-truth connectivity from the clean source signals
true_ab = envelope_correlation(sig_a, sig_b)
true_cb = envelope_correlation(sig_c, sig_b)

# %%
# **LCMV connectivity**: reconstruct each region with an independent
# single-source, unit-gain LCMV, then correlate the envelopes.

lcmv = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori=None,
    weight_norm=None,
)
lcmv_tc = apply_lcmv(evoked, lcmv).data
lcmv_ab = envelope_correlation(lcmv_tc[idx_a], lcmv_tc[idx_b])
lcmv_cb = envelope_correlation(lcmv_tc[idx_c], lcmv_tc[idx_b])

# %%
# **PW-MCMV connectivity**: every pair is reconstructed with a 2-source MCMV. The
# metric is delegated to ``mne-connectivity`` (plain envelope correlation).

conn_pw = pairwise_mcmv_connectivity(
    evoked,
    evoked.info,
    fwd,
    data_cov,
    rois,
    method="envelope",
    noise_cov=noise_cov,
    absolute=False,
)

# %%
# **APW-MCMV connectivity**: treat the (spuriously significant) A-B edge and the
# genuine C-B edge as significant, then re-estimate every significant pair with
# neighbour augmentation. For the A-B pair the conductor ``C`` -- within 4 cm of
# ``A`` and carrying a significant edge -- is added to the beamformer.

significance = np.zeros((3, 3), dtype=bool)
significance[0, 1] = significance[1, 0] = True  # A-B (spurious)
significance[1, 2] = significance[2, 1] = True  # C-B (genuine) -> makes C a conductor
conn_apw = augmented_pairwise_mcmv_connectivity(
    evoked,
    evoked.info,
    fwd,
    data_cov,
    rois,
    conn_pw,
    significance,
    positions=source_rr[rois],
    method="envelope",
    noise_cov=noise_cov,
    absolute=False,
)

# %%
# The leakage coefficient onto the conductor, :math:`\alpha_C`, makes the
# mechanism explicit: the pairwise filter for ``A`` leaks ``C`` (nonzero), while
# adding ``C`` to the beamformer nulls it to machine precision.

filt_pw = make_mcmv(
    evoked.info,
    fwd,
    data_cov,
    sources=[idx_a, idx_b],
    noise_cov=noise_cov,
    weight_norm="unit-gain",
)
filt_apw = make_mcmv(
    evoked.info,
    fwd,
    data_cov,
    sources=[idx_a, idx_b, idx_c],
    noise_cov=noise_cov,
    weight_norm="unit-gain",
)
alpha_c_pw = abs(float(filt_pw["weights"][0] @ lead_c))
alpha_c_apw = abs(float(filt_apw["weights"][0] @ lead_c))

print(f"\n{'':14s}{'A-B (spurious)':>16s}{'C-B (genuine)':>16s}")
print(f"{'ground truth':14s}{true_ab:16.3f}{true_cb:16.3f}")
print(f"{'LCMV':14s}{lcmv_ab:16.3f}{lcmv_cb:16.3f}")
print(f"{'PW-MCMV':14s}{conn_pw[0, 1]:16.3f}{conn_pw[1, 2]:16.3f}")
print(f"{'APW-MCMV':14s}{conn_apw[0, 1]:16.3f}{conn_apw[1, 2]:16.3f}")
print(f"\nleakage alpha_C:  PW-MCMV = {alpha_c_pw:.3f}   APW-MCMV = {alpha_c_apw:.1e}")

# %%
# **The spurious edge and the genuine edge.** The left panel is the false A-B
# connection: LCMV reports a large spurious value, PW-MCMV shrinks it (direct
# leakage removed) but leaves a residual bias, and APW-MCMV recovers the true
# near-zero value (indirect leakage removed). The right panel confirms the
# genuine C-B coupling is preserved throughout.

methods = ["LCMV", "PW-MCMV", "APW-MCMV"]
ab_values = [lcmv_ab, conn_pw[0, 1], conn_apw[0, 1]]
cb_values = [lcmv_cb, conn_pw[1, 2], conn_apw[1, 2]]
colors = ["#c44e52", "#dd8452", "#55a868"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.bar(methods, ab_values, color=colors)
ax1.axhline(true_ab, ls="--", color="k", lw=1, label=f"ground truth ({true_ab:.2f})")
ax1.set_title("A-B edge (true value ~ 0)")
ax1.set_ylabel("envelope correlation")
ax1.legend(loc="upper right", fontsize=9)
ax1.axhline(0, color="k", lw=0.6)

ax2.bar(methods, cb_values, color=colors)
ax2.axhline(true_cb, ls="--", color="k", lw=1, label=f"ground truth ({true_cb:.2f})")
ax2.set_title("C-B edge (genuine coupling)")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="lower right", fontsize=9)
fig.suptitle("PW-/APW-MCMV remove the spurious A-B edge, keep the genuine C-B edge")
fig.tight_layout()

# %%
# **The leakage coefficient.** On a log scale, the conductor leakage
# :math:`\alpha_C = \mathbf{w}_A^{\mathsf T}\mathbf{g}_C` drops from ~0.1 for
# PW-MCMV to machine zero once ``C`` is added by APW-MCMV -- the exact null that
# closes the indirect-leakage path.

fig2, ax = plt.subplots(figsize=(5, 4.2))
ax.bar(
    ["PW-MCMV\n(leaks C)", "APW-MCMV\n(nulls C)"],
    [alpha_c_pw, max(alpha_c_apw, 1e-18)],
    color=["#dd8452", "#55a868"],
)
ax.set_yscale("log")
ax.set_ylabel(r"$|\alpha_C| = |\mathbf{w}_A^{\mathsf{T}} \mathbf{g}_C|$")
ax.set_title("Leakage onto the conductor C")
fig2.tight_layout()

# %%
# The exact null is what distinguishes APW-MCMV: whereas PW-MCMV suppresses the
# conductor only through the data-adaptive inverse (leaving a residual that
# biases connectivity), APW-MCMV removes it by an explicit constraint. In a clean
# three-source scene the envelope-correlation bias is modest, because the
# adaptive inverse already suppresses most of ``C``; in realistic multi-source
# resting-state data the indirect leakage accumulates across many conductors,
# which is where APW-MCMV's advantage is largest (Nunes et al., 2020).
