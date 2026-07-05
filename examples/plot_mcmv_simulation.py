r"""
.. _ex-mcmv-simulation:

======================================================
MCMV versus LCMV for correlated sources: a simulation
======================================================

Two temporally correlated sources are the textbook failure mode of the LCMV
beamformer. Because LCMV minimises output power under a unit-gain constraint, a
second source that is correlated with the target lets the filter place a null
that *cancels the target itself*: the recovered amplitude of a source shrinks by
a factor of :math:`1 - \rho^2` (in root-mean-square, :math:`\sqrt{1 - \rho^2}`)
as its correlation :math:`\rho` with the neighbour approaches one. At
:math:`\rho = 0.95` more than 90% of the power is lost.

The multi-source (MCMV) beamformer constrains both sources in one filter set, so
each filter passes its own source with unit gain while placing an *exact* null
on the other constrained source (Moiseev et al., 2011, Eq. 5). That exact null
is what LCMV lacks, and it is imposed by construction rather than estimated from
the data -- so MCMV recovers the full amplitude at any correlation.

Two points are worth stating because they are easy to get wrong. First, this
cancellation does **not** require the sources to be spatially close: it is driven
by temporal correlation, not leadfield overlap, and two well-separated
(near-orthogonal) sources are cancelled just as strongly as neighbouring ones.
Second, it is an *idealised, high-SNR* effect. Sensor noise and covariance
regularisation both push the recovered amplitude back up towards its true value,
which is exactly why the effect is dramatic in a clean simulation but mild on a
regularised real-data recording of moderately correlated sources.

We use a self-contained EEG sphere model (no dataset download) with a
fixed-orientation forward, inject a pair of sources whose correlation we control
directly, and reconstruct each source with a standard LCMV and with MCMV. To see
the cancellation we must read out the *physical* source amplitude, so both
beamformers use unit-gain weights (``weight_norm=None`` for
:func:`~mne.beamformer.make_lcmv`, ``"unit-gain"`` for
:func:`~advance_beamlab.make_mcmv`); a noise-normalised readout rescales the axis and
obscures the comparison.
"""
# Authors: Sepehr Shirani and Muzhi Wang
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv

from advance_beamlab import apply_mcmv, make_mcmv

# %%
# Build a self-contained EEG forward: a standard 10-20 montage on a
# single-shell sphere, converted to fixed orientation. A fixed-orientation
# forward is essential here -- with free orientation the LCMV
# ``pick_ori="max-power"`` re-optimises each source's orientation and steers
# *around* the cancellation, hiding the very effect we want to show.

montage = mne.channels.make_standard_montage("standard_1020")
ch_names = list(dict.fromkeys(montage.ch_names))
info = mne.create_info(ch_names, sfreq=200.0, ch_types="eeg")
info.set_montage(montage)

sphere = mne.make_sphere_model("auto", "auto", info)
src = mne.setup_volume_source_space(sphere=sphere, pos=15.0)
fwd = mne.make_forward_solution(
    info, trans=None, src=src, bem=sphere, eeg=True, meg=False
)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False)

leadfield = fwd["sol"]["data"]  # (n_channels, n_sources), one column per source
source_rr = fwd["source_rr"]
n_channels = len(ch_names)

# %%
# Pick two sources about 8 cm apart. Injecting each source *along the forward's
# own fixed orientation* (i.e. using its leadfield column directly) keeps the
# simulated topography and the beamformer's model of it perfectly consistent.

center = int(np.argmin(np.linalg.norm(source_rr - source_rr.mean(0), axis=1)))
dist_cm = np.linalg.norm(source_rr - source_rr[center], axis=1) * 100
partner = int(np.argmin(np.abs(dist_cm - 8.0)))
sources = [center, partner]

g0, g1 = leadfield[:, center], leadfield[:, partner]
overlap = abs((g0 @ g1) / (np.linalg.norm(g0) * np.linalg.norm(g1)))
print(
    f"source separation: {dist_cm[partner]:.1f} cm   "
    f"|leadfield overlap|: {overlap:.2f}"
)

# %%
# A helper that simulates one epoched dataset for a target correlation and
# returns the amplitude each beamformer recovers for the first source. The two
# source time courses are 10 Hz oscillations phase-shifted by
# :math:`\varphi = \arccos\rho`, so their correlation is exactly :math:`\rho`.
# Trial noise makes the covariances well conditioned; a pre-stimulus baseline
# supplies the noise covariance.

rng = np.random.default_rng(0)
sfreq, n_times, tmin = 200.0, 160, -0.2
times = np.arange(n_times) / sfreq + tmin
active = times >= 0.0
f0, n_epochs, noise = 10.0, 60, 0.05


def recovered_amplitudes(rho):
    """Return (LCMV, MCMV) recovered RMS amplitude of the first source."""
    phi = np.arccos(rho)
    s0 = np.zeros(n_times)
    s1 = np.zeros(n_times)
    s0[active] = np.sin(2 * np.pi * f0 * times[active])
    s1[active] = np.sin(2 * np.pi * f0 * times[active] + phi)  # corr(s0, s1) = rho

    scale = noise * np.abs(g0).max()
    data = np.stack(
        [
            np.outer(g0, s0)
            + np.outer(g1, s1)
            + scale * rng.standard_normal((n_channels, n_times))
            for _ in range(n_epochs)
        ]
    )
    epochs = mne.EpochsArray(data, info, tmin=tmin, baseline=(None, 0.0), verbose=False)
    epochs.set_eeg_reference("average", projection=True, verbose=False)
    evoked = epochs.average()

    data_cov = mne.compute_covariance(
        epochs, tmin=0.0, tmax=None, method="shrunk", verbose=False
    )
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="shrunk", verbose=False
    )

    lcmv = make_lcmv(
        evoked.info, fwd, data_cov, reg=0.02, noise_cov=noise_cov,
        pick_ori=None, weight_norm=None,  # scalar, unit-gain: reads out true amplitude
    )
    rec_lcmv = apply_lcmv(evoked, lcmv).data[center, active]

    mcmv = make_mcmv(
        evoked.info, fwd, data_cov, sources=sources, noise_cov=noise_cov,
        weight_norm="unit-gain",
    )
    rec_mcmv = apply_mcmv(evoked, mcmv)[0, active]

    return float(np.sqrt(np.mean(rec_lcmv**2))), float(np.sqrt(np.mean(rec_mcmv**2)))


# %%
# Sweep the correlation and normalise each beamformer to its own uncorrelated
# (:math:`\rho = 0`) amplitude, so the two curves start together at 1 and the
# gap that opens is the cancellation itself.

rhos = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
lcmv_amp, mcmv_amp = np.array([recovered_amplitudes(r) for r in rhos]).T
lcmv_rel = lcmv_amp / lcmv_amp[0]
mcmv_rel = mcmv_amp / mcmv_amp[0]

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(rhos, mcmv_rel, "o-", color="C3", label="MCMV (joint)")
ax.plot(rhos, lcmv_rel, "o-", color="C0", label="LCMV (per-source)")
ax.plot(rhos, np.sqrt(1 - rhos**2), "k--", lw=1, label=r"ideal $\sqrt{1-\rho^2}$")
ax.set(
    xlabel=r"source correlation $\rho$",
    ylabel="recovered amplitude (relative to $\\rho=0$)",
    title="LCMV cancels correlated sources; MCMV does not",
    ylim=(0, 1.15),
)
ax.legend()

# %%
# The LCMV curve collapses along the analytic :math:`\sqrt{1-\rho^2}` law while
# MCMV stays flat at one. Now look at the reconstructed time courses at a single
# high correlation to see what that means for a recovered waveform: the LCMV
# estimate is strongly attenuated, while MCMV recovers the injected oscillation.

rho_demo = 0.95
phi = np.arccos(rho_demo)
s0 = np.zeros(n_times)
s1 = np.zeros(n_times)
s0[active] = np.sin(2 * np.pi * f0 * times[active])
s1[active] = np.sin(2 * np.pi * f0 * times[active] + phi)
scale = noise * np.abs(g0).max()
data = np.stack(
    [
        np.outer(g0, s0)
        + np.outer(g1, s1)
        + scale * rng.standard_normal((n_channels, n_times))
        for _ in range(n_epochs)
    ]
)
epochs = mne.EpochsArray(data, info, tmin=tmin, baseline=(None, 0.0), verbose=False)
epochs.set_eeg_reference("average", projection=True, verbose=False)
evoked = epochs.average()
data_cov = mne.compute_covariance(
    epochs, tmin=0.0, tmax=None, method="shrunk", verbose=False
)
noise_cov = mne.compute_covariance(
    epochs, tmin=None, tmax=0.0, method="shrunk", verbose=False
)

# %%
# Spatial check: before comparing waveforms, confirm both beamformers are
# looking at the right place. A whole-grid LCMV power map (the standard
# unit-noise-gain power beamformer) should peak at the two injected sources.
# Both lie at x = 0, so we project onto the sagittal (y, z) plane.

lcmv_grid = make_lcmv(evoked.info, fwd, data_cov, reg=0.02, noise_cov=noise_cov,
                      pick_ori=None, weight_norm="unit-noise-gain")
power = apply_lcmv_cov(data_cov, lcmv_grid).data[:, 0]

rr_mm = source_rr * 1e3
order = np.argsort(power)  # draw faint sources first so the peaks sit on top
fig, ax = plt.subplots(constrained_layout=True)
sctr = ax.scatter(rr_mm[order, 1], rr_mm[order, 2], c=power[order], cmap="hot", s=35)
ax.scatter(
    rr_mm[sources, 1], rr_mm[sources, 2], s=170,
    facecolors="none", edgecolors="tab:cyan", linewidths=2, label="injected sources",
)
fig.colorbar(sctr, ax=ax, label="LCMV power (unit-noise-gain)")
ax.set(
    xlabel="y (mm)", ylabel="z (mm)", aspect="equal",
    title=f"LCMV power localises both sources ($\\rho = {rho_demo}$)",
)
ax.legend(loc="upper right")

# %%
# Now compare the reconstructed waveforms against ground truth. Because both
# beamformers use unit gain, their output is on the same scale as the injected
# signal, so we can overlay the true source directly. We quantify recovery two
# ways: peak amplitude (does the source keep its size?) and waveform fidelity
# (the correlation with the true time course -- does it keep its shape?).

lcmv = make_lcmv(evoked.info, fwd, data_cov, reg=0.02, noise_cov=noise_cov,
                 pick_ori=None, weight_norm=None)
mcmv = make_mcmv(evoked.info, fwd, data_cov, sources=sources, noise_cov=noise_cov,
                 weight_norm="unit-gain")
rec_lcmv = apply_lcmv(evoked, lcmv).data[center]
rec_mcmv = apply_mcmv(evoked, mcmv)[0]

truth = s0  # the injected source-0 waveform
fidelity_lcmv = np.corrcoef(rec_lcmv[active], truth[active])[0, 1]
fidelity_mcmv = np.corrcoef(rec_mcmv[active], truth[active])[0, 1]
peak_lcmv = np.abs(rec_lcmv[active]).max()
peak_mcmv = np.abs(rec_mcmv[active]).max()

fig, ax = plt.subplots(constrained_layout=True, figsize=(7.5, 4))
ax.plot(times * 1e3, truth, color="0.6", lw=3, label="ground truth")
ax.plot(
    times * 1e3, rec_mcmv, color="C3",
    label=f"MCMV: peak {peak_mcmv:.2f}, fidelity {fidelity_mcmv:.2f}",
)
ax.plot(
    times * 1e3, rec_lcmv, color="C0",
    label=f"LCMV: peak {peak_lcmv:.2f}, fidelity {fidelity_lcmv:.2f}",
)
ax.axvline(0, color="k", lw=0.5)
ax.set(
    xlabel="time (ms)",
    ylabel="source amplitude (a.u.)",
    title=f"Recovery of the true source (peak 1.0) at $\\rho = {rho_demo}$",
)
ax.legend(loc="upper right")
