"""
ABMC: localising a low-SNR spike where power-based LCMV struggles
=================================================================

The adaptive Bayesian beamformer with multiple constraints (ABMC) targets
low-power, spike-like sources -- epileptic interictal discharges and delayed
responses to stimulation -- where an ordinary power-based LCMV beamformer is
easily pulled off target by noise. This example simulates one such spike on a
spherical EEG model and localises it with :func:`~advance_beamlab.make_abmc`,
comparing against an LCMV power map on the same data.

The two ingredients are (1) a sparse Bayesian learning covariance
(:func:`~advance_beamlab.sbl_covariance`, estimated internally here) that models
sources as uncorrelated, and (2) a template constraint that locks the beamformer
output onto the known spike morphology at the correct lag. The source is read
out as the location whose output best matches the template.
"""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import mne
import numpy as np

from advance_beamlab import make_abmc

mne.set_log_level("ERROR")


def spike(n, t0, width=8.0):
    """A unit-amplitude biphasic (derivative-of-Gaussian) spike."""
    t = np.arange(n)
    x = -(t - t0) / width * np.exp(-((t - t0) ** 2) / (2 * width**2))
    return x / np.abs(x).max()


# %%
# A spherical EEG forward model (fixed orientation -> scalar leadfield).
montage = mne.channels.make_standard_montage("standard_1020")
ch = list(dict.fromkeys(montage.ch_names))
info = mne.create_info(ch, 250.0, "eeg")
info.set_montage(montage)
sphere = mne.make_sphere_model("auto", "auto", info)
src = mne.setup_volume_source_space(sphere=sphere, pos=15.0)
fwd = mne.convert_forward_solution(
    mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False),
    force_fixed=True,
    use_cps=False,
)
leadfield = fwd["sol"]["data"]
rr = fwd["source_rr"]
n_ch = leadfield.shape[0]

# %%
# Simulate a low-SNR spike at a superficial grid point, buried in sensor noise.
# The ``template`` is the desired-source morphology the caller supplies -- here a
# synthetic spike, in practice an expert-annotated IED or DR -- shifted in time so
# the beamformer must also recover the lag.
rng = np.random.default_rng(0)
n_times = 500
depth = np.linalg.norm(rr - rr.mean(0), axis=1)
i_src = int(np.where(depth > np.percentile(depth, 80))[0][0])
clean = np.outer(leadfield[:, i_src], spike(n_times, 250))
data = clean + 1.5 * np.abs(clean).max() * rng.standard_normal((n_ch, n_times))
template = spike(n_times, 200)  # data spike at 250 -> true lag +50

# %%
# Localise with ABMC. With ``cov=None`` the SBL covariance is estimated from the
# data internally (the intended pipeline).
result = make_abmc(info, fwd, data, template, return_weights=True)
abmc_map = result.template_match
peak = int(np.argmax(abmc_map))
print(f"ABMC localisation error: {np.linalg.norm(rr[peak] - rr[i_src]) * 100:.1f} cm")
print(f"recovered lag: {int(result.lag[peak])} samples (true 50)")

# %%
# A power-based LCMV map on the same data, for comparison.
data_cov = mne.compute_covariance(
    mne.make_fixed_length_epochs(mne.io.RawArray(data, info), duration=1.0)
)
cov = data_cov.data
reg = cov + 0.05 * np.trace(cov) / n_ch * np.eye(n_ch)
lcmv_map = 1.0 / np.einsum("mk,mn,nk->k", leadfield, np.linalg.inv(reg), leadfield)
lcmv_peak = int(np.argmax(lcmv_map))


def error_cm(idx):
    return np.linalg.norm(rr[idx] - rr[i_src]) * 100


# %%
# ABMC's template-match map peaks at the true source; the LCMV power map, driven
# by noise power, peaks elsewhere.
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
panels = [
    (abmc_map, "ABMC (template match)", peak),
    (lcmv_map / lcmv_map.max(), "LCMV (output power)", lcmv_peak),
]
for ax, (mp, name, pk) in zip(axes, panels, strict=True):
    ax.plot(mp, lw=0.8, color="0.6")
    ax.axvline(i_src, color="tab:green", ls="--", label="true source")
    ax.plot(pk, mp[pk], "v", color="tab:red", ms=10,
            label=f"peak ({error_cm(pk):.1f} cm)")
    ax.set_ylabel("normalised value")
    ax.set_title(name)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("grid point")
fig.tight_layout()

# %%
# At the ABMC peak the beamformer output tracks the (lag-aligned) template -- the
# quantity the localiser maximises.
w_peak = result.weights[:, peak]
output = w_peak @ data
lag = int(result.lag[peak])
u_shift = np.zeros_like(template)
if lag >= 0:
    u_shift[lag:] = template[: n_times - lag]
else:
    u_shift[: n_times + lag] = template[-lag:]

t_ms = np.arange(n_times) / info["sfreq"] * 1000
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(t_ms, output / np.abs(output).max(), color="tab:blue", label="ABMC output")
ax2.plot(t_ms, u_shift, color="tab:orange", ls="--", label="template (aligned)")
ax2.set(xlabel="time (ms)", ylabel="normalised amplitude",
        title="Beamformer output at the ABMC peak vs. the template")
ax2.legend(loc="upper right")
fig2.tight_layout()
