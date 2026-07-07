"""
ABMC vs LCMV across the source grid: localisation accuracy
==========================================================

ABMC targets low-power, spike-like sources -- epileptic interictal discharges and
delayed responses to stimulation -- where an ordinary power-based LCMV beamformer
is easily pulled off target by noise. This example places the *same* low-SNR spike
at several grid locations across the volume and, at each, compares how close ABMC
(:func:`~advance_beamlab.make_abmc`, template match) and a power-based LCMV map get
to the true source. Repeating over locations shows the ABMC advantage is
consistent, not a quirk of one spot.

It then reconstructs the source time course at one location with both ABMC and a
unit-gain LCMV filter, and closes with the multi-template API
(:func:`~advance_beamlab.make_abmc_dictionary`), which localises a whole dictionary
of desired waveforms in one call while estimating the sparse Bayesian covariance
only once.
"""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import mne
import numpy as np

from advance_beamlab import make_abmc, make_abmc_dictionary

mne.set_log_level("ERROR")


def spike(n, t0, width=8.0):
    """A unit-amplitude biphasic (derivative-of-Gaussian) spike."""
    t = np.arange(n)
    x = -(t - t0) / width * np.exp(-((t - t0) ** 2) / (2 * width**2))
    return x / np.abs(x).max()


def lcmv_power_map(data, info, leadfield, reg=0.05):
    """A simple power-based LCMV localiser map (one value per grid point)."""
    data_cov = mne.compute_covariance(
        mne.make_fixed_length_epochs(mne.io.RawArray(data, info), duration=1.0)
    )
    cov = data_cov.data
    n_ch = cov.shape[0]
    inv = np.linalg.inv(cov + reg * np.trace(cov) / n_ch * np.eye(n_ch))
    return 1.0 / np.einsum("mk,mn,nk->k", leadfield, inv, leadfield)


# %%
# A spherical EEG forward model (fixed orientation -> scalar leadfield).
montage = mne.channels.make_standard_montage("standard_1020")
ch = list(dict.fromkeys(montage.ch_names))
info = mne.create_info(ch, 250.0, "eeg")
info.set_montage(montage)
sphere = mne.make_sphere_model("auto", "auto", info)
src = mne.setup_volume_source_space(sphere=sphere, pos=20.0)
fwd = mne.convert_forward_solution(
    mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False),
    force_fixed=True,
    use_cps=False,
)
leadfield = fwd["sol"]["data"]
rr = fwd["source_rr"]
n_ch = leadfield.shape[0]

# %%
# Pick several true source locations spread across the (localisable) volume, and
# at each simulate the same biphasic spike buried in sensor noise. The template is
# the same morphology, shifted, so the beamformer must also recover the lag.
n_times = 400
depth = np.linalg.norm(rr - rr.mean(0), axis=1)
shell = np.where(depth > np.percentile(depth, 55))[0]
rng = np.random.default_rng(0)
locations = np.sort(rng.choice(shell, size=8, replace=False))
template = spike(n_times, 180)

abmc_err, lcmv_err = [], []
for i_src in locations:
    clean = np.outer(leadfield[:, i_src], spike(n_times, 230))
    data = clean + 1.3 * np.abs(clean).max() * rng.standard_normal((n_ch, n_times))
    res = make_abmc(info, fwd, data, template)
    abmc_peak = int(np.argmax(res.template_match))
    abmc_err.append(np.linalg.norm(rr[abmc_peak] - rr[i_src]) * 100)
    lcmv_map = lcmv_power_map(data, info, leadfield)
    lcmv_peak = int(np.argmax(lcmv_map))
    lcmv_err.append(np.linalg.norm(rr[lcmv_peak] - rr[i_src]) * 100)

print(f"mean error  ABMC {np.mean(abmc_err):.1f} cm   LCMV {np.mean(lcmv_err):.1f} cm")

# %%
# ABMC lands close to every true source; the power-based LCMV map wanders under
# the same noise.
fig, ax = plt.subplots(figsize=(9, 4))
xp = np.arange(len(locations))
ax.bar(xp - 0.2, abmc_err, 0.4, color="tab:blue",
       label=f"ABMC (mean {np.mean(abmc_err):.1f} cm)")
ax.bar(xp + 0.2, lcmv_err, 0.4, color="tab:orange",
       label=f"LCMV (mean {np.mean(lcmv_err):.1f} cm)")
ax.set_xticks(xp)
ax.set_xticklabels([str(int(i)) for i in locations])
ax.set(xlabel="true source (grid index)", ylabel="localisation error (cm)",
       title="ABMC vs LCMV localisation error across source locations")
ax.legend()
fig.tight_layout()

# %%
# Reconstructed source time course: ABMC vs LCMV. At the true source, compare the
# beamformer output waveforms against the clean spike. Both are unit-gain at the
# source, so both recover the morphology; ABMC's template constraint yields a
# modestly cleaner trace. ABMC's decisive advantage is in *localisation* (the plot
# above) -- the larger waveform gains reported in the paper come with realistic
# iEEG noise rather than this idealised sphere model.
i_demo = int(locations[0])
clean = np.outer(leadfield[:, i_demo], spike(n_times, 230))
data = clean + 1.3 * np.abs(clean).max() * rng.standard_normal((n_ch, n_times))

res = make_abmc(info, fwd, data, template, return_weights=True)
abmc_out = res.weights[:, i_demo] @ data

data_cov = mne.compute_covariance(
    mne.make_fixed_length_epochs(mne.io.RawArray(data, info), duration=1.0)
)
cov = data_cov.data
r_reg = cov + 0.05 * np.trace(cov) / n_ch * np.eye(n_ch)
g = leadfield[:, i_demo]
rinv_g = np.linalg.solve(r_reg, g)
w_lcmv = rinv_g / (g @ rinv_g)
lcmv_out = w_lcmv @ data

cs = spike(n_times, 230)
r_abmc = np.corrcoef(abmc_out, cs)[0, 1]
r_lcmv = np.corrcoef(lcmv_out, cs)[0, 1]
t_ms = np.arange(n_times) / info["sfreq"] * 1000
fig2, ax2 = plt.subplots(figsize=(9, 3.5))
ax2.plot(t_ms, cs, color="0.6", lw=2.5, label="true source")
ax2.plot(t_ms, abmc_out, color="tab:blue", label=f"ABMC (r={r_abmc:.2f})")
ax2.plot(t_ms, lcmv_out, color="tab:orange", alpha=0.8, label=f"LCMV (r={r_lcmv:.2f})")
ax2.set(xlabel="time (ms)", ylabel="source amplitude (a.u.)",
        title="Reconstructed source at the true location (r: correlation with truth)")
ax2.legend(loc="upper right")
fig2.tight_layout()

# %%
# The multi-template API: localise a dictionary of desired waveforms in one call.
# The sparse Bayesian covariance is estimated once and reused for every template.
i_src = i_demo  # reuse the reconstructed-source segment above
templates = {
    "early": spike(n_times, 180),
    "on-time": spike(n_times, 230),
    "late": spike(n_times, 280),
}
results = make_abmc_dictionary(info, fwd, data, templates)
for name, r in results.items():
    peak = int(np.argmax(r.template_match))
    err = np.linalg.norm(rr[peak] - rr[i_src]) * 100
    print(f"{name:>8}: error {err:.1f} cm, lag {int(r.lag[peak])} samples")
