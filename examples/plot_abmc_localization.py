"""
ABMC: localising spike-like sources
===================================

The adaptive Bayesian beamformer with multiple constraints (ABMC) targets
low-power, spike-like sources -- epileptic interictal discharges and delayed
responses to stimulation -- that a power-based LCMV beamformer localises poorly.

This example shows four things on a spherical EEG model:

1. **Localisation maps** at a single source -- ABMC's template-match map versus the
   LCMV power map across the whole grid.
2. **Accuracy across the volume** -- the same spike placed at eight locations, with
   the ABMC and LCMV localisation error at each.
3. **Reconstructed source** -- the recovered time course from both filters.
4. **A dictionary of templates** -- :func:`~advance_beamlab.make_abmc_dictionary`
   localising several desired waveforms in one call.
"""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import mne
import numpy as np

from advance_beamlab import make_abmc, make_abmc_dictionary

mne.set_log_level("ERROR")

# Journal-style figure defaults (high-DPI, clean spines, distinctive palette).
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#9e9e9e",
        "grid.alpha": 0.28,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,
    }
)
C_ABMC, C_LCMV, C_TRUE = "#0072B2", "#D55E00", "#111111"  # Wong colourblind-safe


def spike(n, t0, width=8.0):
    """A unit-amplitude biphasic (derivative-of-Gaussian) spike."""
    t = np.arange(n)
    x = -(t - t0) / width * np.exp(-((t - t0) ** 2) / (2 * width**2))
    return x / np.abs(x).max()


def lcmv_power_map(data, info, leadfield, reg=0.05):
    """A power-based LCMV localiser map (one value per grid point)."""
    cov = data @ data.T / data.shape[1]
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
n_times = 400

# %%
# Place the same biphasic spike at eight grid locations spread across the volume,
# and localise each with ABMC (template match) and a power-based LCMV map. The
# template is the same morphology, shifted, so the lag must also be recovered. We
# keep the maps from the source where LCMV struggles most, to visualise below.
depth = np.linalg.norm(rr - rr.mean(0), axis=1)
shell = np.where(depth > np.percentile(depth, 55))[0]
rng = np.random.default_rng(0)
locations = np.sort(rng.choice(shell, size=8, replace=False))
template = spike(n_times, 180)

abmc_err, lcmv_err, worst = [], [], None
for i_src in locations:
    clean = np.outer(leadfield[:, i_src], spike(n_times, 230))
    data = clean + 1.3 * np.abs(clean).max() * rng.standard_normal((n_ch, n_times))
    res = make_abmc(info, fwd, data, template)
    a_map, a_pk = res.template_match, int(np.argmax(res.template_match))
    a_err = np.linalg.norm(rr[a_pk] - rr[i_src]) * 100
    l_map = lcmv_power_map(data, info, leadfield)
    l_pk = int(np.argmax(l_map))
    l_err = np.linalg.norm(rr[l_pk] - rr[i_src]) * 100
    abmc_err.append(a_err)
    lcmv_err.append(l_err)
    if worst is None or l_err > worst["l_err"]:
        worst = dict(
            i_src=int(i_src),
            data=data,
            a_map=a_map,
            l_map=l_map,
            a_pk=a_pk,
            l_pk=l_pk,
            l_err=l_err,
        )

print(f"mean error  ABMC {np.mean(abmc_err):.1f} cm   LCMV {np.mean(lcmv_err):.1f} cm")

# %%
# **Localisation maps** at the hardest source. ABMC's template-match map peaks
# sharply at the true location; the LCMV power map, driven by noise, peaks away.
i_src = worst["i_src"]
gi = np.arange(leadfield.shape[1])
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
panels = [
    ("ABMC template-match map", worst["a_map"], worst["a_pk"], C_ABMC),
    ("LCMV output-power map", worst["l_map"], worst["l_pk"], C_LCMV),
]
for ax, (name, m, pk, col) in zip(axes, panels, strict=True):
    mm = m / m.max()
    ax.fill_between(gi, mm, color=col, alpha=0.22, lw=0)
    ax.plot(gi, mm, color=col, lw=1.1)
    ax.axvline(i_src, color=C_TRUE, ls=(0, (5, 3)), lw=1.6, label="true source")
    err = np.linalg.norm(rr[pk] - rr[i_src]) * 100
    ax.plot(
        pk,
        mm[pk],
        "v",
        color=col,
        ms=12,
        mec="white",
        mew=1.1,
        label=f"peak ({err:.1f} cm off)",
    )
    ax.set(ylabel="normalised", ylim=(0, 1.08))
    ax.set_title(name, loc="left")
    ax.legend(loc="upper right", ncol=2)
    ax.margins(x=0.01)
    ax.grid(axis="x", visible=False)
axes[-1].set_xlabel("source grid index")
fig.tight_layout()

# %%
# **Accuracy across the volume.** ABMC stays close everywhere; LCMV degrades badly
# at the harder (often deeper) sources. Dashed lines mark the means.
xp = np.arange(len(locations))
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axhline(np.mean(abmc_err), color=C_ABMC, ls="--", lw=1.1, alpha=0.7)
ax.axhline(np.mean(lcmv_err), color=C_LCMV, ls="--", lw=1.1, alpha=0.7)
ax.bar(
    xp - 0.21,
    abmc_err,
    0.42,
    color=C_ABMC,
    zorder=3,
    label=f"ABMC (mean {np.mean(abmc_err):.1f} cm)",
)
ax.bar(
    xp + 0.21,
    lcmv_err,
    0.42,
    color=C_LCMV,
    zorder=3,
    label=f"LCMV (mean {np.mean(lcmv_err):.1f} cm)",
)
ax.set_xticks(xp)
ax.set_xticklabels([str(int(i)) for i in locations])
ax.set(xlabel="true source (grid index)", ylabel="localisation error (cm)")
ax.set_title("ABMC vs LCMV localisation accuracy", loc="left")
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
fig.tight_layout()

# %%
# **Reconstructed source** at that location. Both filters are unit-gain at the
# source, so both recover the spike; ABMC's template constraint yields a modestly
# cleaner trace (``r`` = correlation with the noiseless source). ABMC's decisive
# edge is in *localisation* above -- the larger waveform gains in the paper come
# with realistic iEEG noise, not this idealised sphere.
data = worst["data"]
res = make_abmc(info, fwd, data, template, return_weights=True)
abmc_out = res.weights[:, i_src] @ data
cov = data @ data.T / n_times
r_reg = cov + 0.05 * np.trace(cov) / n_ch * np.eye(n_ch)
g = leadfield[:, i_src]
rinv_g = np.linalg.solve(r_reg, g)
lcmv_out = (rinv_g / (g @ rinv_g)) @ data
cs = spike(n_times, 230)
t_ms = np.arange(n_times) / info["sfreq"] * 1000
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(t_ms, cs, color=C_TRUE, lw=2.6, alpha=0.5, label="true source")
ax.plot(
    t_ms,
    abmc_out,
    color=C_ABMC,
    label=f"ABMC (r={np.corrcoef(abmc_out, cs)[0, 1]:.2f})",
)
ax.plot(
    t_ms,
    lcmv_out,
    color=C_LCMV,
    alpha=0.85,
    label=f"LCMV (r={np.corrcoef(lcmv_out, cs)[0, 1]:.2f})",
)
ax.set(xlabel="time (ms)", ylabel="source amplitude (a.u.)")
ax.set_title("Reconstructed source at the true location", loc="left")
ax.legend(loc="upper right", ncol=3)
fig.tight_layout()

# %%
# **A dictionary of templates.** ``make_abmc_dictionary`` localises several desired
# waveforms in one call, estimating the sparse Bayesian covariance only once.
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
