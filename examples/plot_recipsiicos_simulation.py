r"""
.. _ex-recipsiicos-simulation:

===========================================================
ReciPSIICOS versus LCMV as source correlation is dialled up
===========================================================

An LCMV beamformer minimises output power under a unit-gain constraint. When two
sources are correlated that is a trap: the filter can lower its own output power
by passing a scaled copy of the *other* source, which cancels the target it was
supposed to pass. The recovered amplitude collapses as the correlation rises.

ReciPSIICOS :footcite:`KuznetsovaEtAl2021` attacks this before the beamformer is
built. From the forward model alone it constructs a projector that suppresses the
cross-product (coupling) part of the sensor covariance while sparing the
auto-product (power) part, and applies it to the data covariance. The result
approximates the covariance that would have been measured had the same sources
been *uncorrelated* -- so an ordinary LCMV built on it has nothing to cancel.

This example dials the correlation from 0 to 0.99 and measures what each method
recovers, then shows how to choose the method's single free parameter, the
projection rank.

The head model matters here and is worth stating up front. The sources are
simulated, but they are simulated on the **real BEM forward** of the MNE
``sample`` dataset rather than on a sphere. ReciPSIICOS builds its projector from
the forward, and a single-shell sphere has such low-rank tangential leadfields
that the power and correlation subspaces barely separate -- the rank curve
degenerates and the automatic rank criterion has nothing to lock onto. On a
realistic forward it behaves as the paper describes.

See :ref:`ex-mcmv-simulation` for the same cancellation attacked from the other
direction, by constraining the sources jointly, and
:ref:`ex-recipsiicos-auditory` for this method run end to end on real recorded
data.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import Label
from mne.beamformer import apply_lcmv, make_lcmv
from mne.forward import restrict_forward_to_label

from advance_beamlab import (
    make_recipsiicos_cov,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
)

mne.set_log_level("ERROR")

# %%
# The real gradiometer forward of the ``sample`` dataset, decimated to a coarse
# whole-brain grid. The grid stays whole-brain on purpose: the projector must
# span where the covariance's energy actually lives, so restricting it to a
# region would be wrong. Decimation only keeps the correlation Gram, which is
# quadratic in the number of sources, tractable.

data_path = mne.datasets.sample.data_path()
meg = data_path / "MEG" / "sample"

info = mne.io.read_info(meg / "sample_audvis_filt-0-40_raw.fif")
info = mne.pick_info(info, mne.pick_types(info, meg="grad", eeg=False, exclude="bads"))

fwd = mne.read_forward_solution(meg / "sample_audvis-meg-eeg-oct-6-fwd.fif")
fwd = mne.pick_types_forward(fwd, meg="grad", eeg=False)
fwd = restrict_forward_to_label(
    fwd,
    [
        Label(fwd["src"][h]["vertno"][::28], hemi=hemi, subject="sample")
        for h, hemi in enumerate(("lh", "rh"))
    ],
)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True)

common = [ch for ch in info["ch_names"] if ch in fwd["sol"]["row_names"]]
info = mne.pick_info(info, [info["ch_names"].index(ch) for ch in common])
gain = np.asarray(fwd["sol"]["data"])[
    [fwd["sol"]["row_names"].index(ch) for ch in common]
]
source_rr = fwd["source_rr"]
print(f"{fwd['nsource']} sources, {gain.shape[0]} gradiometers")

# %%
# One source per hemisphere, about 12 cm apart -- the bilateral configuration
# where correlated-source cancellation is the classic failure.

n_lh = len(fwd["src"][0]["vertno"])
left = int(np.argmin(source_rr[:n_lh, 0]))
right = n_lh + int(np.argmax(source_rr[n_lh:, 0]))
sources = [left, right]
sep_cm = np.linalg.norm(source_rr[left] - source_rr[right]) * 100
print(f"source separation: {sep_cm:.1f} cm")

# %%
# **Choosing the projection rank.** The rank is the method's only free parameter.
# ``recipsiicos_rank_curve`` reports, for every rank, how much of the source-power
# subspace and how much of the correlation subspace survive the projection. The
# useful rank keeps the power while depleting the correlation; the 45-degree point
# of that trade-off is what ``return_optimal=True`` returns.

ranks, p_pwr, p_cor, k_opt = recipsiicos_rank_curve(
    fwd, info, method="whitened", return_optimal=True
)
print(
    f"K* = {k_opt}: retains {p_pwr[k_opt - 1]:.1%} of the power subspace "
    f"and {p_cor[k_opt - 1]:.1%} of the correlation subspace"
)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(ranks, p_pwr, color="C3", label="power subspace retained")
ax.plot(ranks, p_cor, color="C0", label="correlation subspace retained")
ax.axvline(k_opt, color="k", ls="--", lw=1.2, label=f"$K^*$ = {k_opt}")
ax.set(xlabel="projection rank $K$", ylabel="retained energy fraction", ylim=(0, 1.05))
ax.set_title("The rank trade-off, and the rank it selects", loc="left")
ax.legend(loc="center right")
fig.tight_layout()

# %%
# Simulate the pair at a given correlation and measure what each beamformer
# recovers. Both read out with **unit gain**, so the recovered amplitude is
# directly comparable with the injected one -- a noise-normalised readout would
# rescale the axis and hide exactly the effect being measured.

rng = np.random.default_rng(0)
n_epochs, n_times = 60, 160
times = np.arange(n_times) / info["sfreq"] - 0.2
active = times >= 0.0
REG = 0.01  # light regularisation: loading of its own masks the cancellation


def recovered(rho):
    """Return (LCMV, ReciPSIICOS) recovered RMS amplitude, as a fraction of truth."""
    phi = np.arccos(rho)
    s = np.zeros((2, n_times))
    s[0, active] = np.cos(2 * np.pi * 10 * times[active])
    s[1, active] = np.cos(2 * np.pi * 10 * times[active] + phi)
    amp = 2e-11 / np.abs(gain[:, sources]).max()
    signal = amp * (gain[:, sources] @ s)
    data = np.stack(
        [
            signal + 0.02 * 2e-11 * rng.standard_normal(signal.shape)
            for _ in range(n_epochs)
        ]
    )
    epochs = mne.EpochsArray(
        data, info.copy(), tmin=times[0], baseline=None, verbose=False
    )
    data_cov = mne.compute_covariance(epochs, tmin=0.0, method="shrunk", verbose=False)
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="shrunk", verbose=False
    )
    evoked = epochs.average()
    truth = amp / np.sqrt(2)  # RMS of a unit-amplitude cosine

    lcmv = make_lcmv(
        info,
        fwd,
        data_cov,
        reg=REG,
        noise_cov=noise_cov,
        pick_ori=None,
        weight_norm=None,
    )
    recip = make_recipsiicos_lcmv(
        info,
        fwd,
        data_cov,
        rank=k_opt,
        method="whitened",
        noise_cov=noise_cov,
        weight_norm=None,
        reg=REG,
    )
    out = []
    for filters in (lcmv, recip):
        tc = apply_lcmv(evoked, filters).data[sources][:, active]
        out.append(float(np.mean(np.sqrt((tc**2).mean(axis=1)) / truth)))
    return out[0], out[1], data_cov, noise_cov


rhos = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
lcmv_amp, recip_amp = [], []
for rho in rhos:
    a, b, data_cov, noise_cov = recovered(rho)
    lcmv_amp.append(a)
    recip_amp.append(b)
    print(f"rho = {rho:4.2f}:  LCMV {a:.3f}   ReciPSIICOS {b:.3f}   ({b / a:.2f}x)")

# %%
# **The headline result.** LCMV loses most of the amplitude by
# :math:`\rho = 0.9` and nearly all of it by :math:`\rho = 0.99`, tracking the
# analytic :math:`\sqrt{1-\rho^2}` law. ReciPSIICOS stays flat at the injected
# amplitude across the whole range, because the covariance it is given no longer
# contains the cross-term the filter would have exploited.

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(
    rhos,
    np.sqrt(1 - rhos**2),
    color="#111111",
    ls=":",
    lw=1.6,
    label=r"$\sqrt{1-\rho^2}$",
)
ax.plot(rhos, lcmv_amp, "o-", color="C0", ms=7, label="LCMV")
ax.plot(rhos, recip_amp, "o-", color="C3", ms=7, label="ReciPSIICOS")
ax.axhline(1.0, color="k", lw=0.6)
ax.set(
    xlabel=r"source correlation $\rho$",
    ylabel="recovered amplitude / true amplitude",
    ylim=(0, 1.2),
)
ax.set_title("ReciPSIICOS holds the amplitude that LCMV cancels away", loc="left")
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# **How sensitive is this to the rank?** Much less than one might fear, which is
# the useful practical finding. Sweeping the rank over more than an order of
# magnitude around :math:`K^*` -- from :math:`K^*/8` to :math:`4K^*` -- moves the
# recovered amplitude only between about 0.96 and 1.08, while LCMV on the same
# data sits at 0.31. On this configuration the choice of rank is not what
# separates the two methods; applying the projection at all is.
#
# That is not a licence to ignore the parameter. The bound at the top end is
# real: once the rank reaches the rank of the power Gram the projector annihilates
# the covariance entirely, and the implementation warns when a requested rank
# crosses it. The point is that between those limits there is a broad plateau
# rather than a knife edge. The sweep below is measured at :math:`\rho = 0.95`.

rho_fixed = 0.95
phi = np.arccos(rho_fixed)
s = np.zeros((2, n_times))
s[0, active] = np.cos(2 * np.pi * 10 * times[active])
s[1, active] = np.cos(2 * np.pi * 10 * times[active] + phi)
amp = 2e-11 / np.abs(gain[:, sources]).max()
signal = amp * (gain[:, sources] @ s)
data = np.stack(
    [signal + 0.02 * 2e-11 * rng.standard_normal(signal.shape) for _ in range(n_epochs)]
)
epochs = mne.EpochsArray(data, info.copy(), tmin=times[0], baseline=None, verbose=False)
data_cov = mne.compute_covariance(epochs, tmin=0.0, method="shrunk", verbose=False)
noise_cov = mne.compute_covariance(
    epochs, tmin=None, tmax=0.0, method="shrunk", verbose=False
)
evoked = epochs.average()
truth = amp / np.sqrt(2)

test_ranks = sorted({max(1, k_opt // 8), k_opt // 3, k_opt, 2 * k_opt, 4 * k_opt})
rank_amp = []
for rk in test_ranks:
    filters = make_recipsiicos_lcmv(
        info,
        fwd,
        data_cov,
        rank=int(rk),
        method="whitened",
        noise_cov=noise_cov,
        weight_norm=None,
        reg=REG,
    )
    tc = apply_lcmv(evoked, filters).data[sources][:, active]
    rank_amp.append(float(np.mean(np.sqrt((tc**2).mean(axis=1)) / truth)))
    print(f"rank {int(rk):4d}: recovered {rank_amp[-1]:.3f}")

lcmv_ref = lcmv_amp[list(rhos).index(0.95)]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.semilogx(test_ranks, rank_amp, "o-", color="C3", ms=7, label="ReciPSIICOS")
ax.axhline(lcmv_ref, color="C0", ls="--", lw=1.2, label="LCMV")
ax.axvline(k_opt, color="k", ls="--", lw=1.2, label=f"$K^*$ = {k_opt}")
ax.set(xlabel="projection rank $K$", ylabel="recovered amplitude / true", ylim=(0, 1.2))
ax.set_title(rf"A broad plateau, not a knife edge ($\rho$ = {rho_fixed})", loc="left")
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# **The modified covariance itself.** :func:`~advance_beamlab.make_recipsiicos_cov`
# returns the cleaned sensor covariance directly, which is useful for inspection
# and for interoperating with other tools. It is the same object
# ``make_recipsiicos_lcmv`` builds internally, mapped back to sensor space.

clean_cov = make_recipsiicos_cov(
    data_cov, fwd, info, rank=k_opt, method="whitened", noise_cov=noise_cov
)
ev_raw = np.linalg.eigvalsh(data_cov.data)[::-1]
ev_clean = np.linalg.eigvalsh(clean_cov.data)[::-1]
keep = int((ev_clean > 1e-12 * ev_clean.max()).sum())
print(f"cleaned covariance rank: {keep} of {len(ev_clean)} sensor dimensions")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.semilogy(ev_raw / ev_raw.max(), color="C0", label="data covariance")
ax.semilogy(
    np.clip(ev_clean / ev_clean.max(), 1e-18, None), color="C3", label="ReciPSIICOS"
)
ax.set(xlabel="eigenvalue index", ylabel="normalised eigenvalue")
ax.set_title(
    "The projection lives in the reduced virtual-sensor space, so the "
    "cleaned covariance is low rank",
    loc="left",
    fontsize=10,
)
ax.legend(loc="lower left")
fig.tight_layout()

# sphinx_gallery_thumbnail_number = 2
