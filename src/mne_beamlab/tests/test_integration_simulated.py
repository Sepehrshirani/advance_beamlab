# Authors: the mne-beamlab contributors
# License: BSD-3-Clause
"""Integration tests on simulated data -- no download, runs in CI.

These exercise the full MCMV and ReciPSIICOS pipelines on a self-contained
forward: a standard EEG electrode montage on a spherical head model, both
bundled with MNE. Two temporally correlated bilateral dipoles are simulated --
the regime both algorithms target, and the one a plain LCMV mishandles.

Why EEG rather than a MEG sphere. A single-shell MEG sphere has a whitened
leadfield of rank ~2 per location, which collapses the ReciPSIICOS
virtual-sensor reduction and cannot exercise the projector/rank-curve/solve
path. An EEG leadfield is full rank per location, so the reduction yields a
healthy working space (q well above the three source orientations).

Why average reference. Re-referencing to the average makes the covariance
rank-deficient by one, so the whitener's null-space handling (auto-detected
rank) is exercised here too -- the same failure mode the real sample dataset
surfaces through its SSP projectors, reproduced without any download.

This module complements ``test_integration_sample.py``, which runs the same
paths on a real BEM forward but is skipped when the dataset is absent.
"""

import mne
import numpy as np
import pytest
from mne.beamformer import apply_lcmv_cov

from mne_beamlab import (
    apply_mcmv_cov,
    make_mcmv,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
)

_SPACING_MM = 25.0  # coarse whole-brain grid: fast in CI, still q >> n_orient


@pytest.fixture(scope="module")
def simulated():
    """EEG sphere forward + two correlated bilateral dipoles + covariances."""
    rng = np.random.default_rng(42)

    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = list(dict.fromkeys(montage.ch_names))
    info = mne.create_info(ch_names, sfreq=200.0, ch_types="eeg")
    info.set_montage(montage)

    sphere = mne.make_sphere_model("auto", "auto", info, verbose=False)
    src = mne.setup_volume_source_space(sphere=sphere, pos=_SPACING_MM, verbose=False)
    fwd = mne.make_forward_solution(
        info, None, src, sphere, eeg=True, meg=False, verbose=False
    )

    gain, rr, n_ch = fwd["sol"]["data"], fwd["source_rr"], len(ch_names)
    # one dipole in each hemisphere (extreme left/right vertices) with a fixed
    # orientation; unit-normalise the topographies so the two contribute equally
    left = int(np.argmin(rr[:, 0]))
    right = int(np.argmax(rr[:, 0]))
    ori = np.array([0.0, 0.0, 1.0])
    g_l = gain[:, 3 * left:3 * left + 3] @ ori
    g_r = gain[:, 3 * right:3 * right + 3] @ ori
    g_l /= np.linalg.norm(g_l)
    g_r /= np.linalg.norm(g_r)

    # 40 epochs; sources active only after t=0.1 s, noise throughout
    n_ep, n_t, sfreq, tmin = 40, 200, 200.0, -0.4
    times = np.arange(n_t) / sfreq + tmin
    active = times >= 0.1
    n_active = int(active.sum())

    data = 0.4 * rng.standard_normal((n_ep, n_ch, n_t))
    for e in range(n_ep):
        shared = rng.standard_normal(n_active)  # common driver -> correlation
        s_l = shared + 0.3 * rng.standard_normal(n_active)
        s_r = shared + 0.3 * rng.standard_normal(n_active)
        amp = 3.0 * (1.0 + 0.2 * rng.standard_normal())  # trial-to-trial gain
        data[e][:, active] += amp * (np.outer(g_l, s_l) + np.outer(g_r, s_r))

    epochs = mne.EpochsArray(
        data, info, tmin=tmin, baseline=(None, 0.0), verbose=False
    )
    epochs.set_eeg_reference("average", projection=True, verbose=False)
    data_cov = mne.compute_covariance(
        epochs, tmin=0.1, tmax=None, method="empirical", verbose=False
    )
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="empirical", verbose=False
    )
    return epochs.info, fwd, data_cov, noise_cov, (left, right, ori, rr)


def _hemisphere_balance(stc, rr):
    """Ratio of the weaker to the stronger hemisphere peak power."""
    lh = rr[:, 0] < 0
    power = stc.data[:, 0]
    lh_peak, rh_peak = power[lh].max(), power[~lh].max()
    return min(lh_peak, rh_peak) / max(lh_peak, rh_peak)


@pytest.mark.filterwarnings("ignore:The spectral-flip")
@pytest.mark.parametrize("method", ["recipsiicos", "whitened"])
def test_recipsiicos_recovers_correlated_bilateral(simulated, method):
    """Reduction stays well-posed and both correlated sources are recovered."""
    info, fwd, data_cov, noise_cov, (left, right, ori, rr) = simulated

    ranks, p_pwr, p_cor, kstar = recipsiicos_rank_curve(
        fwd, info, method=method, noise_cov=noise_cov, return_optimal=True
    )
    # the virtual-sensor reduction must not collapse: q^2 == len(ranks), and the
    # working space must exceed the three source orientations (a low-rank forward
    # or an unstable whitener would drive this to ~2 and make the solve singular)
    q = int(round(len(ranks) ** 0.5))
    assert q > 3, f"virtual-sensor reduction collapsed to q={q}"
    assert ranks[0] == 1 and ranks[-1] == len(ranks)
    assert np.all((p_pwr >= -1e-9) & (p_pwr <= 1 + 1e-9))
    assert np.all((p_cor >= -1e-9) & (p_cor <= 1 + 1e-9))
    assert 1 <= kstar <= len(ranks)

    filters = make_recipsiicos_lcmv(
        info,
        fwd,
        data_cov,
        rank=kstar,
        method=method,
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
    )
    stc = apply_lcmv_cov(data_cov, filters)
    assert np.all(np.isfinite(stc.data))
    # a whole-brain forward keeps the cleaned covariance PSD -> non-negative power
    assert stc.data.min() > -1e-6 * stc.data.max()
    assert stc.data.max() > 0
    # the correlated pair leaves substantial power in both hemispheres
    assert _hemisphere_balance(stc, rr) > 0.3


def test_mcmv_on_simulated_pair(simulated):
    """MCMV builds a finite joint filter on the two known correlated sources."""
    info, fwd, data_cov, noise_cov, (left, right, ori, rr) = simulated
    mcmv = make_mcmv(
        info,
        fwd,
        data_cov,
        sources=[left, right],
        orientations=np.vstack([ori, ori]),
        noise_cov=noise_cov,
    )
    assert np.all(np.isfinite(mcmv["weights"]))
    src_cov = apply_mcmv_cov(data_cov, mcmv)
    assert src_cov.shape == (2, 2)
    assert np.all(np.isfinite(src_cov))
    assert src_cov[0, 0] > 0 and src_cov[1, 1] > 0  # positive source powers
