"""Tests for the Multiple Constrained Minimum Variance (MCMV) beamformer."""

# Authors: Sepehr Shirani

import mne
import numpy as np
import pytest
from mne.beamformer import make_lcmv
from mne.beamformer._compute_beamformer import _reg_pinv
from numpy.testing import assert_allclose

from mne_beamlab import (
    MCMVBeamformer,
    apply_mcmv,
    apply_mcmv_cov,
    make_mcmv,
)
from mne_beamlab._mcmv import _compute_mcmv_weights

mne.set_log_level("ERROR")
REG = 0.05


# --------------------------------------------------------------------------- #
# Fixtures: a tiny, fully analytic (offline) EEG forward solution.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fwd_info():
    """A small free-orientation EEG forward on a sphere + its Info."""
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:32], 200.0, "eeg")
    info.set_montage("standard_1020")
    sphere = mne.make_sphere_model("auto", "auto", info)
    src = mne.setup_volume_source_space(sphere=sphere, pos=30.0)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=False, surf_ori=False)
    return fwd, info


def _random_cov(info, seed=0):
    """A full-rank random SPD covariance as an mne.Covariance."""
    rng = np.random.default_rng(seed)
    n = len(info["ch_names"])
    A = rng.standard_normal((n, n))
    R = A @ A.T + np.eye(n)
    return mne.Covariance(
        R, info["ch_names"], info["bads"], list(info["projs"]), nfree=1
    )


# --------------------------------------------------------------------------- #
# 1. Pure-math properties of the core solve.
# --------------------------------------------------------------------------- #
def test_unit_gain_constraint():
    """W^T H = I_n exactly (Moiseev Eq. (3))."""
    rng = np.random.default_rng(1)
    n_ch, n = 40, 3
    H = rng.standard_normal((n_ch, n))
    A = rng.standard_normal((n_ch, n_ch))
    Rinv = A @ A.T + np.eye(n_ch)
    W = _compute_mcmv_weights(H, Rinv)
    assert W.shape == (n, n_ch)
    assert_allclose(W @ H, np.eye(n), atol=1e-10)


def test_scalar_reduces_to_lcmv_closed_form():
    """For n == 1 the MCMV weight equals the unit-gain LCMV weight."""
    rng = np.random.default_rng(2)
    n_ch = 25
    h = rng.standard_normal((n_ch, 1))
    A = rng.standard_normal((n_ch, n_ch))
    Rinv = A @ A.T + np.eye(n_ch)
    W = _compute_mcmv_weights(h, Rinv)
    w_lcmv = (Rinv @ h) / (h.T @ Rinv @ h)  # R^-1 h / (h^T R^-1 h)
    assert_allclose(W.ravel(), w_lcmv.ravel(), atol=1e-12)


def test_singular_system_raises():
    """Collinear constrained sources -> singular H^T R^-1 H -> RuntimeError."""
    rng = np.random.default_rng(3)
    n_ch = 20
    h = rng.standard_normal((n_ch, 1))
    H = np.hstack([h, h])  # identical columns
    Rinv = np.eye(n_ch)
    with pytest.raises(RuntimeError, match="singular"):
        _compute_mcmv_weights(H, Rinv)


# --------------------------------------------------------------------------- #
# 2. Numerical equivalence with MNE's own beamformer.
# --------------------------------------------------------------------------- #
def test_core_matches_mne_vector_lcmv(fwd_info):
    """MCMV with a location's 3 orthogonal orientations == MNE vector LCMV.

    MNE's vector (free-orientation) LCMV at one location solves exactly the
    3-constraint MCMV system, so this validates the multi-constraint solve and
    the covariance handling against MNE to machine precision.
    """
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    R = np.asarray(data_cov.data)
    filt = make_lcmv(
        info, fwd, data_cov, reg=REG, noise_cov=None,
        pick_ori="vector", weight_norm=None,
    )
    W_lcmv = filt["weights"]  # (n_src * 3, n_ch)

    G = fwd["sol"]["data"]
    Rinv, _, _ = _reg_pinv(R, reg=REG, rank="full")
    for loc in (0, 5, 11):
        L = G[:, 3 * loc : 3 * loc + 3]
        W_mcmv = _compute_mcmv_weights(L, Rinv)  # (3, n_ch)
        assert_allclose(W_mcmv, W_lcmv[3 * loc : 3 * loc + 3], atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. End-to-end public API.
# --------------------------------------------------------------------------- #
def test_make_apply_mcmv_end_to_end(fwd_info):
    """make_mcmv / apply_mcmv / apply_mcmv_cov on a free-orientation forward."""
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    sources = [2, 8]
    oris = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    filt = make_mcmv(info, fwd, data_cov, sources, orientations=oris, reg=REG)

    assert isinstance(filt, MCMVBeamformer)
    assert filt["weights"].shape == (2, len(filt["ch_names"]))
    # Unit-gain / zero-gain constraint holds on the actual leadfield.
    assert_allclose(filt["weights"] @ filt["leadfield"], np.eye(2), atol=1e-9)

    n_times = 50
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((len(filt["ch_names"]), n_times))
    s = apply_mcmv(arr, filt)
    assert s.shape == (2, n_times)

    src_cov = apply_mcmv_cov(data_cov, filt)
    assert src_cov.shape == (2, 2)
    assert_allclose(src_cov, src_cov.T, atol=1e-10)  # symmetric


# --------------------------------------------------------------------------- #
# 4. Scientific validation: leakage suppression for correlated sources.
# --------------------------------------------------------------------------- #
def test_correlated_sources_leakage_suppression(fwd_info):
    """MCMV removes the inter-source leakage that contaminates LCMV.

    Two correlated sources are simulated. Because the constrained sources are
    correlated, single-source LCMV leaks the *independent* part of source 2 into
    the estimate of source 1; MCMV's zero-gain constraint removes it.
    """
    fwd, info = fwd_info
    G = fwd["sol"]["data"]
    s1_idx, s2_idx = 4, 9
    h1 = G[:, 3 * s1_idx : 3 * s1_idx + 3] @ np.array([1.0, 0, 0])
    h2 = G[:, 3 * s2_idx : 3 * s2_idx + 3] @ np.array([0, 1.0, 0])

    rng = np.random.default_rng(11)
    n_t = 4000
    s1 = rng.standard_normal(n_t)
    indep = rng.standard_normal(n_t)
    rho = 0.9
    s2 = rho * s1 + np.sqrt(1 - rho**2) * indep  # corr(s1, s2) ~ 0.9
    noise = 0.01 * rng.standard_normal((len(h1), n_t))
    b = np.outer(h1, s1) + np.outer(h2, s2) + noise
    R = (b @ b.T) / n_t
    # The zero-gain constraint W^T H = I holds for any invertible inverse, so a
    # little regularisation only tames noise amplification on the near-rank-2 R
    # without weakening the leakage-suppression result.
    Rinv, _, _ = _reg_pinv(R, reg=0.05, rank="full")

    # Single-source LCMV for source 1, and 2-source MCMV.
    w1 = (Rinv @ h1) / (h1 @ Rinv @ h1)
    s1_lcmv = w1 @ b
    W = _compute_mcmv_weights(np.column_stack([h1, h2]), Rinv)
    s1_mcmv = W[0] @ b

    # Part of s2 orthogonal to s1 -> a clean "leakage" probe.
    s2_perp = s2 - (np.dot(s2, s1) / np.dot(s1, s1)) * s1

    def corr(a, c):
        return abs(np.corrcoef(a, c)[0, 1])

    leak_lcmv = corr(s1_lcmv, s2_perp)
    leak_mcmv = corr(s1_mcmv, s2_perp)

    # MCMV recovers source 1 almost perfectly and suppresses the leakage.
    assert corr(s1_mcmv, s1) > 0.999
    assert leak_mcmv < 0.02
    # LCMV leaks the independent part of source 2 substantially more.
    assert leak_lcmv > 0.1
    assert leak_mcmv < leak_lcmv / 5


# --------------------------------------------------------------------------- #
# 5. Error contract (mathematically invalid -> raise).
# --------------------------------------------------------------------------- #
def test_duplicate_sources_raise(fwd_info):
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    with pytest.raises(ValueError, match="unique"):
        make_mcmv(info, fwd, data_cov, [3, 3], orientations=np.eye(3)[:2])


def test_out_of_range_sources_raise(fwd_info):
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    big = fwd["nsource"] + 5
    with pytest.raises(ValueError, match="indices must be in"):
        make_mcmv(info, fwd, data_cov, [0, big], orientations=np.eye(3)[:2])


def test_free_ori_requires_orientations(fwd_info):
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    with pytest.raises(ValueError, match="orientations.*required"):
        make_mcmv(info, fwd, data_cov, [1, 2])


def test_fixed_ori_forbids_orientations(fwd_info):
    fwd, info = fwd_info
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=False)
    data_cov = _random_cov(info)
    with pytest.raises(ValueError, match="must be None"):
        make_mcmv(info, fwd_fixed, data_cov, [1, 2], orientations=np.eye(3)[:2])


def test_channel_mismatch_raises(fwd_info):
    fwd, info = fwd_info
    # Covariance over an entirely different channel set.
    bad_info = mne.create_info([f"X{i}" for i in range(5)], 200.0, "eeg")
    cov = _random_cov(bad_info)
    with pytest.raises(ValueError, match="No common good channels"):
        make_mcmv(info, fwd, cov, [1, 2], orientations=np.eye(3)[:2])


# --------------------------------------------------------------------------- #
# 6. Warning contract (valid but limited -> warn, with disclosure).
# --------------------------------------------------------------------------- #
def test_rank_deficient_cov_warns(fwd_info):
    fwd, info = fwd_info
    n = len(info["ch_names"])
    rng = np.random.default_rng(5)
    A = rng.standard_normal((n, n - 3))  # rank n-3
    R = A @ A.T
    cov = mne.Covariance(
        R, info["ch_names"], info["bads"], list(info["projs"]), nfree=1
    )
    with pytest.warns(RuntimeWarning, match="rank-deficient"):
        make_mcmv(info, fwd, cov, [1, 2], orientations=np.eye(3)[:2], reg=0.0)


def test_unit_noise_gain_without_noise_cov_warns(fwd_info):
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    with pytest.warns(RuntimeWarning, match="identity"):
        filt = make_mcmv(
            info, fwd, data_cov, [1, 2], orientations=np.eye(3)[:2],
            weight_norm="unit-noise-gain",
        )
    # Each filter has unit noise gain w.r.t. the identity (||w_i|| == 1).
    assert_allclose(np.linalg.norm(filt["weights"], axis=1), 1.0, atol=1e-9)
