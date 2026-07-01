"""Tests for the ReciPSIICOS covariance modification and beamformer."""

# Authors: the mne-beamlab contributors
# License: BSD-3-Clause

import mne
import numpy as np
import pytest
from mne.beamformer import apply_lcmv_cov
from numpy.testing import assert_allclose

from mne_beamlab import (
    make_recipsiicos_cov,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
)
from mne_beamlab._recipsiicos import (
    _apply_projector,
    _correlation_gram,
    _optimal_rank,
    _power_columns,
    _power_projector,
    _reduction_operator,
    _spectral_flip,
    _tangential_topographies,
    _unvec,
    _vec,
    _whitened_projector,
)

mne.set_log_level("ERROR")


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fwd_info():
    """A small free-orientation EEG sphere forward and its Info (M = 8)."""
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:8], 200.0, "eeg")
    info.set_montage("standard_1020")
    sphere = mne.make_sphere_model("auto", "auto", info)
    src = mne.setup_volume_source_space(sphere=sphere, pos=35.0)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
    return fwd, info


@pytest.fixture(scope="module")
def fwd_fixed(fwd_info):
    """The fixed-orientation version of the fixture forward."""
    fwd, _ = fwd_info
    return mne.convert_forward_solution(fwd, force_fixed=True)


def _cov_from_sources(forward, idx, rho, noise=0.01):
    """Sensor covariance from two unit-variance sources with correlation rho."""
    gain = np.asarray(forward["sol"]["data"], dtype=np.float64)
    g = gain[:, idx]  # (n_channels, 2)
    c_ss = np.array([[1.0, rho], [rho, 1.0]])
    r = g @ c_ss @ g.T + noise * np.eye(gain.shape[0])
    ch = forward["sol"]["row_names"]
    return mne.Covariance(r, ch, [], [], nfree=1)


def _ident_noise_cov(forward, scale=1.0):
    """An identity noise covariance over the forward's channels."""
    ch = forward["sol"]["row_names"]
    return mne.Covariance(scale * np.eye(len(ch)), ch, [], [], nfree=1)


def _orthogonal_topographies(n_channels, n_loc, seed=0):
    """``(n_loc, n_channels, 1)`` mutually orthogonal topographies.

    Columns of a random orthogonal matrix, scaled differently, so that any two
    topographies are orthogonal. A source cross-product is then Frobenius-
    orthogonal to every auto-product, so projecting onto the power subspace
    removes correlation exactly -- the property used to verify the mechanism.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
    scales = 1.0 + 0.5 * np.arange(n_loc)
    return (q[:, :n_loc] * scales).T.reshape(n_loc, n_channels, 1)


# --------------------------------------------------------------------------- #
# Linear-algebra primitives.
# --------------------------------------------------------------------------- #
def test_vec_unvec_roundtrip():
    """vec then unvec restores the matrix (column-major convention)."""
    a = np.arange(49, dtype=float).reshape(7, 7)
    assert_allclose(_unvec(_vec(a), 7), a)


def test_power_projector_is_orthogonal_projector():
    """P = U_K U_K^T is a symmetric idempotent projector of the right trace."""
    topos = _orthogonal_topographies(9, 6)
    g_pwr = _power_columns(topos)
    p, sv = _power_projector(g_pwr, rank=5)
    assert_allclose(p @ p, p, atol=1e-10)
    assert_allclose(p, p.T, atol=1e-10)
    assert_allclose(np.trace(p), 5.0, atol=1e-8)
    assert sv.shape == (g_pwr.shape[1],)


def test_power_columns_shapes_fixed_and_free():
    """One power column per fixed source; three per free-orientation source."""
    gain_fixed = np.random.default_rng(0).standard_normal((6, 4))
    fixed = _tangential_topographies(gain_fixed, fixed=True)
    assert _power_columns(fixed).shape == (36, 4)

    gain_free = np.random.default_rng(1).standard_normal((6, 12))
    free = _tangential_topographies(gain_free, fixed=False)
    assert free.shape[2] == 2
    assert _power_columns(free).shape == (36, 12)


def test_correlation_gram_symmetric_psd():
    """The correlation Gram is symmetric positive semi-definite."""
    topos = _orthogonal_topographies(6, 5)
    c_cor = _correlation_gram(topos)
    assert_allclose(c_cor, c_cor.T, atol=1e-10)
    assert np.linalg.eigvalsh(c_cor).min() > -1e-10


def test_apply_projector_returns_symmetric():
    """Projecting a symmetric covariance yields a symmetric matrix."""
    rng = np.random.default_rng(0)
    topos = _orthogonal_topographies(6, 4)
    g_pwr = _power_columns(topos)
    p, _ = _power_projector(g_pwr, rank=4)
    a = rng.standard_normal((6, 6))
    cov = a @ a.T
    out = _apply_projector(p, cov)
    assert_allclose(out, out.T, atol=1e-10)


def test_spectral_flip_makes_pd_and_reports_energy():
    """The spectral flip makes eigenvalues positive and reports negative energy."""
    eigs = np.array([-2.0, -1.0, 3.0, 4.0, 5.0, 6.0])
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((6, 6)))
    cov = q @ np.diag(eigs) @ q.T
    flipped, neg_energy = _spectral_flip(cov)
    assert_allclose(neg_energy, 3.0 / 21.0, atol=1e-10)
    assert np.linalg.eigvalsh(flipped).min() > 0
    assert_allclose(np.sort(np.linalg.eigvalsh(flipped)), np.sort(np.abs(eigs)))


# --------------------------------------------------------------------------- #
# The core mechanism: exact correlation removal (regime-independent).
# --------------------------------------------------------------------------- #
def test_recipsiicos_removes_correlation_exactly():
    """With orthogonal topographies the projector removes coupling exactly.

    The cross-product of two orthogonal topographies is Frobenius-orthogonal to
    the power subspace, so projecting the coupled covariance onto that subspace
    must reproduce the uncoupled covariance to numerical precision -- and a
    plain LCMV on the coupled covariance must instead cancel the source.
    """
    m, n = 12, 6
    topos = _orthogonal_topographies(m, n)
    gain = topos[:, :, 0].T  # (m, n)
    rho = 0.9
    c_ss = np.eye(n)
    c_ss[0, 1] = c_ss[1, 0] = rho
    r_corr = gain @ c_ss @ gain.T
    r_uncorr = gain @ np.eye(n) @ gain.T

    g_pwr = _power_columns(topos)
    projector, _ = _power_projector(g_pwr, rank=g_pwr.shape[1])
    cleaned, _ = _spectral_flip(_apply_projector(projector, r_corr))

    assert_allclose(cleaned, r_uncorr, atol=1e-8)

    # A plain LCMV on the coupled covariance cancels source 0; on the cleaned
    # covariance it does not.
    g0 = gain[:, 0]
    p_corr = 1.0 / (g0 @ np.linalg.pinv(r_corr) @ g0)
    p_clean = 1.0 / (g0 @ np.linalg.pinv(cleaned) @ g0)
    assert p_corr < 0.2 * p_clean


def test_whitened_reduces_correlation_orthogonal():
    """Whitened ReciPSIICOS reduces the coupling discrepancy.

    Unlike plain ReciPSIICOS, the whitened projector has no exact-removal
    guarantee (power-subspace whitening distorts the Frobenius-orthogonality
    between auto-terms and the coupling direction), but it must still move the
    covariance substantially closer to the uncoupled one and undo the LCMV
    signal cancellation at the coupled source.
    """
    m, n = 12, 6
    # Orthogonal topographies with the COUPLED pair (0, 1) given the largest
    # norms, so that its cross-product is the dominant correlation direction and
    # is therefore among the ones the whitened projector removes.
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    scales = np.array([3.0, 2.7, 1.0, 0.9, 0.8, 0.7])
    topos = (q[:, :n] * scales).T.reshape(n, m, 1)
    gain = topos[:, :, 0].T
    c_ss = np.eye(n)
    c_ss[0, 1] = c_ss[1, 0] = 0.9
    r_corr = gain @ c_ss @ gain.T
    r_uncorr = gain @ np.eye(n) @ gain.T

    g_pwr = _power_columns(topos)
    c_cor = _correlation_gram(topos)
    projector, _ = _whitened_projector(g_pwr, c_cor, rank=1, reg=1e-6)
    cleaned, _ = _spectral_flip(_apply_projector(projector, r_corr))

    base_err = np.linalg.norm(r_corr - r_uncorr)
    clean_err = np.linalg.norm(cleaned - r_uncorr)
    assert clean_err < base_err

    g0 = gain[:, 0]
    p_corr = 1.0 / (g0 @ np.linalg.pinv(r_corr) @ g0)
    p_clean = 1.0 / (g0 @ np.linalg.pinv(cleaned) @ g0)
    assert p_clean > p_corr


# --------------------------------------------------------------------------- #
# Virtual-sensor reduction.
# --------------------------------------------------------------------------- #
def test_reduction_operator_shapes_and_variance(fwd_info):
    """The reduction operator maps M sensors to q <= r virtual sensors."""
    fwd, info = fwd_info
    ch = fwd["sol"]["row_names"]
    b_full, q_full, r = _reduction_operator(
        info, fwd, ch, noise_cov=None, whitener_rank="full",
        pct_var=1.0, n_virtual=None,
    )
    assert b_full.shape == (q_full, len(ch))
    assert q_full == r  # keeping 100% of variance keeps the whitened rank

    b_small, q_small, _ = _reduction_operator(
        info, fwd, ch, noise_cov=None, whitener_rank="full",
        pct_var=0.90, n_virtual=None,
    )
    assert q_small <= q_full

    b_fixed, q_fixed, _ = _reduction_operator(
        info, fwd, ch, noise_cov=None, whitener_rank="full",
        pct_var=1.0, n_virtual=3,
    )
    assert q_fixed == 3


def test_reduction_whitens_noise_to_identity(fwd_info):
    """B applied to the noise covariance yields identity (white working noise)."""
    fwd, info = fwd_info
    ch = fwd["sol"]["row_names"]
    noise = _ident_noise_cov(fwd, scale=2.0)
    b_op, q, _ = _reduction_operator(
        info, fwd, ch, noise_cov=noise, whitener_rank="full",
        pct_var=1.0, n_virtual=None,
    )
    working_noise = b_op @ noise.data @ b_op.T
    assert_allclose(working_noise, np.eye(q), atol=1e-8)


# --------------------------------------------------------------------------- #
# Public covariance API.
# --------------------------------------------------------------------------- #
def test_make_recipsiicos_cov_is_valid_covariance(fwd_fixed):
    """The modified covariance is a symmetric PSD mne.Covariance over the channels."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.9)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    cov = make_recipsiicos_cov(
        data_cov, fwd_fixed, info, rank=20, method="recipsiicos", pct_var=1.0,
    )
    assert isinstance(cov, mne.Covariance)
    assert cov.ch_names == list(fwd_fixed["sol"]["row_names"])
    assert_allclose(cov.data, cov.data.T, atol=1e-10)
    assert np.linalg.eigvalsh(cov.data).min() > -1e-8


@pytest.mark.filterwarnings("ignore:The spectral-flip step carried")
def test_make_recipsiicos_cov_whitened_runs(fwd_fixed):
    """The whitened method also produces a valid covariance.

    On this deliberately tiny (8-channel) fixture the whitened projector at high
    rank trips the module's negative-energy safety warning; that warning is
    expected here and filtered.
    """
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.9)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    cov = make_recipsiicos_cov(
        data_cov, fwd_fixed, info, rank=8, method="whitened", pct_var=1.0,
    )
    assert isinstance(cov, mne.Covariance)
    assert np.linalg.eigvalsh(cov.data).min() > -1e-8


# --------------------------------------------------------------------------- #
# Public beamformer API and MNE interoperability.
# --------------------------------------------------------------------------- #
def test_make_recipsiicos_lcmv_returns_usable_beamformer(fwd_fixed):
    """The returned filters are a Beamformer usable by MNE's apply functions."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.9)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    n_src = fwd_fixed["nsource"]
    filters = make_recipsiicos_lcmv(
        info, fwd_fixed, data_cov, rank=20, method="recipsiicos", pct_var=1.0,
    )
    assert filters["kind"] == "LCMV"
    assert filters["weights"].shape[0] == n_src
    # The reduction operator is stored as the whitener (q x M).
    assert filters["whitener"].shape[1] == len(info["ch_names"])
    stc = apply_lcmv_cov(data_cov, filters)
    assert stc.data.shape[0] == n_src
    assert np.all(np.isfinite(stc.data))
    assert np.all(stc.data >= 0)


def test_beamformer_whitener_folding_matches_manual(fwd_fixed):
    """apply_lcmv_cov applies B then the working weights, as folded in."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.9)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    filters = make_recipsiicos_lcmv(
        info, fwd_fixed, data_cov, rank=20, method="recipsiicos", pct_var=1.0,
    )
    b_op = filters["whitener"]
    w = filters["weights"]
    # Manual: reduce the covariance to the working space, then apply weights.
    manual = np.einsum("ij,jk,ik->i", w @ b_op, data_cov.data, w @ b_op)
    stc = apply_lcmv_cov(data_cov, filters)
    assert_allclose(stc.data.ravel(), manual, rtol=1e-6, atol=1e-12)


def test_free_orientation_beamformer_and_pick_ori(fwd_info):
    """Free-orientation forward supports pick_ori and yields per-source power."""
    fwd, info = fwd_info
    idx = [3, 3 * (fwd["nsource"] - 2)]  # two source columns
    data_cov = _cov_from_sources(fwd, idx=idx, rho=0.9)
    filters = make_recipsiicos_lcmv(
        info, fwd, data_cov, rank=20, method="whitened",
        pick_ori="max-power", pct_var=1.0,
    )
    stc = apply_lcmv_cov(data_cov, filters)
    assert stc.data.shape[0] == fwd["nsource"]
    assert np.all(np.isfinite(stc.data))


# --------------------------------------------------------------------------- #
# Rank curve and automatic rank selection.
# --------------------------------------------------------------------------- #
def test_rank_curve_shapes_and_bounds(fwd_fixed):
    """The rank curve spans 1..q^2 with fractions in [0, 1] and reaches 1."""
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    ranks, p_pwr, p_cor = recipsiicos_rank_curve(
        fwd_fixed, info, method="recipsiicos", pct_var=1.0,
    )
    q_sq = ranks[-1]
    assert ranks.shape == p_pwr.shape == p_cor.shape == (q_sq,)
    assert p_pwr.min() >= -1e-9 and p_pwr.max() <= 1 + 1e-9
    assert p_cor.min() >= -1e-9 and p_cor.max() <= 1 + 1e-9
    # For the power projector the retained power is non-decreasing in rank.
    assert np.all(np.diff(p_pwr) >= -1e-9)
    assert p_pwr[-1] > 0.99


def test_rank_curve_returns_optimal(fwd_fixed):
    """With return_optimal the curve also yields a valid rank K*."""
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    ranks, p_pwr, p_cor, kstar = recipsiicos_rank_curve(
        fwd_fixed, info, method="whitened", pct_var=1.0, return_optimal=True,
    )
    assert 1 <= kstar <= ranks[-1]


def test_optimal_rank_sign_change():
    """K* is the last rank before the first sign change of dP_cor/dk - dP_pwr/dk."""
    # A trailing positive marginal after the crossing must be ignored (the
    # selection is the FIRST crossing, not the last non-negative point).
    p_pwr = np.array([0.0, 0.10, 0.20, 0.30, 0.60, 0.95, 1.00])
    p_cor = np.array([0.0, 0.30, 0.55, 0.75, 0.85, 0.92, 1.00])
    # d_pwr = .10 .10 .10 .30 .35 .05 ; d_cor = .30 .25 .20 .10 .07 .08
    # delta  = +.20 +.15 +.10 -.20 -.28 +.03 -> first negative at index 3
    assert _optimal_rank(p_pwr, p_cor) == 4


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #
def test_invalid_method_raises(fwd_fixed):
    """An unknown method name is rejected."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.5)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="method"):
        make_recipsiicos_cov(data_cov, fwd_fixed, info, rank=5, method="nope")


def test_rank_too_small_raises(fwd_fixed):
    """A non-positive projection rank is rejected."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.5)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="positive"):
        make_recipsiicos_cov(data_cov, fwd_fixed, info, rank=0, pct_var=1.0)


def test_rank_too_large_raises(fwd_fixed):
    """A projection rank above q^2 is rejected."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.5)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    # pct_var=1.0 -> q = 8 -> q^2 = 64.
    with pytest.raises(ValueError, match="exceeds"):
        make_recipsiicos_cov(data_cov, fwd_fixed, info, rank=65, pct_var=1.0)


def test_non_covariance_input_raises(fwd_fixed):
    """A plain array in place of a Covariance is rejected."""
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(TypeError):
        make_recipsiicos_cov(np.eye(8), fwd_fixed, info, rank=5)


def test_pick_ori_requires_free_orientation(fwd_fixed):
    """pick_ori is rejected for a fixed-orientation forward."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.5)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="free-orientation"):
        make_recipsiicos_lcmv(
            info, fwd_fixed, data_cov, rank=5, pick_ori="max-power", pct_var=1.0,
        )


def test_no_common_channels_raises(fwd_fixed):
    """Disjoint channel sets raise a clear error."""
    bad = mne.Covariance(
        np.eye(3), ["X1", "X2", "X3"], [], [], nfree=1,
    )
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="common"):
        make_recipsiicos_cov(bad, fwd_fixed, info, rank=5)