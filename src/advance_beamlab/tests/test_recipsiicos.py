"""Tests for the ReciPSIICOS covariance modification and beamformer."""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import warnings

import mne
import numpy as np
import pytest
from mne.beamformer import apply_lcmv_cov
from numpy.testing import assert_allclose

from advance_beamlab import (
    make_recipsiicos_cov,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
)
from advance_beamlab._recipsiicos import (
    _apply_projector,
    _correlation_blocks,
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


@pytest.fixture(scope="module")
def fwd_rich():
    """A 32-channel, 186-source EEG forward with a non-degenerate rank curve.

    The 8-channel fixture above is deliberately tiny and its rank curve is
    saturated after two or three directions, which is useless for exercising the
    rank-selection criterion. This one is still cheap (well under a second) but
    resolves the power subspace over enough directions for the 45-degree
    crossing to be meaningful.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:32], 200.0, "eeg")
    info.set_montage("standard_1020")
    sphere = mne.make_sphere_model("auto", "auto", info)
    src = mne.setup_volume_source_space(sphere=sphere, pos=20.0)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
    return mne.convert_forward_solution(fwd, force_fixed=True), info


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
        info,
        fwd,
        ch,
        noise_cov=None,
        whitener_rank="full",
        pct_var=1.0,
        n_virtual=None,
    )
    assert b_full.shape == (q_full, len(ch))
    assert q_full == r  # keeping 100% of variance keeps the whitened rank

    b_small, q_small, _ = _reduction_operator(
        info,
        fwd,
        ch,
        noise_cov=None,
        whitener_rank="full",
        pct_var=0.90,
        n_virtual=None,
    )
    assert q_small <= q_full

    b_fixed, q_fixed, _ = _reduction_operator(
        info,
        fwd,
        ch,
        noise_cov=None,
        whitener_rank="full",
        pct_var=1.0,
        n_virtual=3,
    )
    assert q_fixed == 3


def test_reduction_whitens_noise_to_identity(fwd_info):
    """B applied to the noise covariance yields identity (white working noise)."""
    fwd, info = fwd_info
    ch = fwd["sol"]["row_names"]
    noise = _ident_noise_cov(fwd, scale=2.0)
    b_op, q, _ = _reduction_operator(
        info,
        fwd,
        ch,
        noise_cov=noise,
        whitener_rank="full",
        pct_var=1.0,
        n_virtual=None,
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
        data_cov,
        fwd_fixed,
        info,
        rank=20,
        method="recipsiicos",
        pct_var=1.0,
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
        data_cov,
        fwd_fixed,
        info,
        rank=8,
        method="whitened",
        pct_var=1.0,
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
        info,
        fwd_fixed,
        data_cov,
        rank=20,
        method="recipsiicos",
        pct_var=1.0,
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
        info,
        fwd_fixed,
        data_cov,
        rank=20,
        method="recipsiicos",
        pct_var=1.0,
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
        info,
        fwd,
        data_cov,
        rank=20,
        method="whitened",
        pick_ori="max-power",
        pct_var=1.0,
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
        fwd_fixed,
        info,
        method="recipsiicos",
        pct_var=1.0,
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
        fwd_fixed,
        info,
        method="whitened",
        pct_var=1.0,
        return_optimal=True,
    )
    assert 1 <= kstar <= ranks[-1]


def test_rank_curve_matches_bruteforce(fwd_fixed):
    """The closed-form curves equal the per-rank projector computation."""
    from advance_beamlab._recipsiicos import (
        _correlation_gram,
        _forward_gain,
        _power_projector,
        _reduction_operator,
        _tangential_topographies,
        _whitened_projector,
    )

    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    ch = fwd_fixed["sol"]["row_names"]
    b_op, _, _ = _reduction_operator(
        info,
        fwd_fixed,
        ch,
        noise_cov=None,
        whitener_rank="full",
        pct_var=1.0,
        n_virtual=None,
    )
    gain, fixed = _forward_gain(fwd_fixed, ch)
    topos = _tangential_topographies(b_op @ gain, fixed)
    g_pwr = _power_columns(topos)
    c_pwr = g_pwr @ g_pwr.T
    c_cor = _correlation_gram(topos)
    tr_pwr, tr_cor = np.trace(c_pwr), np.trace(c_cor)
    msq = g_pwr.shape[0]
    reg = 0.05

    for method in ("recipsiicos", "whitened"):
        _, p_pwr, p_cor = recipsiicos_rank_curve(
            fwd_fixed,
            info,
            method=method,
            pct_var=1.0,
            reg=reg,
        )
        bp = np.empty(msq)
        bc = np.empty(msq)
        for i, k in enumerate(range(1, msq + 1)):
            if method == "recipsiicos":
                proj, _ = _power_projector(g_pwr, k)
            else:
                proj, _ = _whitened_projector(g_pwr, c_cor, k, reg=reg)
            bp[i] = np.trace(proj @ c_pwr @ proj.T) / tr_pwr
            bc[i] = np.trace(proj @ c_cor @ proj.T) / tr_cor
        assert_allclose(p_pwr, bp, atol=1e-9)
        assert_allclose(p_cor, bc, atol=1e-9)


def test_optimal_rank_direction_aware(fwd_rich):
    """K* is the 45-degree crossing, found per curve direction, with a floor."""
    # recipsiicos: rising curves whose *identity* end is k = q^2, so the rank
    # axis is traversed downward (Fig. 19 of the paper puts the ReciPSIICOS
    # scale in descending order) and K* is one past the LAST negative
    # difference. A measured curve is used because a hand-written one cannot be
    # trusted to have the sign structure a real forward produces.
    fwd, info = fwd_rich
    _, p_pwr, p_cor = recipsiicos_rank_curve(fwd, info, method="recipsiicos")
    kstar = _optimal_rank(p_pwr, p_cor, "recipsiicos")
    delta = np.diff(p_cor) - np.diff(p_pwr)
    # The difference is already negative at the very first step -- the leading
    # power direction is also the one the correlation subspace overlaps most --
    # so an ascending scan would stop at K* = 1 and keep p_pwr[0] ~ 0.5 of the
    # source-power subspace. The descending scan must land far past that.
    assert delta[0] < 0
    assert p_pwr[0] < 0.8
    assert kstar >= 10
    assert p_pwr[kstar - 1] > 0.8
    # The crossing is the last sign change: everything above it is non-negative.
    assert np.all(delta[kstar - 1 :] >= 0)

    # whitened: decreasing curves; K* = last rank before delta turns positive.
    q_pwr = np.array([1.00, 0.80, 0.60, 0.45, 0.35, 0.30, 0.28])
    q_cor = np.array([1.00, 0.60, 0.30, 0.15, 0.10, 0.08, 0.07])
    # d_pwr = -.20 -.20 -.15 -.10 -.05 -.02 ; d_cor = -.40 -.30 -.15 -.05 -.02 -.01
    # delta  = -.20 -.10 .00 +.05 +.03 +.01 -> first positive at index 3 -> K*=4
    assert _optimal_rank(q_pwr, q_cor, "whitened") == 4

    # Degenerate decreasing curve (power emptied almost at once): the floor must
    # pull K* back so the covariance is not zeroed.
    d_pwr = np.array([0.58, 0.02, 0.0, 0.0, 0.0])
    d_cor = np.array([0.55, 0.01, 0.0, 0.0, 0.0])
    assert _optimal_rank(d_pwr, d_cor, "whitened") == 1


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
    """pick_ori is rejected for a fixed-orientation forward, in MNE's words."""
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.5)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="forward operator with free orientation"):
        make_recipsiicos_lcmv(
            info,
            fwd_fixed,
            data_cov,
            rank=5,
            pick_ori="max-power",
            pct_var=1.0,
        )


def test_pick_ori_normal_requires_surface_oriented_forward(fwd_info):
    """'normal' on a surf_ori=False forward raises instead of silently using +Z.

    ``mne.read_forward_solution`` returns ``surf_ori=False`` by default, and the
    filter computation then takes the third orientation column -- the
    head-coordinate +Z axis, not a cortical normal. MNE's make_lcmv refuses this;
    so must we, or the caller gets a plausible SourceEstimate for the wrong
    dipole orientation.
    """
    fwd, info = fwd_info
    assert not fwd["surf_ori"]
    idx = [3, 3 * (fwd["nsource"] - 2)]
    data_cov = _cov_from_sources(fwd, idx=idx, rho=0.5)
    with pytest.raises(ValueError, match="oriented in surface coordinates"):
        make_recipsiicos_lcmv(
            info,
            fwd,
            data_cov,
            rank=5,
            pick_ori="normal",
            pct_var=1.0,
        )


def test_pick_ori_normal_rejects_volume_source_space(fwd_info):
    """'normal' on a volume source space raises MNE's RuntimeError."""
    from copy import deepcopy

    fwd, info = fwd_info
    fwd_vol = deepcopy(fwd)
    # Get past the surf_ori check so the source-space check is the one that
    # fires, then present the source space as a volumetric grid.
    fwd_vol["surf_ori"] = True
    for s in fwd_vol["src"]:
        s["type"] = "vol"
    assert fwd_vol["src"].kind == "volume"
    idx = [3, 3 * (fwd["nsource"] - 2)]
    data_cov = _cov_from_sources(fwd, idx=idx, rho=0.5)
    with pytest.raises(RuntimeError, match="surface or discrete SourceSpaces"):
        make_recipsiicos_lcmv(
            info,
            fwd_vol,
            data_cov,
            rank=5,
            pick_ori="normal",
            pct_var=1.0,
        )


def test_unknown_pick_ori_and_weight_norm_rejected(fwd_info):
    """Unrecognised pick_ori / weight_norm values are rejected up front.

    Without validation an unknown ``pick_ori`` reaches the filter computation,
    which falls through to the free-orientation branch while ``is_free_ori`` is
    set from the requested value, so ``apply_lcmv`` builds a corrupted
    SourceEstimate.
    """
    fwd, info = fwd_info
    idx = [3, 3 * (fwd["nsource"] - 2)]
    data_cov = _cov_from_sources(fwd, idx=idx, rho=0.5)
    with pytest.raises(ValueError, match="Invalid value for the 'pick_ori' parameter"):
        make_recipsiicos_lcmv(
            info, fwd, data_cov, rank=5, pick_ori="normalish", pct_var=1.0
        )
    with pytest.raises(
        ValueError, match="Invalid value for the 'weight_norm' parameter"
    ):
        make_recipsiicos_lcmv(
            info, fwd, data_cov, rank=5, weight_norm="unit-gain", pct_var=1.0
        )


@pytest.mark.filterwarnings("ignore:The spectral-flip")
def test_underdetermined_working_space_raises(fwd_info):
    """A free-orientation solve with q < n_orient errors clearly, not opaquely."""
    fwd, info = fwd_info
    idx = [3, 3 * (fwd["nsource"] - 2)]
    data_cov = _cov_from_sources(fwd, idx=idx, rho=0.9)
    # Force only two virtual sensors: fewer than the three source orientations.
    with pytest.raises(ValueError, match="underdetermined"):
        make_recipsiicos_lcmv(
            info,
            fwd,
            data_cov,
            rank=2,
            method="recipsiicos",
            pick_ori="max-power",
            n_virtual=2,
        )


def test_no_common_channels_raises(fwd_fixed):
    """Disjoint channel sets raise a clear error."""
    bad = mne.Covariance(
        np.eye(3),
        ["X1", "X2", "X3"],
        [],
        [],
        nfree=1,
    )
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="common"):
        make_recipsiicos_cov(bad, fwd_fixed, info, rank=5)


# --------------------------------------------------------------------------- #
# Noise-covariance channel handling.
# --------------------------------------------------------------------------- #
def test_noise_cov_may_cover_a_channel_subset(fwd_fixed):
    """A noise covariance over a subset of the channels restricts, not raises.

    An empty-room (MEG-only) noise covariance combined with a MEG+EEG forward is
    routine, and ``compute_whitener`` raises "Not all channels present in noise
    covariance" if the extra channels are not dropped first. Modelled here with
    an EEG covariance that covers six of the eight fixture channels.
    """
    all_ch = list(fwd_fixed["sol"]["row_names"])
    subset = all_ch[:6]
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.9)
    noise = mne.Covariance(np.eye(len(subset)), subset, [], [], nfree=1)
    info = mne.create_info(all_ch, 200.0, "eeg")
    info.set_montage("standard_1020")

    cov = make_recipsiicos_cov(
        data_cov, fwd_fixed, info, rank=10, noise_cov=noise, pct_var=1.0
    )
    assert cov.ch_names == subset

    filters = make_recipsiicos_lcmv(
        info, fwd_fixed, data_cov, rank=10, noise_cov=noise, pct_var=1.0
    )
    assert filters["ch_names"] == subset
    assert filters["whitener"].shape[1] == len(subset)

    ranks, p_pwr, _ = recipsiicos_rank_curve(
        fwd_fixed, info, noise_cov=noise, pct_var=1.0
    )
    # Six channels survive, so the working space is at most 6 x 6 in vec form.
    assert ranks[-1] <= 36
    assert p_pwr[-1] > 0.99


def test_noise_cov_sharing_no_channels_raises(fwd_fixed):
    """A noise covariance disjoint from the data raises rather than whitening."""
    all_ch = list(fwd_fixed["sol"]["row_names"])
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.5)
    noise = mne.Covariance(np.eye(2), ["Z1", "Z2"], [], [], nfree=1)
    info = mne.create_info(all_ch, 200.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="shares no channels"):
        make_recipsiicos_cov(
            data_cov, fwd_fixed, info, rank=5, noise_cov=noise, pct_var=1.0
        )


# --------------------------------------------------------------------------- #
# Correlation-Gram blocking.
# --------------------------------------------------------------------------- #
def test_correlation_blocks_respect_memory_budget():
    """The block size counts columns of length q^2 and honours the byte budget.

    A free-orientation forward emits four columns per source *pair*, each q^2
    long, so a block size expressed as a pair (or entry) count says nothing about
    the memory a block occupies.
    """
    rng = np.random.default_rng(0)
    q, n_loc = 40, 30
    topos = rng.standard_normal((n_loc, q, 2))
    budget = 1 << 20  # 1 MiB
    blocks = list(_correlation_blocks(topos, max_block_bytes=budget))

    # Rounded down to a whole number of pairs (four columns each).
    max_cols = budget // (8 * q**2) // 4 * 4
    assert max_cols > 4  # the budget is not so tight that it degenerates
    assert all(block.shape[0] == q**2 for block in blocks)
    assert all(block.nbytes <= budget for block in blocks)
    assert max(block.shape[1] for block in blocks) == max_cols
    # Four columns per unordered pair, none lost to the blocking.
    assert sum(block.shape[1] for block in blocks) == 4 * n_loc * (n_loc - 1) // 2

    # The blocking must not change the accumulated Gram.
    full = np.column_stack(list(_correlation_blocks(topos, max_block_bytes=1 << 40)))
    assert_allclose(_correlation_gram(topos), full @ full.T, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Degenerate whitened ranks.
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore:The spectral-flip step carried")
def test_whitened_rank_annihilating_covariance_warns(fwd_fixed):
    """A whitened rank at q(q+1)/2 empties the covariance, and says so.

    The power whitener spans the symmetric subspace, of dimension q(q+1)/2, so
    removing that many correlation directions removes all of its range and the
    projector is exactly zero -- well below the q^2 bound that is the only other
    guard.
    """
    data_cov = _cov_from_sources(fwd_fixed, idx=[2, 20], rho=0.9)
    info = mne.create_info(fwd_fixed["sol"]["row_names"], 200.0, "eeg")
    info.set_montage("standard_1020")
    q = len(fwd_fixed["sol"]["row_names"])  # pct_var=1.0 keeps every direction
    n_sym = q * (q + 1) // 2
    with pytest.warns(RuntimeWarning, match=r"q\(q\+1\)/2"):
        cov = make_recipsiicos_cov(
            data_cov,
            fwd_fixed,
            info,
            rank=n_sym,
            method="whitened",
            pct_var=1.0,
        )
    assert np.linalg.norm(cov.data) < 1e-9 * np.linalg.norm(data_cov.data)

    # Well below the bound the projector passes the covariance through and the
    # warning must not fire.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.filterwarnings("ignore", message="The spectral-flip step carried")
        warnings.filterwarnings("ignore", message="No average EEG reference")
        cov_ok = make_recipsiicos_cov(
            data_cov,
            fwd_fixed,
            info,
            rank=4,
            method="whitened",
            pct_var=1.0,
        )
    assert np.linalg.norm(cov_ok.data) > 1e-3 * np.linalg.norm(data_cov.data)


# --------------------------------------------------------------------------- #
# Real BEM forward (skipped when the ``sample`` dataset is not downloaded).
# --------------------------------------------------------------------------- #
try:
    _SAMPLE = mne.datasets.sample.data_path(download=False) / "MEG" / "sample"
    _HAVE_SAMPLE = (_SAMPLE / "sample_audvis-meg-eeg-oct-6-fwd.fif").is_file()
except Exception:  # pragma: no cover - depends on local dataset state
    _HAVE_SAMPLE = False

requires_sample = pytest.mark.skipif(
    not _HAVE_SAMPLE, reason="MNE sample dataset not downloaded"
)


def _sample_forward(info, step=12):
    """Decimated ``sample`` BEM forward restricted to ``info``'s channels."""
    from mne.forward.forward import _restrict_forward_to_src_sel

    fwd = mne.read_forward_solution(_SAMPLE / "sample_audvis-meg-eeg-oct-6-fwd.fif")
    fwd = mne.pick_channels_forward(fwd, info["ch_names"], ordered=False)
    fwd = _restrict_forward_to_src_sel(fwd, np.arange(0, fwd["nsource"], step))
    return mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)


@requires_sample
def test_optimal_rank_on_real_forward_keeps_the_power_subspace():
    """On a real BEM forward K* sits near the paper's published operating point.

    Kuznetsova et al. (2021), Fig. 5B, put the ReciPSIICOS operating point at
    roughly 95% retained power against 40-50% retained correlation. Scanning the
    rank axis in the wrong direction instead stops at the first negative
    difference, which on this forward is the very first one, and keeps around a
    tenth of the source-power subspace.
    """
    info = mne.io.read_info(_SAMPLE / "sample_audvis-ave.fif")
    info = mne.pick_info(info, mne.pick_types(info, meg="grad", exclude="bads"))
    fwd = _sample_forward(info)
    noise = mne.read_cov(_SAMPLE / "sample_audvis-cov.fif")

    ranks, p_pwr, p_cor, kstar = recipsiicos_rank_curve(
        fwd, info, noise_cov=noise, return_optimal=True
    )
    assert kstar > 50
    assert kstar < ranks[-1] // 2
    assert p_pwr[kstar - 1] > 0.9
    assert p_cor[kstar - 1] < 0.6
    # The correlation subspace is depleted far more than the power subspace.
    assert p_cor[kstar - 1] < 0.6 * p_pwr[kstar - 1]
    # The ascending scan would have stopped immediately, at a useless rank.
    delta = np.diff(p_cor) - np.diff(p_pwr)
    assert np.nonzero(delta < 0)[0][0] < 3
    assert p_pwr[2] < 0.2


@requires_sample
def test_meg_only_noise_cov_with_meg_eeg_forward():
    """An empty-room MEG covariance drives a MEG+EEG forward without raising.

    This is the standard MEG workflow: ``compute_whitener`` rejects the EEG
    channels the empty-room covariance does not cover, so they have to be
    dropped beforehand.
    """
    info = mne.io.read_info(_SAMPLE / "sample_audvis-ave.fif")
    info = mne.pick_info(
        info, mne.pick_types(info, meg="grad", eeg=True, exclude="bads")
    )
    fwd = _sample_forward(info, step=60)
    noise = mne.read_cov(_SAMPLE / "ernoise-cov.fif")
    noise = mne.cov.pick_channels_cov(
        noise, include=[ch for ch in noise.ch_names if ch in info["ch_names"]]
    )
    n_meg = len(noise.ch_names)
    assert 0 < n_meg < len(info["ch_names"])  # EEG present in info, not in noise

    rng = np.random.default_rng(0)
    n_ch = len(info["ch_names"])
    a = rng.standard_normal((n_ch, n_ch + 20))
    data_cov = mne.Covariance(a @ a.T * 1e-24, info["ch_names"], [], [], nfree=1)

    filters = make_recipsiicos_lcmv(info, fwd, data_cov, rank=40, noise_cov=noise)
    assert filters["ch_names"] == noise.ch_names
    assert filters["weights"].shape == (fwd["nsource"], filters["whitener"].shape[0])

    _, p_pwr, _ = recipsiicos_rank_curve(fwd, info, noise_cov=noise)
    assert p_pwr[-1] > 0.99


@requires_sample
def test_several_sensor_types_without_noise_cov_raises():
    """Mixed sensor types without a noise covariance raise MNE's own error."""
    info = mne.io.read_info(_SAMPLE / "sample_audvis-ave.fif")
    info = mne.pick_info(
        info, mne.pick_types(info, meg="grad", eeg=True, exclude="bads")
    )
    fwd = _sample_forward(info, step=60)
    rng = np.random.default_rng(0)
    n_ch = len(info["ch_names"])
    a = rng.standard_normal((n_ch, n_ch + 20))
    data_cov = mne.Covariance(a @ a.T * 1e-24, info["ch_names"], [], [], nfree=1)

    msg = "several sensor types"
    with pytest.raises(ValueError, match=msg):
        make_recipsiicos_lcmv(info, fwd, data_cov, rank=40)
    with pytest.raises(ValueError, match=msg):
        make_recipsiicos_cov(data_cov, fwd, info, rank=40)
    with pytest.raises(ValueError, match=msg):
        recipsiicos_rank_curve(fwd, info)
    # Identical to what make_lcmv does with the same inputs.
    with pytest.raises(ValueError, match=msg):
        mne.beamformer.make_lcmv(info, fwd, data_cov, reg=0.05)
