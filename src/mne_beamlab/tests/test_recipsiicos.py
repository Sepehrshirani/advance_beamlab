"""Tests for the ReciPSIICOS covariance modification."""

# Authors: the mne-beamlab contributors
# License: BSD-3-Clause

import warnings

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
    _local_topographies,
    _power_columns,
    _power_projector,
    _spectral_flip,
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


def _cov_from_sources(forward, info, idx, rho, noise=0.01, seed=0):
    """Sensor covariance from two unit-variance sources with correlation rho.

    Builds ``R = G C_ss G^T + noise * I`` for the two chosen forward columns and
    returns it as an :class:`mne.Covariance` over the forward's channels.
    """
    gain = np.asarray(forward["sol"]["data"], dtype=np.float64)
    g = gain[:, idx]  # (n_channels, 2)
    c_ss = np.array([[1.0, rho], [rho, 1.0]])
    r = g @ c_ss @ g.T + noise * np.eye(gain.shape[0])
    ch = forward["sol"]["row_names"]
    return mne.Covariance(r, ch, [], [], nfree=1)


def _orthogonal_topographies(n_channels, n_loc, seed=0):
    """``(n_loc, n_channels, 1)`` topographies that are mutually orthogonal.

    Columns of a random orthogonal matrix, scaled differently, so that any two
    topographies ``g_i, g_j`` satisfy ``g_i^T g_j = 0``. With orthogonal
    topographies a source cross-product is Frobenius-orthogonal to every
    auto-product, so projecting onto the power subspace removes correlation
    exactly -- the property used to verify the mechanism.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
    scales = 1.0 + 0.5 * np.arange(n_loc)
    topos = (q[:, :n_loc] * scales).T.reshape(n_loc, n_channels, 1)
    return topos


def _lcmv_power(cov, g):
    """Unit-gain LCMV output power 1 / (g^T R^-1 g) for a single topography."""
    r_inv = np.linalg.pinv(cov)
    return (1.0 / (g.T @ r_inv @ g)).item()


# --------------------------------------------------------------------------- #
# 1. vec / unvec and the projector primitives.
# --------------------------------------------------------------------------- #
def test_vec_unvec_roundtrip():
    """unvec(vec(A)) recovers A for a non-symmetric matrix."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 7))
    assert_allclose(_unvec(_vec(a), 7), a)


def test_power_projector_is_orthogonal_projector():
    """P = U_K U_K^T is idempotent and symmetric (an orthogonal projector)."""
    topos = _orthogonal_topographies(8, 8, seed=1)
    g_pwr = _power_columns(topos)
    p, sv = _power_projector(g_pwr, rank=5)
    assert_allclose(p @ p, p, atol=1e-10)
    assert_allclose(p, p.T, atol=1e-10)
    # The projector rank equals the requested K (here below the column count).
    assert_allclose(np.trace(p), 5.0, atol=1e-8)
    assert sv.shape == (g_pwr.shape[1],)


def test_power_columns_shapes_fixed_and_free():
    """Power columns: 1 per location (fixed) and 3 per location (free)."""
    fixed = _orthogonal_topographies(6, 4, seed=2)
    assert _power_columns(fixed).shape == (36, 4)
    # Fake a free-orientation block with two tangential topographies.
    rng = np.random.default_rng(2)
    free = rng.standard_normal((4, 6, 2))
    assert _power_columns(free).shape == (36, 12)


def test_correlation_gram_symmetric_psd():
    """C_cor = G_cor G_cor^T is symmetric and positive semi-definite."""
    topos = _orthogonal_topographies(6, 6, seed=3)
    c_cor = _correlation_gram(topos)
    assert_allclose(c_cor, c_cor.T, atol=1e-10)
    assert np.linalg.eigvalsh(c_cor).min() > -1e-10


def test_apply_projector_returns_symmetric():
    """Projecting a symmetric covariance yields a symmetric matrix."""
    topos = _orthogonal_topographies(6, 6, seed=4)
    p, _ = _power_projector(_power_columns(topos), rank=10)
    rng = np.random.default_rng(4)
    a = rng.standard_normal((6, 6))
    cov = a @ a.T
    out = _apply_projector(p, cov)
    assert_allclose(out, out.T, atol=1e-10)


def test_spectral_flip_makes_pd_and_reports_energy():
    """Negative eigenvalues are flipped and their energy fraction is exact."""
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    eigs = np.array([3.0, 1.0, -2.0])  # |neg| / sum|.| = 2 / 6
    cov = q @ np.diag(eigs) @ q.T
    fixed, neg_energy = _spectral_flip(cov)
    assert_allclose(neg_energy, 2.0 / 6.0, atol=1e-10)
    assert np.linalg.eigvalsh(fixed).min() > 0
    # The flipped spectrum is the absolute value of the original spectrum.
    assert_allclose(np.sort(np.linalg.eigvalsh(fixed)), np.sort(np.abs(eigs)))


# --------------------------------------------------------------------------- #
# 2. The core mechanism: correlation removal for separable sources.
# --------------------------------------------------------------------------- #
def test_recipsiicos_removes_correlation_exactly():
    """With orthogonal topographies, ReciPSIICOS power is rho-independent.

    This is the defining property of the method: the cross-term of two
    orthogonal sources is Frobenius-orthogonal to the power subspace, so the
    projection removes it exactly and the modified covariance equals the
    uncorrelated one. The reconstructed target power is therefore flat in rho,
    whereas a plain LCMV would fall as ``1 - rho**2`` and vanish at rho = 1.
    """
    n_channels, n_loc = 10, 10
    topos = _orthogonal_topographies(n_channels, n_loc, seed=6)
    g_pwr = _power_columns(topos)
    projector, _ = _power_projector(g_pwr, rank=n_loc)
    g1 = topos[0, :, :]  # (n_channels, 1)
    g2 = topos[1, :, :]
    g_pair = np.hstack([g1, g2])

    powers = []
    for rho in (0.0, 0.5, 0.9, 0.99, 1.0):
        c_ss = np.array([[1.0, rho], [rho, 1.0]])
        r = g_pair @ c_ss @ g_pair.T + 0.01 * np.eye(n_channels)
        modified = _apply_projector(projector, r)
        modified, _ = _spectral_flip(modified)
        powers.append(_lcmv_power(modified, g1))

    # All reconstructed powers coincide with the uncorrelated (rho = 0) value.
    assert_allclose(powers, powers[0], rtol=1e-6)
    # And a plain beamformer on the same data would have cancelled the source.
    plain_full_corr = _lcmv_power(
        g_pair @ np.array([[1.0, 1.0], [1.0, 1.0]]) @ g_pair.T
        + 0.01 * np.eye(n_channels),
        g1,
    )
    assert plain_full_corr < 0.05 * powers[0]


def test_whitened_removes_correlation():
    """Whitened ReciPSIICOS also makes the target power rho-independent."""
    n_channels, n_loc = 10, 10
    topos = _orthogonal_topographies(n_channels, n_loc, seed=7)
    g_pwr = _power_columns(topos)
    c_cor = _correlation_gram(topos)
    projector, _ = _whitened_projector(g_pwr, c_cor, rank=4, reg=0.05)
    g1 = topos[0, :, :]
    g2 = topos[1, :, :]
    g_pair = np.hstack([g1, g2])

    powers = []
    for rho in (0.0, 0.5, 0.9):
        c_ss = np.array([[1.0, rho], [rho, 1.0]])
        r = g_pair @ c_ss @ g_pair.T + 0.01 * np.eye(n_channels)
        modified = _apply_projector(projector, r)
        modified, _ = _spectral_flip(modified)
        powers.append(_lcmv_power(modified, g1))
    assert_allclose(powers, powers[0], rtol=1e-6)


# --------------------------------------------------------------------------- #
# 3. Public API and integration with MNE's LCMV.
# --------------------------------------------------------------------------- #
def test_make_recipsiicos_cov_is_valid_covariance(fwd_fixed):
    """The modified covariance is PD, on the common channels, LCMV-ready."""
    info = mne.create_info(
        list(fwd_fixed["sol"]["row_names"]), 200.0, "eeg"
    )
    idx = [0, fwd_fixed["sol"]["data"].shape[1] // 2]
    data_cov = _cov_from_sources(fwd_fixed, info, idx, rho=0.9)
    cov = make_recipsiicos_cov(data_cov, fwd_fixed, rank=20, method="recipsiicos")
    assert isinstance(cov, mne.Covariance)
    assert cov.ch_names == list(fwd_fixed["sol"]["row_names"])
    assert np.linalg.eigvalsh(cov.data).min() > 0
    # The covariance is consumable by MNE's own make_lcmv.
    filters = mne.beamformer.make_lcmv(
        info, fwd_fixed, cov, reg=0.05, weight_norm="unit-noise-gain"
    )
    assert filters["weights"].shape[0] == fwd_fixed["sol"]["data"].shape[1]


def test_make_recipsiicos_cov_whitened_runs(fwd_fixed):
    """The whitened variant returns a valid PD covariance as well."""
    info = mne.create_info(list(fwd_fixed["sol"]["row_names"]), 200.0, "eeg")
    idx = [1, fwd_fixed["sol"]["data"].shape[1] - 2]
    data_cov = _cov_from_sources(fwd_fixed, info, idx, rho=0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # negative-energy warning is expected
        cov = make_recipsiicos_cov(
            data_cov, fwd_fixed, rank=10, method="whitened"
        )
    assert isinstance(cov, mne.Covariance)
    assert np.linalg.eigvalsh(cov.data).min() > 0


def test_make_recipsiicos_lcmv_returns_usable_beamformer(fwd_fixed):
    """The convenience wrapper returns a Beamformer that reconstructs power."""
    info = mne.create_info(list(fwd_fixed["sol"]["row_names"]), 200.0, "eeg")
    n_src = fwd_fixed["sol"]["data"].shape[1]
    idx = [0, n_src // 2]
    data_cov = _cov_from_sources(fwd_fixed, info, idx, rho=0.9)
    filters = make_recipsiicos_lcmv(
        info, fwd_fixed, data_cov, rank=20, method="recipsiicos"
    )
    assert filters["weights"].shape[0] == n_src
    stc = apply_lcmv_cov(data_cov, filters)
    assert stc.data.shape[0] == n_src
    assert np.all(np.isfinite(stc.data))
    assert np.all(stc.data >= 0)


# --------------------------------------------------------------------------- #
# 4. Rank-selection helper.
# --------------------------------------------------------------------------- #
def test_rank_curve_shapes_and_monotonicity(fwd_fixed):
    """Power retention is in [0, 1] and grows with rank for ReciPSIICOS."""
    ranks, p_pwr, p_cor = recipsiicos_rank_curve(
        fwd_fixed, method="recipsiicos"
    )
    msq = len(fwd_fixed["sol"]["row_names"]) ** 2
    assert ranks.shape == p_pwr.shape == p_cor.shape == (msq,)
    assert p_pwr.min() >= -1e-9 and p_pwr.max() <= 1 + 1e-9
    assert p_cor.min() >= -1e-9 and p_cor.max() <= 1 + 1e-9
    # Projecting onto more power directions can only retain more power energy.
    assert np.all(np.diff(p_pwr) >= -1e-9)
    # At full rank essentially all power-subspace energy is retained.
    assert p_pwr[-1] > 0.99


# --------------------------------------------------------------------------- #
# 5. Free-orientation handling.
# --------------------------------------------------------------------------- #
def test_free_orientation_smoke(fwd_info):
    """A free-orientation forward yields two tangential topographies and runs."""
    fwd, info = fwd_info
    topos = _local_topographies(fwd, list(info["ch_names"]))
    assert topos.shape[2] == 2  # two tangential orientations per location
    n_loc = fwd["sol"]["data"].shape[1] // 3  # 3 columns per location
    locs = [0, n_loc // 2]
    # Build a covariance directly from two oriented columns of the gain.
    gain = np.asarray(fwd["sol"]["data"])
    g = gain[:, [3 * locs[0], 3 * locs[1]]]
    r = g @ np.array([[1.0, 0.7], [0.7, 1.0]]) @ g.T + 0.01 * np.eye(g.shape[0])
    data_cov = mne.Covariance(r, list(fwd["sol"]["row_names"]), [], [], nfree=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = make_recipsiicos_cov(data_cov, fwd, rank=20, method="recipsiicos")
    assert np.linalg.eigvalsh(cov.data).min() > 0


# --------------------------------------------------------------------------- #
# 6. Error contract.
# --------------------------------------------------------------------------- #
def test_invalid_method_raises(fwd_fixed):
    """An unknown method name is rejected."""
    info = mne.create_info(list(fwd_fixed["sol"]["row_names"]), 200.0, "eeg")
    idx = [0, 1]
    data_cov = _cov_from_sources(fwd_fixed, info, idx, rho=0.0)
    with pytest.raises(ValueError, match="method"):
        make_recipsiicos_cov(data_cov, fwd_fixed, rank=5, method="nope")


def test_rank_too_small_raises(fwd_fixed):
    """A non-positive rank is rejected."""
    info = mne.create_info(list(fwd_fixed["sol"]["row_names"]), 200.0, "eeg")
    data_cov = _cov_from_sources(fwd_fixed, info, [0, 1], rho=0.0)
    with pytest.raises(ValueError, match="positive"):
        make_recipsiicos_cov(data_cov, fwd_fixed, rank=0)


def test_rank_too_large_raises(fwd_fixed):
    """A rank exceeding M**2 is rejected."""
    info = mne.create_info(list(fwd_fixed["sol"]["row_names"]), 200.0, "eeg")
    data_cov = _cov_from_sources(fwd_fixed, info, [0, 1], rho=0.0)
    msq = len(fwd_fixed["sol"]["row_names"]) ** 2
    with pytest.raises(ValueError, match="exceeds"):
        make_recipsiicos_cov(data_cov, fwd_fixed, rank=msq + 1)


def test_non_covariance_input_raises(fwd_fixed):
    """Passing a bare array instead of a Covariance is rejected."""
    with pytest.raises(TypeError):
        make_recipsiicos_cov(np.eye(8), fwd_fixed, rank=5)


def test_no_common_channels_raises(fwd_fixed):
    """A covariance sharing no channels with the forward is rejected."""
    bad = mne.Covariance(
        np.eye(3), ["X1", "X2", "X3"], [], [], nfree=1
    )
    with pytest.raises(ValueError, match="common"):
        make_recipsiicos_cov(bad, fwd_fixed, rank=5)
