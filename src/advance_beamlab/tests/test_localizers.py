"""Tests for the MCMV localizers and data-driven orientation."""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

from copy import deepcopy

import mne
import numpy as np
import pytest
from mne._fiff.proj import make_eeg_average_ref_proj
from mne.utils import catch_logging
from numpy.testing import assert_allclose

from advance_beamlab import MCMVBeamformer, apply_mcmv, scan_mcmv
from advance_beamlab._localizers import (
    _metric_matrices,
    localizer_value,
    optimal_orientation,
)
from advance_beamlab._mcmv import _cov_as_matrix

mne.set_log_level("ERROR")


# --------------------------------------------------------------------------- #
# Helpers: random well-conditioned SPD matrices and leadfields.
# --------------------------------------------------------------------------- #
def _spd(n, rng, scale=1.0):
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T) + n * np.eye(n)


def _setup(n_ch=16, n_src=1, seed=0, evoked=False):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n_ch, n_src))
    R = _spd(n_ch, rng)
    N = _spd(n_ch, rng, scale=0.3)
    Rbar = _spd(n_ch, rng, scale=0.5) if evoked else None
    return H, R, N, Rbar, rng


# --------------------------------------------------------------------------- #
# 1. Single-source reductions to the published scalar ratios.
# --------------------------------------------------------------------------- #
def test_mai_reduces_to_activity_index_minus_one():
    """MAI at n == 1 equals zeta - 1 = (h^T N^-1 h)/(h^T R^-1 h) - 1."""
    H, R, N, _, _ = _setup(n_src=1, seed=1)
    h = H[:, 0]
    Rinv, Ninv = np.linalg.inv(R), np.linalg.inv(N)
    zeta = (h @ Ninv @ h) / (h @ Rinv @ h)
    assert_allclose(localizer_value("mai", H, R, N), zeta - 1.0, atol=1e-10)


def test_mpz_reduces_to_pseudo_z_minus_one():
    """MPZ at n == 1 equals Z - 1 = (h^T R^-1 h)/(h^T R^-1 N R^-1 h) - 1."""
    H, R, N, _, _ = _setup(n_src=1, seed=2)
    h = H[:, 0]
    Rinv = np.linalg.inv(R)
    z = (h @ Rinv @ h) / (h @ Rinv @ N @ Rinv @ h)
    assert_allclose(localizer_value("mpz", H, R, N), z - 1.0, atol=1e-10)


def test_mer_and_rmer_reduce_to_evoked_ratios():
    """MER/rMER at n == 1 equal the evoked ratios of Moiseev Table 1."""
    H, R, N, Rbar, _ = _setup(n_src=1, seed=3, evoked=True)
    h = H[:, 0]
    Rinv = np.linalg.inv(R)
    num = h @ Rinv @ Rbar @ Rinv @ h
    mer = num / (h @ Rinv @ N @ Rinv @ h)
    rmer = num / (h @ Rinv @ h)
    assert_allclose(localizer_value("mer", H, R, N, evoked_cov=Rbar), mer, atol=1e-10)
    assert_allclose(localizer_value("rmer", H, R, N, evoked_cov=Rbar), rmer, atol=1e-10)


# --------------------------------------------------------------------------- #
# 2. Proven properties: non-negativity and MAI >= MPZ.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_power_localizers_nonnegative_and_ordered(seed):
    """MAI, MPZ >= 0 and MAI >= MPZ for a valid data covariance R = N + signal.

    The Appendix-A proofs assume R is a genuine data covariance built on top of
    the noise floor N (so R - N is PSD); with independent random R and N the
    property need not hold. We therefore construct R = N + H0 C H0^T.
    """
    rng = np.random.default_rng(seed)
    n_ch = 20
    N = _spd(n_ch, rng, scale=0.3)
    G0 = rng.standard_normal((n_ch, 2))  # true-source leadfields
    C = _spd(2, rng)  # PSD source covariance
    R = N + G0 @ C @ G0.T  # a physically valid data covariance (R >= N)
    H = rng.standard_normal((n_ch, 3))  # trial sources

    p_mai = localizer_value("mai", H, R, N)
    p_mpz = localizer_value("mpz", H, R, N)
    assert p_mai >= -1e-9
    assert p_mpz >= -1e-9
    assert p_mai >= p_mpz - 1e-9


# --------------------------------------------------------------------------- #
# 3. Whitening invariance: the localizers are unchanged by an invertible W.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["mai", "mpz", "mer", "rmer"])
def test_localizer_invariant_to_whitening(name):
    """S, G, T, E are invariant under x -> W x, hence so is every localizer."""
    H, R, N, Rbar, rng = _setup(n_ch=14, n_src=2, seed=4, evoked=True)
    W = rng.standard_normal((14, 14))  # a generic invertible transform
    Hw, Rw, Nw = W @ H, W @ R @ W.T, W @ N @ W.T
    Rbw = W @ Rbar @ W.T
    v0 = localizer_value(name, H, R, N, evoked_cov=Rbar)
    v1 = localizer_value(name, Hw, Rw, Nw, evoked_cov=Rbw)
    assert_allclose(v1, v0, rtol=1e-7, atol=1e-9)


# --------------------------------------------------------------------------- #
# 4. Data-driven orientation actually maximises the localizer.
# --------------------------------------------------------------------------- #
def _brute_force_max(name, H_ref, H_loc, R, N, evoked_cov, rng, n=4000):
    """Largest localizer value over many random unit orientations."""
    best = -np.inf
    for _ in range(n):
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        H = (
            np.column_stack([H_ref, H_loc @ u])
            if H_ref.shape[1]
            else ((H_loc @ u)[:, None])
        )
        best = max(best, localizer_value(name, H, R, N, evoked_cov=evoked_cov))
    return best


@pytest.mark.parametrize("name", ["mai", "mpz"])
def test_orientation_maximises_single_source(name):
    """First-source orientation beats every sampled orientation."""
    rng = np.random.default_rng(5)
    n_ch = 18
    H_loc = rng.standard_normal((n_ch, 3))
    R, N = _spd(n_ch, rng), _spd(n_ch, rng, scale=0.3)
    H_ref = np.empty((n_ch, 0))

    u = optimal_orientation(name, H_ref, H_loc, R, N)
    p_opt = localizer_value(name, (H_loc @ u)[:, None], R, N)
    p_brute = _brute_force_max(name, H_ref, H_loc, R, N, None, rng)
    assert p_opt >= p_brute - 1e-9


@pytest.mark.parametrize("name", ["mai", "mpz"])
def test_orientation_maximises_with_reference(name):
    """Second-source orientation (Eqs. 13-14) beats every sampled orientation."""
    rng = np.random.default_rng(6)
    n_ch = 18
    H_ref = rng.standard_normal((n_ch, 1))  # one already-found source
    H_loc = rng.standard_normal((n_ch, 3))
    R, N = _spd(n_ch, rng), _spd(n_ch, rng, scale=0.3)

    u = optimal_orientation(name, H_ref, H_loc, R, N)
    H_full = np.column_stack([H_ref, H_loc @ u])
    p_opt = localizer_value(name, H_full, R, N)
    p_brute = _brute_force_max(name, H_ref, H_loc, R, N, None, rng)
    assert p_opt >= p_brute - 1e-9


# --------------------------------------------------------------------------- #
# 5. Error contract.
# --------------------------------------------------------------------------- #
def test_unknown_localizer_raises():
    H, R, N, _, _ = _setup()
    with pytest.raises(ValueError, match="localizer must be one of"):
        localizer_value("bogus", H, R, N)


def test_event_related_requires_evoked_cov():
    H, R, N, _, _ = _setup()
    with pytest.raises(ValueError, match="event-related"):
        localizer_value("mer", H, R, N)


def test_orientation_requires_three_columns():
    rng = np.random.default_rng(7)
    n_ch = 12
    R, N = _spd(n_ch, rng), _spd(n_ch, rng, scale=0.3)
    H_loc2 = rng.standard_normal((n_ch, 2))  # only 2 columns, not x/y/z
    with pytest.raises(ValueError, match="3 columns"):
        optimal_orientation("mai", np.empty((n_ch, 0)), H_loc2, R, N)


# --------------------------------------------------------------------------- #
# 6. Sequential source search: recover implanted sources on a sphere forward.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sphere_fwd():
    """A small free-orientation EEG forward on a sphere + its Info."""
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:32], 200.0, "eeg")
    info.set_montage("standard_1020")
    # The EEG forward is computed against an average reference, so the recording
    # must carry the average-reference projector -- MNE's inverse machinery
    # (``_check_reference``) refuses EEG data without it.
    with info._unlock():
        info["projs"] = [make_eeg_average_ref_proj(info, verbose=False)]
    sphere = mne.make_sphere_model("auto", "auto", info)
    src = mne.setup_volume_source_space(sphere=sphere, pos=30.0)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=False, surf_ori=False)
    return fwd, info


def _implanted_cov(fwd, info, locs, oris, corr=0.0, snr=9.0):
    """Data covariance R = N + scale * H C H^T for implanted sources (N = I)."""
    G = fwd["sol"]["data"]
    cols = [
        G[:, 3 * loc : 3 * loc + 3] @ (u / np.linalg.norm(u))
        for loc, u in zip(locs, oris, strict=True)
    ]
    H = np.column_stack(cols)
    k = len(locs)
    C = np.full((k, k), corr) + (1.0 - corr) * np.eye(k)  # pairwise correlation
    N = np.eye(G.shape[0])
    sig = H @ C @ H.T
    R = N + snr * np.trace(N) / np.trace(sig) * sig  # target rough SNR
    cov = mne.Covariance(
        R, info["ch_names"], info["bads"], list(info["projs"]), nfree=1
    )
    return cov


@pytest.mark.parametrize("localizer", ["mai", "mpz"])
def test_scan_recovers_single_source(sphere_fwd, localizer):
    """A single implanted source is recovered exactly (it lies on the grid)."""
    fwd, info = sphere_fwd
    true_loc, true_ori = 26, np.array([1.0, -1.0, 0.5])
    cov = _implanted_cov(fwd, info, [true_loc], [true_ori])

    res = scan_mcmv(info, fwd, cov, localizer=localizer, n_sources=1)
    assert res["sources"] == [true_loc]
    # Orientation is recovered up to sign.
    cos = np.abs(res["orientations"][0] @ (true_ori / np.linalg.norm(true_ori)))
    assert cos > 0.99


def test_scan_recovers_two_correlated_sources(sphere_fwd):
    """Two correlated sources are both recovered by the sequential search."""
    fwd, info = sphere_fwd
    locs = [10, 45]
    oris = [np.array([1.0, 0.0, 0.3]), np.array([0.0, 1.0, -0.2])]
    cov = _implanted_cov(fwd, info, locs, oris, corr=0.8, snr=16.0)

    res = scan_mcmv(info, fwd, cov, localizer="mpz", n_sources=2)
    assert set(res["sources"]) == set(locs)


def test_scan_result_is_usable(sphere_fwd):
    """The result exposes filters, pseudo-Z and maps of the right shapes."""
    fwd, info = sphere_fwd
    cov = _implanted_cov(fwd, info, [15, 33], [[1, 0, 0], [0, 0, 1]], corr=0.5)

    res = scan_mcmv(info, fwd, cov, localizer="mai", n_sources=2)
    assert isinstance(res, dict)
    assert isinstance(res["filters"], MCMVBeamformer)
    assert res["pseudo_z"].shape == (2,)
    assert len(res["maps"]) == 2 and res["maps"][0].shape == (fwd["nsource"],)
    # The jointly-optimal filters apply to sensor data.
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((len(res["filters"]["ch_names"]), 40))
    assert apply_mcmv(arr, res["filters"]).shape == (2, 40)


def test_scan_rejects_bad_n_sources(sphere_fwd):
    fwd, info = sphere_fwd
    cov = _implanted_cov(fwd, info, [20], [[1, 0, 0]])
    with pytest.raises(ValueError, match="n_sources must be >= 1"):
        scan_mcmv(info, fwd, cov, n_sources=0)
    with pytest.raises(ValueError, match="exceeds the number of grid"):
        scan_mcmv(info, fwd, cov, n_sources=fwd["nsource"] + 1)


# --------------------------------------------------------------------------- #
# 7. Coverage: repr, fixed-orientation scan, event-related scan, noise_cov.
# --------------------------------------------------------------------------- #
def test_scan_result_repr(sphere_fwd):
    """MCMVScanResult has an informative repr."""
    fwd, info = sphere_fwd
    cov = _implanted_cov(fwd, info, [22], [[1, 0, 0]])
    res = scan_mcmv(info, fwd, cov, localizer="mai", n_sources=1)
    assert "MCMVScanResult" in repr(res) and "MAI" in repr(res)


def test_scan_fixed_orientation(sphere_fwd):
    """Scanning a fixed-orientation forward recovers the source, ori is None."""
    fwd, info = sphere_fwd
    fwd_fix = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
    G = fwd_fix["sol"]["data"]  # (n_ch, n_loc), one column per location
    loc = 18
    h = G[:, loc]
    N = np.eye(G.shape[0])
    R = N + 12.0 * np.outer(h, h) / (h @ h)
    cov = mne.Covariance(R, info["ch_names"], [], list(info["projs"]), nfree=1)

    res = scan_mcmv(info, fwd_fix, cov, localizer="mai", n_sources=1)
    assert res["sources"] == [loc]
    assert res["orientations"] is None


def test_scan_event_related_with_diagonal_evoked_cov(sphere_fwd):
    """MER scan runs with an evoked covariance, incl. the diagonal-cov path."""
    fwd, info = sphere_fwd
    loc, ori = 30, np.array([0.0, 1.0, 0.0])
    cov = _implanted_cov(fwd, info, [loc], [ori], snr=16.0)
    # A diagonal evoked covariance exercises the 1-D covariance branch.
    evoked_cov = mne.Covariance(
        np.ones(len(info["ch_names"])), info["ch_names"], [], [], nfree=1
    )
    res = scan_mcmv(info, fwd, cov, localizer="mer", n_sources=1, evoked_cov=evoked_cov)
    assert len(res["sources"]) == 1


def test_scan_noise_cov_subset(sphere_fwd):
    """Scan restricts to noise_cov channels when it covers a subset."""
    fwd, info = sphere_fwd
    cov = _implanted_cov(fwd, info, [12], [[1, 0, 0]])
    sub = info["ch_names"][:24]
    ncov = mne.Covariance(np.eye(24), sub, [], [], nfree=1)
    res = scan_mcmv(info, fwd, cov, localizer="mai", n_sources=1, noise_cov=ncov)
    assert res["filters"]["ch_names"] == sub


def test_scan_noise_cov_disjoint_raises(sphere_fwd):
    """A disjoint noise_cov is an error in the scan too."""
    fwd, info = sphere_fwd
    cov = _implanted_cov(fwd, info, [12], [[1, 0, 0]])
    ncov = mne.Covariance(np.eye(2), ["ZZ1", "ZZ2"], [], [], nfree=1)
    with pytest.raises(ValueError, match="shares no channels"):
        scan_mcmv(info, fwd, cov, n_sources=1, noise_cov=ncov)


def _evoked_cov(fwd, info, locs, oris, snr=16.0, n_ave=50):
    """Covariance of the epoch-averaged field of the implanted sources.

    ``evoked_cov`` is the covariance of the *averaged* field, so it must be
    built from the same sources as the data covariance -- the evoked topography
    plus sensor noise attenuated by the number of averaged epochs. A covariance
    unrelated to the implanted sources describes a different evoked response,
    and the event-related localizers correctly peak wherever *that* response is
    strongest instead.
    """
    G = fwd["sol"]["data"]
    H = np.column_stack(
        [
            G[:, 3 * loc : 3 * loc + 3] @ (np.asarray(u, float) / np.linalg.norm(u))
            for loc, u in zip(locs, oris, strict=True)
        ]
    )
    n_ch = G.shape[0]
    sig = H @ H.T
    Rbar = snr * n_ch / np.trace(sig) * sig + np.eye(n_ch) / n_ave
    return mne.Covariance(Rbar, info["ch_names"], [], list(info["projs"]), nfree=1)


@pytest.mark.parametrize("localizer", ["mer", "rmer"])
def test_scan_mer_with_full_evoked_cov(sphere_fwd, localizer):
    """The event-related localizers recover the source of the evoked field.

    Uses a dense (2-D) evoked covariance, exercising that branch as well.
    """
    fwd, info = sphere_fwd
    loc, ori = 28, np.array([1.0, 0.0, 0.0])
    cov = _implanted_cov(fwd, info, [loc], [ori], snr=16.0)
    evoked_cov = _evoked_cov(fwd, info, [loc], [ori])
    assert evoked_cov.data.ndim == 2  # the dense-covariance path

    res = scan_mcmv(
        info, fwd, cov, localizer=localizer, n_sources=1, evoked_cov=evoked_cov
    )
    assert res["sources"] == [loc]
    cos = np.abs(res["orientations"][0] @ (ori / np.linalg.norm(ori)))
    assert cos > 0.99


# --------------------------------------------------------------------------- #
# 8. Cached metric matrices: identical results, one inversion per scan.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["mai", "mpz", "mer", "rmer"])
def test_cached_metrics_match_recomputed(name):
    """Passing pre-computed ``metrics`` gives bit-comparable results."""
    H, R, N, Rbar, rng = _setup(n_ch=15, n_src=2, seed=8, evoked=True)
    metrics = _metric_matrices(R, N, Rbar)

    v0 = localizer_value(name, H, R, N, evoked_cov=Rbar)
    v1 = localizer_value(name, H, R, N, evoked_cov=Rbar, metrics=metrics)
    assert_allclose(v1, v0, rtol=1e-12, atol=0)

    H_ref, H_loc = H[:, :1], rng.standard_normal((15, 3))
    u0 = optimal_orientation(name, H_ref, H_loc, R, N, evoked_cov=Rbar)
    u1 = optimal_orientation(name, H_ref, H_loc, R, N, evoked_cov=Rbar, metrics=metrics)
    assert_allclose(u1, u0, rtol=1e-12, atol=0)


def test_cached_metrics_are_used_verbatim():
    """``metrics`` short-circuits the inversions rather than being merged in."""
    H, R, N, _, _ = _setup(n_ch=10, n_src=1, seed=9)
    # A deliberately wrong cache: if it were ignored, the value would not change.
    metrics = _metric_matrices(2.0 * R, N)
    v_plain = localizer_value("mai", H, R, N)
    v_cached = localizer_value("mai", H, R, N, metrics=metrics)
    assert not np.isclose(v_plain, v_cached)
    assert_allclose(v_cached, localizer_value("mai", H, 2.0 * R, N), rtol=1e-12)


# --------------------------------------------------------------------------- #
# 9. Rank handling: the scan must use the same regularised, rank-truncated
#    inverse as the filters it returns.
# --------------------------------------------------------------------------- #
def _rank_deficient_cov(fwd, info, loc, ori, n_null=6, snr=16.0, seed=0):
    """Data covariance whose rank deficiency is invisible to the whitener.

    Mimics ICA/SSP-cleaned data whose projectors never reached ``info['projs']``:
    ``n_null`` spatial directions -- chosen orthogonal to the source topography,
    so the source itself survives intact -- are removed from the covariance while
    the noise covariance stays full rank.
    """
    G = fwd["sol"]["data"]
    n_ch = G.shape[0]
    h = G[:, 3 * loc : 3 * loc + 3] @ (np.asarray(ori, float) / np.linalg.norm(ori))
    sig = np.outer(h, h)
    R = np.eye(n_ch) + snr * n_ch / np.trace(sig) * sig
    V = np.random.default_rng(seed).standard_normal((n_ch, n_null))
    q = h / np.linalg.norm(h)
    V -= np.outer(q, q @ V)  # keep the source out of the removed subspace
    V = np.linalg.qr(V)[0]
    P = np.eye(n_ch) - V @ V.T
    projs = list(info["projs"])
    data_cov = mne.Covariance(P @ R @ P, info["ch_names"], [], projs, nfree=1)
    noise_cov = mne.Covariance(np.eye(n_ch), info["ch_names"], [], projs, nfree=1)
    return data_cov, noise_cov


def test_scan_inverse_agrees_with_returned_filters(sphere_fwd):
    """Scan pseudo-Z equals the pseudo-Z of the filter row the scan returns.

    The two are equal only if the scan inverts the whitened data covariance the
    way :func:`make_mcmv` does -- regularised *and* rank-truncated. With a plain
    ``inv(R_w + loading * I)`` the near-null directions survive and the two
    disagree on a rank-deficient covariance.
    """
    fwd, info = sphere_fwd
    loc, ori = 26, np.array([1.0, -1.0, 0.5])
    data_cov, noise_cov = _rank_deficient_cov(fwd, info, loc, ori)

    with pytest.warns(RuntimeWarning, match="rank-deficient"):
        res = scan_mcmv(
            info, fwd, data_cov, localizer="mai", n_sources=1, noise_cov=noise_cov
        )
    assert res["sources"] == [loc]

    ch = res["filters"]["ch_names"]
    R = _cov_as_matrix(data_cov, ch)
    N = _cov_as_matrix(noise_cov, ch)
    w = res["filters"]["weights"][-1]
    assert_allclose(res["pseudo_z"][-1], (w @ R @ w) / (w @ N @ w), rtol=1e-9)


def test_scan_rejects_more_sources_than_data_rank(sphere_fwd):
    """``n_sources`` above the data-covariance rank is refused up front."""
    fwd, info = sphere_fwd
    loc, ori = 26, np.array([1.0, -1.0, 0.5])
    data_cov, noise_cov = _rank_deficient_cov(fwd, info, loc, ori, n_null=24)
    with pytest.raises(ValueError, match=r"n_sources=20 exceeds the data-covariance"):
        scan_mcmv(info, fwd, data_cov, n_sources=20, noise_cov=noise_cov)


def test_scan_rank_conventions(sphere_fwd):
    """``rank`` is translated for each consumer, and follows MNE's convention.

    ``'info'`` is valid for the whitener but not for the covariance inverse, so
    it only works if the two conventions are split rather than passed through.
    """
    fwd, info = sphere_fwd
    loc = 12
    cov = _implanted_cov(fwd, info, [loc], [[1, 0, 0]], snr=16.0)
    assert scan_mcmv(info, fwd, cov, n_sources=1, rank="info")["sources"] == [loc]
    with pytest.raises(TypeError, match="rank must be an instance of"):
        scan_mcmv(info, fwd, cov, n_sources=1, rank=5)


def test_scan_validates_reg_before_scanning(sphere_fwd, monkeypatch):
    """``reg`` is checked before the grid scan, not after it."""
    fwd, info = sphere_fwd
    cov = _implanted_cov(fwd, info, [12], [[1, 0, 0]])

    def _boom(*args, **kwargs):
        raise AssertionError("the grid scan must not start with an invalid reg")

    monkeypatch.setattr("advance_beamlab._localizers.optimal_orientation", _boom)
    with pytest.raises(ValueError, match=r"reg must be in \[0, 1\]"):
        scan_mcmv(info, fwd, cov, n_sources=1, reg=1.5)


# --------------------------------------------------------------------------- #
# 10. Degenerate grid locations are skipped, never selected.
# --------------------------------------------------------------------------- #
def test_scan_skips_location_with_nonfinite_leadfield(sphere_fwd):
    """A location whose leadfield block is non-finite is skipped, not fatal.

    ``scipy.linalg.eigh`` reports this as a plain ``ValueError`` rather than a
    ``LinAlgError``, so it is only survivable if both are caught.
    """
    fwd, info = sphere_fwd
    fwd = deepcopy(fwd)
    loc, ori, bad = 26, np.array([1.0, -1.0, 0.5]), 5
    cov = _implanted_cov(fwd, info, [loc], [ori], snr=16.0)
    fwd["sol"]["data"][:, 3 * bad : 3 * bad + 3] = np.nan

    res = scan_mcmv(info, fwd, cov, localizer="mai", n_sources=1)
    assert res["sources"] == [loc]
    assert np.isnan(res["maps"][0][bad])
    assert np.isfinite(res["maps"][0][loc])


def test_scan_ignores_nan_localizer_values(sphere_fwd):
    """A location evaluating to NaN never wins the argmax.

    ``numpy.linalg.solve`` propagates NaN silently instead of raising, so the
    fixed-orientation path can produce a NaN localizer value; ``np.argmax`` would
    return that location.
    """
    fwd, info = sphere_fwd
    fwd_fix = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
    fwd_fix = deepcopy(fwd_fix)
    G = fwd_fix["sol"]["data"]
    loc, bad = 18, 40
    h = G[:, loc]
    n_ch = G.shape[0]
    R = np.eye(n_ch) + 12.0 * np.outer(h, h) / (h @ h)
    cov = mne.Covariance(R, info["ch_names"], [], list(info["projs"]), nfree=1)
    fwd_fix["sol"]["data"][:, bad] = np.nan

    res = scan_mcmv(info, fwd_fix, cov, localizer="mai", n_sources=1)
    assert res["sources"] == [loc]
    assert np.isnan(res["maps"][0][bad])


# --------------------------------------------------------------------------- #
# 11. ``verbose`` is honoured (the decorator is present).
# --------------------------------------------------------------------------- #
def test_scan_verbose_controls_logging(sphere_fwd):
    """verbose='debug' emits the per-iteration trace; the default does not."""
    fwd, info = sphere_fwd
    cov = _implanted_cov(fwd, info, [12], [[1, 0, 0]])

    with catch_logging() as log:
        scan_mcmv(info, fwd, cov, localizer="mai", n_sources=1, verbose="debug")
    assert "Iteration 1/1" in log.getvalue()

    with catch_logging() as log:
        scan_mcmv(info, fwd, cov, localizer="mai", n_sources=1, verbose="error")
    assert "Iteration 1/1" not in log.getvalue()


# --------------------------------------------------------------------------- #
# 12. Orientations come back in head coordinates for either representation.
# --------------------------------------------------------------------------- #
def _rotate_to_local_frame(fwd, seed=0):
    """Re-express a free-orientation forward in a random per-source local frame.

    Returns a physically identical forward with ``surf_ori=True``: each source's
    three gain columns are expressed in a local frame whose axes, in head
    coordinates, are the rows of that source's ``source_nn`` block. This is the
    representation :func:`mne.convert_forward_solution` produces for a surface
    source space.
    """
    rng = np.random.default_rng(seed)
    fwd = deepcopy(fwd)
    G = fwd["sol"]["data"]
    nn = np.asarray(fwd["source_nn"], dtype=np.float64).copy()
    for s in range(fwd["nsource"]):
        Q = np.linalg.qr(rng.standard_normal((3, 3)))[0]  # rows = local axes
        G[:, 3 * s : 3 * s + 3] = G[:, 3 * s : 3 * s + 3] @ Q.T
        nn[3 * s : 3 * s + 3] = Q
    fwd["source_nn"] = nn
    fwd["surf_ori"] = True
    return fwd


def test_scan_returns_head_coordinate_orientations(sphere_fwd):
    """The same physical forward gives the same orientation in either frame."""
    fwd, info = sphere_fwd
    loc, ori = 26, np.array([1.0, -1.0, 0.5])
    cov = _implanted_cov(fwd, info, [loc], [ori], snr=16.0)
    fwd_rot = _rotate_to_local_frame(fwd)

    res = scan_mcmv(info, fwd, cov, localizer="mai", n_sources=1)
    res_rot = scan_mcmv(info, fwd_rot, cov, localizer="mai", n_sources=1)

    assert res_rot["sources"] == res["sources"] == [loc]
    # Head-coordinate orientations agree up to the eigenvector sign.
    assert_allclose(
        np.abs(res_rot["orientations"][0] @ res["orientations"][0]), 1.0, atol=1e-6
    )
    # ... and therefore so do the filters built from them.
    assert_allclose(
        np.abs(res_rot["filters"]["weights"]),
        np.abs(res["filters"]["weights"]),
        rtol=1e-6,
    )


# --------------------------------------------------------------------------- #
# 13. What each reported quantity is measured on.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("localizer", ["mer", "rmer"])
def test_event_related_scan_recovers_both_correlated_sources(sphere_fwd, localizer):
    """A correlated pair driving one evoked response is recovered in full.

    The event-related localizers reach the averaged field only through the
    Table-2 matrix E, in which the averaged-field covariance is weighted by the
    data-covariance inverse on *both* sides; that two-sided form is what makes E
    the evoked power seen through the beamformer rather than raw evoked power.
    A single strong source is found whichever way the three factors are
    arranged, because the peak stays put; a correlated second source is not. Get
    E wrong and the scan still returns the requested number of sources -- with
    the second one sitting somewhere no source was ever implanted, and nothing
    in the result to say so.
    """
    fwd, info = sphere_fwd
    locs = [10, 45]
    oris = [np.array([1.0, 0.0, 0.3]), np.array([0.0, 1.0, -0.2])]
    cov = _implanted_cov(fwd, info, locs, oris, corr=0.8, snr=16.0)
    evoked_cov = _evoked_cov(fwd, info, locs, oris)

    res = scan_mcmv(
        info, fwd, cov, localizer=localizer, n_sources=2, evoked_cov=evoked_cov
    )
    assert res["sources"] == locs
    for ori, u in zip(oris, res["orientations"], strict=True):
        assert np.abs(u @ (ori / np.linalg.norm(ori))) > 0.95


def test_last_pseudo_z_is_measured_on_the_newest_filter_row(sphere_fwd):
    """Each pseudo-Z belongs to the source its own iteration added.

    Every row of a joint MCMV filter has its own pseudo-Z, and the rows differ
    substantially: an early row has been sharpened by the constraints added
    after it. ``pseudo_z[k]`` is what the user reads to judge whether the k-th
    source is genuine, so it has to be measured on that source's row -- the row
    the k-th iteration just added -- of the k+1-source filter. Measured on an
    older row it would keep re-reporting the strength of a source already
    accepted, and every further source would look as convincing as the first.
    """
    fwd, info = sphere_fwd
    locs = [10, 45]
    oris = [np.array([1.0, 0.0, 0.3]), np.array([0.0, 1.0, -0.2])]
    cov = _implanted_cov(fwd, info, locs, oris, corr=0.8, snr=16.0)
    # An explicit unit noise covariance, so the pseudo-Z of a filter row is
    # exactly (w R w) / (w w) in sensor space.
    ncov = mne.Covariance(
        np.eye(len(info["ch_names"])),
        info["ch_names"],
        [],
        list(info["projs"]),
        nfree=1,
    )

    res = scan_mcmv(info, fwd, cov, localizer="mpz", n_sources=2, noise_cov=ncov)
    assert res["sources"] == locs

    ch = res["filters"]["ch_names"]
    R, N = _cov_as_matrix(cov, ch), _cov_as_matrix(ncov, ch)
    z = np.array([(w @ R @ w) / (w @ N @ w) for w in res["filters"]["weights"]])
    assert_allclose(res["pseudo_z"][-1], z[-1], rtol=1e-9)
    # The rows are far apart here, so the reported value does identify its row.
    assert z[0] > 2.0 * z[-1]


def test_scan_raises_when_no_location_can_be_evaluated(sphere_fwd):
    """A grid on which nothing can be evaluated is reported, not guessed at.

    Skipping a degenerate location is safe only while some other location
    survives. When none does -- a corrupted forward, a covariance whose inverse
    has lost every direction -- the candidate values are all still at the
    placeholder the scan starts them from. Those placeholders are not source
    strengths, so no location may be chosen from them: the user must be told the
    scan failed rather than handed grid location 0 with a pseudo-Z attached to
    it.
    """
    fwd, info = sphere_fwd
    fwd = deepcopy(fwd)
    cov = _implanted_cov(fwd, info, [26], [[1.0, -1.0, 0.5]], snr=16.0)
    fwd["sol"]["data"][:] = np.nan  # every location is now degenerate

    with pytest.raises(RuntimeError, match="No valid source location"):
        scan_mcmv(info, fwd, cov, localizer="mai", n_sources=1)


@pytest.mark.parametrize("name", ["mai", "mpz", "mer", "rmer"])
def test_orientation_dominant_component_is_never_negative(name):
    """The orientation's sign follows a fixed convention, not the eigensolver.

    An eigenvector is defined only up to sign and every localizer is even in the
    orientation, so LAPACK may return either of the two. The convention settles
    it: the largest component of the returned unit vector is positive. Without
    it the polarity of the reconstructed time course, and the direction of a
    dipole drawn from ``orientations``, would flip from one location, run or
    machine to the next with nothing in the data behind the change -- and source
    estimates that ought to be identical could no longer be averaged, subtracted
    or compared.
    """
    H, R, N, Rbar, rng = _setup(n_ch=16, n_src=1, seed=13, evoked=True)
    for _ in range(20):
        H_loc = rng.standard_normal((16, 3))
        for n_ref in (0, 1):  # first source, then one already-found reference
            u = optimal_orientation(name, H[:, :n_ref], H_loc, R, N, evoked_cov=Rbar)
            assert_allclose(np.linalg.norm(u), 1.0, atol=1e-12)
            # A unit 3-vector's dominant component has magnitude >= 1/sqrt(3),
            # so the convention puts it comfortably above zero.
            assert u[np.argmax(np.abs(u))] > 0.5
