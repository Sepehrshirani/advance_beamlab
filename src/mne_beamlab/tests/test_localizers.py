"""Tests for the MCMV localizers and data-driven orientation."""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_beamlab import MCMVBeamformer, apply_mcmv, scan_mcmv
from mne_beamlab._localizers import localizer_value, optimal_orientation

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
    assert_allclose(
        localizer_value("rmer", H, R, N, evoked_cov=Rbar), rmer, atol=1e-10
    )


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
        H = np.column_stack([H_ref, H_loc @ u]) if H_ref.shape[1] else (
            (H_loc @ u)[:, None]
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
    res = scan_mcmv(
        info, fwd, cov, localizer="mer", n_sources=1, evoked_cov=evoked_cov
    )
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


def test_scan_mer_with_full_evoked_cov(sphere_fwd):
    """MER scan with a dense (2-D) evoked covariance."""
    fwd, info = sphere_fwd
    loc, ori = 28, np.array([1.0, 0.0, 0.0])
    cov = _implanted_cov(fwd, info, [loc], [ori], snr=16.0)
    rng = np.random.default_rng(9)
    A = rng.standard_normal((len(info["ch_names"]), 3))
    Rbar = A @ A.T + np.eye(len(info["ch_names"]))  # dense evoked covariance
    evoked_cov = mne.Covariance(Rbar, info["ch_names"], [], [], nfree=1)
    res = scan_mcmv(
        info, fwd, cov, localizer="mer", n_sources=1, evoked_cov=evoked_cov
    )
    assert len(res["sources"]) == 1
