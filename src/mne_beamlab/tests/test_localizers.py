"""Tests for the MCMV localizers and data-driven orientation."""

# Authors: the mne-beamlab contributors
# License: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_beamlab._localizers import localizer_value, optimal_orientation


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
