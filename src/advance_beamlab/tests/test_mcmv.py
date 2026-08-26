"""Tests for the Multiple Constrained Minimum Variance (MCMV) beamformer."""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import mne
import numpy as np
import pytest
from mne.beamformer import make_lcmv
from mne.beamformer._compute_beamformer import _reg_pinv
from numpy.testing import assert_allclose

from advance_beamlab import (
    MCMVBeamformer,
    apply_mcmv,
    apply_mcmv_cov,
    make_mcmv,
)
from advance_beamlab._mcmv import (
    _check_source_separation,
    _compute_mcmv_weights,
    _make_whitener,
)

mne.set_log_level("ERROR")
REG = 0.05


# --------------------------------------------------------------------------- #
# Fixtures: a tiny, fully analytic (offline) EEG forward solution.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fwd_info():
    """A small free-orientation EEG forward on a sphere + its Info.

    The Info carries an average EEG reference projector, which MNE requires for
    inverse modelling (the forward is computed against one) and which
    :func:`make_mcmv` enforces just as :func:`mne.beamformer.make_lcmv` does.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:32], 200.0, "eeg")
    info.set_montage("standard_1020")
    info = (
        mne.io.RawArray(np.zeros((len(info["ch_names"]), 2)), info, verbose=False)
        .set_eeg_reference("average", projection=True, verbose=False)
        .info
    )
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


def test_numerically_silent_constraint_raises():
    """A source the array cannot see is refused rather than silently amplified.

    The condition number of ``H^T R^-1 H`` is scale-invariant, so it says
    nothing about a constraint whose leadfield has shrunk to numerical silence:
    with one source that matrix is 1x1 and its condition number is exactly 1
    whatever the column holds. The solve then returns weights of order
    ``1/||h||`` -- here 1e12 times too large -- and their output is amplified
    noise wearing the name of a source, with nothing in the result to betray it.
    Whitening is what supplies the missing absolute scale, since every whitened
    channel carries unit noise variance.
    """
    rng = np.random.default_rng(4)
    n_ch = 20
    h = rng.standard_normal((n_ch, 1))
    h = 1e-12 * h / np.linalg.norm(h)  # a leadfield the array cannot see
    Rinv = np.eye(n_ch)
    assert np.linalg.cond(h.T @ Rinv @ h) == pytest.approx(1.0)  # blind to it
    with pytest.raises(ValueError, match="numerically silent"):
        _compute_mcmv_weights(h, Rinv)

    # The same direction at a normal strength is accepted, so it is the scale
    # that is being rejected and not the geometry.
    weights = _compute_mcmv_weights(h / np.linalg.norm(h), Rinv)
    assert np.isfinite(weights).all()


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
    filt = make_lcmv(
        info,
        fwd,
        data_cov,
        reg=REG,
        noise_cov=None,
        pick_ori="vector",
        weight_norm=None,
    )
    W_lcmv = filt["weights"]  # (n_src * 3, n_ch)

    # ``make_lcmv`` applies ``info['projs']`` -- here the average EEG reference --
    # to both the leadfield and the covariance before solving. Apply the same
    # projection so the manual comparison is like for like, and let ``_reg_pinv``
    # estimate the rank, since projecting makes the covariance rank-deficient.
    from mne._fiff.proj import make_projector

    proj, _, _ = make_projector(info["projs"], info["ch_names"])
    G = proj @ fwd["sol"]["data"]
    R = proj @ np.asarray(data_cov.data) @ proj.T
    Rinv, _, _ = _reg_pinv(R, reg=REG, rank=None)
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
    """No noise_cov -> normalisation is against the ad-hoc model, and warns."""
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    with pytest.warns(RuntimeWarning, match="ad-hoc"):
        filt = make_mcmv(
            info,
            fwd,
            data_cov,
            [1, 2],
            orientations=np.eye(3)[:2],
            weight_norm="unit-noise-gain",
        )
    # Unit noise gain is defined w.r.t. the noise model actually used (ad-hoc):
    # w_i^T C_n w_i == 1, which is *not* ||w_i|| == 1 once C_n != I.
    nad = mne.make_ad_hoc_cov(info)
    cov_ch = list(nad.ch_names)
    idx = [cov_ch.index(ch) for ch in filt["ch_names"]]
    d = np.asarray(nad.data)
    N = np.diag(d[idx]) if d.ndim == 1 else d[np.ix_(idx, idx)]
    ng = np.einsum("ij,jk,ik->i", filt["weights"], N, filt["weights"])
    assert_allclose(ng, 1.0, atol=1e-9)


def test_unit_noise_gain_with_explicit_noise_cov(fwd_info):
    """With a measured noise_cov, w_i^T C_n w_i == 1 exactly (no warning)."""
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    rng = np.random.default_rng(13)
    d = rng.uniform(0.5, 2.0, len(info["ch_names"]))
    ncov = mne.Covariance(
        np.diag(d), info["ch_names"], info["bads"], list(info["projs"]), nfree=1
    )
    filt = make_mcmv(
        info,
        fwd,
        data_cov,
        [1, 2],
        orientations=np.eye(3)[:2],
        noise_cov=ncov,
        weight_norm="unit-noise-gain",
    )
    ng = np.einsum("ij,jk,ik->i", filt["weights"], np.diag(d), filt["weights"])
    assert_allclose(ng, 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# 6. Whitening: device-agnostic correctness.
# --------------------------------------------------------------------------- #
def test_whitened_mcmv_matches_closed_form_whitened_lcmv():
    """n == 1 whitened MCMV equals the closed-form whitened LCMV, folded back."""
    rng = np.random.default_rng(11)
    n_ch = 24
    h = rng.standard_normal((n_ch, 1))
    A = rng.standard_normal((n_ch, n_ch))
    R = A @ A.T + np.eye(n_ch)
    B = rng.standard_normal((n_ch, n_ch))
    N = B @ B.T + np.eye(n_ch)  # non-trivial SPD noise covariance

    # Whitener with W N W^T = I (full rank here), via eigendecomposition.
    evals, evecs = np.linalg.eigh(N)
    W = (evecs * evals**-0.5).T  # (n_ch, n_ch)
    assert_allclose(W @ N @ W.T, np.eye(n_ch), atol=1e-10)

    h_w, R_w = W @ h, W @ R @ W.T
    Rinv_w = np.linalg.inv(R_w)
    # Closed-form whitened unit-gain LCMV, then folded to sensor space.
    w_closed = ((Rinv_w @ h_w) / (h_w.T @ Rinv_w @ h_w)).T @ W  # (1, n_ch)
    w_mcmv = _compute_mcmv_weights(h_w, Rinv_w) @ W  # (1, n_ch)

    assert_allclose(w_mcmv, w_closed, atol=1e-10)
    # Unit-gain constraint holds in the original (unwhitened) sensor space.
    assert_allclose((w_mcmv @ h).item(), 1.0, atol=1e-10)


def test_single_sensor_type_invariant_to_whitening():
    """A scalar (single-type) noise covariance cancels from the unit-gain filter.

    This is why a CTF gradiometer-only array gives identical results with or
    without whitening: the per-type scaling is a global scalar.
    """
    rng = np.random.default_rng(12)
    n_ch = 20
    H = rng.standard_normal((n_ch, 2))
    A = rng.standard_normal((n_ch, n_ch))
    R = A @ A.T + np.eye(n_ch)
    Rinv = np.linalg.inv(R)
    w_raw = _compute_mcmv_weights(H, Rinv)  # unwhitened solve

    c = 3.7  # arbitrary single-type noise level
    N = c**2 * np.eye(n_ch)
    evals, evecs = np.linalg.eigh(N)
    W = (evecs * evals**-0.5).T
    H_w, R_w = W @ H, W @ R @ W.T
    w_white = _compute_mcmv_weights(H_w, np.linalg.inv(R_w)) @ W

    assert_allclose(w_white, w_raw, atol=1e-9)


def test_make_whitener_retains_every_sensor_type():
    """Per-type rank: a mixed mag+grad array keeps *both* types after whitening.

    A single global eigenvalue threshold would discard the smaller-unit type as
    null space; MNE's per-type handling (used by ``_make_whitener``) does not.
    """
    ch_names = ["MAG1", "MAG2", "MAG3", "GRAD1", "GRAD2", "GRAD3"]
    ch_types = ["mag", "mag", "mag", "grad", "grad", "grad"]
    info = mne.create_info(ch_names, 1000.0, ch_types)
    W = _make_whitener(info, None, ch_names, "full")

    # Both sensor types survive: the whitener spans all 6 channels, not just one.
    assert W.shape == (6, 6)
    nad = mne.make_ad_hoc_cov(info)
    d = np.asarray(nad.data)
    N = np.diag(d) if d.ndim == 1 else d
    assert_allclose(W @ N @ W.T, np.eye(6), atol=1e-8)


# --------------------------------------------------------------------------- #
# 7. Coverage: fixed orientation, noise_cov intersection, apply, reprs.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fwd_fixed(fwd_info):
    """The same forward converted to fixed orientation."""
    fwd, info = fwd_info
    return mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True), info


def test_beamformer_repr(fwd_info):
    """MCMVBeamformer has an informative repr."""
    fwd, info = fwd_info
    filt = make_mcmv(
        info, fwd, _random_cov(info), [1, 2], orientations=np.eye(3)[:2], reg=REG
    )
    text = repr(filt)
    assert "MCMVBeamformer" in text and "source(s)" in text


def test_fixed_orientation_forward(fwd_fixed):
    """make_mcmv on a fixed-orientation forward: no orientations, constraint holds."""
    fwd, info = fwd_fixed
    data_cov = _random_cov(info)
    filt = make_mcmv(info, fwd, data_cov, [3, 7], reg=REG)  # orientations=None
    assert filt["is_fixed_orient"] is True
    assert_allclose(filt["weights"] @ filt["leadfield"], np.eye(2), atol=1e-9)
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((len(filt["ch_names"]), 20))
    assert apply_mcmv(arr, filt).shape == (2, 20)


def test_noise_cov_channel_subset(fwd_info):
    """A noise_cov covering only some channels narrows the common set."""
    fwd, info = fwd_info
    data_cov = _random_cov(info)
    sub = info["ch_names"][:20]  # noise_cov on a subset of channels
    rng = np.random.default_rng(2)
    ncov = mne.Covariance(np.diag(rng.uniform(0.5, 2.0, 20)), sub, [], [], nfree=1)
    filt = make_mcmv(
        info, fwd, data_cov, [1, 2], orientations=np.eye(3)[:2], noise_cov=ncov
    )
    assert filt["ch_names"] == sub  # restricted to the noise_cov channels
    assert filt["weights"].shape == (2, 20)


def test_noise_cov_disjoint_raises(fwd_info):
    """A noise_cov sharing no channels is an error."""
    fwd, info = fwd_info
    ncov = mne.Covariance(np.eye(2), ["ZZ1", "ZZ2"], [], [], nfree=1)
    with pytest.raises(ValueError, match="shares no channels"):
        make_mcmv(
            info,
            fwd,
            _random_cov(info),
            [1, 2],
            orientations=np.eye(3)[:2],
            noise_cov=ncov,
        )


def test_non_unit_orientations_warn(fwd_info):
    """Non-unit orientations are normalised, with a warning."""
    fwd, info = fwd_info
    with pytest.warns(RuntimeWarning, match="unit vectors"):
        filt = make_mcmv(
            info,
            fwd,
            _random_cov(info),
            [1, 2],
            orientations=np.array([[2.0, 0, 0], [0, 3.0, 0]]),
        )
    assert_allclose(filt["weights"] @ filt["leadfield"], np.eye(2), atol=1e-9)


def test_apply_mcmv_on_evoked(fwd_info):
    """apply_mcmv accepts an MNE object and picks channels by name."""
    fwd, info = fwd_info
    filt = make_mcmv(
        info, fwd, _random_cov(info), [4, 9], orientations=np.eye(3)[:2], reg=REG
    )
    ev_info = mne.create_info(filt["ch_names"], 200.0, "eeg")
    rng = np.random.default_rng(3)
    evoked = mne.EvokedArray(rng.standard_normal((len(filt["ch_names"]), 25)), ev_info)
    assert apply_mcmv(evoked, filt).shape == (2, 25)


def test_source_separation_warns_directly():
    """_check_source_separation warns for near-coincident sources."""
    fwd_like = {"source_rr": np.array([[0, 0, 0.05], [0, 0, 0.0501], [0, 0, 0.09]])}
    with pytest.warns(RuntimeWarning, match="mm apart"):
        _check_source_separation(fwd_like, [0, 1])
    # No source_rr -> silently does nothing.
    _check_source_separation({}, [0, 1])


def test_zero_orientation_raises(fwd_info):
    """A zero orientation vector is rejected."""
    fwd, info = fwd_info
    with pytest.raises(ValueError, match="zero vector"):
        make_mcmv(
            info,
            fwd,
            _random_cov(info),
            [1, 2],
            orientations=np.array([[1.0, 0, 0], [0, 0, 0]]),
        )


def test_forward_nonfinite_raises(fwd_info):
    """A forward with non-finite entries is rejected."""
    fwd, info = fwd_info
    bad = fwd.copy()
    bad["sol"] = dict(fwd["sol"])
    data = np.array(fwd["sol"]["data"], dtype=float)
    data[0, 4] = np.nan  # a column within source index 1's (x,y,z) block
    bad["sol"]["data"] = data
    with pytest.raises(ValueError, match="forward solution contains non-finite"):
        make_mcmv(info, bad, _random_cov(info), [1, 2], orientations=np.eye(3)[:2])


def test_apply_shape_mismatch_raises(fwd_info):
    """apply_mcmv rejects a raw array with the wrong channel count."""
    fwd, info = fwd_info
    filt = make_mcmv(
        info, fwd, _random_cov(info), [1, 2], orientations=np.eye(3)[:2], reg=REG
    )
    with pytest.raises(ValueError):
        apply_mcmv(np.zeros((len(filt["ch_names"]) - 1, 10)), filt)


def test_dropped_channels_still_work(fwd_info):
    """A data_cov missing a channel just narrows the common set (logged path)."""
    fwd, info = fwd_info
    keep = info["ch_names"][:-1]  # drop one channel from the covariance
    rng = np.random.default_rng(4)
    A = rng.standard_normal((len(keep), len(keep)))
    cov = mne.Covariance(A @ A.T + np.eye(len(keep)), keep, [], [], nfree=1)
    filt = make_mcmv(info, fwd, cov, [1, 2], orientations=np.eye(3)[:2])
    assert len(filt["ch_names"]) == len(keep)


def test_order_exceeds_rank_raises(fwd_info):
    """Requesting more sources than the data-covariance rank is an error."""
    fwd, info = fwd_info
    rng = np.random.default_rng(5)
    n = len(info["ch_names"])
    B = rng.standard_normal((n, 3))  # rank-3 data covariance
    cov = mne.Covariance(B @ B.T, info["ch_names"], [], [], nfree=1)
    oris = np.eye(3)[[0, 1, 2, 0, 1]]
    with (
        pytest.warns(RuntimeWarning, match="rank-deficient"),
        pytest.raises(ValueError, match="exceeds the data-covariance rank"),
    ):
        make_mcmv(info, fwd, cov, [1, 2, 3, 4, 5], orientations=oris, reg=0.0)


# --------------------------------------------------------------------------- #
# 9. Interoperability with the rest of the MNE pipeline.
# --------------------------------------------------------------------------- #
def test_diagonal_covariance_is_accepted(fwd_info):
    """A diagonal mne.Covariance (1-D ``.data``) works, as it does in make_lcmv.

    ``mne.make_ad_hoc_cov`` and MNE's diagonal covariance files store only the
    diagonal, so indexing ``cov.data`` as a square matrix raises. MNE's own
    beamformers go through ``Covariance._get_square``; so do we.
    """
    fwd, info = fwd_info
    diag_cov = mne.make_ad_hoc_cov(info)
    assert diag_cov["diag"] and np.asarray(diag_cov.data).ndim == 1

    ori = np.tile([0.0, 0.0, 1.0], (2, 1))
    filters = make_mcmv(info, fwd, diag_cov, [0, 5], orientations=ori)
    assert filters["weights"].shape == (2, len(filters["ch_names"]))
    assert np.all(np.isfinite(filters["weights"]))
    # ``apply_mcmv_cov`` must accept one too.
    src_cov = apply_mcmv_cov(diag_cov, filters)
    assert src_cov.shape == (2, 2) and np.all(np.isfinite(src_cov))


@pytest.mark.parametrize("rank", [None, "full", "info", {"eeg": 20}])
def test_rank_accepts_mne_conventions(fwd_info, rank):
    """``rank`` takes MNE's None | 'full' | 'info' | dict, as make_lcmv does."""
    fwd, info = fwd_info
    cov = _random_cov(info)
    ori = np.tile([0.0, 0.0, 1.0], (1, 1))
    filters = make_mcmv(info, fwd, cov, [0], orientations=ori, rank=rank)
    assert np.all(np.isfinite(filters["weights"]))


@pytest.mark.parametrize(("rank", "error"), [(20, TypeError), ("bogus", ValueError)])
def test_rank_rejects_unsupported_values(fwd_info, rank, error):
    """An integer or an unknown string is refused up front, not deep inside MNE."""
    fwd, info = fwd_info
    cov = _random_cov(info)
    ori = np.tile([0.0, 0.0, 1.0], (1, 1))
    with pytest.raises(error):
        make_mcmv(info, fwd, cov, [0], orientations=ori, rank=rank)


def test_several_sensor_types_require_noise_cov():
    """Refuse the ad-hoc fallback for mixed sensor types, exactly as make_lcmv does.

    The ad-hoc noise model is only a harmless global scaling for a *single*
    sensor type. With more than one its arbitrary per-type variances set the
    relative weighting of the types, so MNE refuses it; the error text asserted
    here is MNE's verbatim.
    """
    from advance_beamlab._mcmv import _check_noise_cov_required

    mixed = mne.create_info(
        ["MEG 0111", "MEG 0112", "MEG 0113", "MEG 0122"],
        200.0,
        ["grad", "grad", "mag", "grad"],
    )
    with pytest.raises(ValueError, match="several sensor types"):
        _check_noise_cov_required(mixed, mixed["ch_names"], None)

    # A single sensor type keeps the ad-hoc fallback.
    single = mne.create_info(["MEG 0111", "MEG 0112"], 200.0, ["grad", "grad"])
    _check_noise_cov_required(single, single["ch_names"], None)


def test_beamformer_copy_is_deep(fwd_info):
    """``.copy()`` deep-copies, matching mne.beamformer.Beamformer.copy."""
    fwd, info = fwd_info
    cov = _random_cov(info)
    ori = np.tile([0.0, 0.0, 1.0], (2, 1))
    filters = make_mcmv(info, fwd, cov, [0, 5], orientations=ori)
    before = filters["weights"].copy()
    duplicate = filters.copy()
    duplicate["weights"][:] = 0.0
    assert_allclose(filters["weights"], before)


def test_orientations_are_head_coordinates(fwd_info):
    """The same head-coordinate orientation means the same dipole for either forward.

    ``convert_forward_solution(..., surf_ori=True)`` expresses each source's three
    gain columns in its local surface frame. Orientations are documented as head
    coordinates, so the two representations of one forward must agree.
    """
    fwd, info = fwd_info
    cov = _random_cov(info)
    fwd_head = mne.convert_forward_solution(fwd, surf_ori=False, force_fixed=False)
    fwd_surf = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False)
    assert not fwd_head["surf_ori"] and fwd_surf["surf_ori"]

    sources = [0, 5]
    ori = np.asarray(fwd_head["source_nn"])[2::3][sources]
    ori = ori / np.linalg.norm(ori, axis=1, keepdims=True)
    w_head = make_mcmv(info, fwd_head, cov, sources, orientations=ori)["weights"]
    w_surf = make_mcmv(info, fwd_surf, cov, sources, orientations=ori)["weights"]
    assert_allclose(w_surf, w_head, rtol=1e-8, atol=0)


def test_array_gain_states_its_constraint_on_the_whitened_leadfield(fwd_info):
    """``w'h = ||h_w||``, and the zero-gain constraint survives the rescaling.

    Array gain exists because unit gain is biased towards deep sources: a deep
    leadfield is small, so the filter has to amplify to reach unit gain, and the
    amplified noise comes with it. Normalising the leadfield first divides that
    amplification back out.

    The norm is taken in the whitened space, where it equals
    ``sqrt(h^T C_n^-1 h)``. That is the only scale on which channels of
    different sensor types are commensurable (see the mixed-array test below);
    the price, visible here, is that the amplitudes carry the noise level with
    them and so are not in the units of the array.
    """
    fwd, info = fwd_info
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False)
    rng = np.random.default_rng(0)
    n_ch = len(info["ch_names"])
    data = rng.standard_normal((n_ch, 4000)) * 1e-8
    cov = mne.compute_covariance(
        mne.EpochsArray(data[None], info, verbose=False),
        method="empirical",
        verbose=False,
    )
    gain = fwd["sol"]["data"]
    sources = [3, 11]

    bf = make_mcmv(
        info,
        fwd,
        cov,
        sources,
        noise_cov=cov,
        reg=0.05,
        weight_norm="array-gain",
        verbose=False,
    )
    weights = bf["weights"]
    # The scale is the leadfield measured against the noise it has to be seen
    # through: ``sqrt(h^T C_n^+ h)``, taken through the same rank-aware whitener
    # the filter itself was built with. The pseudo-inverse is not a detail here
    # -- the average-reference projector leaves this EEG noise covariance one
    # dimension short, and a naive ``C_n^-1`` would divide by that null space.
    whitener = _make_whitener(info, cov, bf["ch_names"], None)
    scale = np.linalg.norm(whitener @ gain[:, sources], axis=0)
    for i, s in enumerate(sources):
        assert weights[i] @ gain[:, s] == pytest.approx(scale[i], rel=1e-6)
    # Rescaling each row cannot break the null on the other source. The
    # residual is judged against the row's own gain, which the normalisation
    # has just made large.
    assert abs(weights[0] @ gain[:, sources[1]]) < 1e-9 * scale[0]
    assert abs(weights[1] @ gain[:, sources[0]]) < 1e-9 * scale[1]

    # And it is a pure rescaling of the unit-gain filter, one factor per row.
    plain = make_mcmv(
        info,
        fwd,
        cov,
        sources,
        noise_cov=cov,
        reg=0.05,
        weight_norm="unit-gain",
        verbose=False,
    )["weights"]
    for i in range(len(sources)):
        np.testing.assert_allclose(weights[i], plain[i] * scale[i], rtol=1e-6)


# --------------------------------------------------------------------------- #
# 10. Mixed sensor types: the one scale on which they can be compared at all.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fwd_info_mixed():
    """A magnetometer + gradiometer sphere array: forward, Info and noise.

    Every other fixture in this file is a single-sensor-type EEG array, where a
    leadfield norm may be taken in the recorded units without coming to harm.
    This one records in two units at once -- T and T/m -- which is where a
    quantity summed over the whole array stops being a physical quantity.
    """
    n = 16
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - k / n)  # a spiral over the upper half of a shell
    theta = np.pi * (1 + 5**0.5) * k
    pos = (
        0.11
        * np.c_[np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    names = [f"MEG{i:03d}1" for i in range(n)] + [f"MEG{i:03d}2" for i in range(n)]
    info = mne.create_info(names, 200.0, ["mag"] * n + ["grad"] * n)
    for i, ch in enumerate(info["chs"]):
        p = pos[i % n]
        ez = p / np.linalg.norm(p)  # the coil normal points away from the head
        ex = np.cross([0.0, 0.0, 1.0], ez)
        ex /= np.linalg.norm(ex)
        ch["loc"] = np.concatenate([p, ex, np.cross(ez, ex), ez])
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.09)
    src = mne.setup_volume_source_space(sphere=sphere, pos=30.0)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=False, meg=True)
    # Per-type noise levels standing in for an empty-room covariance: MNE's own
    # ad-hoc figures of 20 fT for a magnetometer and 5e-13 T/m for a
    # gradiometer, numbers that are not in the same units to begin with.
    std = np.r_[np.full(n, 20e-15), np.full(n, 5e-13)]
    noise_cov = mne.Covariance(np.diag(std**2), names, [], [], nfree=1)
    return fwd, info, noise_cov


def _joint_leadfield(fwd, sources, orientations):
    """The joint leadfield H of the given sources on a free-orientation forward."""
    G = fwd["sol"]["data"]
    return np.column_stack(
        [
            G[:, 3 * s : 3 * s + 3] @ u
            for s, u in zip(sources, orientations, strict=True)
        ]
    )


def test_array_gain_is_not_set_by_whichever_type_has_the_larger_numbers(
    fwd_info_mixed,
):
    """Array gain must not turn on a unit convention.

    Squared leadfield entries summed over a mixed array add T^2 to (T/m)^2. On
    this array over 99% of that sum comes from the gradiometers, so a norm taken
    in the recorded units normalises for the gradiometers alone -- the
    magnetometers might as well not be there -- and it moves if the same
    recording is written in T/cm instead of T/m. A user reading amplitudes off
    such a beamformer is reading a depth correction that belongs to one sensor
    type and a unit choice. Whitened, each type is measured against its own
    noise: the constraint becomes ``sqrt(h^T C_n^-1 h)`` and does not move.
    """
    fwd, info, noise_cov = fwd_info_mixed
    n = len(info["ch_names"]) // 2
    sources = [10, 60]
    oris = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    H = _joint_leadfield(fwd, sources, oris)
    N = np.asarray(noise_cov.data)
    # Two 10 nA m sources on top of the sensor noise.
    R = N + 1e-16 * (H @ H.T)
    data_cov = mne.Covariance(R, info["ch_names"], [], [], nfree=1)
    kwargs = dict(noise_cov=noise_cov, reg=0.05, weight_norm="array-gain")
    weights = make_mcmv(info, fwd, data_cov, sources, orientations=oris, **kwargs)[
        "weights"
    ]
    for i in range(len(sources)):
        expected = np.sqrt(H[:, i] @ np.linalg.solve(N, H[:, i]))
        assert weights[i] @ H[:, i] == pytest.approx(expected, rel=1e-6)

    # The gradiometers own the norm taken in the recorded units; whitened, both
    # types are heard.
    def _grad_share(M):
        return np.linalg.norm(M[n:], axis=0) ** 2 / np.linalg.norm(M, axis=0) ** 2

    assert _grad_share(H).min() > 0.98
    whitened = _grad_share(_make_whitener(info, noise_cov, info["ch_names"], None) @ H)
    assert whitened.min() > 0.05 and whitened.max() < 0.95

    # Re-express the gradiometers in T/cm -- the same recording, written
    # differently -- and the filter must land in the same place.
    alpha = np.r_[np.ones(n), np.full(n, 1e-2)]
    fwd_cm = fwd.copy()
    fwd_cm["sol"]["data"] = alpha[:, None] * fwd["sol"]["data"]
    noise_cm, data_cm = (
        mne.Covariance(alpha[:, None] * M * alpha, info["ch_names"], [], [], nfree=1)
        for M in (N, R)
    )
    weights_cm = make_mcmv(
        info,
        fwd_cm,
        data_cm,
        sources,
        orientations=oris,
        noise_cov=noise_cm,
        reg=0.05,
        weight_norm="array-gain",
    )["weights"]
    H_cm = alpha[:, None] * H
    for i in range(len(sources)):
        assert weights_cm[i] @ H_cm[:, i] == pytest.approx(
            weights[i] @ H[:, i], rel=1e-6
        )
    # And the test has teeth: the norm in the recorded units did move.
    assert (np.linalg.norm(H, axis=0) / np.linalg.norm(H_cm, axis=0)).min() > 10


def test_source_the_array_cannot_see_is_refused(fwd_info_mixed):
    """A dipole at the centre of a spherical model produces no MEG field at all.

    Its leadfield column is exactly zero whatever orientation is asked for, so
    every filter satisfying the unit-gain constraint is infinite. Before the
    absolute-scale guard the only thing between the caller and a filter built on
    nothing was the condition number of ``H^T R^-1 H``, which reported
    "collinear sources" -- true of neither -- for two sources and, for one, saw
    nothing wrong at all.
    """
    fwd, info, noise_cov = fwd_info_mixed
    centre = int(np.argmin(np.linalg.norm(fwd["source_rr"], axis=1)))
    assert np.linalg.norm(fwd["source_rr"][centre]) == 0.0
    assert not fwd["sol"]["data"][:, 3 * centre : 3 * centre + 3].any()
    ori = np.array([[1.0, 0.0, 0.0]])
    data_cov = mne.Covariance(
        np.asarray(noise_cov.data).copy(), info["ch_names"], [], [], nfree=1
    )
    with pytest.raises(ValueError, match="numerically silent"):
        make_mcmv(info, fwd, data_cov, [centre], orientations=ori, noise_cov=noise_cov)

    # A source the array does see, in the same call shape, still builds.
    filters = make_mcmv(
        info, fwd, data_cov, [10], orientations=ori, noise_cov=noise_cov
    )
    assert np.isfinite(filters["weights"]).all()
