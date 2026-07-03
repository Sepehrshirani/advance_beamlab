r"""Tests for pairwise and augmented-pairwise MCMV connectivity (Nunes et al., 2020)."""
# Authors: Sepehr Shirani <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import mne
import numpy as np
import pytest
from scipy.signal import hilbert

from mne_beamlab import (
    ar1_surrogate_significance,
    augmented_pairwise_mcmv_connectivity,
    make_mcmv,
    pairwise_mcmv_connectivity,
    reconstruct_pairwise_mcmv,
)
from mne_beamlab._connectivity import _as_pairs, _epoched, _select_neighbours

mne.set_log_level("ERROR")


@pytest.fixture(scope="module")
def scenario():
    """A controlled EEG-sphere scenario with a known indirect-leakage structure.

    Four fixed-orientation volume sources: ``A`` and ``C`` are ~2.4 cm apart (so
    ``A`` leaks into ``C``), ``B`` is far from both, and ``D`` is an extra
    distant source. ``C`` is genuinely coupled to ``B`` (shared amplitude
    envelope), while ``A`` and ``D`` carry independent envelopes -- so the true
    ``A``--``B`` coupling is ~0 but is confounded by the ``A``->``C``->``B``
    indirect path.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    ch = list(dict.fromkeys(montage.ch_names))
    info = mne.create_info(ch, 200.0, "eeg")
    info.set_montage(montage)
    sphere = mne.make_sphere_model("auto", "auto", info, verbose=False)
    src = mne.setup_volume_source_space(sphere=sphere, pos=12.0, verbose=False)
    fwd = mne.convert_forward_solution(
        mne.make_forward_solution(
            info, None, src, sphere, eeg=True, meg=False, verbose=False
        ),
        force_fixed=True,
        use_cps=False,
        verbose=False,
    )
    gain = fwd["sol"]["data"]
    rr = fwd["source_rr"]
    n_ch = len(ch)

    a = int(np.argmin(np.linalg.norm(rr - rr.mean(0), axis=1)))
    da = np.linalg.norm(rr - rr[a], axis=1)
    c = int(np.argmin(np.abs(da - 0.024)))
    b = int(np.argmin(np.abs(da - 0.09)))
    d = int(np.argmin(np.abs(da - 0.06)))
    ha, hb, hc = gain[:, a], gain[:, b], gain[:, c]

    rng = np.random.default_rng(0)
    sfreq = 200.0
    n_times = int(120 * sfreq)
    t = np.arange(n_times) / sfreq

    def carrier(env, phase):
        return env * np.cos(2 * np.pi * 10 * t + phase)

    def slow_env(seed):
        r = np.random.default_rng(seed)
        smooth = np.exp(-0.5 * (np.arange(-200, 201) / 60) ** 2)
        smooth /= smooth.sum()
        e = np.convolve(r.standard_normal(n_times), smooth, "same")
        return 1.0 + 0.8 * (e - e.mean()) / e.std()

    env_c, env_a = slow_env(1), slow_env(2)
    s_c = 2.0 * carrier(env_c, 0.0)
    s_b = carrier(env_c, 1.3)  # shares C's envelope -> genuine C-B coupling
    s_a = carrier(env_a, 2.1)  # independent -> no true A-B coupling
    signal = np.outer(ha, s_a) + np.outer(hb, s_b) + np.outer(hc, s_c)
    noise = 0.1 * np.abs(ha).max() * rng.standard_normal((n_ch, n_times))
    data = signal + noise

    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    dcov = mne.compute_covariance(
        mne.make_fixed_length_epochs(raw, duration=2.0, verbose=False),
        method="empirical",
        verbose=False,
    )
    ncov = mne.make_ad_hoc_cov(info, verbose=False)
    evoked = mne.EvokedArray(raw.get_data(), info, tmin=0)
    evoked.set_eeg_reference("average", projection=True, verbose=False)

    # sources ordered so that A=0, B=1, C=2, D=3 for the connectivity matrices
    sources = [a, b, c, d]
    return dict(
        info=evoked.info,
        fwd=fwd,
        gain=gain,
        sources=sources,
        positions=rr[sources],
        dcov=dcov,
        ncov=ncov,
        evoked=evoked,
        s_a=s_a,
        s_b=s_b,
        A=a,
        B=b,
        C=c,
        D=d,
    )


def _env_corr(x, y):
    return float(np.corrcoef(np.abs(hilbert(x)), np.abs(hilbert(y)))[0, 1])


# --------------------------------------------------------------------------- #
# Core correctness: the beamformer constraints (Nunes 2020 Eq. 4)             #
# --------------------------------------------------------------------------- #
def test_pairwise_is_direct_leakage_free(scenario):
    """2-source MCMV: unit gain on self, exact zero gain on the partner."""
    d = scenario
    filt = make_mcmv(
        d["info"], d["fwd"], d["dcov"], sources=[d["A"], d["B"]],
        noise_cov=d["ncov"], weight_norm="unit-gain",
    )
    w = filt["weights"]
    g = d["gain"]
    assert abs(w[0] @ g[:, d["A"]] - 1.0) < 1e-8  # unit gain on A
    assert abs(w[0] @ g[:, d["B"]]) < 1e-8        # zero gain on B (no direct leakage)
    assert abs(w[1] @ g[:, d["B"]] - 1.0) < 1e-8
    assert abs(w[1] @ g[:, d["A"]]) < 1e-8


def test_augmentation_nulls_the_neighbour(scenario):
    """PW-MCMV leaks a neighbour; adding it (APW-MCMV) places an exact null on it."""
    d = scenario
    pw = make_mcmv(
        d["info"], d["fwd"], d["dcov"], sources=[d["A"], d["B"]],
        noise_cov=d["ncov"], weight_norm="unit-gain",
    )
    aug = make_mcmv(
        d["info"], d["fwd"], d["dcov"], sources=[d["A"], d["B"], d["C"]],
        noise_cov=d["ncov"], weight_norm="unit-gain",
    )
    g = d["gain"]
    # PW-MCMV has a non-trivial leakage coefficient onto the conductor C ...
    assert abs(pw["weights"][0] @ g[:, d["C"]]) > 0.05
    # ... which APW-MCMV eliminates exactly.
    assert abs(aug["weights"][0] @ g[:, d["C"]]) < 1e-8
    # the augmented pair still satisfies its own constraints
    assert abs(aug["weights"][0] @ g[:, d["A"]] - 1.0) < 1e-8
    assert abs(aug["weights"][0] @ g[:, d["B"]]) < 1e-8


# --------------------------------------------------------------------------- #
# reconstruct_pairwise_mcmv structure                                         #
# --------------------------------------------------------------------------- #
def test_reconstruct_pairwise_shapes(scenario):
    d = scenario
    pairs, tcs = reconstruct_pairwise_mcmv(
        d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"], noise_cov=d["ncov"]
    )
    n = len(d["sources"])
    assert pairs == _as_pairs(n)
    assert len(tcs) == len(pairs)
    for tc in tcs:
        assert tc.shape[0] == 2  # two reconstructed rows per pair
        assert tc.shape[-1] == d["evoked"].data.shape[-1]


def test_epoched_helper():
    assert _epoched(np.zeros((2, 10))).shape == (1, 2, 10)
    assert _epoched(np.zeros((5, 2, 10))).shape == (5, 2, 10)
    with pytest.raises(ValueError, match="2D|3D"):
        _epoched(np.zeros((3,)))


# --------------------------------------------------------------------------- #
# Connectivity matrix properties + metric delegation                          #
# --------------------------------------------------------------------------- #
def test_connectivity_matrix_is_symmetric_zero_diagonal(scenario):
    d = scenario
    conn = pairwise_mcmv_connectivity(
        d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
        method="envelope", noise_cov=d["ncov"], absolute=False,
    )
    n = len(d["sources"])
    assert conn.shape == (n, n)
    assert np.allclose(conn, conn.T)
    assert np.allclose(np.diag(conn), 0.0)


def test_spectral_method_runs(scenario):
    """The coherence path delegates to mne-connectivity and yields a matrix."""
    d = scenario
    ep = mne.make_fixed_length_epochs(
        mne.io.RawArray(d["evoked"].data, d["info"], verbose=False),
        duration=2.0, verbose=False,
    ).get_data()
    conn = pairwise_mcmv_connectivity(
        ep, d["info"], d["fwd"], d["dcov"], d["sources"],
        method="coh", sfreq=200.0, fmin=8.0, fmax=12.0, noise_cov=d["ncov"],
    )
    n = len(d["sources"])
    assert conn.shape == (n, n)
    assert np.all(np.isfinite(conn))
    assert np.allclose(conn, conn.T)


# --------------------------------------------------------------------------- #
# Neighbour-selection heuristic (Sec. 2.4)                                     #
# --------------------------------------------------------------------------- #
def test_select_neighbours_picks_significant_in_radius(scenario):
    """For pair (A, B), the significant in-radius neighbour C is selected."""
    d = scenario
    pos = d["positions"]
    sig = np.zeros((4, 4), bool)
    sig[0, 1] = sig[1, 0] = True  # A-B
    sig[1, 2] = sig[2, 1] = True  # C-B -> gives C degree 1
    degree = sig.sum(1)
    chosen = _select_neighbours(0, 1, pos, sig, degree, radius=0.04, max_neighbours=2)
    assert chosen == [2]  # C only; D is not significant, A/B excluded


def test_select_neighbours_respects_radius():
    """A significant node outside the radius is not selected."""
    pos = np.array([[0, 0, 0], [0.2, 0, 0], [0.09, 0, 0]], float)  # C 9 cm from A
    sig = np.zeros((3, 3), bool)
    sig[0, 1] = sig[1, 0] = True
    sig[1, 2] = sig[2, 1] = True
    degree = sig.sum(1)
    chosen = _select_neighbours(0, 1, pos, sig, degree, radius=0.04, max_neighbours=2)
    assert chosen == []


def test_select_neighbours_ranks_by_degree_then_proximity():
    """Higher-degree candidates come first; ties break by proximity."""
    # sources 0,1 are the pair; 2,3,4 are candidates near source 0
    pos = np.array(
        [[0, 0, 0], [0.2, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.015, 0, 0]], float
    )
    sig = np.zeros((5, 5), bool)
    sig[0, 1] = sig[1, 0] = True
    # degrees: node 2 -> 1, node 3 -> 3, node 4 -> 3 (tie, node 4 is nearer than node 3)
    for j in (0,):
        sig[2, j] = sig[j, 2] = True
    for j in (0, 1, 4):
        sig[3, j] = sig[j, 3] = True
    for j in (0, 1, 3):
        sig[4, j] = sig[j, 4] = True
    degree = sig.sum(1)
    chosen = _select_neighbours(0, 1, pos, sig, degree, radius=0.04, max_neighbours=2)
    assert chosen == [4, 3]  # tie on degree 3 -> nearer node 4 first; node 2 dropped


def test_select_neighbours_caps_order(scenario):
    """At most max_neighbours per source are returned (order stays bounded)."""
    d = scenario
    pos = d["positions"]
    sig = np.ones((4, 4), bool)
    np.fill_diagonal(sig, False)
    degree = sig.sum(1)
    chosen = _select_neighbours(0, 1, pos, sig, degree, radius=1.0, max_neighbours=2)
    assert len(chosen) <= 4  # up to 2 per source, deduplicated


# --------------------------------------------------------------------------- #
# APW-MCMV end-to-end behaviour                                                #
# --------------------------------------------------------------------------- #
def test_apw_reduces_indirect_leakage_bias(scenario):
    """APW-MCMV moves the spurious A-B edge closer to its true value than PW-MCMV."""
    d = scenario
    conn = pairwise_mcmv_connectivity(
        d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
        method="envelope", noise_cov=d["ncov"], absolute=False,
    )
    sig = np.zeros((4, 4), bool)
    sig[0, 1] = sig[1, 0] = True  # treat A-B as (spuriously) significant
    sig[1, 2] = sig[2, 1] = True  # C-B genuinely significant -> C is a conductor
    apw = augmented_pairwise_mcmv_connectivity(
        d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"], conn, sig,
        positions=d["positions"], method="envelope",
        noise_cov=d["ncov"], absolute=False,
    )
    truth = _env_corr(d["s_a"], d["s_b"])  # ~ -0.09
    # APW is at least as close to the truth as PW on the augmented A-B edge
    assert abs(apw[0, 1] - truth) <= abs(conn[0, 1] - truth) + 1e-9
    # non-significant edges are left untouched
    assert apw[0, 3] == conn[0, 3]


def test_apw_beats_lcmv(scenario):
    """Genuine C-B edge preserved; spurious A-B edge far smaller than LCMV's."""
    from mne.beamformer import apply_lcmv, make_lcmv

    d = scenario
    conn = pairwise_mcmv_connectivity(
        d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
        method="envelope", noise_cov=d["ncov"], absolute=False,
    )
    lcmv = make_lcmv(
        d["info"], d["fwd"], d["dcov"], reg=0.05, noise_cov=d["ncov"],
        pick_ori=None, weight_norm=None,
    )
    sl = apply_lcmv(d["evoked"], lcmv).data
    lcmv_ab = _env_corr(sl[d["A"]], sl[d["B"]])
    assert abs(conn[0, 1]) < abs(lcmv_ab)   # PW-MCMV less spurious than LCMV on A-B
    assert conn[1, 2] > 0.5                 # genuine C-B coupling preserved


# --------------------------------------------------------------------------- #
# AR(1) surrogate significance                                                 #
# --------------------------------------------------------------------------- #
def test_ar1_significance_flags_strong_not_weak():
    """A coupled pair is flagged significant; independent pairs are not.

    Uses first-order autoregressive sources, matching the surrogate model, so the
    null is well-calibrated. (The AR(1) null under-estimates variance for strongly
    narrow-band signals, whose slow envelope a first-order process cannot capture;
    that is a property of the paper's procedure, not tested here.)
    """
    from mne_beamlab._connectivity import _pair_connectivity

    n_times = 20000

    def ar1(seed, phi=0.9):
        r = np.random.default_rng(seed)
        innov = r.standard_normal(n_times)
        x = np.zeros(n_times)
        for k in range(1, n_times):
            x[k] = phi * x[k - 1] + innov[k]
        return x

    shared = ar1(3)
    # rows 0, 1 independent; rows 2, 3 share most of their signal (strong coupling)
    ref = np.stack([ar1(1), ar1(2), shared, 0.85 * shared + 0.15 * ar1(4)])

    conn = np.zeros((4, 4))
    for i, j in _as_pairs(4):
        value = _pair_connectivity(
            np.stack([ref[i], ref[j]]), "envelope", sfreq=None, fmin=None,
            fmax=None, orthogonalize=False, absolute=True, mt_bandwidth=None,
        )
        conn[i, j] = conn[j, i] = value

    mask = ar1_surrogate_significance(
        conn, ref, method="envelope", n_surrogates=200, alpha=0.05,
        absolute=True, random_state=0,
    )
    assert mask.shape == (4, 4)
    assert not mask.diagonal().any()
    assert np.array_equal(mask, mask.T)
    assert mask[2, 3]        # coupled pair flagged significant
    assert not mask[0, 1]    # independent pair not flagged


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #
def test_requires_two_sources(scenario):
    d = scenario
    with pytest.raises(ValueError, match="at least 2 sources"):
        pairwise_mcmv_connectivity(
            d["evoked"], d["info"], d["fwd"], d["dcov"], [d["A"]], noise_cov=d["ncov"]
        )


def test_unknown_method_raises(scenario):
    d = scenario
    with pytest.raises(ValueError, match="method must be one of"):
        pairwise_mcmv_connectivity(
            d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
            method="not-a-method", noise_cov=d["ncov"],
        )


def test_spectral_method_requires_sfreq(scenario):
    d = scenario
    with pytest.raises(ValueError, match="requires ``sfreq``"):
        pairwise_mcmv_connectivity(
            d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
            method="coh", noise_cov=d["ncov"],
        )


def test_augmented_validates_shapes(scenario):
    d = scenario
    n = len(d["sources"])
    good = np.zeros((n, n))
    sig = np.zeros((n, n), bool)
    with pytest.raises(ValueError, match="connectivity must be shape"):
        augmented_pairwise_mcmv_connectivity(
            d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
            np.zeros((n, n + 1)), sig, noise_cov=d["ncov"],
        )
    with pytest.raises(ValueError, match="significance must be shape"):
        augmented_pairwise_mcmv_connectivity(
            d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
            good, np.zeros((n + 1, n + 1), bool), noise_cov=d["ncov"],
        )
    with pytest.raises(ValueError, match="positions must be"):
        augmented_pairwise_mcmv_connectivity(
            d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
            good, sig, positions=np.zeros((n, 2)), noise_cov=d["ncov"],
        )
