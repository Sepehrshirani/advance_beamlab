r"""Tests for pairwise and augmented-pairwise MCMV connectivity (Nunes et al., 2020)."""
# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import lfilter

from advance_beamlab import (
    ar1_surrogate_significance,
    augmented_pairwise_mcmv_connectivity,
    make_mcmv,
    pairwise_mcmv_connectivity,
    reconstruct_pairwise_mcmv,
)
from advance_beamlab._connectivity import (
    _SPECTRAL_METHODS,
    _ar1_fit,
    _ar1_surrogate,
    _as_pairs,
    _benjamini_hochberg,
    _envelope_corr_matrix,
    _epoched,
    _fisher_z,
    _pair_connectivity,
    _select_neighbours,
    _spectral_conn_matrix,
)

mne.set_log_level("ERROR")

SFREQ = 200.0


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


def _env_corr(x, y, envelope_lowpass=0.5):
    """Envelope correlation of two source time courses, module defaults."""
    return float(
        _envelope_corr_matrix(
            np.stack([x, y]),
            sfreq=SFREQ,
            orthogonalize=False,
            absolute=False,
            envelope_lowpass=envelope_lowpass,
            envelope_resample=None,
        )[0, 1]
    )


# --------------------------------------------------------------------------- #
# Core correctness: the beamformer constraints (Nunes 2020 Eq. 4)             #
# --------------------------------------------------------------------------- #
def test_pairwise_is_direct_leakage_free(scenario):
    """2-source MCMV: unit gain on self, exact zero gain on the partner."""
    d = scenario
    filt = make_mcmv(
        d["info"],
        d["fwd"],
        d["dcov"],
        sources=[d["A"], d["B"]],
        noise_cov=d["ncov"],
        weight_norm="unit-gain",
    )
    w = filt["weights"]
    g = d["gain"]
    assert abs(w[0] @ g[:, d["A"]] - 1.0) < 1e-8  # unit gain on A
    assert abs(w[0] @ g[:, d["B"]]) < 1e-8  # zero gain on B (no direct leakage)
    assert abs(w[1] @ g[:, d["B"]] - 1.0) < 1e-8
    assert abs(w[1] @ g[:, d["A"]]) < 1e-8


def test_augmentation_nulls_the_neighbour(scenario):
    """PW-MCMV leaks a neighbour; adding it (APW-MCMV) places an exact null on it."""
    d = scenario
    pw = make_mcmv(
        d["info"],
        d["fwd"],
        d["dcov"],
        sources=[d["A"], d["B"]],
        noise_cov=d["ncov"],
        weight_norm="unit-gain",
    )
    aug = make_mcmv(
        d["info"],
        d["fwd"],
        d["dcov"],
        sources=[d["A"], d["B"], d["C"]],
        noise_cov=d["ncov"],
        weight_norm="unit-gain",
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


def test_ndarray_must_be_restricted_to_good_channels(scenario):
    """With bad channels, a raw array must already be cut down to the good ones.

    ``make_mcmv`` drops ``info['bads']``, so an array still holding every
    ``info['ch_names']`` row has the wrong length; only the restricted array (or
    an MNE object, matched by name) reproduces the reconstruction.
    """
    d = scenario
    info = d["info"].copy()
    info["bads"] = [info["ch_names"][3]]
    evoked = d["evoked"].copy()
    evoked.info["bads"] = [evoked.ch_names[3]]
    pair = d["sources"][:2]
    kwargs = dict(noise_cov=d["ncov"])

    filters = make_mcmv(info, d["fwd"], d["dcov"], sources=pair, **kwargs)
    assert len(filters["ch_names"]) == len(info["ch_names"]) - 1
    good = mne.pick_channels(evoked.ch_names, filters["ch_names"], ordered=True)

    _, from_array = reconstruct_pairwise_mcmv(
        evoked.data[good], info, d["fwd"], d["dcov"], pair, **kwargs
    )
    _, from_evoked = reconstruct_pairwise_mcmv(
        evoked, info, d["fwd"], d["dcov"], pair, **kwargs
    )
    assert np.array_equal(from_array[0], from_evoked[0])

    with pytest.raises(ValueError, match="channels"):
        reconstruct_pairwise_mcmv(
            evoked.data, info, d["fwd"], d["dcov"], pair, **kwargs
        )


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
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        method="envelope",
        noise_cov=d["ncov"],
        absolute=False,
    )
    n = len(d["sources"])
    assert conn.shape == (n, n)
    assert np.allclose(conn, conn.T)
    assert np.allclose(np.diag(conn), 0.0)


def test_envelope_matches_mne_connectivity_without_lowpass():
    """With ``envelope_lowpass=None`` the estimator reproduces mne-connectivity."""
    from mne_connectivity import envelope_correlation

    x = np.random.default_rng(0).standard_normal((4, 3000))
    off = ~np.eye(4, dtype=bool)
    for orthogonalize in (False, "pairwise"):
        ref = envelope_correlation(
            x[np.newaxis], orthogonalize=orthogonalize, absolute=False, verbose=False
        ).get_data("dense")[0, :, :, 0]
        got = _envelope_corr_matrix(
            x,
            sfreq=SFREQ,
            orthogonalize=orthogonalize,
            absolute=False,
            envelope_lowpass=None,
            envelope_resample=None,
        )
        assert np.abs(ref[off] - got[off]).max() < 1e-12


def test_absolute_is_honoured_without_orthogonalisation():
    """``absolute=True`` takes |r| even when ``orthogonalize=False``.

    ``mne_connectivity.envelope_correlation`` silently ignores ``absolute``
    unless ``orthogonalize='pairwise'``, so a signed value used to come back.
    """
    x = np.random.default_rng(1).standard_normal((3, 3000))
    kwargs = dict(
        sfreq=SFREQ,
        orthogonalize=False,
        envelope_lowpass=None,
        envelope_resample=None,
    )
    signed = _envelope_corr_matrix(x, absolute=False, **kwargs)
    magnitude = _envelope_corr_matrix(x, absolute=True, **kwargs)
    off = ~np.eye(3, dtype=bool)
    assert (signed[off] < 0).any()  # there is a sign to lose
    assert np.allclose(magnitude[off], np.abs(signed[off]))


def test_envelope_lowpass_is_applied():
    """The 0.5 Hz envelope low-pass changes the estimate, on data and surrogates.

    A narrow-band signal has a slow envelope plus fast sub-band ripple; removing
    the ripple raises the correlation of two envelopes that share only their slow
    part, so the smoothed and unsmoothed estimates must differ substantially.
    """
    rng = np.random.default_rng(0)
    n_times = 12000
    t = np.arange(n_times) / SFREQ
    slow = mne.filter.filter_data(
        rng.standard_normal(n_times), SFREQ, None, 0.3, verbose=False
    )
    slow = 1.0 + 0.8 * (slow - slow.mean()) / slow.std()
    x = np.stack(
        [
            slow * np.cos(2 * np.pi * 10 * t) + 0.3 * rng.standard_normal(n_times),
            slow * np.cos(2 * np.pi * 10 * t + 1.1)
            + 0.3 * rng.standard_normal(n_times),
        ]
    )
    kwargs = dict(sfreq=SFREQ, orthogonalize=False, absolute=False)
    raw = _envelope_corr_matrix(
        x, envelope_lowpass=None, envelope_resample=None, **kwargs
    )[0, 1]
    smoothed = _envelope_corr_matrix(
        x, envelope_lowpass=0.5, envelope_resample=None, **kwargs
    )[0, 1]
    assert smoothed - raw > 0.1
    # downsampling the smoothed envelopes leaves the value essentially unchanged
    downsampled = _envelope_corr_matrix(
        x, envelope_lowpass=0.5, envelope_resample=4.0, **kwargs
    )[0, 1]
    assert abs(downsampled - smoothed) < 0.02


def test_envelope_lowpass_reaches_the_public_api(scenario):
    """``envelope_lowpass`` is threaded through the connectivity functions."""
    d = scenario
    kwargs = dict(method="envelope", noise_cov=d["ncov"], absolute=False)
    smoothed = pairwise_mcmv_connectivity(
        d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"], **kwargs
    )
    raw = pairwise_mcmv_connectivity(
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        envelope_lowpass=None,
        **kwargs,
    )
    assert not np.allclose(smoothed, raw)
    # the default really is the paper's 0.5 Hz
    explicit = pairwise_mcmv_connectivity(
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        envelope_lowpass=0.5,
        **kwargs,
    )
    assert np.allclose(smoothed, explicit)


def test_cohy_is_rejected(scenario):
    """Complex coherency cannot be stored in the real matrix, so it is refused."""
    d = scenario
    with pytest.raises(ValueError, match="complex coherency"):
        pairwise_mcmv_connectivity(
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            method="cohy",
            sfreq=SFREQ,
            fmin=8.0,
            fmax=12.0,
            noise_cov=d["ncov"],
        )


def test_spectral_method_runs(scenario):
    """The coherence path delegates to mne-connectivity and yields a matrix."""
    d = scenario
    ep = mne.make_fixed_length_epochs(
        mne.io.RawArray(d["evoked"].data, d["info"], verbose=False),
        duration=2.0,
        verbose=False,
    ).get_data()
    conn = pairwise_mcmv_connectivity(
        ep,
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        method="coh",
        sfreq=200.0,
        fmin=8.0,
        fmax=12.0,
        noise_cov=d["ncov"],
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
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        method="envelope",
        noise_cov=d["ncov"],
        absolute=False,
    )
    sig = np.zeros((4, 4), bool)
    sig[0, 1] = sig[1, 0] = True  # treat A-B as (spuriously) significant
    sig[1, 2] = sig[2, 1] = True  # C-B genuinely significant -> C is a conductor
    apw = augmented_pairwise_mcmv_connectivity(
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        conn,
        sig,
        positions=d["positions"],
        method="envelope",
        noise_cov=d["ncov"],
        absolute=False,
    )
    truth = _env_corr(d["s_a"], d["s_b"])  # ~ -0.153
    pw_error = abs(conn[0, 1] - truth)
    apw_error = abs(apw[0, 1] - truth)
    # A "<=" inequality here would also be satisfied by an APW that did nothing.
    # Nulling the conductor C removes essentially all of the indirect-leakage
    # bias: the residual error drops by more than an order of magnitude
    # (measured: 0.0597 -> 0.0021, a factor of 28).
    assert pw_error > 0.03  # PW-MCMV really is biased on this edge
    assert apw_error < 0.01
    assert pw_error / apw_error > 10.0
    # the genuine C-B edge is not degraded by the augmentation
    assert apw[1, 2] > 0.99
    # non-significant edges are left untouched
    assert apw[0, 3] == conn[0, 3]


def test_apw_beats_lcmv(scenario):
    """Genuine C-B edge preserved; spurious A-B edge far smaller than LCMV's."""
    from mne.beamformer import apply_lcmv, make_lcmv

    d = scenario
    conn = pairwise_mcmv_connectivity(
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        method="envelope",
        noise_cov=d["ncov"],
        absolute=False,
    )
    lcmv = make_lcmv(
        d["info"],
        d["fwd"],
        d["dcov"],
        reg=0.05,
        noise_cov=d["ncov"],
        pick_ori=None,
        weight_norm=None,
    )
    sl = apply_lcmv(d["evoked"], lcmv).data
    lcmv_ab = _env_corr(sl[d["A"]], sl[d["B"]])
    assert abs(conn[0, 1]) < abs(lcmv_ab)  # PW-MCMV less spurious than LCMV on A-B
    assert conn[1, 2] > 0.5  # genuine C-B coupling preserved


# --------------------------------------------------------------------------- #
# AR(1) surrogate significance                                                 #
# --------------------------------------------------------------------------- #
def test_ar1_significance_flags_strong_not_weak():
    """A coupled pair is flagged significant; independent pairs are not.

    Uses first-order autoregressive sources, matching the surrogate model, so the
    null is well-calibrated. The calibration of the null for narrow-band signals
    -- which depends on the 0.5 Hz envelope low-pass of Nunes et al. (2020),
    Sec. 2.5 -- is tested separately in
    ``test_ar1_null_false_positive_rate_at_or_below_alpha``.
    """
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
            np.stack([ref[i], ref[j]]),
            "envelope",
            sfreq=SFREQ,
            fmin=None,
            fmax=None,
            orthogonalize=False,
            absolute=True,
            mt_bandwidth=None,
        )
        conn[i, j] = conn[j, i] = value

    mask = ar1_surrogate_significance(
        conn,
        ref,
        method="envelope",
        n_surrogates=200,
        alpha=0.05,
        sfreq=SFREQ,
        absolute=True,
        random_state=0,
    )
    assert mask.shape == (4, 4)
    assert not mask.diagonal().any()
    assert np.array_equal(mask, mask.T)
    assert mask[2, 3]  # coupled pair flagged significant
    assert not mask[0, 1]  # independent pair not flagged
    assert conn[2, 3] > 0.5  # and it really is a strong envelope correlation


def test_ar1_null_false_positive_rate_at_or_below_alpha():
    """Under a complete null the AR(1) test rejects at most ``alpha`` of edges.

    Eight mutually independent 9-11 Hz sources: every edge is a true negative.
    A first-order autoregressive process cannot reproduce the slow amplitude
    fluctuation of such a narrow-band signal, so the surrogate null is too
    narrow unless the envelopes are low-pass filtered to 0.5 Hz first, as
    Nunes et al. (2020), Sec. 2.5, prescribe. Without that step the measured
    rejection rate here is 7.5 per cent of edges at ``alpha=0.05``; with the
    low-pass it is 0.4 per cent.
    """
    n, n_times, alpha = 8, 12000, 0.05
    pairs = _as_pairs(n)
    rates = {}
    for lowpass in (0.5, None):
        n_false = 0
        for seed in range(10):
            r = np.random.default_rng(seed)
            x = mne.filter.filter_data(
                r.standard_normal((n, n_times)), SFREQ, 9.0, 11.0, verbose=False
            )
            conn = _envelope_corr_matrix(
                x,
                sfreq=SFREQ,
                orthogonalize=False,
                absolute=False,
                envelope_lowpass=lowpass,
                envelope_resample=None,
            )
            np.fill_diagonal(conn, 0.0)
            mask = ar1_surrogate_significance(
                conn,
                x,
                sfreq=SFREQ,
                n_surrogates=100,
                alpha=alpha,
                envelope_lowpass=lowpass,
                random_state=seed,
            )
            n_false += sum(bool(mask[i, j]) for i, j in pairs)
        rates[lowpass] = n_false / (10 * len(pairs))
    assert rates[0.5] <= alpha
    # and the low-pass is what buys the calibration
    assert rates[None] > alpha
    assert rates[0.5] < rates[None] / 4.0


def test_ar1_degenerate_null_is_not_significant():
    """A collapsed surrogate null must never read as a significant edge.

    Constant reference courses give zero-variance AR(1) surrogates, so every
    surrogate connectivity is identically zero and the null standard deviation
    is exactly zero. Replacing that zero by ``eps`` (the previous behaviour)
    turns every edge into a ~1e16 z-score and a p-value of 0.
    """
    ref = np.zeros((3, 4000))
    conn = np.full((3, 3), 0.9)
    np.fill_diagonal(conn, 0.0)
    # the module sets the log level to ERROR, which silences mne.utils.warn
    with mne.use_log_level("warning"), pytest.warns(RuntimeWarning, match="degenerate"):
        mask = ar1_surrogate_significance(
            conn, ref, sfreq=SFREQ, n_surrogates=20, random_state=0
        )
    assert not mask.any()


def test_ar1_requires_at_least_two_surrogates():
    ref = np.random.default_rng(0).standard_normal((3, 4000))
    with pytest.raises(ValueError, match="n_surrogates must be at least 2"):
        ar1_surrogate_significance(
            np.zeros((3, 3)), ref, sfreq=SFREQ, n_surrogates=1, random_state=0
        )


def test_ar1_surrogate_is_the_scalar_recursion():
    """The vectorised AR(1) draw reproduces the per-sample recursion exactly."""
    ref = np.cumsum(np.random.default_rng(5).standard_normal((4, 3000)), axis=1)
    phis, sigmas = _ar1_fit(ref)
    n_times = 3000
    fast = _ar1_surrogate(phis, sigmas, n_times, np.random.default_rng(11))
    noise = np.random.default_rng(11).standard_normal((4, n_times))
    slow = np.empty((4, n_times))
    for k in range(4):
        slow[k, 0] = noise[k, 0] * sigmas[k]
        for t in range(1, n_times):
            slow[k, t] = phis[k] * slow[k, t - 1] + sigmas[k] * noise[k, t]
    assert np.array_equal(fast, slow)


def test_benjamini_hochberg_matches_scipy():
    """The step-up rejection set equals ``false_discovery_control(p) <= alpha``."""
    from scipy.stats import false_discovery_control

    rng = np.random.default_rng(0)
    for alpha in (0.01, 0.05, 0.2):
        for _ in range(200):
            m = int(rng.integers(1, 40))
            pvals = rng.random(m)
            if rng.random() < 0.3:  # force ties and tiny values
                pvals = np.round(pvals, 2)
            if rng.random() < 0.3:
                pvals *= 0.02
            expected = false_discovery_control(pvals, method="bh") <= alpha
            assert np.array_equal(_benjamini_hochberg(pvals, alpha), expected)
    # non-finite p-values are treated as "not rejected"
    keep = _benjamini_hochberg(np.array([np.nan, 1e-9, np.inf]), 0.05)
    assert keep.tolist() == [False, True, False]


# --------------------------------------------------------------------------- #
# AR(1) surrogate significance: the spectral extension                        #
# --------------------------------------------------------------------------- #
BAND = dict(fmin=8.0, fmax=12.0, mt_bandwidth=None)
# Realisations of the complete null averaged over by ``_spectral_null_rate``.
NULL_SEEDS = tuple(range(8))


def _epoch(x, n_epochs, n_times):
    """Cut a continuous ``(n_sources, n_epochs * n_times)`` block into epochs."""
    return x.reshape(len(x), n_epochs, n_times).transpose(1, 0, 2)


def _ar1_epochs(n_sources, n_epochs, n_times, phi=0.95, seed=0):
    """Independent AR(1) sources: exactly the model the surrogate null assumes."""
    rng = np.random.default_rng(seed)
    innov = rng.standard_normal((n_sources, n_epochs * n_times))
    return _epoch(lfilter([1.0], [1.0, -phi], innov, axis=-1), n_epochs, n_times)


def _narrowband_epochs(n_sources, n_epochs, n_times, seed=0):
    """Independent 9-11 Hz sources, whose in-band power AR(1) cannot reproduce."""
    rng = np.random.default_rng(seed)
    x = mne.filter.filter_data(
        rng.standard_normal((n_sources, n_epochs * n_times)),
        SFREQ,
        9.0,
        11.0,
        verbose=False,
    )
    return _epoch(x, n_epochs, n_times)


def _spectral_matrix(ref, method):
    """Band-averaged connectivity matrix in the pair ordering the estimator uses.

    ``_spectral_conn_matrix`` fills only the lower triangle, and entry
    ``[j, i]`` holds the *ordered* pair ``(j, i)``, which is the entry
    ``_pair_connectivity`` reads; ``imcoh`` changes sign with that order, so the
    matrix handed to the test must be built the same way.
    """
    dense = _spectral_conn_matrix(ref, method, sfreq=SFREQ, **BAND)
    n = ref.shape[1]
    conn = np.zeros((n, n))
    for i, j in _as_pairs(n):
        conn[i, j] = conn[j, i] = dense[j, i]
    return conn


def _spectral_null_rate(method, sources, seeds=NULL_SEEDS, alpha=0.05):
    """Fraction of true-null edges flagged over ``seeds`` complete-null datasets.

    ``sources`` builds ``(n_epochs, n_sources, n_times)`` mutually independent
    signals, so every edge is a true negative and the returned fraction is the
    per-edge false-positive rate of the mask (that is, after Benjamini-Hochberg,
    since that is what the function returns).
    """
    n_sources, n_epochs, n_times = 6, 24, 400
    pairs = _as_pairs(n_sources)
    seeds = list(seeds)
    n_flagged = 0
    for seed in seeds:
        ref = sources(n_sources, n_epochs, n_times, seed=seed)
        mask = ar1_surrogate_significance(
            _spectral_matrix(ref, method),
            ref,
            method=method,
            sfreq=SFREQ,
            n_surrogates=60,
            alpha=alpha,
            random_state=seed,
            **BAND,
        )
        n_flagged += sum(bool(mask[i, j]) for i, j in pairs)
    return n_flagged / (len(seeds) * len(pairs))


@pytest.mark.parametrize("method", _SPECTRAL_METHODS)
def test_ar1_spectral_mask_is_symmetric_and_flags_the_coupled_pair(method):
    """The spectral path returns a well-formed mask and finds a lagged coupling.

    Sources 2 and 3 share a 9-11 Hz oscillation a quarter cycle apart, which
    every supported metric can see (a zero-lag coupling would be invisible to
    ``imcoh``, ``pli`` and ``wpli``); the other two sources are independent, so
    the only edge that may be flagged is ``(2, 3)``.
    """
    n_epochs, n_times, lag = 24, 400, 5  # 5 samples is a quarter cycle at 10 Hz
    rng = np.random.default_rng(0)
    x = mne.filter.filter_data(
        rng.standard_normal((4, n_epochs * n_times)), SFREQ, 9.0, 11.0, verbose=False
    )
    common = mne.filter.filter_data(
        rng.standard_normal(n_epochs * n_times), SFREQ, 9.0, 11.0, verbose=False
    )
    x[2] = 0.7 * common + 0.3 * x[2]
    x[3] = 0.7 * np.roll(common, lag) + 0.3 * x[3]
    ref = _epoch(x, n_epochs, n_times)

    conn = _spectral_matrix(ref, method)
    # The null is compared against the estimator's own value for the pair, so
    # the matrix read out of the joint call must equal the two-signal call.
    assert_allclose(
        conn[2, 3],
        _pair_connectivity(
            ref[:, [2, 3]],
            method,
            sfreq=SFREQ,
            orthogonalize=False,
            absolute=False,
            **BAND,
        ),
        atol=1e-10,
    )

    mask = ar1_surrogate_significance(
        conn, ref, method=method, sfreq=SFREQ, n_surrogates=60, random_state=0, **BAND
    )
    assert mask.shape == (4, 4)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, mask.T)
    assert not mask.diagonal().any()
    assert [(i, j) for i, j in _as_pairs(4) if mask[i, j]] == [(2, 3)]


@pytest.mark.parametrize("method", ("coh", "plv"))
def test_ar1_spectral_null_false_positive_rate_at_or_below_alpha(method):
    """Under a complete null the spectral mask flags at most ``alpha`` of edges.

    Six mutually independent AR(1) sources (:math:`\\varphi = 0.95`) over eight
    realisations, so all 120 edge decisions are true negatives and the sources
    are exactly the model the surrogate assumes. Measured here the mask flags
    0.000 (coh) and 0.008 (plv) of them. This is the rate of the returned mask,
    that is after Benjamini-Hochberg, which only ever rejects a subset of the
    edges with an uncorrected p at or below ``alpha``, so it says nothing about
    the uncorrected rate; that one is above nominal, as the warning in
    ``ar1_surrogate_significance`` documents. Sources that are not AR(1) can
    push even the corrected rate above ``alpha`` (0.092 for coherence on
    narrow-band sources), which
    ``test_ar1_spectral_null_is_anticonservative_for_narrow_band_sources``
    measures.
    """
    alpha = 0.05
    assert _spectral_null_rate(method, _ar1_epochs, alpha=alpha) <= alpha


def test_ar1_spectral_null_is_anticonservative_for_narrow_band_sources():
    """The spectral null only holds when the sources really are AR(1).

    An AR(1) process can match the lag-1 autocorrelation of a narrow-band source
    and still put almost none of its power in the analysis band, so its
    band-averaged null comes out too narrow. With the same six independent
    sources, the same 8-12 Hz coherence and the same seeds, replacing the AR(1)
    sources by 9-11 Hz ones takes the fraction of flagged true-null edges from
    0.000 to 0.092, that is from below ``alpha`` to above it. The excess moves
    with the realisation: over five blocks of eight seeds it measured 0.042 to
    0.133 for the narrow-band sources against 0.000 to 0.017 for the AR(1) ones.
    Only the ordering is asserted, since the size of the gap is a property of
    the simulation rather than of the code.
    """
    alpha = 0.05
    matched = _spectral_null_rate("coh", _ar1_epochs, alpha=alpha)
    narrow = _spectral_null_rate("coh", _narrowband_epochs, alpha=alpha)
    assert matched <= alpha
    assert narrow > matched


def test_ar1_nonnegative_metric_is_tested_one_sided():
    """A coherence far *below* its null is no evidence of coupling.

    With four independent 9-11 Hz sources the surrogate null of the
    band-averaged coherence sits at about 0.07, so a coherence of exactly zero
    lies 2.4 to 3.8 null standard deviations below it, which a two-sided reading
    would reject at ``alpha=0.05`` on every edge (p 0.0002 to 0.015, measured on
    the 60 surrogates drawn here). The one-sided rule for the non-negative
    metrics flags none of them, while the mirror-image case of an implausibly
    high coherence is flagged.
    """
    ref = _narrowband_epochs(4, 24, 400, seed=0)
    masks = []
    for value in (0.0, 0.99):
        conn = np.full((4, 4), value)
        np.fill_diagonal(conn, 0.0)
        masks.append(
            ar1_surrogate_significance(
                conn,
                ref,
                method="coh",
                sfreq=SFREQ,
                n_surrogates=60,
                random_state=0,
                **BAND,
            )
        )
    assert not masks[0].any()
    assert masks[1].sum() == 2 * len(_as_pairs(4))


def _envelope_reference():
    """Four courses with two very different kinds of envelope coupling.

    Rows 0 and 1 share a slow amplitude but have independent 9-11 Hz carriers,
    the amplitude coupling that survives pairwise orthogonalisation; rows 2 and
    3 are near copies of one another, an instantaneous coupling that
    orthogonalisation removes. The contrast is what makes the pinned masks below
    sensitive to ``orthogonalize``.
    """
    rng = np.random.default_rng(0)
    n_times = 8000

    def ar1(phi=0.9):
        return lfilter([1.0], [1.0, -phi], rng.standard_normal(n_times))

    slow = np.abs(ar1(0.995))
    carrier = mne.filter.filter_data(
        rng.standard_normal((2, n_times)), SFREQ, 9.0, 11.0, verbose=False
    )
    x2 = ar1()
    x3 = 0.85 * x2 + 0.15 * ar1()
    return np.stack([slow * carrier[0], slow * carrier[1], x2, x3])


ENVELOPE_CASES = [
    ({}, [True, False, False, False, False, True]),
    ({"absolute": True}, [True, False, False, False, False, True]),
    ({"orthogonalize": "pairwise"}, [True, False, False, False, False, False]),
    ({"envelope_lowpass": None}, [True, False, False, False, False, True]),
    ({"envelope_resample": 10.0}, [True, False, False, False, False, True]),
]


@pytest.mark.parametrize("kwargs, expected", ENVELOPE_CASES)
def test_ar1_envelope_mask_is_unchanged_by_the_spectral_extension(kwargs, expected):
    """The envelope path still gives the mask it gave before the extension.

    The expected masks were generated by running the implementation of commit
    c0fb92b, before the spectral metrics were added, on this reference and these
    settings; they are reproduced exactly by the current code. They are also
    stable, being unchanged over four surrogate seeds and 60, 100 and 200
    surrogates, so a difference means a change of behaviour rather than of
    sampling.
    """
    ref = _envelope_reference()
    conn = _envelope_corr_matrix(
        ref,
        sfreq=SFREQ,
        orthogonalize=kwargs.get("orthogonalize", False),
        absolute=kwargs.get("absolute", False),
        envelope_lowpass=kwargs.get("envelope_lowpass", 0.5),
        envelope_resample=kwargs.get("envelope_resample", None),
    )
    np.fill_diagonal(conn, 0.0)
    mask = ar1_surrogate_significance(
        conn, ref, sfreq=SFREQ, n_surrogates=100, random_state=0, **kwargs
    )
    assert [bool(mask[i, j]) for i, j in _as_pairs(4)] == expected


def test_ar1_envelope_is_tested_two_sided():
    """The signed envelope correlation is flagged on either side of its null.

    Unlike coherence, a strongly *negative* envelope correlation is a real
    effect, so the envelope keeps the two-sided p-value it has always had. The
    null-centred value is the control: it is flagged on neither side.
    """
    ref = _ar1_epochs(4, 1, 8000, phi=0.9, seed=1)[0]
    flagged = {}
    for value in (-0.9, 0.0, 0.9):
        conn = np.full((4, 4), value)
        np.fill_diagonal(conn, 0.0)
        mask = ar1_surrogate_significance(
            conn, ref, sfreq=SFREQ, n_surrogates=100, random_state=0
        )
        flagged[value] = int(mask.sum())
    n_edges = 2 * len(_as_pairs(4))
    assert flagged == {-0.9: n_edges, 0.0: 0, 0.9: n_edges}


def test_ar1_envelope_two_dimensional_reference_equals_a_one_epoch_reference():
    """Wrapping the paper's single continuous segment as one epoch changes nothing."""
    ref = _envelope_reference()
    conn = _envelope_corr_matrix(
        ref,
        sfreq=SFREQ,
        orthogonalize=False,
        absolute=False,
        envelope_lowpass=0.5,
        envelope_resample=None,
    )
    np.fill_diagonal(conn, 0.0)
    kwargs = dict(sfreq=SFREQ, n_surrogates=50, random_state=3)
    assert np.array_equal(
        ar1_surrogate_significance(conn, ref, **kwargs),
        ar1_surrogate_significance(conn, ref[np.newaxis], **kwargs),
    )


def test_ar1_spectral_requires_an_epoched_reference():
    """A spectral null needs the epoch geometry the estimate came from.

    The null of a band-averaged spectral metric moves with the number of epochs,
    and on a single segment plv, pli and wpli are identically 1, so a continuous
    reference is refused rather than silently tested against a degenerate null.
    ``method='envelope'``, the case of Nunes et al. (2020), still takes it.
    """
    ref = _ar1_epochs(3, 1, 4000)[0]
    assert ref.shape == (3, 4000)
    with pytest.raises(ValueError, match="at least 2 epochs"):
        ar1_surrogate_significance(
            np.zeros((3, 3)),
            ref,
            method="plv",
            sfreq=SFREQ,
            n_surrogates=10,
            random_state=0,
            **BAND,
        )


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
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            method="not-a-method",
            noise_cov=d["ncov"],
        )


def test_spectral_method_requires_sfreq(scenario):
    d = scenario
    with pytest.raises(ValueError, match="requires ``sfreq``"):
        pairwise_mcmv_connectivity(
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            method="coh",
            noise_cov=d["ncov"],
        )


def test_spectral_method_requires_band(scenario):
    d = scenario
    with pytest.raises(ValueError, match="requires a frequency band"):
        pairwise_mcmv_connectivity(
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            method="coh",
            sfreq=200.0,
            noise_cov=d["ncov"],
        )


def test_augmented_validates_shapes(scenario):
    d = scenario
    n = len(d["sources"])
    good = np.zeros((n, n))
    sig = np.zeros((n, n), bool)
    with pytest.raises(ValueError, match="connectivity must be shape"):
        augmented_pairwise_mcmv_connectivity(
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            np.zeros((n, n + 1)),
            sig,
            noise_cov=d["ncov"],
        )
    with pytest.raises(ValueError, match="significance must be shape"):
        augmented_pairwise_mcmv_connectivity(
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            good,
            np.zeros((n + 1, n + 1), bool),
            noise_cov=d["ncov"],
        )
    with pytest.raises(ValueError, match="positions must be"):
        augmented_pairwise_mcmv_connectivity(
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            good,
            sig,
            positions=np.zeros((n, 2)),
            noise_cov=d["ncov"],
        )


def test_orientations_must_match_the_sources(scenario):
    """A wrong-sized ``orientations`` is refused before anything is built.

    The rows are taken by position in ``sources``, so a mismatched array is not
    a harmless one: too few rows raise an ``IndexError`` from inside the pair
    loop, with an index nobody can trace back to the argument, and too many --
    or an array assembled for a different source list -- quietly orient a
    region along another region's dipole and return a connectivity matrix that
    looks perfectly ordinary. ``data``, ``forward`` and ``data_cov`` are
    ``None`` here to pin the check ahead of the beamformers: nothing may be
    solved before the caller is told.
    """
    d = scenario
    n = len(d["sources"])
    for n_rows in (n - 1, n + 1):
        bad = np.tile([0.0, 0.0, 1.0], (n_rows, 1))
        with pytest.raises(ValueError, match="orientations must be"):
            reconstruct_pairwise_mcmv(
                None, d["info"], None, None, d["sources"], orientations=bad
            )
        with pytest.raises(ValueError, match="orientations must be"):
            pairwise_mcmv_connectivity(
                None, d["info"], None, None, d["sources"], orientations=bad
            )
        with pytest.raises(ValueError, match="orientations must be"):
            augmented_pairwise_mcmv_connectivity(
                None,
                d["info"],
                None,
                None,
                d["sources"],
                np.zeros((n, n)),
                np.zeros((n, n), bool),
                positions=d["positions"],
                orientations=bad,
            )


def test_spectral_band_must_be_a_single_scalar_band(scenario):
    """A sequence of band edges would be reduced to its first band in silence.

    :func:`mne_connectivity.spectral_connectivity_epochs` accepts tuples and
    returns one matrix per band, but a connectivity matrix here has room for a
    single value per edge, so a caller asking for alpha and beta in one call
    would get alpha back and never learn that beta had been dropped.
    """
    d = scenario
    with pytest.raises(ValueError, match="estimates one band per call"):
        pairwise_mcmv_connectivity(
            None,
            d["info"],
            None,
            None,
            d["sources"],
            method="coh",
            sfreq=SFREQ,
            fmin=(8.0, 20.0),
            fmax=(12.0, 30.0),
        )
    with pytest.raises(ValueError, match="estimates one band per call"):
        ar1_surrogate_significance(
            np.zeros((2, 2)),
            np.zeros((2, 2, 400)),
            method="coh",
            sfreq=SFREQ,
            fmin=(8.0, 20.0),
            fmax=(12.0, 30.0),
        )


def test_orthogonalize_true_is_rejected(scenario):
    """``True`` is not a value any of these estimators can honour.

    Pairwise orthogonalisation is the only one implemented, so ``True`` could
    only be read as a guess at ``'pairwise'`` -- a different estimator, and one
    that double-corrects data the MCMV has already unmixed. It is refused at
    the entry point, with ``data``, ``forward`` and ``data_cov`` ``None`` here
    to pin that no beamformer is solved before the caller hears about it.
    """
    d = scenario
    n = len(d["sources"])
    with pytest.raises(ValueError, match="orthogonalize must be False"):
        pairwise_mcmv_connectivity(
            None, d["info"], None, None, d["sources"], orthogonalize=True
        )
    with pytest.raises(ValueError, match="orthogonalize must be False"):
        augmented_pairwise_mcmv_connectivity(
            None,
            d["info"],
            None,
            None,
            d["sources"],
            np.zeros((n, n)),
            np.zeros((n, n), bool),
            positions=d["positions"],
            orthogonalize=True,
        )
    with pytest.raises(ValueError, match="orthogonalize must be False"):
        ar1_surrogate_significance(
            np.zeros((2, 2)), np.zeros((2, 400)), sfreq=SFREQ, orthogonalize=True
        )


def test_max_neighbours_and_radius_are_used_as_they_were_validated(scenario):
    """The run must use the values the validation coerced, not the raw ones.

    ``max_neighbours`` is checked through ``int()`` and ``radius`` through
    ``float()``, so a count written as ``1.0`` and a bound read out of a text
    configuration both pass. If the raw value travels on from there,
    ``max_neighbours`` arrives at the neighbour search as a slice bound and
    ``radius`` as the right-hand side of a distance comparison, and the run
    dies with a ``TypeError`` about slice indices or numpy loops -- nowhere
    near the argument that caused it, and only for the callers whose data
    happen to have a significant edge to augment.
    """
    d = scenario
    sig = np.zeros((4, 4), bool)
    sig[0, 1] = sig[1, 0] = True
    sig[2, 3] = sig[3, 2] = True  # every region ends up with degree 1
    args = (
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        np.zeros((4, 4)),
        sig,
    )
    kwargs = dict(positions=d["positions"], method="envelope", noise_cov=d["ncov"])
    coerced = augmented_pairwise_mcmv_connectivity(
        *args, radius="0.2", max_neighbours=1.0, **kwargs
    )
    plain = augmented_pairwise_mcmv_connectivity(
        *args, radius=0.2, max_neighbours=1, **kwargs
    )
    # The augmented edges were re-estimated off a zero matrix, so a value that
    # moved away from zero is proof the neighbour search really ran.
    assert plain[0, 1] != 0.0
    assert_allclose(coerced, plain, rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------- #
# Previously untested paths: free-orientation forwards, the default
# ``positions``, and epoched input to the augmented estimator.
# --------------------------------------------------------------------------- #
def _free_scenario():
    """A free-orientation EEG scenario with three well-separated sources."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch = list(dict.fromkeys(montage.ch_names))[:32]
    info = mne.create_info(ch, 200.0, "eeg")
    info.set_montage(montage)
    info = (
        mne.io.RawArray(np.zeros((len(ch), 2)), info, verbose=False)
        .set_eeg_reference("average", projection=True, verbose=False)
        .info
    )
    sphere = mne.make_sphere_model("auto", "auto", info, verbose=False)
    src = mne.setup_volume_source_space(sphere=sphere, pos=30.0, verbose=False)
    fwd = mne.make_forward_solution(
        info, None, src, sphere, eeg=True, meg=False, verbose=False
    )
    assert not mne.forward.is_fixed_orient(fwd)

    rng = np.random.default_rng(5)
    gain = fwd["sol"]["data"]
    rr = fwd["source_rr"]
    # three sources spread across the volume
    order = np.argsort(np.linalg.norm(rr - rr.mean(0), axis=1))
    sources = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    oris = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    n_times = 4000
    t = np.arange(n_times) / 200.0
    env = 1.0 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    tcs = np.stack(
        [
            env * np.cos(2 * np.pi * 10 * t),
            env * np.cos(2 * np.pi * 10 * t + 0.4),  # shares the envelope
            np.cos(2 * np.pi * 10 * t + 1.1),  # flat envelope, independent
        ]
    )
    data = sum(
        np.outer(gain[:, 3 * s : 3 * s + 3] @ u, tc)
        for s, u, tc in zip(sources, oris, tcs, strict=True)
    )
    data = data + 0.05 * np.abs(data).max() * rng.standard_normal(data.shape)

    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=5.0, verbose=False)
    dcov = mne.compute_covariance(epochs, method="empirical", verbose=False)
    evoked = mne.EvokedArray(raw.get_data(), info, tmin=0.0, verbose=False)
    evoked.set_eeg_reference("average", projection=True, verbose=False)
    return info, fwd, dcov, sources, oris, evoked, epochs


def test_free_orientation_connectivity_runs_and_is_leakage_free():
    """PW-MCMV works on a free-orientation forward with explicit orientations."""
    info, fwd, dcov, sources, oris, evoked, _ = _free_scenario()

    conn = pairwise_mcmv_connectivity(
        evoked,
        info,
        fwd,
        dcov,
        sources,
        orientations=oris,
        method="envelope",
        noise_cov=mne.make_ad_hoc_cov(info, verbose=False),
        sfreq=info["sfreq"],
    )
    assert conn.shape == (3, 3)
    assert np.allclose(conn, conn.T)
    assert np.allclose(np.diag(conn), 0.0)
    assert np.all(np.abs(conn) <= 1.0)

    # The defining property: each pair's filter nulls the other pair member.
    filters = make_mcmv(
        info,
        fwd,
        dcov,
        sources=[sources[0], sources[1]],
        orientations=oris[:2],
        noise_cov=mne.make_ad_hoc_cov(info, verbose=False),
    )
    assert_allclose(filters["weights"] @ filters["leadfield"], np.eye(2), atol=1e-9)


def test_augmented_defaults_positions_from_the_forward():
    """``positions=None`` uses ``forward['source_rr'][sources]``."""
    info, fwd, dcov, sources, oris, evoked, _ = _free_scenario()
    ncov = mne.make_ad_hoc_cov(info, verbose=False)
    conn = pairwise_mcmv_connectivity(
        evoked,
        info,
        fwd,
        dcov,
        sources,
        orientations=oris,
        method="envelope",
        noise_cov=ncov,
        sfreq=info["sfreq"],
    )
    sig = np.zeros((3, 3), dtype=bool)
    sig[0, 1] = sig[1, 0] = True

    explicit = augmented_pairwise_mcmv_connectivity(
        evoked,
        info,
        fwd,
        dcov,
        sources,
        conn,
        sig,
        positions=np.asarray(fwd["source_rr"])[sources],
        orientations=oris,
        method="envelope",
        noise_cov=ncov,
        sfreq=info["sfreq"],
    )
    defaulted = augmented_pairwise_mcmv_connectivity(
        evoked,
        info,
        fwd,
        dcov,
        sources,
        conn,
        sig,
        positions=None,
        orientations=oris,
        method="envelope",
        noise_cov=ncov,
        sfreq=info["sfreq"],
    )
    assert_allclose(defaulted, explicit, rtol=1e-12)


def test_augmented_accepts_epoched_data():
    """The augmented estimator handles (n_epochs, n_channels, n_times) input."""
    info, fwd, dcov, sources, oris, _, epochs = _free_scenario()
    ncov = mne.make_ad_hoc_cov(info, verbose=False)
    data = epochs.get_data(copy=True)
    picks = [epochs.ch_names.index(c) for c in epochs.ch_names]
    data = data[:, picks]

    conn = pairwise_mcmv_connectivity(
        data,
        info,
        fwd,
        dcov,
        sources,
        orientations=oris,
        method="envelope",
        noise_cov=ncov,
        sfreq=info["sfreq"],
    )
    sig = np.zeros((3, 3), dtype=bool)
    sig[0, 1] = sig[1, 0] = True
    out = augmented_pairwise_mcmv_connectivity(
        data,
        info,
        fwd,
        dcov,
        sources,
        conn,
        sig,
        orientations=oris,
        method="envelope",
        noise_cov=ncov,
        sfreq=info["sfreq"],
    )
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T)
    # Non-significant edges keep their PW value; the significant one is redone.
    assert out[0, 2] == conn[0, 2]


def test_the_pair_is_reconstructed_in_the_order_it_was_asked_for(scenario):
    """Row 0 of a pair's reconstruction is the pair's first source.

    Checked against an independently built two-source MCMV rather than against
    the same function called the other way round. Comparing the function with
    itself cannot see this: reversing the order inside the loop reverses both
    calls, so the relationship between them is unchanged and the swap is
    invisible. Only an outside reference pins which row is which.
    """
    d = scenario
    pair = [d["A"], d["B"]]
    pairs, tcs = reconstruct_pairwise_mcmv(
        d["evoked"].data, d["info"], d["fwd"], d["dcov"], pair, noise_cov=d["ncov"]
    )
    assert pairs == [(0, 1)]

    reference = (
        make_mcmv(
            d["info"],
            d["fwd"],
            d["dcov"],
            sources=pair,
            noise_cov=d["ncov"],
            weight_norm="unit-gain",
        )["weights"]
        @ d["evoked"].data
    )
    assert_allclose(tcs[0], reference, rtol=1e-5, atol=1e-14)
    # And the rows are not interchangeable: the two sources differ, so a swap
    # would be a real error rather than a relabelling.
    assert not np.allclose(reference[0], reference[1], rtol=1e-3)


def test_neighbours_are_gathered_around_both_sources_out_to_the_radius():
    """Conductors beside either end of a pair are found, the boundary included.

    Indirect leakage on an edge is carried by regions close to *either* of its
    two sources, so the search has to run from both. Were it run from the first
    alone, every conductor sitting beside the second source would go unmodelled
    and the edge would keep exactly the indirect bias the augmentation exists to
    remove -- silently, since the returned matrix looks no different. The radius
    is likewise a "within radius" criterion rather than a strictly-inside one:
    on a regular volume grid whole shells of candidates sit at exactly the
    search radius, and excluding them would drop real conductors.
    """
    # 0 and 1 are the pair; 2 lies exactly at the radius from source 0, while 3
    # and 4 lie close to source 1 and far outside the radius of source 0.
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.19, 0.0, 0.0],
            [0.2, 0.0, 0.02],
        ],
        float,
    )
    radius = 0.04
    assert np.linalg.norm(pos[2] - pos[0]) == radius  # on the boundary, exactly
    sig = np.zeros((5, 5), bool)
    sig[0, 1] = sig[1, 0] = True
    for k in (2, 3, 4):  # each conductor carries one significant connection
        sig[k, 1] = sig[1, k] = True
    degree = sig.sum(1)

    chosen = _select_neighbours(0, 1, pos, sig, degree, radius, max_neighbours=2)
    # the boundary region of source 0 first, then source 1's two, nearer first
    assert chosen == [2, 3, 4]


def test_default_positions_keep_each_region_with_its_own_source(scenario):
    """``positions=None`` reads ``source_rr`` in the order ``sources`` was given.

    Those defaults are the only thing telling the neighbour search where each
    region sits. Permute them against ``sources`` and the search still returns a
    plausible-looking neighbour set, merely the wrong one: here the conductor C,
    2.4 cm from A, is no longer recognised as A's neighbour, so the spurious
    A--B edge is rebuilt without nulling it and keeps its full indirect-leakage
    bias. Nothing downstream can flag that, hence the check against positions
    passed in explicitly -- and against the reversed labelling, to show the
    geometry really does tell the two apart.
    """
    d = scenario
    rr = np.asarray(d["fwd"]["source_rr"], float)
    sig = np.zeros((4, 4), bool)
    sig[0, 1] = sig[1, 0] = True  # the spurious A-B edge, treated as significant
    sig[1, 2] = sig[2, 1] = True  # C-B genuine -> C is a conductor of degree 1
    # Only significant edges are re-estimated, so the matrix being refined has
    # no bearing on what is compared here.
    kwargs = dict(
        method="envelope",
        noise_cov=d["ncov"],
        absolute=False,
    )
    args = (
        d["evoked"],
        d["info"],
        d["fwd"],
        d["dcov"],
        d["sources"],
        np.zeros((4, 4)),
        sig,
    )
    defaulted = augmented_pairwise_mcmv_connectivity(*args, positions=None, **kwargs)
    explicit = augmented_pairwise_mcmv_connectivity(
        *args, positions=rr[d["sources"]], **kwargs
    )
    misordered = augmented_pairwise_mcmv_connectivity(
        *args, positions=rr[d["sources"][::-1]], **kwargs
    )

    assert_allclose(defaulted, explicit, rtol=1e-12, atol=1e-12)
    # With the regions correctly placed, nulling C brings the A-B edge back to
    # its true value; mislabelled, C is out of range of both ends of the pair,
    # no neighbour is added and the leakage bias survives untouched.
    truth = _env_corr(d["s_a"], d["s_b"])  # ~ -0.153
    assert abs(explicit[0, 1] - truth) < 0.01
    assert abs(misordered[0, 1] - truth) > 0.03


def test_filling_the_neighbour_allowance_is_not_reported_as_exceeding_it(scenario):
    """An order of exactly ``2 + 2 * max_neighbours`` is the intended limit.

    The order warning exists to catch a neighbour set that has escaped its cap.
    A pair that merely spends its whole allowance -- one neighbour from each
    source at ``max_neighbours=1`` -- is the ordinary case in a well-connected
    analysis, and warning on it would attach a RuntimeWarning to most edges,
    burying the genuine alarm and breaking anyone who runs with warnings as
    errors.
    """
    import warnings

    d = scenario
    sig = np.zeros((4, 4), bool)
    sig[0, 1] = sig[1, 0] = True
    sig[2, 3] = sig[3, 2] = True  # every region ends up with degree 1
    degree = sig.sum(1)
    radius, max_neighbours = 0.2, 1
    neighbours = _select_neighbours(
        0, 1, d["positions"], sig, degree, radius, max_neighbours
    )
    # the allowance is genuinely spent: one neighbour from each source
    assert 2 + len(neighbours) == 2 + 2 * max_neighbours

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        conn = augmented_pairwise_mcmv_connectivity(
            d["evoked"],
            d["info"],
            d["fwd"],
            d["dcov"],
            d["sources"],
            np.zeros((4, 4)),
            sig,
            positions=d["positions"],
            method="envelope",
            noise_cov=d["ncov"],
            radius=radius,
            max_neighbours=max_neighbours,
        )
    assert [r for r in records if "exceeds the cap" in str(r.message)] == []
    assert np.all(np.isfinite(conn))


def test_ar1_fit_recovers_the_coefficient_and_innovation_scale_it_was_given():
    """The surrogate null is only ever as good as the AR(1) summary it is built on.

    Every surrogate is drawn from these two numbers alone, so they fix the
    spectrum and the variance of the null against which a real connectivity
    value is judged. A coefficient read off the wrong lag would whiten the
    surrogates and narrow the null, and an innovation scale that keeps the part
    of the signal the coefficient already explains would widen it; either way the
    mask stops reflecting the smoothness of the data it came from. Rows ranging
    from nearly a random walk to nearly white noise, each with its own
    innovation scale, are recovered here well inside the tolerance the null
    needs.
    """
    phis = np.array([0.95, 0.9, 0.7, 0.3, 0.05])
    sigmas = np.array([0.5, 2.0, 1.0, 3.0, 1.5])
    rng = np.random.default_rng(0)
    ref = np.stack(
        [
            sigma * lfilter([1.0], [1.0, -phi], rng.standard_normal(40000))
            for phi, sigma in zip(phis, sigmas, strict=True)
        ]
    )
    fitted_phis, fitted_sigmas = _ar1_fit(ref)
    assert_allclose(fitted_phis, phis, atol=0.02)
    assert_allclose(fitted_sigmas, sigmas, rtol=0.02)


def test_every_surrogate_epoch_keeps_the_autocorrelation_of_its_own_region(
    monkeypatch,
):
    """A region's null must be drawn from that region's own temporal smoothness.

    The coefficients are fitted on the epochs concatenated and the surrogates are
    drawn as one continuous block per region, so the region axis is folded away
    and unfolded again either side of the draw. If those two steps disagree, the
    surrogates are still AR(1) and the mask still looks plausible, but a smooth
    region is handed a white region's null: the smooth region has far fewer
    independent samples, its null is far wider, and judging it against the narrow
    one turns chance coincidences into significant edges. Here one region is
    nearly a random walk and the other nearly white, and every epoch of every
    surrogate is required to carry the lag-1 correlation of the region it stands
    for.
    """
    phis = (0.95, 0.05)
    n_surrogates, n_epochs, n_times = 8, 6, 1024
    ref = np.concatenate(
        [
            _ar1_epochs(1, n_epochs, n_times, phi=phi, seed=k)
            for k, phi in enumerate(phis)
        ],
        axis=1,
    )
    assert ref.shape == (n_epochs, len(phis), n_times)

    drawn = []
    spectral_conn_matrix = _spectral_conn_matrix

    def _record(surrogate, *args, **kwargs):
        drawn.append(np.array(surrogate))
        return spectral_conn_matrix(surrogate, *args, **kwargs)

    monkeypatch.setattr("advance_beamlab._connectivity._spectral_conn_matrix", _record)
    ar1_surrogate_significance(
        np.zeros((len(phis), len(phis))),
        ref,
        method="plv",
        sfreq=SFREQ,
        n_surrogates=n_surrogates,
        random_state=0,
        **BAND,
    )

    surrogates = np.stack(drawn)
    assert surrogates.shape == (n_surrogates, n_epochs, len(phis), n_times)
    lag1 = np.array(
        [
            [
                [np.corrcoef(block[1:], block[:-1])[0, 1] for block in epoch]
                for epoch in surrogate
            ]
            for surrogate in surrogates
        ]
    ).mean(axis=0)
    assert lag1.shape == (n_epochs, len(phis))
    assert_allclose(lag1, np.broadcast_to(phis, lag1.shape), atol=0.1)


def test_a_pair_is_reconstructed_with_the_orientations_of_its_own_two_sources():
    """Each pair gets its own two orientation rows, in the order it was asked for.

    On a free-orientation forward the orientations are what collapse the
    three-column leadfield of a region down to the single column the beamformer
    constrains. A pair handed its partner's orientation is therefore constrained
    to dipoles pointing in directions nobody asked about: the unit-gain and
    zero-gain conditions then hold for those dipoles rather than for the
    requested regions, and the reconstruction -- with any connectivity computed
    from it -- describes different sources than the caller believes. The
    amplitude is pinned in the same breath, because the reconstruction is a
    physical time course: falling back to unit gain instead of the requested
    normalisation rescales every row by a couple of hundred here, and a user
    comparing regions across analyses would be comparing different units.
    """
    info, fwd, dcov, sources, oris, evoked, _ = _free_scenario()
    kwargs = dict(
        noise_cov=mne.make_ad_hoc_cov(info, verbose=False),
        reg=0.1,
        weight_norm="array-gain",
    )
    pairs, tcs = reconstruct_pairwise_mcmv(
        evoked, info, fwd, dcov, sources, orientations=oris, **kwargs
    )
    assert pairs == _as_pairs(len(sources))

    for (i, j), tc in zip(pairs, tcs, strict=True):
        reference = (
            make_mcmv(
                info,
                fwd,
                dcov,
                sources=[sources[i], sources[j]],
                orientations=oris[[i, j]],
                **kwargs,
            )["weights"]
            @ evoked.data
        )
        assert_allclose(tc, reference, rtol=1e-9, atol=0.0)
        # The two orientation rows are not interchangeable: taking them the
        # other way round is a different beamformer, not a relabelling.
        swapped = (
            make_mcmv(
                info,
                fwd,
                dcov,
                sources=[sources[i], sources[j]],
                orientations=oris[[j, i]],
                **kwargs,
            )["weights"]
            @ evoked.data
        )
        assert np.linalg.norm(swapped[0] - reference[0]) > 0.5 * np.linalg.norm(
            reference[0]
        )

    # Array gain returns a value in measurement units; unit gain returns a
    # dipole moment, smaller here by more than two orders of magnitude.
    unit_gain = (
        make_mcmv(
            info,
            fwd,
            dcov,
            sources=[sources[0], sources[1]],
            orientations=oris[[0, 1]],
            noise_cov=kwargs["noise_cov"],
            reg=kwargs["reg"],
            weight_norm=None,
        )["weights"]
        @ evoked.data
    )
    assert np.linalg.norm(tcs[0][0]) > 10 * np.linalg.norm(unit_gain[0])


def test_a_pair_with_no_neighbours_keeps_its_pairwise_value():
    """With nothing to augment, the rebuilt edge must be the pairwise edge itself.

    When the neighbour search returns nothing -- because the augmentation is
    switched off, or because no region near the pair carries a significant
    connection of its own -- the augmented beamformer constrains exactly the two
    regions of the pair, so it *is* the PW-MCMV filter and the edge cannot
    change. If it does, something the caller asked for went astray on the way to
    the beamformer, and the user reads the difference between a significant edge
    and its neighbours as an effect of the augmentation when it is really a
    disagreement between the two estimators. ``imcoh`` is the metric here
    deliberately: it changes sign with the order of the pair, so it also sees
    the two rows arriving the other way round, which a symmetric metric such as
    the envelope correlation cannot. The weight normalisation is the one thing
    the value itself cannot show -- every metric here is invariant to rescaling
    a row -- so it is pinned through the caveat that ``'unit-noise-gain'``
    without a measured noise covariance carries: that caveat has to reach the
    user for the rebuilt edge as well, not only for the pairwise pass.
    """
    info, fwd, dcov, sources, oris, _, epochs = _free_scenario()
    kwargs = dict(
        method="imcoh",
        sfreq=info["sfreq"],
        fmin=8.0,
        fmax=12.0,
        orientations=oris,
        noise_cov=None,
        reg=0.1,
        weight_norm="unit-noise-gain",
    )
    with pytest.warns(RuntimeWarning, match="unit-noise-gain"):
        pw = pairwise_mcmv_connectivity(epochs, info, fwd, dcov, sources, **kwargs)
    sig = np.zeros((3, 3), dtype=bool)
    sig[0, 1] = sig[1, 0] = True

    with pytest.warns(RuntimeWarning, match="unit-noise-gain"):
        aug = augmented_pairwise_mcmv_connectivity(
            epochs, info, fwd, dcov, sources, pw, sig, max_neighbours=0, **kwargs
        )
    # The edge sits well away from zero, so a sign flip or a filter rebuilt with
    # a different regularisation is a visible change rather than round-off.
    assert abs(pw[0, 1]) > 0.05
    assert_allclose(aug, pw, rtol=0.0, atol=1e-9)


def test_ar1_threshold_is_set_by_the_unbiased_null_spread_and_two_tails():
    """How much connectivity an edge needs before it is flagged, and why.

    The function returns a decision and nothing else, so the way that decision
    is formed is only visible in the connectivity value at which it flips. For
    this reference and these three surrogates the single edge needs 0.175: an
    envelope correlation of 0.15 is not evidence of coupling and 0.20 is. Two
    things put the threshold there. The null spread is the unbiased sample
    estimate over the surrogates, which for a handful of them is appreciably
    wider than the population one: dividing by the number of surrogates rather
    than by one less inflates every z-score by 22 per cent here and would move
    the threshold down to 0.126. And the signed envelope correlation is judged
    on a two-sided p-value, so spending the whole of ``alpha`` on the upper tail
    would move it down to 0.133. Either way an edge this null cannot support is
    reported to the user as a genuine coupling. Three surrogates is the sparse
    extreme of the permitted range, where the correction is largest; it falls
    off as one over the surrogate count and is 0.25 per cent at the default 200.
    """
    ref = _ar1_epochs(2, 1, 3000, phi=0.9, seed=1)[0]
    flagged = {}
    for value in (0.15, 0.20):
        conn = np.array([[0.0, value], [value, 0.0]])
        mask = ar1_surrogate_significance(
            conn, ref, sfreq=SFREQ, n_surrogates=3, random_state=0
        )
        assert np.array_equal(mask, mask.T)
        flagged[value] = bool(mask[0, 1])
    assert flagged == {0.15: False, 0.20: True}


def test_ar1_null_is_standardised_on_the_same_scale_as_the_value_tested():
    """The Fisher transform is applied to the surrogates, not only to the data.

    The z-score is a distance in Fisher-:math:`z` units, so the null it is
    measured against has to be in those units too: the transform stretches the
    upper end of the range hard (0.6 goes to 0.69 but 0.9 goes to 1.47), and a
    null whose values sit up there is much wider after it than before. Phase
    locking on the minimum of two epochs is the clearest case: its null sits at
    0.63 with a spread of 0.19, which transformed is 0.83 with a spread of 0.40,
    so an untransformed null is centred too low and less than half as wide. The
    consequence is a threshold in the wrong place: over eight surrogate seeds a
    plv of 0.80 between two independent sources is below the threshold every
    time (it ranges over 0.84 to 0.90) and would be above an untransformed one
    every time (0.70 to 0.74). The genuinely extreme 0.95 is flagged, so what is
    pinned is a threshold and not a mask that never fires.
    """
    ref = _ar1_epochs(2, 2, 400, phi=0.95, seed=1)
    flagged = {}
    for value in (0.80, 0.95):
        conn = np.array([[0.0, value], [value, 0.0]])
        mask = ar1_surrogate_significance(
            conn,
            ref,
            method="plv",
            sfreq=SFREQ,
            n_surrogates=60,
            random_state=0,
            **BAND,
        )
        flagged[value] = bool(mask[0, 1])
    assert flagged == {0.80: False, 0.95: True}


def test_ar1_degenerate_edge_is_silenced_however_strong_its_connectivity():
    """A collapsed null silences its own edge, and only its own edge.

    A degenerate edge is given an infinite null spread rather than a nominal
    one, so its z-score is zero whatever connectivity was measured: with no
    surrogate spread to compare against, there is no strength of coupling that
    could be evidence. A nominal spread of one Fisher-:math:`z` unit would
    instead flag anything above 0.96, which is exactly the region a collapsed
    reconstruction lands in. Here the first source is flat, so the two edges
    that touch it are degenerate while the third is not, and a near-perfect
    0.999 is placed on all three: the live edge is flagged and the two dead ones
    are not, so the substitution is per edge and does not silence a real
    coupling that happens to share the matrix with a dead one.
    """
    ref = np.vstack([np.zeros(4000), _ar1_epochs(2, 1, 4000, phi=0.9, seed=1)[0]])
    conn = np.full((3, 3), 0.999)
    np.fill_diagonal(conn, 0.0)
    # the module sets the log level to ERROR, which silences mne.utils.warn
    with mne.use_log_level("warning"), pytest.warns(RuntimeWarning, match="2 of 3"):
        mask = ar1_surrogate_significance(
            conn, ref, sfreq=SFREQ, n_surrogates=20, random_state=0
        )
    assert [bool(mask[i, j]) for i, j in _as_pairs(3)] == [False, False, True]


def test_a_perfect_correlation_survives_the_fisher_transform():
    """A metric that reaches exactly 1 must not decide its own test.

    ``pli`` and ``plv`` are bounded magnitudes that attain 1 exactly -- every
    epoch agreeing on the sign of the imaginary coherency, or on the phase
    difference, is all it takes, and a reference at the two-epoch minimum
    reaches it readily. The z-score is a distance measured in Fisher-:math:`z`
    units, so an unbounded transform breaks the test in both directions: an edge
    whose measured value is 1 is significant however wide its null, and a single
    surrogate that reaches 1 takes that edge's null mean and spread with it and
    has it silenced as degenerate instead. Either way the decision is made by
    the transform rather than by the evidence. Holding it just inside the
    singularity keeps a perfect correlation what it should be -- the top of the
    scale -- while leaving everything a real estimate can resolve untouched.
    """
    r = np.array([-1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0])
    z = _fisher_z(r)
    assert np.isfinite(z).all()
    # still a strictly increasing odd map, and untouched away from the ends
    assert (np.diff(z) > 0).all()
    assert_allclose(z, -z[::-1], rtol=1e-12)
    assert_allclose(z[1:-1], np.arctanh(r[1:-1]), rtol=1e-12)
    # the bound sits within a millionth of the singularity, so no correlation
    # two estimates could tell apart is flattened onto it
    assert 1.0 - np.tanh(z[-1]) < 1e-5

    # a null one of whose surrogates reached 1 still has a usable mean and spread
    null_z = _fisher_z(np.array([0.4, 0.6, 1.0]))
    assert np.isfinite(null_z.mean()) and np.isfinite(null_z.std(ddof=1))


def test_imcoh_is_antisymmetric_while_the_other_metrics_are_symmetric():
    """ImCoh changes sign with the pair order, so the matrix must too.

    Every other supported metric is a symmetric function of its two arguments,
    and writing one scalar into both triangles is right for them. The imaginary
    part of coherency is not: ImCoh(a, b) = -ImCoh(b, a), because swapping the
    pair conjugates the cross-spectrum. Mirroring it would report the same lag
    direction for both orders of every pair, so half the matrix would say the
    opposite of the truth about which region leads -- and it would do so with no
    outward sign, since a mirrored matrix looks exactly like a correct symmetric
    one. mne-connectivity avoids the question by filling only one triangle.
    """
    info, fwd, dcov, sources, oris, _, epochs = _free_scenario()
    common = dict(
        sfreq=info["sfreq"],
        fmin=8.0,
        fmax=12.0,
        orientations=oris,
        noise_cov=None,
        reg=0.1,
    )
    imcoh = np.asarray(
        pairwise_mcmv_connectivity(
            epochs, info, fwd, dcov, sources, method="imcoh", **common
        )
    )
    assert_allclose(imcoh, -imcoh.T, atol=1e-12)
    # Not trivially zero: an all-zero matrix would satisfy the line above.
    assert np.abs(imcoh).max() > 1e-6

    for method in ("coh", "plv"):
        sym = np.asarray(
            pairwise_mcmv_connectivity(
                epochs, info, fwd, dcov, sources, method=method, **common
            )
        )
        assert_allclose(sym, sym.T, atol=1e-12)
