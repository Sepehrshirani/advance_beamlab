r"""Tests for pairwise and augmented-pairwise MCMV connectivity (Nunes et al., 2020)."""
# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import mne
import numpy as np
import pytest

from advance_beamlab import (
    ar1_surrogate_significance,
    augmented_pairwise_mcmv_connectivity,
    make_mcmv,
    pairwise_mcmv_connectivity,
    reconstruct_pairwise_mcmv,
)
from advance_beamlab._connectivity import (
    _ar1_fit,
    _ar1_surrogate,
    _as_pairs,
    _benjamini_hochberg,
    _envelope_corr_matrix,
    _epoched,
    _pair_connectivity,
    _select_neighbours,
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


def test_ar1_rejects_non_envelope_methods():
    """The AR(1) surrogate is defined for envelope correlation only."""
    ref = np.random.default_rng(0).standard_normal((3, 4000))
    conn = np.zeros((3, 3))
    with pytest.raises(ValueError, match="method='envelope' only"):
        ar1_surrogate_significance(
            conn, ref, method="plv", sfreq=SFREQ, n_surrogates=10, random_state=0
        )


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
