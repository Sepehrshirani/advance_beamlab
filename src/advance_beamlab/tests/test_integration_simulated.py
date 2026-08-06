# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause
"""Cross-algorithm integration tests on simulated data with known ground truth.

These tests need no downloaded dataset, so unlike ``test_integration_sample.py``
they run on every CI machine. They are the suite's only end-to-end coverage when
the ``sample`` dataset is absent, and they are written against *ground truth*
rather than self-consistency: sources are injected at known grid points with a
known correlation, and each algorithm is asked to recover them.

Two properties are deliberately exercised here and nowhere else:

* **Physical units.** The simulated amplitudes are chosen so the sensor data
  lands in the SI range MNE actually produces (volts for EEG), rather than the
  convenient O(1) values a hand-rolled fixture tends to use. Algorithms that
  quietly assume O(1) leadfields or data -- the failure mode that made the sparse
  Bayesian covariance singular on real recordings -- are caught here.
* **Scale invariance.** Where a quantity is mathematically invariant to the units
  of the data (a beamformer's unit-gain output, a correlation, a model covariance
  up to its scale), the test asserts that invariance directly.
"""

import mne
import numpy as np
import pytest
from mne.beamformer import apply_lcmv, make_lcmv

from advance_beamlab import (
    apply_mcmv,
    apply_mcmv_cov,
    make_abmc,
    make_mcmv,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
    sbl_covariance,
    scan_mcmv,
)

# Sensor-level amplitude the simulation targets, in volts -- the order of
# magnitude of a real scalp EEG recording.
_EEG_SCALE = 2e-5


@pytest.fixture(scope="module")
def sphere_setup():
    """A self-contained EEG sphere model with a coarse volume source grid."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = list(dict.fromkeys(montage.ch_names))
    info = mne.create_info(ch_names, sfreq=200.0, ch_types="eeg")
    info.set_montage(montage)
    info["bads"] = []

    sphere = mne.make_sphere_model("auto", "auto", info, verbose=False)
    src = mne.setup_volume_source_space(sphere=sphere, pos=25.0, verbose=False)
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
    )
    fwd = mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=False, verbose=False
    )
    return info, fwd


def _pick_two_sources(fwd, target_cm=7.0):
    """Two grid indices roughly ``target_cm`` apart, the first near the centre."""
    rr = fwd["source_rr"]
    first = int(np.argmin(np.linalg.norm(rr - rr.mean(0), axis=1)))
    dist_cm = np.linalg.norm(rr - rr[first], axis=1) * 100
    second = int(np.argmin(np.abs(dist_cm - target_cm)))
    return [first, second]


def _simulate(info, fwd, sources, rho, *, n_epochs=40, n_times=120, seed=0):
    """Epoched EEG with two sources of exact correlation ``rho``.

    The two time courses are 10 Hz oscillations phase-shifted by ``arccos(rho)``,
    so their correlation is exactly ``rho``. Returns the epochs together with
    shrinkage-regularised active and baseline covariances -- the estimator that is
    correct for real recordings and the one the examples use.
    """
    rng = np.random.default_rng(seed)
    gain = fwd["sol"]["data"]
    lead = gain[:, sources]  # (n_channels, 2)
    n_channels = lead.shape[0]

    times = np.arange(n_times) / info["sfreq"] - 0.2
    active = times >= 0.0
    phi = np.arccos(rho)

    # Scale the source amplitude so the sensor field lands at _EEG_SCALE.
    amp = _EEG_SCALE / np.abs(lead).max()

    data = np.empty((n_epochs, n_channels, n_times))
    for e in range(n_epochs):
        phase = rng.uniform(0, 2 * np.pi)
        s = np.zeros((2, n_times))
        s[0, active] = np.cos(2 * np.pi * 10 * times[active] + phase)
        s[1, active] = np.cos(2 * np.pi * 10 * times[active] + phase + phi)
        noise = 0.05 * _EEG_SCALE * rng.standard_normal((n_channels, n_times))
        data[e] = amp * (lead @ s) + noise

    epochs = mne.EpochsArray(
        data, info.copy(), tmin=times[0], baseline=None, verbose=False
    )
    data_cov = mne.compute_covariance(
        epochs, tmin=0.0, tmax=None, method="shrunk", verbose=False
    )
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="shrunk", verbose=False
    )
    return epochs, data_cov, noise_cov, amp


# --------------------------------------------------------------------------- #
# MCMV against LCMV, with ground truth
# --------------------------------------------------------------------------- #
def test_mcmv_recovers_correlated_amplitude_where_lcmv_cancels(sphere_setup):
    """MCMV holds the true amplitude at rho=0.95; LCMV loses most of it.

    This is the core claim of Moiseev et al. (2011) and the one the simulation
    example illustrates. Both beamformers read out *unit gain*, so the recovered
    amplitude is directly comparable to the injected one.
    """
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    epochs, data_cov, noise_cov, amp = _simulate(info, fwd, sources, rho=0.95)
    evoked = epochs.average()

    mcmv = make_mcmv(
        info, fwd, data_cov, sources, noise_cov=noise_cov, weight_norm="unit-gain"
    )
    lcmv = make_lcmv(
        info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori=None,
        weight_norm=None,
        depth=None,
        verbose=False,
    )
    s_mcmv = apply_mcmv(evoked, mcmv)
    s_lcmv = apply_lcmv(evoked, lcmv, verbose=False).data[sources]

    post = evoked.times >= 0.0
    truth = amp / np.sqrt(2)  # RMS of a unit-amplitude cosine
    rms_mcmv = np.sqrt(np.mean(s_mcmv[:, post] ** 2, axis=1))
    rms_lcmv = np.sqrt(np.mean(s_lcmv[:, post] ** 2, axis=1))

    # MCMV recovers the injected amplitude; LCMV is strongly attenuated.
    assert np.all(rms_mcmv > 0.7 * truth)
    assert np.all(rms_lcmv < 0.6 * rms_mcmv)


def test_mcmv_zero_gain_constraint_is_exact(sphere_setup):
    """W H = I holds to machine precision in the raw sensor space."""
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    _, data_cov, noise_cov, _ = _simulate(info, fwd, sources, rho=0.9)

    filters = make_mcmv(info, fwd, data_cov, sources, noise_cov=noise_cov)
    gain = fwd["sol"]["data"]
    fwd_ch = list(fwd["sol"]["row_names"])
    idx = [fwd_ch.index(ch) for ch in filters["ch_names"]]
    H = gain[np.ix_(idx, sources)]
    assert_almost = filters["weights"] @ H
    np.testing.assert_allclose(assert_almost, np.eye(len(sources)), atol=1e-10)


def test_mcmv_unit_gain_is_scale_invariant(sphere_setup):
    """Scaling the data by s scales the unit-gain output by exactly s."""
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    epochs, data_cov, noise_cov, _ = _simulate(info, fwd, sources, rho=0.9)
    evoked = epochs.average()

    s = 1e3
    scaled_data = mne.Covariance(
        data_cov.data * s**2,
        data_cov.ch_names,
        list(data_cov["bads"]),
        list(data_cov["projs"]),
        nfree=data_cov.nfree,
    )
    scaled_noise = mne.Covariance(
        noise_cov.data * s**2,
        noise_cov.ch_names,
        list(noise_cov["bads"]),
        list(noise_cov["projs"]),
        nfree=noise_cov.nfree,
    )
    a = apply_mcmv(evoked, make_mcmv(info, fwd, data_cov, sources, noise_cov=noise_cov))
    scaled_evoked = evoked.copy()
    scaled_evoked.data *= s
    b = apply_mcmv(
        scaled_evoked,
        make_mcmv(info, fwd, scaled_data, sources, noise_cov=scaled_noise),
    )
    np.testing.assert_allclose(b, a * s, rtol=1e-8)


def test_apply_mcmv_cov_matches_time_domain(sphere_setup):
    """W R W^T equals the covariance of the reconstructed time courses."""
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    epochs, data_cov, noise_cov, _ = _simulate(info, fwd, sources, rho=0.8)

    filters = make_mcmv(info, fwd, data_cov, sources, noise_cov=noise_cov)
    src_cov = apply_mcmv_cov(data_cov, filters)

    picks = [epochs.ch_names.index(ch) for ch in filters["ch_names"]]
    active = epochs.copy().crop(tmin=0.0).get_data(copy=True)[:, picks]
    tc = np.einsum("sc,ect->est", filters["weights"], active)
    flat = tc.transpose(1, 0, 2).reshape(len(sources), -1)
    empirical = np.cov(flat)
    # Same structure up to the covariance estimator's shrinkage.
    corr_a = src_cov / np.outer(*(2 * [np.sqrt(np.diag(src_cov))]))
    corr_b = empirical / np.outer(*(2 * [np.sqrt(np.diag(empirical))]))
    assert abs(corr_a[0, 1] - corr_b[0, 1]) < 0.15


# --------------------------------------------------------------------------- #
# Sequential source discovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("localizer", ["mai", "mpz"])
def test_scan_discovers_both_injected_sources(sphere_setup, localizer):
    """The greedy MCMV scan recovers the two injected grid points."""
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    _, data_cov, noise_cov, _ = _simulate(info, fwd, sources, rho=0.9)

    result = scan_mcmv(
        info,
        fwd,
        data_cov,
        localizer=localizer,
        n_sources=2,
        noise_cov=noise_cov,
    )
    assert set(result["sources"]) == set(sources)
    # Both are genuine sources, so both pseudo-Z values are well above baseline.
    assert np.all(np.asarray(result["pseudo_z"]) > 1.0)


def test_scan_filters_agree_with_make_mcmv(sphere_setup):
    """The filters the scan returns are exactly make_mcmv's for that source set."""
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    _, data_cov, noise_cov, _ = _simulate(info, fwd, sources, rho=0.9)

    result = scan_mcmv(
        info, fwd, data_cov, localizer="mai", n_sources=2, noise_cov=noise_cov
    )
    direct = make_mcmv(info, fwd, data_cov, result["sources"], noise_cov=noise_cov)
    np.testing.assert_allclose(
        result["filters"]["weights"], direct["weights"], rtol=1e-10
    )


# --------------------------------------------------------------------------- #
# ReciPSIICOS end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["recipsiicos", "whitened"])
def test_recipsiicos_end_to_end(sphere_setup, method):
    """The rank curve, the chosen rank and the beamformer are all well formed."""
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    _, data_cov, noise_cov, _ = _simulate(info, fwd, sources, rho=0.9)

    ranks, p_pwr, p_cor, kstar = recipsiicos_rank_curve(
        fwd, info, method=method, noise_cov=noise_cov, return_optimal=True
    )
    assert ranks[0] == 1 and ranks[-1] == len(ranks)
    assert np.all((p_pwr >= -1e-9) & (p_pwr <= 1 + 1e-9))
    assert np.all((p_cor >= -1e-9) & (p_cor <= 1 + 1e-9))
    assert 1 <= kstar <= len(ranks)
    # The selected rank must retain most of the source-power subspace; a rank
    # that empties the covariance is the failure mode this guards against.
    assert p_pwr[kstar - 1] > 0.5

    filters = make_recipsiicos_lcmv(
        info, fwd, data_cov, rank=kstar, method=method, noise_cov=noise_cov
    )
    stc = apply_lcmv(_simulate(info, fwd, sources, rho=0.9)[0].average(), filters)
    assert stc.data.shape[0] == fwd["nsource"]
    assert np.all(np.isfinite(stc.data))


# --------------------------------------------------------------------------- #
# ABMC on realistically scaled data
# --------------------------------------------------------------------------- #
def _spike(n, t0, width=6.0):
    """A unit-amplitude biphasic (derivative-of-Gaussian) transient."""
    t = np.arange(n)
    x = -(t - t0) / width * np.exp(-((t - t0) ** 2) / (2 * width**2))
    return x / np.abs(x).max()


def test_sbl_covariance_on_si_scale_data(sphere_setup):
    """The sparse Bayesian covariance must not depend on the units of the data.

    A hyperparameter initialisation that assumes O(1) leadfields makes
    ``R = G a G^T + Lambda`` singular for data in SI units, which is what real
    MNE recordings always are. This asserts both that the fit succeeds and that
    it is equivariant under a change of units.
    """
    info, fwd = sphere_setup
    sources = _pick_two_sources(fwd)
    epochs, data_cov, noise_cov, _ = _simulate(info, fwd, sources, rho=0.5)

    cov = sbl_covariance(info, fwd, data_cov, noise_cov=noise_cov)
    assert np.all(np.isfinite(cov.data))
    # A covariance must be positive definite.
    assert np.linalg.eigvalsh(cov.data).min() > 0

    s = 1e6
    scaled = mne.Covariance(
        data_cov.data * s**2,
        data_cov.ch_names,
        list(data_cov["bads"]),
        list(data_cov["projs"]),
        nfree=data_cov.nfree,
    )
    scaled_noise = mne.Covariance(
        noise_cov.data * s**2,
        noise_cov.ch_names,
        list(noise_cov["bads"]),
        list(noise_cov["projs"]),
        nfree=noise_cov.nfree,
    )
    cov_scaled = sbl_covariance(info, fwd, scaled, noise_cov=scaled_noise)
    np.testing.assert_allclose(cov_scaled.data, cov.data * s**2, rtol=1e-6)


def test_abmc_localises_a_spike_at_si_scale(sphere_setup):
    """ABMC finds the grid point carrying a template-shaped transient."""
    info, fwd = sphere_setup
    gain = fwd["sol"]["data"]
    rr = fwd["source_rr"]
    rng = np.random.default_rng(3)

    n_times = 200
    i_src = int(np.argmin(np.linalg.norm(rr - rr.mean(0), axis=1)))
    source = _spike(n_times, 110)
    clean = np.outer(gain[:, i_src], source)
    clean *= _EEG_SCALE / np.abs(clean).max()
    data = clean + 0.2 * np.abs(clean).max() * rng.standard_normal(clean.shape)

    evoked = mne.EvokedArray(data, info.copy(), tmin=0.0, verbose=False)
    result = make_abmc(info, fwd, evoked, _spike(n_times, 110))

    peak = int(np.argmax(result.template_match))
    error_cm = np.linalg.norm(rr[peak] - rr[i_src]) * 100
    assert error_cm < 3.0
    # ``template_match`` is a correlation, so it is bounded.
    assert result.template_match.max() <= 1.0 + 1e-9


def test_abmc_template_match_is_scale_invariant(sphere_setup):
    """The localiser is a correlation and must not depend on any amplitude."""
    info, fwd = sphere_setup
    gain = fwd["sol"]["data"]
    rr = fwd["source_rr"]
    rng = np.random.default_rng(4)

    n_times = 160
    i_src = int(np.argmin(np.linalg.norm(rr - rr.mean(0), axis=1)))
    clean = np.outer(gain[:, i_src], _spike(n_times, 80))
    clean *= _EEG_SCALE / np.abs(clean).max()
    data = clean + 0.2 * np.abs(clean).max() * rng.standard_normal(clean.shape)
    evoked = mne.EvokedArray(data, info.copy(), tmin=0.0, verbose=False)

    template = _spike(n_times, 80)
    reference = make_abmc(info, fwd, evoked, template).template_match
    for scale in (1e-6, 1e-10):
        scaled = make_abmc(info, fwd, evoked, template * scale).template_match
        np.testing.assert_allclose(scaled, reference, rtol=1e-6)
