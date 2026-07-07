"""Tests for the ABMC beamformer (SBL covariance + template-constrained filter)."""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from advance_beamlab import (
    ABMCResult,
    make_abmc,
    make_abmc_dictionary,
    sbl_covariance,
)

mne.set_log_level("ERROR")


@pytest.fixture(scope="module")
def sphere_fwd():
    """A fixed-orientation EEG sphere forward + Info (scalar leadfield, Eqs 1-13)."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch = list(dict.fromkeys(montage.ch_names))
    info = mne.create_info(ch, 250.0, "eeg")
    info.set_montage(montage)
    sphere = mne.make_sphere_model("auto", "auto", info)
    src = mne.setup_volume_source_space(sphere=sphere, pos=20.0)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False)
    return fwd, info


def _spike(n, t0, w=8.0):
    t = np.arange(n)
    x = -(t - t0) / w * np.exp(-((t - t0) ** 2) / (2 * w**2))
    return x / np.abs(x).max()


def _shell_sources(fwd, n=2):
    rr = fwd["source_rr"]
    depth = np.linalg.norm(rr - rr.mean(0), axis=1)
    shell = np.where(depth > np.percentile(depth, 75))[0]
    order = shell[np.argsort(rr[shell, 0])]
    return [int(order[0]), int(order[-1])][:n]


def _data_cov(x, info):
    raw = mne.io.RawArray(x, info)
    return mne.compute_covariance(mne.make_fixed_length_epochs(raw, duration=1.0))


# --- Stage 1: SBL covariance (Eqs 5-13) --------------------------------------
def test_sbl_returns_valid_covariance(sphere_fwd):
    """sbl_covariance returns a positive-definite mne.Covariance over the channels."""
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    rng = np.random.default_rng(0)
    (i,) = _shell_sources(fwd, 1)
    x = np.outer(lf[:, i], _spike(400, 200))
    x = x + 0.3 * np.abs(x).max() * rng.standard_normal(x.shape)
    r_cov = sbl_covariance(info, fwd, _data_cov(x, info))
    assert isinstance(r_cov, mne.Covariance)
    assert r_cov.data.shape[0] == len(info["ch_names"])
    assert np.linalg.eigvalsh(r_cov.data).min() > 0


def test_sbl_localises_correlated_sources(sphere_fwd):
    """SBL puts both correlated sources on top and beats the sample covariance."""
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    n_ch, _ = lf.shape
    rng = np.random.default_rng(1)
    i_a, i_b = _shell_sources(fwd, 2)
    t = np.arange(600) / 250.0
    s1 = np.sin(2 * np.pi * 8 * t) + 0.3 * rng.standard_normal(600)
    s2 = 0.9 * s1 + np.sqrt(1 - 0.81) * (
        np.sin(2 * np.pi * 8 * t + 0.5) + 0.3 * rng.standard_normal(600)
    )
    xs = np.outer(lf[:, i_a], s1) + np.outer(lf[:, i_b], s2)
    x = xs + 0.4 * np.abs(xs).max() * rng.standard_normal((n_ch, 600))
    dcov = _data_cov(x, info)
    _, alpha = sbl_covariance(info, fwd, dcov, return_source_power=True)
    top = list(np.argsort(alpha)[::-1][:6])
    assert i_a in top and i_b in top
    cov = dcov.data
    inv = np.linalg.inv(cov + 0.05 * np.trace(cov) / n_ch * np.eye(n_ch))
    p_sample = 1.0 / np.einsum("mk,mn,nk->k", lf, inv, lf)
    assert list(np.argsort(alpha)[::-1]).index(i_a) <= list(
        np.argsort(p_sample)[::-1]
    ).index(i_a)


def test_sbl_input_validation(sphere_fwd):
    """Invalid iteration controls raise ValueError."""
    fwd, info = sphere_fwd
    dcov = _data_cov(np.outer(fwd["sol"]["data"][:, 0], _spike(300, 150)), info)
    with pytest.raises(ValueError, match="max_iter"):
        sbl_covariance(info, fwd, dcov, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        sbl_covariance(info, fwd, dcov, tol=0.0)


# --- Stage 2: template-constrained beamformer (Eqs 14-19) --------------------
def test_abmc_localises_spike(sphere_fwd):
    """make_abmc localises a spike, recovers its lag, returns an ABMCResult."""
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    rng = np.random.default_rng(2)
    (i,) = _shell_sources(fwd, 1)
    xs = np.outer(lf[:, i], _spike(400, 250))
    x = xs + 0.4 * np.abs(xs).max() * rng.standard_normal(xs.shape)
    res = make_abmc(info, fwd, x, _spike(400, 200))  # data spike at 250 -> lag +50
    assert isinstance(res, ABMCResult)
    assert isinstance(res.stc, mne.VolSourceEstimate)
    peak = int(np.argmax(res.template_match))
    assert np.linalg.norm(fwd["source_rr"][peak] - fwd["source_rr"][i]) < 0.03
    assert abs(int(res.lag[peak]) - 50) <= 3
    assert res.converged


def test_abmc_distortionless_constraint(sphere_fwd):
    """At convergence the beamformer holds G^T W = f = 1 at every grid point."""
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    rng = np.random.default_rng(3)
    (i,) = _shell_sources(fwd, 1)
    x = np.outer(lf[:, i], _spike(400, 250))
    x = x + 0.3 * np.abs(x).max() * rng.standard_normal(x.shape)
    res = make_abmc(info, fwd, x, _spike(400, 250), return_weights=True)
    assert_allclose(np.einsum("mk,mk->k", lf, res.weights), 1.0, atol=1e-6)


def test_abmc_blowup_reported(sphere_fwd):
    """Small P is stable and convergent; blowup_fraction is reported in [0, 1)."""
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    rng = np.random.default_rng(4)
    (i,) = _shell_sources(fwd, 1)
    x = np.outer(lf[:, i], _spike(400, 250))
    x = x + 0.3 * np.abs(x).max() * rng.standard_normal(x.shape)
    res = make_abmc(info, fwd, x, _spike(400, 250), P=0.02)
    assert res.converged
    assert 0.0 <= res.blowup_fraction < 0.05


def test_abmc_template_length_check(sphere_fwd):
    """A template whose length differs from the data raises ValueError."""
    fwd, info = sphere_fwd
    x = np.outer(fwd["sol"]["data"][:, 0], _spike(400, 250))
    with pytest.raises(ValueError, match="template length"):
        make_abmc(info, fwd, x, _spike(300, 150))


def test_abmc_free_orientation():
    """make_abmc handles a free-orientation forward and recovers the orientation."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch = list(dict.fromkeys(montage.ch_names))
    info = mne.create_info(ch, 250.0, "eeg")
    info.set_montage(montage)
    sphere = mne.make_sphere_model("auto", "auto", info)
    src = mne.setup_volume_source_space(sphere=sphere, pos=20.0)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
    lf = fwd["sol"]["data"]
    rr = fwd["source_rr"]
    rng = np.random.default_rng(5)
    depth = np.linalg.norm(rr - rr.mean(0), axis=1)
    i = int(np.where(depth > np.percentile(depth, 75))[0][0])
    x = np.outer(lf[:, 3 * i + 2], _spike(400, 250))
    x = x + 0.3 * np.abs(x).max() * rng.standard_normal(x.shape)
    res = make_abmc(info, fwd, x, _spike(400, 250))
    assert len(res.power) == fwd["nsource"]
    assert res.orientation[int(np.argmax(res.template_match))] == 2


def test_abmc_dictionary(sphere_fwd):
    """make_abmc_dictionary runs one scan per template, reusing one covariance."""
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    rng = np.random.default_rng(6)
    (i,) = _shell_sources(fwd, 1)
    x = np.outer(lf[:, i], _spike(400, 250))
    x = x + 0.3 * np.abs(x).max() * rng.standard_normal(x.shape)
    templates = {"early": _spike(400, 200), "late": _spike(400, 300)}
    results = make_abmc_dictionary(info, fwd, x, templates)
    assert set(results) == {"early", "late"}
    assert all(isinstance(r, ABMCResult) for r in results.values())
    # every template localises the same underlying source
    for r in results.values():
        peak = int(np.argmax(r.template_match))
        assert np.linalg.norm(fwd["source_rr"][peak] - fwd["source_rr"][i]) < 0.03
    # the two templates are offset in opposite directions -> different lags
    e_peak = int(np.argmax(results["early"].template_match))
    l_peak = int(np.argmax(results["late"].template_match))
    assert results["early"].lag[e_peak] > results["late"].lag[l_peak]


def test_abmc_dictionary_accepts_sequence(sphere_fwd):
    """A plain sequence of templates is labelled by integer position."""
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    rng = np.random.default_rng(7)
    x = np.outer(lf[:, 0], _spike(400, 250))
    x = x + 0.3 * np.abs(x).max() * rng.standard_normal(x.shape)
    results = make_abmc_dictionary(info, fwd, x, [_spike(400, 200), _spike(400, 250)])
    assert set(results) == {0, 1}


def test_abmc_dictionary_validation(sphere_fwd):
    """Empty templates and length mismatches raise ValueError."""
    fwd, info = sphere_fwd
    x = np.outer(fwd["sol"]["data"][:, 0], _spike(400, 250))
    with pytest.raises(ValueError, match="non-empty"):
        make_abmc_dictionary(info, fwd, x, {})
    with pytest.raises(ValueError, match="must match data"):
        make_abmc_dictionary(info, fwd, x, {"bad": _spike(300, 150)})
