"""Tests for the ABMC beamformer (SBL covariance + template-constrained filter)."""

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import warnings

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from advance_beamlab import (
    ABMCResult,
    abmc_stability_curve,
    make_abmc,
    make_abmc_dictionary,
    sbl_covariance,
)

mne.set_log_level("ERROR")


def _avg_ref(info):
    """Add the average EEG reference projector MNE requires for inverse modelling."""
    return (
        mne.io.RawArray(np.zeros((len(info["ch_names"]), 2)), info, verbose=False)
        .set_eeg_reference("average", projection=True, verbose=False)
        .info
    )


# A realistic scalp-EEG amplitude (volts). The SI scale is the whole point of
# several tests below: the leadfield of a sphere model is O(100) V/(A m) while
# the data covariance is O(1e-10) V^2, and a hyperparameter initialisation that
# ignores that gap makes ``R = G a G^T + Lambda`` singular on the first update.
_EEG_SCALE = 2e-5

# ---------------------------------------------------------------------------
# the MNE ``sample`` dataset is only needed for the surface-source-space tests
# ---------------------------------------------------------------------------
try:
    _DATA_PATH = mne.datasets.sample.data_path(download=False)
    _SAMPLE = _DATA_PATH / "MEG" / "sample"
    _FWD = _SAMPLE / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    _HAVE_SAMPLE = _SAMPLE.is_dir() and _FWD.is_file()
except Exception:  # pragma: no cover - depends on local dataset state
    _HAVE_SAMPLE = False


@pytest.fixture(scope="module")
def sphere_fwd():
    """A fixed-orientation EEG sphere forward + Info (scalar leadfield, Eqs 1-13)."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch = list(dict.fromkeys(montage.ch_names))
    info = _avg_ref(mne.create_info(ch, 250.0, "eeg"))
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


def _cov(x, ch_names):
    """The plain sample covariance ``X X^T / T`` as an mne.Covariance."""
    return mne.Covariance(
        x @ x.T / x.shape[1], list(ch_names), bads=[], projs=[], nfree=x.shape[1]
    )


def _si_data(fwd, index, *, n=400, t0=200, seed=0, snr=0.3):
    """One spike-like source at a realistic scalp-EEG amplitude, plus noise."""
    clean = np.outer(fwd["sol"]["data"][:, index], _spike(n, t0))
    clean *= _EEG_SCALE / np.abs(clean).max()
    rng = np.random.default_rng(seed)
    return clean + snr * np.abs(clean).max() * rng.standard_normal(clean.shape)


def _type2_cost(cov, model):
    r"""The Eq. 6 cost :math:`F=\mathrm{tr}(CR^{-1})+\log|R|`."""
    logdet = 2.0 * np.log(np.diag(np.linalg.cholesky(model))).sum()
    return float(np.trace(np.linalg.solve(model, cov)) + logdet)


def _numpy_warnings(records):
    """The floating-point warnings numpy raises out of a broken linear solve."""
    return [
        str(r.message)
        for r in records
        if any(
            key in str(r.message)
            for key in ("divide by zero", "overflow", "invalid value", "Singular")
        )
    ]


# --- Stage 1: SBL covariance (Eqs 5-13) --------------------------------------
def test_sbl_cost_decreases_and_concentrates_on_the_source(sphere_fwd):
    """The type-II cost falls monotonically and alpha concentrates on the source.

    This pins the fit itself, not merely the shape of the returned object: a
    do-nothing implementation returns a valid positive-definite covariance but
    a flat cost sequence and a flat ``alpha``.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, seed=0)
    dcov = _cov(x, info["ch_names"])

    costs = []
    for n_iter in range(1, 9):
        with warnings.catch_warnings():  # truncated runs warn about convergence
            warnings.simplefilter("ignore", RuntimeWarning)
            r_cov = sbl_covariance(info, fwd, dcov, max_iter=n_iter)
        assert isinstance(r_cov, mne.Covariance)
        assert np.linalg.eigvalsh(r_cov.data).min() > 0
        costs.append(_type2_cost(dcov.data, r_cov.data))

    costs = np.asarray(costs)
    assert np.all(np.diff(costs) < 0), costs
    # the fit must actually move: a no-op SBL leaves the cost where it started
    assert costs[0] - costs[-1] > 1.0, costs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, alpha = sbl_covariance(info, fwd, dcov, return_source_power=True)
    peak = int(np.argmax(alpha))
    assert np.linalg.norm(fwd["source_rr"][peak] - fwd["source_rr"][i]) < 0.03
    # and it must be sparse, not the flat prior it started from
    assert alpha.max() > 50 * np.median(alpha)


def test_sbl_runs_on_si_unit_data(sphere_fwd):
    """The fit succeeds for data in volts, where a dimensionless prior is singular.

    ``alpha = 1`` against an O(100) sphere leadfield puts ``G alpha G^T`` and
    ``Lambda`` fourteen orders of magnitude apart, and ``R`` is singular on the
    first iteration.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    dcov = _cov(_si_data(fwd, i, seed=1), info["ch_names"])
    assert np.trace(dcov.data) < 1e-6  # genuinely SI-scaled

    r_cov = sbl_covariance(info, fwd, dcov)
    assert np.all(np.isfinite(r_cov.data))
    assert np.linalg.eigvalsh(r_cov.data).min() > 0
    assert np.linalg.cond(r_cov.data) < 1e6


def test_sbl_is_invariant_to_the_units_of_the_data(sphere_fwd):
    """Rescaling the data by ``s`` must rescale the model covariance by ``s**2``."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, seed=2)
    ch = list(info["ch_names"])
    s = 1e6

    with warnings.catch_warnings():  # a truncated run is enough to compare
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = sbl_covariance(info, fwd, _cov(x, ch), max_iter=40)
        scaled = sbl_covariance(info, fwd, _cov(x * s, ch), max_iter=40)
        assert_allclose(scaled.data, reference.data * s**2, rtol=1e-8)

        # ... and with a noise covariance, rescaled along with the data
        var = (0.3 * _EEG_SCALE) ** 2
        ncov = mne.Covariance(var * np.eye(len(ch)), ch, [], [], nfree=1)
        ncov_s = mne.Covariance(var * s**2 * np.eye(len(ch)), ch, [], [], nfree=1)
        reference = sbl_covariance(info, fwd, _cov(x, ch), noise_cov=ncov, max_iter=40)
        scaled = sbl_covariance(
            info, fwd, _cov(x * s, ch), noise_cov=ncov_s, max_iter=40
        )
    assert_allclose(scaled.data, reference.data * s**2, rtol=1e-8)


def test_sbl_emits_no_numpy_warnings(sphere_fwd):
    """The cost is a Cholesky log-determinant, so no numpy warning is raised.

    Computing ``slogdet(inv(R^-1))`` emitted three RuntimeWarnings per iteration
    and, once the sign guard returned ``+inf``, the tolerance test could never
    fire again and the fit silently burned every iteration.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    dcov = _cov(_si_data(fwd, i, seed=3), info["ch_names"])
    with warnings.catch_warnings(record=True) as records, np.errstate(all="warn"):
        warnings.simplefilter("always")
        sbl_covariance(info, fwd, dcov)
    assert _numpy_warnings(records) == []
    # the convergence test stays live, so the fit stops on tol, not on max_iter
    assert [str(r.message) for r in records] == []


def test_sbl_excludes_bad_channels(sphere_fwd):
    """Bad channels are dropped, as :func:`mne.compute_covariance` drops them."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, seed=4)
    bad = info["ch_names"][3]
    info_bad = info.copy()
    info_bad["bads"] = [bad]

    r_cov = sbl_covariance(info_bad, fwd, _cov(x, info["ch_names"]))
    assert bad not in r_cov.ch_names
    assert len(r_cov.ch_names) == len(info["ch_names"]) - 1

    # a covariance that already dropped the bad channel must line up too
    good = [ch for ch in info["ch_names"] if ch != bad]
    keep = [info["ch_names"].index(ch) for ch in good]
    r_cov2 = sbl_covariance(info_bad, fwd, _cov(x[keep], good))
    assert r_cov2.ch_names == r_cov.ch_names


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
    assert_allclose(res.stc.vertices[0], fwd["src"][0]["vertno"])
    assert res.stc.data.shape == (fwd["nsource"], 1)
    peak = int(np.argmax(res.template_match))
    assert np.linalg.norm(fwd["source_rr"][peak] - fwd["source_rr"][i]) < 0.03
    assert abs(int(res.lag[peak]) - 50) <= 3
    assert res.converged


@pytest.mark.skipif(not _HAVE_SAMPLE, reason="MNE sample dataset not downloaded")
@pytest.mark.parametrize("hemis", [("lh", "rh"), ("lh",)])
def test_abmc_surface_forward_returns_a_surface_stc(hemis):
    """A surface forward gives a SourceEstimate carrying the forward's vertices.

    Hard-wiring ``vertices=[forward['src'][0]['vertno']]`` and
    ``VolSourceEstimate`` either raises a length error after the whole scan (two
    hemispheres) or -- when a single-hemisphere label leaves every in-use vertex
    in the first source space -- silently returns the wrong class carrying
    surface vertex numbers.
    """
    from mne import Label
    from mne.forward import restrict_forward_to_label

    fwd = mne.read_forward_solution(_FWD, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg="grad", eeg=False)
    labels = [
        Label(fwd["src"]["lh rh".split().index(h)]["vertno"][::120], hemi=h)
        for h in hemis
    ]
    fwd = restrict_forward_to_label(fwd, labels)
    fwd = mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=True, verbose=False
    )
    info = mne.io.read_info(_SAMPLE / "sample_audvis-ave.fif", verbose=False)
    info = mne.pick_info(
        info, mne.pick_channels(info["ch_names"], fwd["sol"]["row_names"])
    )

    n_times = 120
    rng = np.random.default_rng(11)
    clean = np.outer(fwd["sol"]["data"][:, 0], _spike(n_times, 60))
    clean *= 1e-11 / np.abs(clean).max()
    x = clean + 0.3 * np.abs(clean).max() * rng.standard_normal(clean.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = make_abmc(info, fwd, x, _spike(n_times, 60))

    assert isinstance(res.stc, mne.SourceEstimate)
    assert not isinstance(res.stc, mne.VolSourceEstimate)
    assert len(res.stc.vertices) == 2
    assert sum(len(v) for v in res.stc.vertices) == fwd["nsource"]
    wanted = [s["vertno"] for s in fwd["src"]]
    for got, want in zip(res.stc.vertices, wanted, strict=True):
        assert_allclose(got, want)


def test_abmc_accepts_raw(sphere_fwd):
    """A Raw segment is read with ``get_data()``; ``Raw.data`` does not exist."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=5)
    raw = mne.io.RawArray(x, info.copy(), verbose=False)
    res = make_abmc(info, fwd, raw, _spike(200, 100))
    peak = int(np.argmax(res.template_match))
    assert np.linalg.norm(fwd["source_rr"][peak] - fwd["source_rr"][i]) < 0.03
    # identical to feeding the same numbers as an array
    assert_allclose(
        res.template_match, make_abmc(info, fwd, x, _spike(200, 100)).template_match
    )


def test_abmc_rejects_epochs(sphere_fwd):
    """Epochs are refused with an actionable message, not silently mishandled."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    raw = mne.io.RawArray(_si_data(fwd, i, seed=6), info.copy(), verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=0.4, verbose=False).load_data()
    with pytest.raises(TypeError, match="Epochs are not supported"):
        make_abmc(info, fwd, epochs, _spike(100, 50))


def test_abmc_excludes_bad_channels(sphere_fwd):
    """A covariance that dropped ``info['bads']`` still lines up with the data."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=7)
    bad = info["ch_names"][5]
    info_bad = info.copy()
    info_bad["bads"] = [bad]
    good = [ch for ch in info["ch_names"] if ch != bad]
    keep = [info["ch_names"].index(ch) for ch in good]
    # ``data`` still carries the bad channel, as an Evoked/Raw always does
    evoked = mne.EvokedArray(x, info_bad.copy(), tmin=0.0, verbose=False)
    user_cov = _cov(x[keep], good)

    res = make_abmc(
        info_bad,
        fwd,
        evoked,
        _spike(200, 100),
        cov=user_cov,
        return_weights=True,
    )
    assert res.weights.shape[0] == len(good)
    peak = int(np.argmax(res.template_match))
    assert np.linalg.norm(fwd["source_rr"][peak] - fwd["source_rr"][i]) < 0.03


def test_abmc_template_match_is_invariant_to_template_amplitude(sphere_fwd):
    """Only the *shape* of the template may matter, never its amplitude.

    An absolute floor on the correlation denominator carries the units of the
    data times the template, so a small-amplitude template moved the peak and
    returned an unbounded value.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=8)
    template = _spike(200, 100)
    reference = make_abmc(info, fwd, x, template)
    for amplitude in (1e-4, 1e-8, 1e-10, 1e6):
        res = make_abmc(info, fwd, x, template * amplitude)
        assert_allclose(res.template_match, reference.template_match, rtol=1e-8)
        assert_allclose(res.lag, reference.lag)
    assert reference.template_match.max() <= 1.0 + 1e-9


def test_abmc_map_is_invariant_to_the_units_of_the_data(sphere_fwd):
    """The whole scan is unit free: rescaling the data leaves the map unchanged."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=9)
    template = _spike(200, 100)
    reference = make_abmc(info, fwd, x, template).template_match
    scaled = make_abmc(info, fwd, x * 1e6, template).template_match
    assert_allclose(scaled, reference, rtol=1e-8)


def test_abmc_template_constraint_is_active(sphere_fwd):
    """``P`` is a dimensionless trade-off, so the default changes the map.

    Without rescaling the constraint column to its leadfield column, ``P g^T c``
    and ``g^T g`` carry different units and on SI-scale data the paper's second
    constraint is numerically inert -- ABMC silently degenerates into an
    iterative LCMV.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=10)
    template = _spike(200, 100)

    with_p = make_abmc(info, fwd, x, template, P=0.03).template_match
    with pytest.warns(RuntimeWarning, match="numerically inert"):
        without_p = make_abmc(info, fwd, x, template, P=1e-14).template_match
    rel = np.abs(with_p - without_p).max() / np.abs(without_p).max()
    assert rel > 1e-3, rel


def test_abmc_matches_the_fixed_point_of_the_papers_iteration(sphere_fwd):
    """The closed-form solve equals the limit of the paper's Eqs. 17-19 descent.

    ``make_abmc`` solves the template-constrained beamformer at its fixed point
    rather than descending to it. This pins the two together: running the paper's
    own gradient iteration to a very tight tolerance reproduces the weights the
    closed form returns.
    """
    from advance_beamlab._abmc import (
        _aligned_leadfield_and_data,
        _noise_scaling,
        _restrict_to_cov,
        _shift_template,
        sbl_covariance,
    )

    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=12)
    template = _spike(200, 100)
    P, reg = 0.03, 0.05
    res = make_abmc(info, fwd, x, template, P=P, reg=reg, return_weights=True)

    # Rebuild exactly what the solver used, in the same noise-scaled space.
    lf, xx, ch = _aligned_leadfield_and_data(info, fwd, x)
    dcov = mne.Covariance(xx @ xx.T / xx.shape[1], ch, [], [], nfree=xx.shape[1])
    lf, xx, ch, cov_mat = _restrict_to_cov(sbl_covariance(info, fwd, dcov), lf, xx, ch)
    sd = _noise_scaling(info, ch, None)
    lf, xx = lf / sd[:, None], xx / sd[:, None]
    cov_mat = cov_mat / np.outer(sd, sd)
    r = 0.5 * (cov_mat + cov_mat.T)
    n_ch, n_col = lf.shape
    r_reg = r + reg * np.trace(r) / n_ch * np.eye(n_ch)

    w0 = np.linalg.solve(r_reg, lf)
    w0 = w0 / np.einsum("mk,mk->k", lf, w0)[None, :]
    y0 = w0.T @ xx
    lags = np.arange(-(len(template) - 1), len(template))
    c = np.empty((n_ch, n_col))
    for k in range(n_col):
        xc = np.correlate(y0[k], template, mode="full")
        c[:, k] = xx @ _shift_template(template, int(lags[np.argmax(np.abs(xc))]))
    c *= (
        np.linalg.norm(lf, axis=0)
        / np.clip(np.linalg.norm(c, axis=0), np.finfo(float).tiny, None)
    )[None, :]

    gg = np.einsum("mk,mk->k", lf, lf)
    gc = np.einsum("mk,mk->k", lf, c)
    mu = 1.0 / np.linalg.eigvalsh(r_reg).max()
    denom = mu * (gg + P * gc)
    w = np.zeros((n_ch, n_col))
    for _ in range(100000):
        rw = r_reg @ w
        beta1 = (
            1.0 - np.einsum("mk,mk->k", lf, w) + mu * np.einsum("mk,mk->k", lf, rw)
        ) / denom
        w_new = w - mu * (rw - lf * beta1[None, :] - c * (P * beta1)[None, :])
        step = np.linalg.norm(w_new - w)
        w = w_new
        if step <= 1e-14 * max(np.linalg.norm(w), 1e-300):
            break

    assert_allclose(res.weights, w / sd[:, None], rtol=1e-4, atol=0)
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
    info = _avg_ref(mne.create_info(ch, 250.0, "eeg"))
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


# --------------------------------------------------------------------------- #
# Hyperparameter selection, and the paper's literal descent.
# --------------------------------------------------------------------------- #
def test_stability_curve_finds_the_plateau(sphere_fwd):
    """The curve reports a run of P over which the localised peak is fixed.

    The paper states that P "is empirically adjusted" and reports no value, so
    the useful setting has to be found on the data at hand. Selection is by
    stability rather than by template match, which rises with P by construction.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=7)
    template = _spike(200, 100)

    P, peak, match, blowup, coupling, P_opt = abmc_stability_curve(
        info, fwd, x, template, return_optimal=True
    )
    assert P.ndim == peak.ndim == 1
    assert np.all(np.diff(P) > 0)  # ascending and deduplicated
    assert len(P) == len(peak) == len(match) == len(blowup) == len(coupling)
    assert np.all(coupling > 0)

    # The selected P sits inside the plateau it was chosen from, and that
    # plateau localises the implanted source.
    inside = np.argmin(np.abs(P - P_opt))
    assert peak[inside] == np.argmax(np.bincount(peak))
    err = np.linalg.norm(fwd["source_rr"][peak[inside]] - fwd["source_rr"][i])
    assert err < 0.03

    # Coupling is monotone in P: it is P times a fixed ratio.
    assert np.all(np.diff(coupling) >= -1e-12)


def test_auto_p_matches_an_explicit_p_from_the_plateau(sphere_fwd):
    """``P='auto'`` reproduces what the curve selected."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=7)
    template = _spike(200, 100)

    *_, P_opt = abmc_stability_curve(info, fwd, x, template, return_optimal=True)
    auto = make_abmc(info, fwd, x, template, P="auto")
    explicit = make_abmc(info, fwd, x, template, P=P_opt)
    assert_allclose(auto.template_match, explicit.template_match, rtol=1e-10)


def test_iterative_method_reaches_the_closed_form(sphere_fwd):
    """The paper's descent, run to convergence, gives the closed-form answer.

    This is the check that the two are one estimator rather than two: Eqs. 17-19
    iterated to a tight tolerance must reproduce the closed form, and the
    convergence test is the true distance to that fixed point rather than the
    size of the step -- the latter goes small on an ill-conditioned R precisely
    when the descent is furthest from converging.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=7)
    template = _spike(200, 100)

    exact = make_abmc(info, fwd, x, template, P=0.03, return_weights=True)
    descended = make_abmc(
        info,
        fwd,
        x,
        template,
        P=0.03,
        method="iterative",
        max_iter=200000,
        tol=1e-8,
        return_weights=True,
    )
    assert descended.converged
    assert descended.n_iter > 1  # it really did iterate
    rel = np.linalg.norm(descended.weights - exact.weights) / np.linalg.norm(
        exact.weights
    )
    assert rel < 1e-6
    assert_allclose(descended.template_match, exact.template_match, atol=1e-6)


def test_iterative_method_warns_when_stopped_short(sphere_fwd):
    """A truncated descent is reported as unconverged, not silently returned."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=7)
    with pytest.warns(RuntimeWarning, match="did not reach the fixed point"):
        res = make_abmc(
            info, fwd, x, _spike(200, 100), method="iterative", max_iter=3, tol=1e-12
        )
    assert not res.converged
    assert res.n_iter == 3


def test_unknown_method_is_rejected(sphere_fwd):
    """``method`` is validated up front."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=7)
    with pytest.raises(ValueError, match="method"):
        make_abmc(info, fwd, x, _spike(200, 100), method="steepest-descent")
