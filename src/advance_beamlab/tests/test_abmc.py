"""Tests for the ABMC beamformer (SBL covariance + template-constrained filter)."""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
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
from advance_beamlab._abmc import _abmc_prepare, _critical_p

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


def _local_forward(fwd, index, n_sources):
    """Restrict ``fwd`` to the ``n_sources`` grid points nearest ``index``."""
    rr = fwd["source_rr"]
    near = np.argsort(np.linalg.norm(rr - rr[index], axis=1))[:n_sources]
    vertno = np.sort(fwd["src"][0]["vertno"][near])
    stc = mne.VolSourceEstimate(np.ones((n_sources, 1)), [vertno], 0.0, 1.0)
    return mne.forward.restrict_forward_to_stc(fwd, stc)


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


def _forward_from_indices(fwd, indices):
    """Restrict ``fwd`` to an explicit list of grid indices."""
    vertno = np.sort(fwd["src"][0]["vertno"][np.asarray(indices)])
    stc = mne.VolSourceEstimate(np.ones((len(vertno), 1)), [vertno], 0.0, 1.0)
    return mne.forward.restrict_forward_to_stc(fwd, stc)


def test_abmc_critical_p_is_the_smallest_pole_of_the_gain_denominator(sphere_fwd):
    """``critical_p`` is ``min(-g^T g / g^T c)`` over the columns with ``g^T c < 0``."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=400, t0=250, seed=4)
    template = _spike(400, 250)

    prep = _abmc_prepare(info, fwd, x, template, None, None, 0.0, None)
    gg, gc = prep["gg"], prep["gc"]
    negative = gc < 0
    assert negative.any()  # the premise: part of this grid does have a pole

    res = make_abmc(info, fwd, x, template, P=0.03)
    assert_allclose(res.critical_p, (-gg[negative] / gc[negative]).min(), rtol=1e-12)

    # What the number means: at that ``P`` the gain denominator of Eq. 19 really
    # does vanish for one column, and for no column before it.
    denominator = gg + res.critical_p * gc
    assert abs(denominator.min()) < 1e-9 * gg.max()
    assert (gg + 0.99 * res.critical_p * gc > 0).all()

    # Because ``c`` is rescaled to the norm of ``g``, the pole of a column sits
    # at 1 / |cos(g, c)|, so no ``P`` below 1 can destabilise any dataset.
    cos = gc / np.sqrt(gg * np.einsum("mk,mk->k", prep["c"], prep["c"]))
    assert_allclose(res.critical_p, 1.0 / abs(cos.min()), rtol=1e-10)
    assert res.critical_p >= 1.0


def test_abmc_unstable_fraction_matches_the_sign_of_gc(sphere_fwd):
    """``unstable_fraction`` is the share of columns with ``g^T c < 0``, in [0, 1]."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=400, t0=250, seed=4)
    template = _spike(400, 250)

    gc = _abmc_prepare(info, fwd, x, template, None, None, 0.0, None)["gc"]
    res = make_abmc(info, fwd, x, template, P=0.03)
    assert 0.0 <= res.unstable_fraction <= 1.0
    assert res.unstable_fraction == float((gc < 0).mean())
    assert 0.0 < res.unstable_fraction < 1.0  # the premise: this grid is mixed

    # ``c`` is linear in the data, so negating the data flips every ``g^T c`` and
    # the two fractions have to partition the grid between them.
    flipped = make_abmc(info, fwd, -x, template, P=0.03)
    assert_allclose(flipped.unstable_fraction, 1.0 - res.unstable_fraction, rtol=1e-12)
    assert np.isfinite(flipped.critical_p)


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


@pytest.mark.parametrize(
    "gg, gc, expected_p, expected_fraction",
    [
        # No column can reach a pole, so there is no P that destabilises.
        ([1.0, 2.0, 3.0], [0.5, 1.0, 2.0], np.inf, 0.0),
        # Every column can, and the smallest pole wins: -2/-0.5 = 4 beats
        # -1/-0.5 = 2, so the answer is 2.
        ([1.0, 2.0], [-0.5, -0.5], 2.0, 1.0),
        # Mixed: only the negative columns contribute a pole.
        ([1.0, 4.0, 9.0], [1.0, -2.0, -1.5], 2.0, 2.0 / 3.0),
        # A zero is not negative, so it contributes no pole.
        ([1.0, 1.0], [0.0, 1.0], np.inf, 0.0),
    ],
)
def test_critical_p_contract(gg, gc, expected_p, expected_fraction):
    """The pole and the unstable share, on sign patterns chosen by hand.

    Deliberately a unit test on arrays rather than on a forward. The sign of
    ``g^T c`` is not a property of a single column, so a test cannot obtain a
    chosen sign pattern by selecting grid points; an earlier version of these
    tests assumed the pattern a spherical grid happened to produce locally and
    failed on every CI runner.
    """
    got_p, got_fraction = _critical_p(np.array(gg), np.array(gc))
    assert got_p == expected_p
    assert got_fraction == pytest.approx(expected_fraction)


def test_abmc_reports_the_pole_its_own_prep_predicts(sphere_fwd):
    """``ABMCResult`` carries exactly what ``_critical_p`` gives for that data.

    Asserted against whatever sign pattern this grid actually has, so the test
    holds on any source space and any BLAS.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    local = _local_forward(fwd, i, 12)
    x = _si_data(fwd, i, n=400, t0=250, seed=4)
    template = _spike(400, 250)

    prep = _abmc_prepare(info, local, x, template, None, None, 0.0, None)
    expected_p, expected_fraction = _critical_p(prep["gg"], prep["gc"])

    res = make_abmc(info, local, x, template, P=0.03)
    assert res.critical_p == expected_p
    assert res.unstable_fraction == pytest.approx(expected_fraction)
    assert 0.0 <= res.unstable_fraction <= 1.0
    # The rescaling of each constraint column to its leadfield column's norm
    # makes every pole 1/|cos|, so a finite one can never fall below 1.
    assert res.critical_p >= 1.0


def test_abmc_warns_from_the_critical_p_upwards_and_not_below(sphere_fwd):
    """The pre-solve warning fires at ``P >= critical_p`` and is silent below."""
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    # Twelve rather than three: a small neighbourhood may contain no column with
    # g^T c < 0 at all, and then there is no pole for this test to cross.
    local = _local_forward(fwd, i, 12)
    x = _si_data(fwd, i, n=400, t0=250, seed=4)
    template = _spike(400, 250)
    critical = _abmc_prepare(info, local, x, template, None, None, 0.0, None)[
        "critical_p"
    ]
    if not np.isfinite(critical):
        pytest.skip("this grid has no unstable column, so there is no pole to cross")

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        below = make_abmc(info, local, x, template, P=0.99 * critical)
    assert [r for r in records if "critical value" in str(r.message)] == []
    assert_allclose(below.critical_p, critical, rtol=1e-12)

    # Caught by class rather than by message because past the pole the older
    # blow-up warning fires as well, and both belong to this regime.
    for P in (critical, 1.5 * critical):
        with pytest.warns(RuntimeWarning) as records:
            make_abmc(info, local, x, template, P=P)
        assert any("at or above the critical value" in str(r.message) for r in records)


def test_template_match_is_a_centred_correlation(sphere_fwd):
    """It is documented as |corr(W'X, u)|, so check it against that definition.

    The map centres both the filter output and the lag-aligned template before
    taking their inner product. Drop either and the quantity silently becomes a
    cosine similarity, which rewards a constant offset shared with the template
    rather than a shared shape.

    Tested on the scoring step directly, with a deliberate offset on both
    arguments. Through the full pipeline it is invisible: a spike and broadband
    noise are already near zero mean, so the centring changes nothing and the
    bug goes unnoticed -- which is exactly what happened.
    """
    from advance_beamlab._abmc import _abmc_map

    fwd, info = sphere_fwd
    del info
    rng = np.random.default_rng(7)
    n_col = fwd["nsource"]
    n_ch, n_times = 8, 300

    # A template and an output that both carry a large constant. Correlation is
    # blind to it by construction; an uncentred inner product is dominated by it.
    base = _spike(n_times, 120)
    u_shift = np.tile(base + 6.0, (n_col, 1))
    x = rng.standard_normal((n_ch, n_times)) + 9.0
    w = rng.standard_normal((n_ch, n_col))
    prep = dict(
        x=x,
        u_shift=u_shift,
        n_columns=n_col,
        col_lag=np.zeros(n_col, dtype=int),
        r=np.eye(n_ch),
        sd=np.ones(n_ch),
    )
    tmatch = np.asarray(_abmc_map(prep, w, fwd)["template_match"])

    out = w.T @ x
    out_c = out - out.mean(axis=1, keepdims=True)
    us_c = u_shift - u_shift.mean(axis=1, keepdims=True)
    expected = np.abs(
        np.einsum("kt,kt->k", out_c, us_c)
        / (np.linalg.norm(out_c, axis=1) * np.linalg.norm(us_c, axis=1))
    )
    assert_allclose(tmatch, expected, rtol=1e-9, atol=1e-12)
    # And a correlation magnitude is bounded, which the uncentred form is not
    # once a shared offset dominates both arguments.
    assert tmatch.max() <= 1.0 + 1e-9


def test_the_fit_reads_every_channel_relative_to_its_own_noise_level(sphere_fwd):
    """Rescaling a channel's gain, recording and noise together only rescales R.

    Stage 1 is fitted after dividing every channel by its noise standard
    deviation, which is what lets magnetometers, gradiometers and EEG be fitted
    together and what makes the fit independent of the units the data arrive in.
    The consequence is that the inferred model depends on a channel only through
    its signal-to-noise ratio: express one channel in different units -- its
    leadfield row, its recording and its noise variance all by the same factor --
    and the same model must come back, in the new units. Apply that normalisation
    the wrong way round and the arithmetic still runs, but the noisiest channels
    are the ones the fit trusts most, and on a mixed-sensor array the source
    powers are decided by whichever sensor type carries the larger numbers.
    """
    fwd, info = sphere_fwd
    (i,) = _shell_sources(fwd, 1)
    ch = list(info["ch_names"])
    x = _si_data(fwd, i, n=200, t0=100, seed=13)
    var = (0.3 * _EEG_SCALE) ** 2
    ncov = mne.Covariance(var * np.eye(len(ch)), ch, [], [], nfree=1)

    # A per-channel change of units spanning a factor of forty.
    s = np.geomspace(0.2, 8.0, len(ch))
    fwd_s = fwd.copy()
    fwd_s["sol"]["data"] = fwd_s["sol"]["data"] * s[:, None]
    ncov_s = mne.Covariance(var * np.diag(s**2), ch, [], [], nfree=1)

    with warnings.catch_warnings():  # a truncated run is enough to compare
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = sbl_covariance(info, fwd, _cov(x, ch), noise_cov=ncov, max_iter=30)
        scaled = sbl_covariance(
            info, fwd_s, _cov(s[:, None] * x, ch), noise_cov=ncov_s, max_iter=30
        )
    assert_allclose(scaled.data, reference.data * np.outer(s, s), rtol=1e-8)


def test_the_weights_deliver_the_distortionless_gain_that_was_asked_for(sphere_fwd):
    """``G^T W`` must equal ``f``, which is what fixes the amplitude of the output.

    The gain constraint is the only thing tying the filter output to the physical
    amplitude of the source: a caller who asks for ``f`` and gets 1 reads every
    recovered waveform, and every power, off by a constant factor with nothing in
    the result to say so. The default ``f=1`` hides such a slip, since there the
    right answer and the ungained one coincide.
    """
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    (i,) = _shell_sources(fwd, 1)
    x = _si_data(fwd, i, n=200, t0=100, seed=14)
    template = _spike(200, 100)

    reference = make_abmc(info, fwd, x, template, return_weights=True)
    assert_allclose(np.einsum("mk,mk->k", lf, reference.weights), 1.0, rtol=1e-8)
    for f in (2.5, 0.4):
        res = make_abmc(info, fwd, x, template, f=f, return_weights=True)
        assert_allclose(np.einsum("mk,mk->k", lf, res.weights), f, rtol=1e-8)
        # The gain scales the filter, so the localiser -- a correlation -- is
        # untouched by it and the peak cannot move with the requested amplitude.
        assert_allclose(res.template_match, reference.template_match, rtol=1e-9)
        assert_allclose(res.weights, reference.weights * f, rtol=1e-9)


def test_the_power_is_half_the_output_variance_over_all_three_orientations(sphere_fwd):
    """``power`` is the filter's own objective at each grid point, not a share of it.

    It is documented as ``1/2 W^T R W`` summed over orientations, and it is the
    map a reader compares against an LCMV one, so both halves of that matter. A
    free-orientation grid point radiates through three columns; reporting only
    the largest of the three understates every source that is not aligned with an
    axis, by up to a factor of three, and makes the diagnostic disagree with the
    variance the returned weights actually produce on the data.
    """
    _, info = sphere_fwd
    sphere = mne.make_sphere_model("auto", "auto", info)
    src = mne.setup_volume_source_space(sphere=sphere, pos=30.0)
    free = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
    lf = free["sol"]["data"]
    n_sources = free["nsource"]
    assert lf.shape[1] == 3 * n_sources

    rng = np.random.default_rng(15)
    xs = np.outer(lf[:, 1], _spike(200, 100))
    x = xs + 0.3 * np.abs(xs).max() * rng.standard_normal(xs.shape)
    ch = list(info["ch_names"])
    cov = _cov(x, ch)

    res = make_abmc(info, free, x, _spike(200, 100), cov=cov, return_weights=True)
    per_column = 0.5 * np.einsum("mk,mn,nk->k", res.weights, cov.data, res.weights)
    per_column = per_column.reshape(n_sources, 3)
    assert_allclose(res.power, per_column.sum(1), rtol=1e-8)
    # Every grid point spreads its variance over the three orientations, so the
    # sum and the largest single share are far apart compared with round-off.
    assert (per_column.sum(1) > 1.1 * per_column.max(1)).all()
    assert np.median(per_column.sum(1) / per_column.max(1)) > 1.5


def test_every_grid_point_reports_its_own_lag_inside_an_inclusive_window(sphere_fwd):
    """The lag is per grid point, and ``max_lag`` bounds it inclusively.

    The lag says where in the segment each location's output matches the
    template, so it is read alongside the map to time the event; one number
    copied across the whole grid would date every candidate location by whatever
    the first grid point happened to prefer. ``max_lag`` is documented as
    ``|j| <= max_lag``: an exclusive bound would quietly forbid the window the
    caller asked for, and ``max_lag=0``, the way to pin the template where it was
    handed in, would leave no lag admissible at all.
    """
    fwd, info = sphere_fwd
    lf = fwd["sol"]["data"]
    rng = np.random.default_rng(16)
    (i,) = _shell_sources(fwd, 1)
    xs = np.outer(lf[:, i], _spike(400, 250))
    x = xs + 0.4 * np.abs(xs).max() * rng.standard_normal(xs.shape)
    template = _spike(400, 200)  # the data spike sits 50 samples later

    res = make_abmc(info, fwd, x, template)
    col_lag = _abmc_prepare(info, fwd, x, template, None, None, 0.0, None)["col_lag"]
    assert_allclose(res.lag, col_lag)
    peak = int(np.argmax(res.template_match))
    assert abs(int(res.lag[peak]) - 50) <= 3
    # the grid disagrees about the lag, so a single copied value cannot pass
    assert (res.lag != res.lag[peak]).mean() > 0.3
    assert res.lag.std() > 5.0

    narrow = make_abmc(info, fwd, x, template, max_lag=3)
    assert np.abs(narrow.lag).max() == 3
    assert (np.abs(narrow.lag) == 3).mean() > 0.2  # the boundary is reachable
    assert (np.abs(make_abmc(info, fwd, x, template, max_lag=0).lag) == 0).all()


def test_only_a_vanishing_gain_denominator_makes_a_column_degenerate():
    """A negative denominator flips a column's sign; it does not disqualify it.

    Eq. 19 divides by ``g^T R^-1 (g + P c)``, which is positive while the
    constraint column agrees with the leadfield and passes through zero as ``P``
    grows for a column where it does not. Only the crossing itself is degenerate:
    past it the weights are finite, still hold ``G^T W = f``, and give the same
    absolute correlation with the template, because a beamformer and its negative
    describe the same source. Rejecting the sign rather than the magnitude would
    blank those columns out of the map and warn about a fault that is not there,
    over exactly the range of ``P`` where the template constraint has started to
    do something.

    Written on arrays rather than on a grid: the sign of the denominator is a
    joint property of a column, the covariance and ``P``, so no choice of grid
    points can be relied on to produce one.
    """
    from advance_beamlab._abmc import _abmc_fixed_point

    leadfield = np.eye(3)
    # Constraint columns that agree with, oppose, and exactly cancel their own
    # leadfield column at P = 1: the denominators are 2, -1 and 0.
    c = np.array([[1.0, 0.0, -1.0], [0.0, -2.0, 0.0], [0.0, 0.0, -1.0]])
    f = 2.5

    w, degenerate = _abmc_fixed_point(
        dict(leadfield=leadfield, c=c, r_reg=np.eye(3)), 1.0, f
    )

    assert list(degenerate) == [False, False, True]
    assert_allclose(np.einsum("mk,mk->k", leadfield, w), [f, f, 0.0], rtol=1e-12)
    # the opposed column is the flipped one, and it is not blanked
    assert_allclose(w[:, 1], [0.0, f, 0.0], rtol=1e-12)
    assert_allclose(w[:, 2], 0.0)
