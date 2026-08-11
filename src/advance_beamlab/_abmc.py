r"""ABMC: adaptive Bayesian beamformer with multiple constraints.

This module implements the beamforming pipeline of Shirani et al. (2024)
:footcite:`Shirani2024`, "Do interictal epileptiform discharges and brain
responses to electrical stimulation come from the same location? An advanced
source localization solution". The method localizes low-power, spike-like
transients (interictal epileptiform discharges, IEDs, and delayed responses,
DRs, to single-pulse electrical stimulation) that ordinary LCMV localizes poorly
because it is power-sensitive and collapses nearby correlated sources.

ABMC has two stages:

- **Stage 1: sparse Bayesian learning (SBL) covariance** (:func:`sbl_covariance`,
  Eqs. 5-13). A Champagne-style type-II maximum-likelihood fit of per-source
  prior variances :math:`\alpha` and a diagonal sensor-noise covariance
  :math:`\Lambda` yields a *model* covariance :math:`R = G\alpha G^\mathsf{T} +
  \Lambda`. Because the sources are modelled as mutually uncorrelated
  (:math:`\alpha` diagonal), :math:`R` does not carry the cross-source
  correlation structure that makes an LCMV beamformer cancel correlated sources.
- **Stage 2: template-constrained beamformer** (Eqs. 14-19). A
  minimum-variance beamformer with the usual distortionless constraint *plus* a
  maximum-cross-correlation-to-template constraint that locks the output onto the
  known DR/IED morphology. The paper reaches it by gradient descent; this
  implementation solves at that descent's fixed point, which is available in
  closed form, and offers the descent itself as ``method='iterative'`` for exact
  reproduction. The two agree to ~1e-8 once the descent has converged.

Choosing the trade-off :math:`P`
--------------------------------
The paper states that :math:`P` "is empirically adjusted according to the
convergence rate" and reports no value for it, nor for the step size, iteration
count or tolerance. That is not an omission: the useful setting depends on the
recording, and the paper's own data is 20-32 subdural contacts, a different
regime from a whole-head MEG array. :func:`abmc_stability_curve` performs that
adjustment on the data at hand. It runs a coarse logarithmic sweep to find the
range where the constraint is neither inert nor divergent, then refines around
the widest run over which the localised peak does not move. ``P='auto'`` uses
it. Selection is by stability rather than by template match, because the match
rises with :math:`P` by construction and maximising it would be circular.

Two things are known about :math:`P` before any sweep is run. First, rescaling
each constraint column to the norm of its leadfield column (see below) makes that
column's pole exactly :math:`1/|\cos(g_n, c_n)|`, so no :math:`P` below 1 can
destabilise any dataset; the smallest pole over the grid is reported as
``ABMCResult.critical_p``. Second, on the 94-channel spherical-EEG fixture of
``examples/plot_abmc_localization.py`` (301 grid points, eight simulated spikes)
the constraint changed the weights by 1 to 18 per cent over
:math:`P\in[0.01, 0.18]` while leaving the localised peak exactly where the
:math:`P\to 0` limit put it, which is the measurement behind the 0.01-0.1 range
quoted in :func:`make_abmc`. By :math:`P=1` half of those peaks had moved and
the mean peak error had risen from 0.85 cm to 2.20 cm, well before any weight
blew up: the poles sat at 2.3 to 4.4 on that fixture.

Localization follows the paper's criterion: the source is the grid location whose
beamformer output has the **maximum cross-correlation with the desired template**
:math:`u` at the best lag (not the output power, which is what LCMV maximizes).
The template is supplied by the caller (in the paper, expert-annotated IED or DR
morphologies), so ABMC can be steered to any known target waveform, not a fixed
shape. Both stages are provided here: :func:`sbl_covariance` (Stage 1) and
:func:`make_abmc` (Stage 2).

Working in a noise-normalised sensor space
------------------------------------------
The paper's recordings are single-sensor-type intracranial EEG, so its equations
are written directly in the recorded units. Two things break when they are
transcribed literally for MEG/EEG data in SI units. First, the type-II fit of
Stage 1 needs an initial :math:`(\alpha, \Lambda)` whose two terms are of
comparable size; a dimensionless initialisation against a gradiometer leadfield
puts them eighteen orders of magnitude apart and :math:`R` is numerically
singular on the first iteration. Second, a *diagonal* :math:`\Lambda` is only
meaningful when the channels share a unit, which magnetometers (T),
gradiometers (T/m) and EEG (V) do not.

Both are handled by rescaling every channel by its noise standard deviation
(from ``noise_cov`` when one is supplied, otherwise from MNE's ad-hoc per-type
model), fitting there, and undoing the scaling on the returned covariance. A
diagonal rescaling maps the model class onto itself (a diagonal :math:`\Lambda`
stays diagonal), so this is a change of working units rather than a change of
model, and for a single sensor type it is a global scalar that leaves the fit
exactly unchanged. It does make the whole of Stage 1 invariant to the physical
units of the data, and makes mixed sensor types commensurable.

Stage 2 needs the same treatment for a different reason: the template-constraint
column :math:`c_n = X u_{j^*}^\mathsf{T}` of Eq. 19 carries the units of the data
times the template, while the leadfield column :math:`g_n` carries the units of
the forward model, so the trade-off :math:`P` in
:math:`g_n^\mathsf{T} g_n + P\,g_n^\mathsf{T} c_n` would otherwise be a
dimensional quantity. Each :math:`c_n` is therefore rescaled to the norm of its
leadfield column, which makes :math:`P` the dimensionless trade-off the paper
describes.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import warnings
from dataclasses import dataclass

import numpy as np
from mne import Covariance
from mne.forward.forward import _subject_from_forward
from mne.source_estimate import _get_src_type, _make_stc
from mne.source_space._source_space import _get_vertno
from mne.utils import _check_option, _validate_type, logger, verbose
from scipy.linalg import cho_factor, cho_solve

# NOTE: as in ``_mcmv``, the stdlib ``warnings.warn`` (RuntimeWarning) is used
# rather than ``mne.utils.warn`` so that warnings are emitted regardless of the
# logging level and are reliably catchable by ``pytest.warns``.
# Reuse the channel-selection helpers from the MCMV module so that every
# algorithm in the package reads the forward, the data and the covariances in
# the same channel space. In particular, that makes all of them drop
# ``info['bads']``, which ``mne.compute_covariance`` has already dropped.
from ._mcmv import (
    _align_channels,
    _check_eeg_reference,
    _check_noise_cov_required,
    _cov_as_matrix,
    _intersect_noise_cov,
)

_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps


def _forward_rows(forward, ch_names):
    """Return the forward gain matrix with its rows ordered as ``ch_names``."""
    fwd_ch = list(forward["sol"]["row_names"])
    idx = [fwd_ch.index(ch) for ch in ch_names]
    return np.asarray(forward["sol"]["data"], float)[idx]


def _aligned_leadfield_and_cov(info, forward, data_cov, noise_cov=None):
    """Return leadfield ``G`` and covariance ``C`` on the common good channels.

    ``G`` is ``(n_channels, n_columns)``: one column per source for a
    fixed-orientation forward, three per source (x, y, z) for a free-orientation
    forward. ``C`` is the matching ``(n_channels, n_channels)`` block of
    ``data_cov``, and both are ordered by the returned ``ch_names``. Bad channels
    are excluded and, when a noise covariance is given, channels it does not
    cover are dropped, exactly as in :func:`~advance_beamlab.make_mcmv`.
    """
    ch_names, cov = _align_channels(info, forward, data_cov)
    ch_names, cov = _intersect_noise_cov(ch_names, cov, noise_cov)
    if len(ch_names) < 2:
        raise ValueError(
            "fewer than 2 good channels are common to info, forward and "
            "data_cov; cannot estimate a covariance."
        )
    return _forward_rows(forward, ch_names), cov, ch_names


def _noise_scaling(info, ch_names, noise_cov):
    """Per-channel noise standard deviations over ``ch_names``.

    Dividing every channel by its noise standard deviation puts magnetometers,
    gradiometers and EEG on a common, dimensionless footing and removes the
    dependence of the SBL fit on the physical units of the data. When no noise
    covariance is available MNE's ad-hoc per-type model is used; for a single
    sensor type that is a global scalar, so the fit is bit-for-bit unchanged.
    """
    from mne import make_ad_hoc_cov

    cov = make_ad_hoc_cov(info, verbose=False) if noise_cov is None else noise_cov
    sd = np.sqrt(np.abs(np.diag(_cov_as_matrix(cov, ch_names))))
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(
            "noise_cov has a non-positive or non-finite variance on one of the "
            "selected channels; it cannot be used to normalise the sensors."
        )
    return sd


def _model_precision(leadfield, alpha, lam):
    r"""Return the inverse and the log-determinant of :math:`R`.

    Here :math:`R = G\alpha G^\mathsf{T} + \Lambda`.

    Both come from a single Cholesky factorisation. Taking the log-determinant
    from the factor is exact and warning-free, whereas evaluating
    ``slogdet(inv(R^-1))`` (an inverse of an inverse) loses accuracy and, for a
    covariance in SI units, under- or overflows on the way.
    """
    model = (leadfield * alpha) @ leadfield.T
    model[np.diag_indices_from(model)] += lam
    model = 0.5 * (model + model.T)  # symmetrise against round-off
    factor = cho_factor(model, lower=True, check_finite=False)
    precision = cho_solve(factor, np.eye(model.shape[0]), check_finite=False)
    precision = 0.5 * (precision + precision.T)
    logdet = 2.0 * float(np.log(np.diag(factor[0])).sum())
    return precision, logdet


@verbose
def sbl_covariance(
    info,
    forward,
    data_cov,
    *,
    noise_cov=None,
    max_iter=100,
    tol=1e-5,
    return_source_power=False,
    verbose=None,
):
    r"""Estimate a model covariance by sparse Bayesian learning (ABMC Stage 1).

    Fits the generative model of Shirani et al. (2024) :footcite:`Shirani2024`,
    Eqs. 2-5,

    .. math::

        x(t) = G\,s(t) + \varepsilon(t), \qquad
        x(t)\sim\mathcal N(0, R), \qquad
        R = G\,\alpha\,G^\mathsf{T} + \Lambda,

    with independent zero-mean source priors of variance
    :math:`\alpha=\mathrm{diag}(\alpha_1,\dots)` (one per leadfield column) and a
    diagonal sensor-noise covariance :math:`\Lambda=\mathrm{diag}(\lambda_1,
    \dots,\lambda_M)`, by type-II maximum likelihood. The hyperparameters are
    updated with the convex-bounding (majorization-minimization) rules of
    Eqs. 9-13, which in terms of the data covariance :math:`C` read

    .. math::

        \alpha_n \leftarrow \alpha_n
            \sqrt{\frac{g_n^\mathsf{T} R^{-1} C R^{-1} g_n}
                       {g_n^\mathsf{T} R^{-1} g_n}}, \qquad
        \lambda_m \leftarrow \lambda_m
            \sqrt{\frac{(R^{-1} C R^{-1})_{mm}}{(R^{-1})_{mm}}} .

    Because the sources are modelled as uncorrelated, the returned
    :math:`R=G\alpha G^\mathsf{T}+\Lambda` suppresses the cross-source
    correlation that causes an LCMV beamformer to cancel correlated sources; it
    is intended as the covariance fed to the ABMC beamformer.

    The fit is run in a noise-normalised sensor space (see the module notes), so
    it is invariant to the physical units of the data and valid for arrays that
    mix sensor types.

    Parameters
    ----------
    info : mne.Info
        Measurement info. Defines the channels used and, with ``noise_cov=None``,
        the ad-hoc per-sensor-type noise model. Channels in ``info['bads']`` are
        excluded, as they are by :func:`mne.compute_covariance`.
    forward : mne.Forward
        Forward solution. A fixed-orientation forward gives one source per grid
        point; a free-orientation forward gives three columns (x, y, z) per grid
        point, each treated as an independent source in the prior.
    data_cov : mne.Covariance
        Sensor data covariance :math:`C=\tfrac1T XX^\mathsf{T}`, e.g. from
        :func:`mne.compute_covariance` over the analysis segment.
    noise_cov : mne.Covariance | None
        Noise covariance. Its diagonal both normalises the sensors and
        initialises :math:`\Lambda`; the noise variances are then learned
        (Eq. 10) either way. If ``None``, MNE's ad-hoc per-sensor-type model
        (:func:`mne.make_ad_hoc_cov`) is used for the normalisation and
        :math:`\Lambda` starts isotropic, carrying half of
        :math:`\mathrm{tr}(C)`. Supplying a measured ``noise_cov`` is strongly
        recommended when several sensor types are present, since the ad-hoc
        variances then set their relative weighting.
    max_iter : int
        Maximum number of update iterations (default 100).
    tol : float
        Stop when the relative change in the cost :math:`F` (Eq. 6) between
        iterations falls below this (default 1e-5).
    return_source_power : bool
        If ``True``, also return the fitted source-power vector :math:`\alpha`
        (one value per leadfield column). Note it carries the usual SBL depth
        bias, since the raw (un-normalised) leadfield is used.
    %(verbose)s

    Returns
    -------
    cov : mne.Covariance
        The model covariance :math:`R` over the common good channels.
    source_power : ndarray, shape (n_columns,)
        The fitted :math:`\alpha`; returned only if ``return_source_power`` is
        ``True``.

    Notes
    -----
    For a free-orientation forward the prior of Eq. 3 gives each of the three
    leadfield columns of a grid point its own variance and no cross-terms, i.e.
    the x, y and z components are treated as three independent scalar sources.
    That diagonal-per-axis prior is not covariant under a rotation of the source
    frame: the same physical dipole fitted in a rotated frame is not in general
    assigned the same total variance. Use a fixed-orientation forward when the
    orientation is known.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(data_cov, Covariance, "data_cov")
    _validate_type(noise_cov, (Covariance, None), "noise_cov")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}.")

    leadfield, cov, ch_names = _aligned_leadfield_and_cov(
        info, forward, data_cov, noise_cov
    )
    _check_noise_cov_required(info, ch_names, noise_cov)
    _check_eeg_reference(info, ch_names)
    if not np.all(np.isfinite(cov)):
        raise ValueError("data_cov contains non-finite values.")

    # Move into the noise-normalised sensor space: every channel is divided by
    # its noise standard deviation, which is what makes the fit independent of
    # the units of the data and of the sensor type.
    sd = _noise_scaling(info, ch_names, noise_cov)
    g = leadfield / sd[:, None]
    c = cov / np.outer(sd, sd)
    c = 0.5 * (c + c.T)
    n_channels, n_columns = g.shape

    trace_c = float(np.trace(c))
    if not np.isfinite(trace_c) or trace_c <= 0:
        raise ValueError(
            "data_cov has a non-positive trace over the selected channels; "
            "there is no signal to fit."
        )

    # Initialise so that the source and noise terms of R each carry half of the
    # measured power. Any fixed number here would be wrong: alpha multiplies the
    # squared leadfield, so its scale is set by the data *and* by the forward
    # model, and an O(1) start makes R singular for SI-unit recordings.
    gn2 = np.einsum("mk,mk->k", g, g)
    alpha = np.full(n_columns, 0.5 * trace_c / max(float(gn2.sum()), _TINY))
    if noise_cov is None:
        lam = np.full(n_channels, 0.5 * trace_c / n_channels)
    else:
        lam = np.ones(n_channels)  # the normalisation used noise_cov itself
    # Floor keeping R strictly positive definite (and so Cholesky-factorable)
    # even if a channel's learned noise variance collapses.
    lam_floor = _EPS * trace_c / n_channels
    # Scaling data by ``s`` shifts log|R| by a constant M log(s^2), so a
    # *relative* stopping rule on the bare Eq. 6 cost would stop after a
    # different number of iterations for the same data in different units.
    # Referring the cost to an isotropic covariance of the same total power
    # removes that constant and leaves the comparison unit free; it is an
    # additive constant, so the minimiser is untouched.
    cost_offset = n_channels * np.log(trace_c / n_channels)

    prev_cost = np.inf
    converged = False
    for iteration in range(max_iter):
        precision, logdet = _model_precision(g, alpha, lam)

        # Type-II ML cost F = tr(C R^-1) + log|R| (Eq. 6).
        cost = float(np.einsum("ij,ji->", c, precision) + logdet - cost_offset)
        if (
            np.isfinite(cost)
            and np.isfinite(prev_cost)
            and abs(prev_cost - cost) <= tol * max(abs(prev_cost), _TINY)
        ):
            converged = True
            logger.info(f"SBL covariance converged after {iteration + 1} iterations.")
            break
        prev_cost = cost

        prec_c_prec = precision @ c @ precision

        # alpha update (Eq. 9): needs diag(G^T (R^-1 C R^-1) G) and diag(G^T R^-1 G)
        num_alpha = np.einsum("mk,mk->k", g, prec_c_prec @ g)
        den_alpha = np.einsum("mk,mk->k", g, precision @ g)
        alpha = alpha * np.sqrt(
            np.clip(num_alpha, 0, None) / np.clip(den_alpha, _TINY, None)
        )

        # lambda update (Eq. 10): diagonals of R^-1 C R^-1 and R^-1
        num_lam = np.clip(np.diag(prec_c_prec), 0, None)
        den_lam = np.clip(np.diag(precision), _TINY, None)
        lam = np.clip(lam * np.sqrt(num_lam / den_lam), lam_floor, None)

    if not converged:
        warnings.warn(
            f"SBL covariance did not converge in {max_iter} iterations "
            f"(relative cost change above tol={tol}); returning current estimate.",
            RuntimeWarning,
            stacklevel=2,
        )

    model = (g * alpha) @ g.T
    model[np.diag_indices_from(model)] += lam
    model = 0.5 * (model + model.T)  # symmetrise against round-off
    model *= np.outer(sd, sd)  # back to the recorded units
    r_cov = Covariance(
        model,
        ch_names,
        bads=list(data_cov["bads"]),
        projs=list(data_cov["projs"]),
        nfree=data_cov.nfree,
    )
    if return_source_power:
        return r_cov, alpha
    return r_cov


@dataclass
class ABMCResult:
    r"""Result of an ABMC beamformer scan (Shirani et al., 2024, Stage 2).

    See :func:`make_abmc` and :footcite:`Shirani2024`. This is a
    :func:`~dataclasses.dataclass`, so the fields below are both its constructor
    parameters and its attributes; :func:`make_abmc` builds it for you.

    Parameters
    ----------
    stc : mne.SourceEstimate | mne.VolSourceEstimate | mne.MixedSourceEstimate
        The localization map: the per-grid-point template match (see
        ``template_match``), as a single-time source estimate for plotting. Its
        class follows the source-space type of the forward, so it can be passed
        to :meth:`~mne.SourceEstimate.plot`, source morphing
        and :func:`mne.extract_label_time_course` unchanged.
    template_match : ndarray, shape (n_sources,)
        Primary localizer: :math:`|\mathrm{corr}(W^\mathsf{T}X, u_{j^*})|`, the
        absolute correlation between the beamformer output and the lag-aligned
        template, maximised over orientation at each grid point. The peak is the
        estimated source.
    power : ndarray, shape (n_sources,)
        The beamformer output variance :math:`\tfrac12 W^\mathsf{T}RW` summed
        over orientations. This is the filter's minimisation objective, and what
        LCMV would localise on. It is reported as a diagnostic only: neither the
        scan nor the orientation choice uses it, both being driven by
        ``template_match``.
    lag : ndarray, shape (n_sources,)
        Template lag :math:`j^*` (samples) fixed per grid point for the winning
        orientation.
    orientation : ndarray, shape (n_sources,)
        Index (0, 1, 2) of the orientation with the strongest template match at
        each grid point; all zeros for a fixed-orientation forward.
    weights : ndarray | None
        Beamformer weights ``(n_channels, n_columns)`` if ``return_weights``,
        else ``None``.
    n_iter : int
        Iterations actually run.
    converged : bool
        Whether the weight update met the tolerance before ``max_iter``.
    blowup_fraction : float
        Fraction of grid columns whose weights grew anomalously large, measured
        on the weights *after* the solve. Above ~0.05, lower ``P``. A small
        value is not an all-clear: this only detects the neighbourhood of the
        poles described under ``critical_p``, and returns to exactly zero for
        ``P`` well beyond them, where the weights are finite but the
        localisation is wrong. Use ``critical_p`` to decide that ``P`` is safe.
    critical_p : float
        Smallest ``P`` at which some grid column's gain denominator
        :math:`g_n^\mathsf{T}g_n + P\,g_n^\mathsf{T}c_n` vanishes, that is the
        smallest :math:`-g_n^\mathsf{T}g_n / g_n^\mathsf{T}c_n` over the columns
        with :math:`g_n^\mathsf{T}c_n < 0`, and ``inf`` when there are none.
        Unlike ``blowup_fraction``, it is predicted from the forward and the
        constraint columns *before* the solve, and it remains valid for every
        larger ``P`` rather than only near the pole. Because each constraint
        column is rescaled to the norm of its leadfield column, a column's pole
        is exactly :math:`1/|\cos(g_n, c_n)|`, so this is never below 1 for any
        dataset.
    unstable_fraction : float
        Fraction of grid columns with :math:`g_n^\mathsf{T}c_n < 0`, that is the
        share of the grid that has a finite pole at all. Also predicted before
        the solve. It says how much of the grid ``critical_p`` speaks for, not
        that anything has gone wrong at the ``P`` actually used.

    References
    ----------
    .. footbibliography::
    """

    stc: object
    template_match: np.ndarray
    power: np.ndarray
    lag: np.ndarray
    orientation: np.ndarray
    weights: object
    n_iter: int
    converged: bool
    blowup_fraction: float
    critical_p: float
    unstable_fraction: float


def _aligned_leadfield_and_data(info, forward, data):
    """Return leadfield ``G`` and sensor data ``X`` on the common good channels."""
    from mne import Evoked
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw

    if isinstance(data, BaseEpochs):
        raise TypeError(
            "data must be a single continuous segment; Epochs are not supported "
            "because the template constraint is defined on one (n_channels, "
            "n_times) segment. Pass epochs.average(), a single epoch, or an "
            "array of concatenated epochs together with a matching template."
        )
    if isinstance(data, (Evoked, BaseRaw)):
        # ``BaseRaw`` has no ``.data``; ``get_data()`` is the accessor both share.
        x_full = np.asarray(data.get_data(), float)
        data_ch = list(data.ch_names)
        bads = set(info["bads"]) | set(data.info["bads"])
    else:
        x_full = np.asarray(data, float)
        if x_full.ndim != 2:
            raise ValueError(
                f"data array must be 2-D (n_channels, n_times), got shape "
                f"{x_full.shape}."
            )
        data_ch = list(info["ch_names"])
        bads = set(info["bads"])
        if x_full.shape[0] != len(data_ch):
            raise ValueError(
                f"data has {x_full.shape[0]} rows but info has {len(data_ch)} "
                "channels; pass an Evoked/Raw or a matching array."
            )
    fwd_set, data_set = set(forward["sol"]["row_names"]), set(data_ch)
    # Preserve the channel order of ``info`` for determinism, and drop bads so
    # that the leadfield, the covariance and the filter share a channel set.
    ch_names = [
        ch
        for ch in info["ch_names"]
        if ch in fwd_set and ch in data_set and ch not in bads
    ]
    if len(ch_names) < 2:
        raise ValueError(
            "fewer than 2 good channels are common to info, forward and data."
        )
    leadfield = _forward_rows(forward, ch_names)
    x = x_full[[data_ch.index(ch) for ch in ch_names]]
    return leadfield, x, ch_names


def _restrict_to_cov(cov, leadfield, x, ch_names):
    """Drop channels the beamformer covariance does not cover, and return ``R``."""
    cov_set = set(cov.ch_names)
    keep = [i for i, ch in enumerate(ch_names) if ch in cov_set]
    if len(keep) < 2:
        raise ValueError(
            "fewer than 2 channels are shared by the beamformer covariance and "
            "info/forward/data."
        )
    if len(keep) < len(ch_names):
        logger.info(
            f"    Dropping {len(ch_names) - len(keep)} channel(s) not covered by "
            "the beamformer covariance."
        )
        ch_names = [ch_names[i] for i in keep]
        leadfield = leadfield[keep]
        x = x[keep]
    return leadfield, x, ch_names, _cov_as_matrix(cov, ch_names)


def _shift_template(u, lag):
    """Shift ``u`` by ``lag`` samples with zero padding (positive = delay)."""
    out = np.zeros_like(u)
    if lag >= 0:
        out[lag:] = u[: len(u) - lag] if lag < len(u) else 0.0
    else:
        out[: len(u) + lag] = u[-lag:]
    return out


def _abmc_stc(forward, values):
    """Build the source estimate matching the forward's source-space type."""
    vertno = _get_vertno(forward["src"])
    n_vert = int(sum(len(v) for v in vertno))
    if n_vert != values.shape[0]:
        raise RuntimeError(
            f"the forward reports {values.shape[0]} sources but its source "
            f"spaces hold {n_vert} vertices; the source estimate cannot be "
            "built. Check that the forward has not been modified in place."
        )
    return _make_stc(
        values[:, None],
        vertno,
        src_type=_get_src_type(forward["src"], vertno),
        tmin=0.0,
        tstep=1.0,
        subject=_subject_from_forward(forward),
    )


def _critical_p(gg, gc):
    r"""Return the smallest pole of the gain denominator, and the share of the grid.

    Eq. 19 divides by :math:`g^\mathsf{T}g + P\,g^\mathsf{T}c`, so a column with
    :math:`g^\mathsf{T}c < 0` has a pole at :math:`-g^\mathsf{T}g/g^\mathsf{T}c`
    and a column with :math:`g^\mathsf{T}c \ge 0` has none. The first return
    value is ``inf`` when no column has one.

    Kept separate from :func:`_abmc_prepare` so the contract can be tested on
    arrays with a chosen sign pattern. It cannot be tested by picking grid
    points: the sign of :math:`g^\mathsf{T}c` is not a property of a column on
    its own, and restricting a forward to a subset of its columns changes it.
    """
    negative = gc < 0
    if not negative.any():
        return np.inf, 0.0
    return float((-gg[negative] / gc[negative]).min()), float(negative.mean())


def _abmc_prepare(info, forward, data, template, cov, noise_cov, reg, max_lag):
    """Assemble everything in Stage 2 that does not depend on ``P``.

    The sparse Bayesian covariance, the noise scaling, the per-column template
    lag and the template-constraint columns are all independent of the
    constraint trade-off ``P``. Separating them means a sweep over ``P`` costs
    one linear solve per value instead of rebuilding the whole problem, which is
    what makes :func:`~advance_beamlab.abmc_stability_curve` affordable.
    """
    leadfield, x, ch_names = _aligned_leadfield_and_data(info, forward, data)
    u = np.asarray(template, float).ravel()
    if u.shape[0] != x.shape[1]:
        raise ValueError(
            f"template length {u.shape[0]} must match data length {x.shape[1]}."
        )

    if cov is None:
        data_cov = Covariance(
            x @ x.T / x.shape[1], ch_names, bads=[], projs=[], nfree=x.shape[1]
        )
        cov = sbl_covariance(info, forward, data_cov, noise_cov=noise_cov)
    else:
        _validate_type(cov, Covariance, "cov")
    leadfield, x, ch_names, cov_mat = _restrict_to_cov(cov, leadfield, x, ch_names)
    n_channels, n_columns = leadfield.shape
    _check_noise_cov_required(info, ch_names, noise_cov)
    _check_eeg_reference(info, ch_names)

    # Stage 2 runs in the noise-scaled space, for the same reason Stage 1 does:
    # in recorded units a mixed-sensor covariance spans ~16 orders of magnitude
    # and the solve is driven by whichever type has the larger numerical scale.
    sd = _noise_scaling(info, ch_names, noise_cov)
    leadfield = leadfield / sd[:, None]
    x = x / sd[:, None]
    cov_mat = cov_mat / np.outer(sd, sd)

    r = 0.5 * (cov_mat + cov_mat.T)
    r_reg = r + reg * np.trace(r) / n_channels * np.eye(n_channels)

    # Seed the per-column template lag from an initial distortionless output.
    r_inv_g = np.linalg.solve(r_reg, leadfield)
    w0 = r_inv_g / np.einsum("mk,mk->k", leadfield, r_inv_g)[None, :]
    y0 = w0.T @ x
    lags_full = np.arange(-(len(u) - 1), len(u))
    lag_mask = (
        np.ones_like(lags_full, dtype=bool)
        if max_lag is None
        else np.abs(lags_full) <= max_lag
    )
    c = np.empty((n_channels, n_columns))
    u_shift = np.empty((n_columns, len(u)))
    col_lag = np.empty(n_columns, dtype=int)
    for k in range(n_columns):
        xc = np.correlate(y0[k], u, mode="full")
        j = int(lags_full[lag_mask][np.argmax(np.abs(xc[lag_mask]))])
        col_lag[k] = j
        u_shift[k] = _shift_template(u, j)
        c[:, k] = x @ u_shift[k]

    # Put the template-constraint column on the same scale as the leadfield
    # column it competes with in Eq. 19, so ``P`` is the dimensionless trade-off
    # the paper describes rather than a quantity carrying the units of the data.
    c *= (
        np.linalg.norm(leadfield, axis=0)
        / np.clip(np.linalg.norm(c, axis=0), _TINY, None)
    )[None, :]

    gg = np.einsum("mk,mk->k", leadfield, leadfield)
    gc = np.einsum("mk,mk->k", leadfield, c)
    # Predict the instability rather than wait to observe it. The gain
    # denominator of Eq. 19 is g^T g + P g^T c, so a column with g^T c < 0 has a
    # pole at P = -g^T g / g^T c and its weights are worthless from there on.
    # Two reasons to compute it here: it is available before the solve (one pass
    # over the columns, measured at 0.1 ms for eight 301-point grids), and it
    # stays true for every larger P, whereas the blow-up share measured after the
    # solve is only large in the immediate neighbourhood of the poles. Above
    # them the weights settle onto the finite but wrong P -> infinity limit
    # f R^-1 c / (g^T R^-1 c), where no post-hoc statistic complains at all.
    critical_p, unstable_fraction = _critical_p(gg, gc)

    return dict(
        leadfield=leadfield,
        x=x,
        ch_names=ch_names,
        sd=sd,
        r=r,
        r_reg=r_reg,
        c=c,
        u_shift=u_shift,
        col_lag=col_lag,
        gg=gg,
        gc=gc,
        critical_p=critical_p,
        unstable_fraction=unstable_fraction,
        n_channels=n_channels,
        n_columns=n_columns,
    )


def _abmc_fixed_point(prep, P, f):
    """Closed-form solution of Eqs. 17-19, plus the degenerate-column mask.

    Setting the update of Eq. 17 to zero gives ``R w = (g + P c) beta1``, and the
    consistency of Eq. 19 then forces ``g^T w = f``, so the estimator the paper's
    descent converges to is available directly:

        ``w* = f R^-1 (g + P c) / (g^T R^-1 (g + P c))``.
    """
    leadfield = prep["leadfield"]
    v = leadfield + P * prep["c"]
    ri_v = np.linalg.solve(prep["r_reg"], v)
    denom = np.einsum("mk,mk->k", leadfield, ri_v)
    scale = _EPS * np.linalg.norm(leadfield, axis=0) * np.linalg.norm(ri_v, axis=0)
    degenerate = np.abs(denom) <= scale
    w = np.where(degenerate[None, :], 0.0, ri_v / np.where(degenerate, 1.0, denom))
    return w * f, degenerate


def _abmc_descend(prep, P, f, mu, max_iter, tol):
    """Run the paper's gradient descent, Eqs. 17-19, verbatim.

    Convergence is measured as the distance to the fixed point, not as the size
    of the step. The two are not interchangeable: on an ill-conditioned ``R`` the
    steps become small precisely because the descent is crawling along a shallow
    direction, so a step-size rule reports convergence while the weights are
    still far from the solution. Since the fixed point is available in closed
    form, the honest test is available too.
    """
    leadfield, c, r_reg = prep["leadfield"], prep["c"], prep["r_reg"]
    target, degenerate = _abmc_fixed_point(prep, P, f)
    target_norm = max(float(np.linalg.norm(target)), _TINY)
    if mu is None:
        mu = 1.0 / float(np.linalg.eigvalsh(r_reg).max())
    denom = mu * (prep["gg"] + P * prep["gc"])

    w = np.zeros_like(leadfield)
    n_run, converged, distance = 0, False, np.inf
    for iteration in range(int(max_iter)):
        rw = r_reg @ w
        beta1 = (
            f
            - np.einsum("mk,mk->k", leadfield, w)
            + mu * np.einsum("mk,mk->k", leadfield, rw)
        ) / denom
        w = w - mu * (rw - leadfield * beta1[None, :] - c * (P * beta1)[None, :])
        n_run = iteration + 1
        distance = float(np.linalg.norm(w - target)) / target_norm
        if distance < tol:
            converged = True
            break
    return w, degenerate, n_run, converged, distance, mu


def _abmc_map(prep, w, forward):
    """Per-grid-point template match, orientation, lag, power and blow-up share.

    Kept separate from :func:`_abmc_readout` so a sweep over ``P`` can score a
    solution without paying to build a source estimate for every value.
    """
    x, u_shift, n_columns = prep["x"], prep["u_shift"], prep["n_columns"]
    out = np.einsum("mk,mt->kt", w, x)
    out_c = out - out.mean(1, keepdims=True)
    us_c = u_shift - u_shift.mean(1, keepdims=True)
    out_norm = np.linalg.norm(out_c, axis=1)
    us_norm = np.linalg.norm(us_c, axis=1)
    num_corr = np.abs(np.einsum("kt,kt->k", out_c, us_c))
    # Per column and overflow-safe: a single blown-up column can make its own
    # norm overflow to inf while every weight entry is still finite, and a global
    # floor built from ``max()`` would then be inf and zero the entire map.
    ref = np.abs(out_c).max(axis=1) * np.sqrt(out_c.shape[1])
    good = (
        np.isfinite(out_norm)
        & np.isfinite(us_norm)
        & (out_norm > _EPS * ref)
        & (us_norm > 0)
    )
    tmatch = np.zeros(n_columns)
    tmatch[good] = num_corr[good] / (out_norm[good] * us_norm[good])
    power = 0.5 * np.einsum("mk,mk->k", w, prep["r"] @ w)

    col_norm = np.linalg.norm(w, axis=0)
    blowup = float((col_norm > 10 * np.median(col_norm)).mean())

    # ``out``, ``tmatch`` and ``power`` are invariant to the noise scaling, but
    # the returned weights must act on the recorded data.
    w_out = w / prep["sd"][:, None]

    n_sources = forward["nsource"]
    n_ori = n_columns // n_sources
    if n_ori * n_sources != n_columns:
        raise RuntimeError("leadfield columns are not an integer multiple of sources.")
    tmatch_s = tmatch.reshape(n_sources, n_ori)
    orientation = tmatch_s.argmax(1)
    lag_grid = prep["col_lag"].reshape(n_sources, n_ori)[
        np.arange(n_sources), orientation
    ]
    return dict(
        template_match=tmatch_s.max(1),
        power=power.reshape(n_sources, n_ori).sum(1),
        lag=lag_grid,
        orientation=orientation if n_ori > 1 else np.zeros(n_sources, int),
        weights=w_out,
        blowup=blowup,
    )


def _abmc_readout(prep, w, forward, P, f, return_weights, n_iter, converged):
    """Turn the per-column weights into an :class:`ABMCResult`."""
    m = _abmc_map(prep, w, forward)
    return ABMCResult(
        stc=_abmc_stc(forward, m["template_match"]),
        template_match=m["template_match"],
        power=m["power"],
        lag=m["lag"],
        orientation=m["orientation"],
        weights=m["weights"] if return_weights else None,
        n_iter=n_iter,
        converged=converged,
        blowup_fraction=m["blowup"],
        critical_p=prep["critical_p"],
        unstable_fraction=prep["unstable_fraction"],
    )


def _plateau(peaks, viable):
    """Longest run of consecutive viable entries sharing one peak location.

    Returns ``(start, stop)`` inclusive, or ``None`` if nothing is viable. The
    plateau, not the best score, is the selection criterion: the template match
    rises with ``P`` by construction (the constraint pushes the output towards
    the template), so maximising it would be circular. A range of ``P`` over
    which the *localised source does not move* is evidence that the answer is
    determined by the data rather than by the setting.
    """
    best = None
    i = 0
    n = len(peaks)
    while i < n:
        if not viable[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and viable[j + 1] and peaks[j + 1] == peaks[i]:
            j += 1
        if best is None or (j - i) > (best[1] - best[0]):
            best = (i, j)
        i = j + 1
    return best


@verbose
def abmc_stability_curve(
    info,
    forward,
    data,
    template,
    *,
    cov=None,
    noise_cov=None,
    P_range=(1e-4, 1e4),
    n_coarse=17,
    n_refine=9,
    reg=0.0,
    f=1.0,
    max_lag=None,
    return_optimal=False,
    verbose=None,
):
    r"""Explore the template-constraint trade-off :math:`P`, then refine it.

    Shirani et al. (2024) :footcite:`Shirani2024` state that :math:`P` "is
    empirically adjusted" and report no value for it, because the useful setting
    depends on the recording: the paper's own data is 20-32 subdural contacts,
    which is a different regime from a whole-head MEG array. This function does
    that adjustment on *your* data, in two stages.

    **Coarse exploration.** :math:`P` is swept logarithmically across
    ``P_range``. Each value is scored on three things that together say whether
    the method is operating at all: the *coupling*
    :math:`P\,g^\mathsf{T}c / g^\mathsf{T}g`, which must be non-negligible or the
    template constraint is inert and ABMC degenerates to a plain LCMV; the
    fraction of grid weights that have blown up, which marks the paper's
    non-convergence regime at large :math:`P`; and the location of the localiser
    peak.

    **Refinement.** The widest run of consecutive viable values over which the
    peak *does not move* is the plateau. Its edges are then re-sampled more
    finely and the geometric centre of the refined plateau is returned.

    The upper edge is not arbitrary. Eliminating :math:`\beta_1` from Eq. 17
    using Eq. 19 leaves an affine recursion whose linear part is
    :math:`\Pi(I - \mu R)`, with :math:`\Pi = I - d g^\mathsf{T}/(g^\mathsf{T}d)`
    an *oblique* projector and :math:`d = g + Pc`. The obliquity, and hence the
    norm of :math:`\Pi`, grows as :math:`g^\mathsf{T}d = g^\mathsf{T}g +
    P\,g^\mathsf{T}c` shrinks. Consequently, wherever :math:`g^\mathsf{T}c < 0`
    there is a :math:`P` beyond which the paper's descent is unstable. That is
    the "threshold for each segment" the paper reports without deriving; the
    smallest such :math:`P` over the grid is reported directly as
    ``ABMCResult.critical_p``. The threshold is what bounds the sweep from
    above, but not where the answer stops being trustworthy: the peak can move
    well below it, which is why the viable range has to be found per dataset
    rather than assumed.

    Selection is by stability rather than by score, deliberately. The template
    match increases with :math:`P` by construction: a stronger constraint pulls
    the output towards the template whether or not the location is right.
    Choosing the :math:`P` that maximises it would therefore be circular. A
    plateau over which the answer is unchanged is evidence about the data; a
    maximum of a self-referential score is not.

    Parameters
    ----------
    info : instance of mne.Info
        Measurement info (channel set only).
    forward : instance of mne.Forward
        Forward solution, as for :func:`make_abmc`.
    data : instance of mne.Evoked | instance of mne.io.Raw | ndarray
        The sensor data segment, as for :func:`make_abmc`.
    template : ndarray, shape (n_times,)
        The desired-source waveform, as for :func:`make_abmc`.
    cov : instance of mne.Covariance | None
        Covariance for the beamformer. ``None`` estimates it once with
        :func:`sbl_covariance` and reuses it for every :math:`P`.
    noise_cov : instance of mne.Covariance | None
        Noise covariance, as for :func:`make_abmc`.
    P_range : tuple of float
        Inclusive ``(low, high)`` bounds of the logarithmic sweep.
    n_coarse : int
        Number of points in the coarse sweep.
    n_refine : int
        Number of points in the refinement sweep. Set to ``0`` to skip
        refinement and return the coarse plateau centre.
    reg : float
        Diagonal loading, as for :func:`make_abmc`.
    f : float
        Distortionless gain, as for :func:`make_abmc`.
    max_lag : int | None
        Template-lag search window, as for :func:`make_abmc`.
    return_optimal : bool
        If ``True``, also return the selected :math:`P`.
    %(verbose)s

    Returns
    -------
    p_values : ndarray, shape (n_points,)
        The values of :math:`P` evaluated, ascending (coarse and refinement
        points merged).
    peak : ndarray of int, shape (n_points,)
        Index of the localiser peak at each :math:`P`.
    template_match : ndarray, shape (n_points,)
        Peak template-match value at each :math:`P`. Reported for inspection;
        it is deliberately *not* the selection criterion.
    blowup : ndarray, shape (n_points,)
        Fraction of grid weights that blew up at each :math:`P`.
    coupling : ndarray, shape (n_points,)
        The realised constraint coupling at each :math:`P`.
    p_opt : float
        The selected :math:`P`; returned only if ``return_optimal`` is ``True``.

    See Also
    --------
    make_abmc : Pass ``P='auto'`` to use this selection directly.

    References
    ----------
    .. footbibliography::
    """
    prep = _abmc_prepare(info, forward, data, template, cov, noise_cov, reg, max_lag)
    out = _abmc_select_p(prep, forward, f, P_range, n_coarse, n_refine)
    return out if return_optimal else out[:-1]


def _abmc_select_p(prep, forward, f, P_range, n_coarse, n_refine):
    """Coarse sweep then local refinement of ``P``; see the public wrapper."""
    lo, hi = (float(v) for v in P_range)
    if not 0 < lo < hi:
        raise ValueError(f"P_range must satisfy 0 < low < high, got {P_range}.")
    if int(n_coarse) < 3:
        raise ValueError(f"n_coarse must be >= 3, got {n_coarse}.")

    gg, gc = prep["gg"], prep["gc"]

    def score(p_vals):
        peaks, matches, blows, coups = [], [], [], []
        for p in p_vals:
            w, _ = _abmc_fixed_point(prep, float(p), f)
            m = _abmc_map(prep, w, forward)
            tm = m["template_match"]
            peaks.append(int(np.argmax(tm)))
            matches.append(float(tm.max()))
            blows.append(float(m["blowup"]))
            coups.append(float(np.max(np.abs(p * gc / np.clip(gg, _TINY, None)))))
        return (
            np.asarray(peaks),
            np.asarray(matches),
            np.asarray(blows),
            np.asarray(coups),
        )

    p_coarse = np.geomspace(lo, hi, int(n_coarse))
    peaks, matches, blows, coups = score(p_coarse)
    # Three conditions, and the third is not redundant. Beyond every column's
    # pole the weights settle onto the finite P -> infinity limit
    # f R^-1 c / (g^T R^-1 c), which is a fixed point: its peak sits perfectly
    # still over decades of P and blows up nowhere, so it looks exactly like a
    # plateau to the rule below while localising badly. Measured on the
    # ``plot_abmc_localization`` fixture, dropping this condition and narrowing
    # the range to (1e-2, 1e4) makes that false plateau win on seven of eight
    # segments, selecting P = 334 to 750 and a mean peak error of 7.91 cm
    # against 0.85 cm, with no warning raised anywhere. Requiring P below the
    # smallest pole removes it: every real plateau there ends by P = 5.01 and
    # every false one starts at P >= 10. It changes nothing on the default
    # range, where the selections are bit-identical.
    viable = (coups >= 1e-6) & (blows <= 0.05) & (p_coarse < prep["critical_p"])
    span = _plateau(peaks, viable)

    p_all, pk_all, tm_all, bl_all, cp_all = p_coarse, peaks, matches, blows, coups
    if span is None:
        warnings.warn(
            "no value of P in the requested range gave a usable solution: the "
            "template constraint is inert at the low end and the weights blow "
            "up at the high end. Widen P_range, or check that the template "
            "resembles anything in the data.",
            RuntimeWarning,
            stacklevel=2,
        )
        p_opt = float(np.sqrt(lo * hi))
    else:
        i, j = span
        p_opt = float(np.sqrt(p_coarse[i] * p_coarse[j]))
        if int(n_refine) > 0:
            # Re-sample between the neighbours of the plateau, so its edges are
            # located more precisely than the coarse grid can resolve.
            left = p_coarse[max(i - 1, 0)]
            right = p_coarse[min(j + 1, len(p_coarse) - 1)]
            p_fine = np.geomspace(left, right, int(n_refine))
            fp, fm, fb, fc = score(p_fine)
            merged = np.concatenate([p_coarse, p_fine])
            order = np.argsort(merged)
            # The refinement brackets the plateau using coarse grid points, so
            # the two sweeps share their endpoints; drop the repeats.
            keep = order[np.concatenate([[True], np.diff(merged[order]) > 0])]
            p_all = merged[keep]
            pk_all = np.concatenate([peaks, fp])[keep]
            tm_all = np.concatenate([matches, fm])[keep]
            bl_all = np.concatenate([blows, fb])[keep]
            cp_all = np.concatenate([coups, fc])[keep]
            fine_viable = (fc >= 1e-6) & (fb <= 0.05) & (p_fine < prep["critical_p"])
            fine_span = _plateau(fp, fine_viable)
            if fine_span is not None and fp[fine_span[0]] == peaks[i]:
                p_opt = float(np.sqrt(p_fine[fine_span[0]] * p_fine[fine_span[1]]))
        logger.info(
            f"    ABMC stability: peak {peaks[i]} held over P in "
            f"[{p_coarse[i]:.3g}, {p_coarse[j]:.3g}]; selected P = {p_opt:.4g}."
        )
        if i == j:
            warnings.warn(
                f"the localiser peak is stable at only one value of P "
                f"({p_coarse[i]:.3g}); the result is sensitive to this setting. "
                "Treat the localisation with caution, and consider a denser "
                "sweep (larger n_coarse) or a longer data segment.",
                RuntimeWarning,
                stacklevel=2,
            )

    return p_all, pk_all, tm_all, bl_all, cp_all, p_opt


@verbose
def make_abmc(
    info,
    forward,
    data,
    template,
    *,
    cov=None,
    noise_cov=None,
    P=0.03,
    reg=0.0,
    f=1.0,
    max_lag=None,
    method="closed-form",
    mu=None,
    max_iter=100000,
    tol=1e-6,
    return_weights=False,
    verbose=None,
):
    r"""Localise a spike-like source with the ABMC beamformer.

    Runs the template-constrained beamformer of Shirani et al. (2024)
    :footcite:`Shirani2024`, Eqs. 14-19, over the source grid. For each grid point
    and orientation the weight vector solves

    .. math::

        \min_W \tfrac12 W^\mathsf{T} R W \quad\text{s.t.}\quad
        G^\mathsf{T} W = f \;\text{ and }\; \max_W (W^\mathsf{T} X \cdot u),

    a distortionless minimum-variance beamformer with an added
    maximum-cross-correlation-to-template constraint. The unconstrained
    Lagrangian (Eqs. 15-17) has :math:`\beta_1` eliminated via the
    gain constraint (Eq. 19) and :math:`\beta_2 = P\beta_1`. The template lag
    :math:`j^*` is fixed once per grid point (seeded from an initial LCMV output),
    per the confirmed reading of the paper.

    Following the paper, the source is localised by the **maximum cross-correlation
    between the beamformer output and the template**,
    :math:`|\mathrm{corr}(W^\mathsf{T}X, u_{j^*})|`, maximised over orientation:
    the estimate is the grid location whose output best matches the desired
    morphology at the best lag. That same criterion picks the orientation at each
    grid point. The output variance :math:`\tfrac12 W^\mathsf{T}RW` is the
    beamformer's minimisation *objective*, not the localiser, and is returned
    only as a diagnostic.

    Parameters
    ----------
    info : mne.Info
        Measurement info. Defines the channels used; ``info['bads']`` are
        excluded.
    forward : mne.Forward
        Forward solution, on a surface, volume, discrete or mixed source space.
        Fixed- or free-orientation.
    data : mne.Evoked | mne.io.Raw | ndarray, shape (n_channels, n_times)
        The sensor data segment :math:`X` to localise. Epochs are not accepted;
        average or concatenate them first.
    template : ndarray, shape (n_times,)
        The desired-source waveform :math:`u` to localise. This is the morphology
        of the target activity, supplied by the caller and the same length as
        ``data``. In the paper these are expert-annotated IED or DR templates, but
        any known target morphology may be passed; ABMC is steered to the location
        whose output best matches this template (at the best lag). Only its shape
        matters: the readout is invariant to its amplitude.
    cov : mne.Covariance | None
        Covariance :math:`R` for the beamformer. If ``None``, it is estimated
        from ``data`` by :func:`sbl_covariance` (the intended ABMC pipeline).
    noise_cov : mne.Covariance | None
        Passed to :func:`sbl_covariance` when ``cov`` is ``None``.
    P : float
        Ratio :math:`\beta_2/\beta_1` weighting the template constraint against
        the distortionless one (default 0.03). The constraint column is rescaled
        to the norm of its leadfield column before Eq. 19 is applied, so ``P`` is
        a dimensionless trade-off and its useful range does not depend on the
        units of the data. Work at ``P`` of order 0.01-0.1. On the 94-channel
        spherical-EEG fixture of ``examples/plot_abmc_localization.py`` the
        constraint changed the weights by 1 to 18 per cent over ``P`` in
        [0.01, 0.18] while the localised peak stayed exactly where the
        :math:`P\to 0` (plain LCMV) limit put it on all eight simulated spikes;
        at :math:`P=1` half of those peaks had moved and the mean peak error had
        risen from 0.85 cm to 2.20 cm. ``P`` far below that range reduces ABMC
        to an iterative LCMV (a warning fires). Above it, the test for ``P``
        being too large is ``result.critical_p``, the first ``P`` at which a
        column's gain denominator vanishes; ``result.blowup_fraction`` sees only
        the immediate neighbourhood of that value. Where exactly the stable
        range ends is a property of the recording rather than of the method, so
        use ``P='auto'`` (:func:`abmc_stability_curve`) to place it on your own
        data.
    reg : float
        Diagonal loading of :math:`R`, as a fraction of
        :math:`\mathrm{tr}(R)/M`. The default is **0**, which is what the paper
        does: its :math:`R = G\alpha G^\mathsf{T} + \Lambda` has an estimated
        per-channel noise term :math:`\Lambda` and is positive definite by
        construction, so no loading is needed (its condition number is about 20
        on the MNE ``sample`` data). Raise it only when supplying your own
        ill-conditioned ``cov``; at 0.05 the Stage-2 weights move by a couple of
        per cent on a well-conditioned :math:`R`, and considerably more on a
        poorly conditioned one.
    f : float
        Distortionless gain (default 1.0).
    max_lag : int | None
        Restrict the template lag search to :math:`|j|\le` ``max_lag`` samples.
        ``None`` searches all lags.
    method : 'closed-form' | 'iterative'
        How Stage 2 is solved. ``'closed-form'`` (default) evaluates the fixed
        point of Eqs. 17-19 directly. ``'iterative'`` runs the paper's gradient
        descent verbatim, which is slower (its step count grows with the
        condition number of :math:`R`) and is provided for exact reproduction;
        the two agree once the descent has converged.
    mu : float | None
        Descent step size, used only when ``method='iterative'``. ``None`` uses
        :math:`1/\lambda_{\max}(R)`.
    max_iter : int
        Maximum descent iterations, used only when ``method='iterative'``.
    tol : float
        Convergence tolerance for the descent, used only when
        ``method='iterative'``. It is the *relative distance to the fixed point*,
        not the size of the step: on an ill-conditioned :math:`R` the steps
        become small precisely because the descent is crawling, so a step-size
        rule reports convergence while the weights are still far away.
    return_weights : bool
        If ``True``, include the beamformer weights in the result.
    %(verbose)s

    Returns
    -------
    result : ABMCResult
        The localization map and diagnostics; see :class:`ABMCResult`.

    References
    ----------
    .. footbibliography::
    """
    if max_lag is not None and int(max_lag) < 0:
        raise ValueError(f"max_lag must be >= 0 samples, got {max_lag}.")
    _check_option("method", method, ("closed-form", "iterative"))
    prep = _abmc_prepare(info, forward, data, template, cov, noise_cov, reg, max_lag)

    if P == "auto":
        P = _abmc_select_p(prep, forward, f, (1e-4, 1e4), 17, 9)[-1]
        logger.info(f"    ABMC: selected P = {P:.4g} from the stability curve.")
    P = float(P)
    if not P > 0:
        raise ValueError(f"P must be > 0 or 'auto', got {P}.")

    coupling = float(np.max(np.abs(P * prep["gc"] / np.clip(prep["gg"], _TINY, None))))
    if coupling < 1e-6:
        warnings.warn(
            "the template constraint is numerically inert (P g^T c / g^T g is "
            f"at most {coupling:.2e}); ABMC reduces to a plain LCMV beamformer. "
            "Raise P, or check that the template is not orthogonal to the data.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Checked before the solve, because past the first pole the weights are not
    # large but simply wrong, so no post-hoc statistic will say so.
    critical_p = prep["critical_p"]
    if P >= critical_p:
        warnings.warn(
            f"P={P:.4g} is at or above the critical value {critical_p:.4g} at "
            "which the gain denominator g^T g + P g^T c of a grid column "
            f"vanishes ({prep['unstable_fraction']:.1%} of columns have "
            "g^T c < 0). Beyond it the weights of those columns are past their "
            "pole and the localisation is unreliable, without necessarily "
            "blowing up. Lower P below the critical value, or pass P='auto'.",
            RuntimeWarning,
            stacklevel=2,
        )

    if method == "closed-form":
        w, degenerate = _abmc_fixed_point(prep, P, f)
        n_iter, converged = 0, True
    else:
        w, degenerate, n_iter, converged, distance, mu_used = _abmc_descend(
            prep, P, f, mu, max_iter, tol
        )
        logger.info(
            f"    ABMC: descent ran {n_iter} iteration(s), mu = {mu_used:.3e}, "
            f"relative distance to the fixed point {distance:.2e}."
        )
        if not converged:
            warnings.warn(
                f"the ABMC descent did not reach the fixed point in {max_iter} "
                f"iterations (relative distance {distance:.2e} > tol={tol}). The "
                "number of steps needed grows with the condition number of R; "
                "raise max_iter, or use the default method='closed-form', which "
                "solves for the same estimator directly.",
                RuntimeWarning,
                stacklevel=2,
            )

    if degenerate.any():
        warnings.warn(
            f"{int(degenerate.sum())} of {prep['n_columns']} grid columns have a "
            "degenerate gain constraint (g^T R^-1 (g + P c) is numerically "
            "zero); their localiser value is set to 0. Reduce P, or check the "
            "forward model at those locations.",
            RuntimeWarning,
            stacklevel=2,
        )

    result = _abmc_readout(prep, w, forward, P, f, return_weights, n_iter, converged)
    if result.blowup_fraction > 0.05:
        warnings.warn(
            f"{result.blowup_fraction:.0%} of grid weights blew up; P={P:.4g} is "
            "likely too large for this segment (cf. the paper's non-convergence "
            "regime).",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


@verbose
def make_abmc_dictionary(
    info,
    forward,
    data,
    templates,
    *,
    cov=None,
    noise_cov=None,
    P=0.03,
    reg=0.0,
    f=1.0,
    max_lag=None,
    return_weights=False,
    verbose=None,
):
    r"""Localise with ABMC for each template in a dictionary of desired waveforms.

    Shirani et al. (2024) :footcite:`Shirani2024` match several expert-annotated
    templates per case. Because the sparse Bayesian covariance :math:`R` depends
    only on the data and not on the template, it is estimated **once** here and
    reused for every template, so this is materially cheaper than calling
    :func:`make_abmc` once per template. Stage 2 (the template-constrained
    beamformer) is then run independently for each template, and the
    per-template results are returned in a dictionary.

    Parameters
    ----------
    info : mne.Info
        Measurement info. Defines the channels used; ``info['bads']`` are
        excluded.
    forward : mne.Forward
        Forward solution, on a surface, volume, discrete or mixed source space.
        Fixed- or free-orientation.
    data : mne.Evoked | mne.io.Raw | ndarray, shape (n_channels, n_times)
        The sensor data segment :math:`X` to localise. Epochs are not accepted;
        average or concatenate them first.
    templates : dict of {label: ndarray} | sequence of ndarray
        The desired-source waveforms :math:`u` to localise, each the same length
        as ``data``. A plain sequence is labelled by integer position. Each entry
        is handled independently: ABMC is run once per template.
    cov : mne.Covariance | None
        Covariance :math:`R` for the beamformer, shared across all templates. If
        ``None`` (default) it is estimated once from ``data`` by
        :func:`sbl_covariance`.
    noise_cov : mne.Covariance | None
        Passed to :func:`sbl_covariance` when ``cov`` is ``None``.
    P : float
        Ratio :math:`\beta_2/\beta_1` for the template constraint; see
        :func:`make_abmc`. Applied to every template.
    reg : float
        Diagonal loading of :math:`R`, as in :func:`make_abmc` (default 0).
    f : float
        Distortionless gain, as in :func:`make_abmc`.
    max_lag : int | None
        Template-lag search window in samples, as in :func:`make_abmc`.
    return_weights : bool
        Whether to include the beamformer weights in each result, as in
        :func:`make_abmc`.
    %(verbose)s

    Returns
    -------
    results : dict
        One ABMC scan per template, returned as ``{label: ABMCResult}`` and keyed
        by the dictionary key (or by integer position for a sequence).

    See Also
    --------
    make_abmc : Localise a single template.

    References
    ----------
    .. footbibliography::
    """
    if isinstance(templates, dict):
        items = list(templates.items())
    else:
        items = list(enumerate(templates))
    if not items:
        raise ValueError("templates must be a non-empty dict or sequence.")

    leadfield, x, ch_names = _aligned_leadfield_and_data(info, forward, data)
    n_times = x.shape[1]
    for label, template in items:
        length = np.asarray(template, float).ravel().shape[0]
        if length != n_times:
            raise ValueError(
                f"template {label!r} has length {length}, must match data "
                f"length {n_times}."
            )

    # estimate the SBL covariance once: it does not depend on the template
    if cov is None:
        data_cov = Covariance(
            x @ x.T / n_times, ch_names, bads=[], projs=[], nfree=n_times
        )
        cov = sbl_covariance(info, forward, data_cov, noise_cov=noise_cov)

    return {
        label: make_abmc(
            info,
            forward,
            data,
            template,
            cov=cov,
            noise_cov=noise_cov,
            P=P,
            reg=reg,
            f=f,
            max_lag=max_lag,
            return_weights=return_weights,
        )
        for label, template in items
    }
