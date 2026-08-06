r"""ABMC: adaptive Bayesian beamformer with multiple constraints.

This module implements the beamforming pipeline of Shirani et al. (2024)
:footcite:`Shirani2024`, "Do interictal epileptiform discharges and brain
responses to electrical stimulation come from the same location? An advanced
source localization solution". The method localizes low-power, spike-like
transients (interictal epileptiform discharges, IEDs, and delayed responses,
DRs, to single-pulse electrical stimulation) that ordinary LCMV localizes poorly
because it is power-sensitive and collapses nearby correlated sources.

ABMC has two stages:

- **Stage 1 -- sparse Bayesian learning (SBL) covariance** (:func:`sbl_covariance`,
  Eqs. 5-13). A Champagne-style type-II maximum-likelihood fit of per-source
  prior variances :math:`\alpha` and a diagonal sensor-noise covariance
  :math:`\Lambda` yields a *model* covariance :math:`R = G\alpha G^\mathsf{T} +
  \Lambda`. Because the sources are modelled as mutually uncorrelated
  (:math:`\alpha` diagonal), :math:`R` does not carry the cross-source
  correlation structure that makes an LCMV beamformer cancel correlated sources.
- **Stage 2 -- template-constrained beamformer** (Eqs. 14-19). A
  minimum-variance beamformer with the usual distortionless constraint *plus* a
  maximum-cross-correlation-to-template constraint that locks the output onto the
  known DR/IED morphology. The paper reaches it by gradient descent; this
  implementation solves it at that descent's fixed point, which is available in
  closed form (see :func:`make_abmc`).

Localization follows the paper's criterion: the source is the grid location whose
beamformer output has the **maximum cross-correlation with the desired template**
:math:`u` at the best lag (not the output power, which is what LCMV maximizes).
The template is supplied by the caller -- in the paper, expert-annotated IED or DR
morphologies -- so ABMC can be steered to any known target waveform, not a fixed
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

Both are handled by rescaling every channel by its noise standard deviation --
from ``noise_cov`` when one is supplied, otherwise from MNE's ad-hoc per-type
model -- fitting there, and undoing the scaling on the returned covariance. A
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

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import warnings
from dataclasses import dataclass

import numpy as np
from mne import Covariance
from mne.forward.forward import _subject_from_forward
from mne.source_estimate import _get_src_type, _make_stc
from mne.source_space._source_space import _get_vertno
from mne.utils import _validate_type, logger, verbose
from scipy.linalg import cho_factor, cho_solve

# NOTE: as in ``_mcmv``, the stdlib ``warnings.warn`` (RuntimeWarning) is used
# rather than ``mne.utils.warn`` so that warnings are emitted regardless of the
# logging level and are reliably catchable by ``pytest.warns``.
# Reuse the channel-selection helpers from the MCMV module so that every
# algorithm in the package reads the forward, the data and the covariances in
# the same channel space -- in particular, so that all of them drop
# ``info['bads']``, which ``mne.compute_covariance`` has already dropped.
from ._mcmv import (
    _align_channels,
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

    ``G`` is ``(n_channels, n_columns)`` -- one column per source for a
    fixed-orientation forward, three per source (x, y, z) for a free-orientation
    forward -- and ``C`` is the matching ``(n_channels, n_channels)`` block of
    ``data_cov``, both ordered by the returned ``ch_names``. Bad channels are
    excluded and, when a noise covariance is given, channels it does not cover
    are dropped, exactly as in :func:`~advance_beamlab.make_mcmv`.
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
    ``slogdet(inv(R^-1))`` -- an inverse of an inverse -- loses accuracy and, for
    a covariance in SI units, under- or overflows on the way.
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
        The localization map -- the per-grid-point template match (see
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
        over orientations -- the filter's minimisation objective, and what LCMV
        would localise on. It is reported as a diagnostic only: neither the scan
        nor the orientation choice uses it, both being driven by
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
        Fraction of grid columns whose weights grew anomalously large -- a signal
        that ``P`` is too large for this segment (cf. the paper's non-convergence
        regime). Above ~0.05, lower ``P``.

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
    reg=0.05,
    f=1.0,
    max_lag=None,
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
    :math:`|\mathrm{corr}(W^\mathsf{T}X, u_{j^*})|`, maximised over orientation --
    the grid location whose output best matches the desired morphology at the best
    lag. That same criterion picks the orientation at each grid point. The output
    variance :math:`\tfrac12 W^\mathsf{T}RW` is the beamformer's minimisation
    *objective*, not the localiser, and is returned only as a diagnostic.

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
        The desired-source waveform :math:`u` to localise -- the morphology of the
        target activity, supplied by the caller and the same length as ``data``. In
        the paper these are expert-annotated IED or DR templates, but any known
        target morphology may be passed; ABMC is steered to the location whose
        output best matches this template (at the best lag). Only its shape
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
        units of the data: ``P`` of order 0.01-0.1 gives the template a
        perceptible but subordinate weight, ``P`` far below that reduces ABMC to
        an iterative LCMV, and large ``P`` enters the paper's weight-blow-up /
        non-convergence regime. Watch ``result.blowup_fraction``.
    reg : float
        Diagonal loading (fraction of :math:`\mathrm{tr}(R)/M`) for the inverse
        used to seed the lag (default 0.05).
    f : float
        Distortionless gain (default 1.0).
    max_lag : int | None
        Restrict the template lag search to :math:`|j|\le` ``max_lag`` samples.
        ``None`` searches all lags.
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
    if not 0 < P:
        raise ValueError(f"P must be > 0, got {P}.")
    leadfield, x, ch_names = _aligned_leadfield_and_data(info, forward, data)
    u = np.asarray(template, float).ravel()
    if u.shape[0] != x.shape[1]:
        raise ValueError(
            f"template length {u.shape[0]} must match data length {x.shape[1]}."
        )

    # covariance R: use the SBL estimate by default (the ABMC pipeline)
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

    # Stage 2 runs in the noise-scaled space, for the same reason Stage 1 does.
    # In recorded units a magnetometer/gradiometer/EEG covariance spans ~16
    # orders of magnitude, R is numerically indefinite, and the solve is driven
    # by whichever sensor type has the larger numerical scale -- on a
    # Neuromag + EEG array the EEG channels contributed essentially all of the
    # filter output and the 305 MEG channels ~1e-10 of it. Dividing each channel
    # by its noise standard deviation makes the types commensurable and R
    # positive definite (condition number ~1e6 rather than ~1e16). The filter is
    # scale-covariant, so the localiser, the output and the power are unchanged;
    # only the returned weights are mapped back to recorded units at the end.
    sd = _noise_scaling(info, ch_names, noise_cov)
    leadfield = leadfield / sd[:, None]
    x = x / sd[:, None]
    cov_mat = cov_mat / np.outer(sd, sd)

    r = 0.5 * (cov_mat + cov_mat.T)

    # seed the per-column lag from an initial (distortionless LCMV) output
    r_reg = r + reg * np.trace(r) / n_channels * np.eye(n_channels)
    r_inv_g = np.linalg.solve(r_reg, leadfield)
    w0 = r_inv_g / np.einsum("mk,mk->k", leadfield, r_inv_g)[None, :]
    y0 = w0.T @ x
    lags_full = np.arange(-(len(u) - 1), len(u))
    if max_lag is not None:
        lag_mask = np.abs(lags_full) <= max_lag
    else:
        lag_mask = np.ones_like(lags_full, dtype=bool)
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
    # column it competes with in Eq. 19. ``c = X u^T`` carries the units of the
    # data times the template while ``g`` carries those of the forward model, so
    # without this the ratio P g^T c / g^T g is a dimensional quantity: on
    # SI-unit MEG it is ~1e-19 at any sane P and the paper's second constraint
    # is inert. Rescaled, P is the dimensionless trade-off the paper describes
    # and the whole scan is invariant to the units of the data and the template.
    c *= (
        np.linalg.norm(leadfield, axis=0)
        / np.clip(np.linalg.norm(c, axis=0), _TINY, None)
    )[None, :]

    gg = np.einsum("mk,mk->k", leadfield, leadfield)
    gc = np.einsum("mk,mk->k", leadfield, c)
    coupling = float(np.max(np.abs(P * gc / np.clip(gg, _TINY, None))))
    if coupling < 1e-6:
        warnings.warn(
            "the template constraint is numerically inert (P g^T c / g^T g is "
            f"at most {coupling:.2e}); ABMC reduces to an iterative LCMV "
            "beamformer. Raise P, or check that the template is not orthogonal "
            "to the data.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Solve Eqs. 17-19 at their fixed point rather than descending to it.
    # Setting the update step to zero gives R w = (g + P c) beta1, and the
    # consistency of Eq. 19 then forces g^T w = f, so the estimator the paper's
    # iteration converges to is available in closed form:
    #
    #     w* = f R^-1 (g + P c) / (g^T R^-1 (g + P c)).
    #
    # This is the same estimator, not a different one -- the shipped iteration's
    # own relative step at w* is ~2e-12 -- but it removes the tuning. The
    # gradient descent needed several thousand steps on a real gradiometer
    # covariance, and its tolerance test (on the size of the *step*) reported
    # convergence while the weights were still ~40% away from w*, moving the
    # localiser peak by ~9 cm. ``r_reg`` rather than ``r`` is inverted, so ``reg``
    # regularises this solve exactly as it does in ``make_mcmv`` and ``make_lcmv``.
    v = leadfield + P * c
    ri_v = np.linalg.solve(r_reg, v)
    denom_w = np.einsum("mk,mk->k", leadfield, ri_v)
    # A column whose gain constraint is degenerate cannot be normalised; it is
    # zeroed and excluded from the localiser below rather than poisoning the map.
    scale = _EPS * np.linalg.norm(leadfield, axis=0) * np.linalg.norm(ri_v, axis=0)
    degenerate = np.abs(denom_w) <= scale
    w = np.where(degenerate[None, :], 0.0, ri_v / np.where(degenerate, 1.0, denom_w))
    w *= f
    n_run, converged = 0, True
    if degenerate.any():
        warnings.warn(
            f"{int(degenerate.sum())} of {n_columns} grid columns have a "
            "degenerate gain constraint (g^T R^-1 (g + P c) is numerically "
            "zero); their localiser value is set to 0. Reduce P, or check the "
            "forward model at those locations.",
            RuntimeWarning,
            stacklevel=2,
        )

    # readouts per column
    out = np.einsum("mk,mt->kt", w, x)
    out_c = out - out.mean(1, keepdims=True)
    us_c = u_shift - u_shift.mean(1, keepdims=True)
    out_norm = np.linalg.norm(out_c, axis=1)
    us_norm = np.linalg.norm(us_c, axis=1)
    denom_corr = out_norm * us_norm
    num_corr = np.abs(np.einsum("kt,kt->k", out_c, us_c))
    # A correlation is scale free, so the guard against a constant (zero
    # variance) signal must be scale free too. An absolute floor carries the
    # units of the data times the template and silently returns a wrong -- and
    # unbounded -- value for a small-amplitude template.
    # Per column, and overflow-safe. A single blown-up column can make its own
    # norm overflow to inf while every weight entry is still finite; a global
    # floor built from ``max()`` would then be inf and zero the entire map.
    ref = np.abs(out_c).max(axis=1) * np.sqrt(out_c.shape[1])
    good = (
        np.isfinite(out_norm)
        & np.isfinite(us_norm)
        & (out_norm > _EPS * ref)
        & (us_norm > 0)
    )
    tmatch = np.zeros(n_columns)
    tmatch[good] = num_corr[good] / denom_corr[good]
    power = 0.5 * np.einsum("mk,mk->k", w, r @ w)

    col_norm = np.linalg.norm(w, axis=0)
    blowup = float((col_norm > 10 * np.median(col_norm)).mean())
    if blowup > 0.05:
        warnings.warn(
            f"{blowup:.0%} of grid weights blew up; P={P} is likely too large "
            "for this segment (cf. the paper's non-convergence regime).",
            RuntimeWarning,
            stacklevel=2,
        )

    # ``out``, ``tmatch`` and ``power`` are invariant to the noise scaling, but
    # the returned weights must act on the recorded data, so map them back:
    # the solve enforced (g / sd)^T w = f, which is g^T (w / sd) = f.
    w = w / sd[:, None]

    # combine orientations -> one value per grid point
    n_sources = forward["nsource"]
    n_ori = n_columns // n_sources
    if n_ori * n_sources != n_columns:
        raise RuntimeError("leadfield columns are not an integer multiple of sources.")
    tmatch_s = tmatch.reshape(n_sources, n_ori)
    orientation = tmatch_s.argmax(1)
    tmatch_grid = tmatch_s.max(1)
    power_grid = power.reshape(n_sources, n_ori).sum(1)
    lag_grid = col_lag.reshape(n_sources, n_ori)[np.arange(n_sources), orientation]

    return ABMCResult(
        stc=_abmc_stc(forward, tmatch_grid),
        template_match=tmatch_grid,
        power=power_grid,
        lag=lag_grid,
        orientation=orientation if n_ori > 1 else np.zeros(n_sources, int),
        weights=w if return_weights else None,
        n_iter=n_run,
        converged=converged,
        blowup_fraction=blowup,
    )


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
    reg=0.05,
    f=1.0,
    max_lag=None,
    return_weights=False,
    verbose=None,
):
    r"""Localise with ABMC for each template in a dictionary of desired waveforms.

    Shirani et al. (2024) :footcite:`Shirani2024` match several expert-annotated
    templates per case. Because the sparse Bayesian covariance :math:`R` depends
    only on the data -- not on the template -- it is estimated **once** here and
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
        is handled independently -- ABMC is run once per template.
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
        Diagonal loading of :math:`R`, as in :func:`make_abmc`.
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
        ``{label: ABMCResult}`` -- one ABMC scan per template, keyed by the
        dictionary key (or by integer position for a sequence).

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

    # estimate the SBL covariance once -- it does not depend on the template
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
