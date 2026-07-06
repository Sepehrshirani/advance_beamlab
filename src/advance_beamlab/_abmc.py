r"""ABMC: adaptive Bayesian beamformer with multiple constraints (Shirani et al., 2024).

This module implements the beamforming pipeline of Shirani, Abdi-Sargezeh,
Valentin, Alarcon, Bird & Sanei (2024), "Do interictal epileptiform discharges
and brain responses to electrical stimulation come from the same location? An
advanced source localization solution", IEEE TBME 71(9):2771-2781
(doi:10.1109/TBME.2024.3392603). The method localizes low-power, spike-like
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
- **Stage 2 -- template-constrained iterative beamformer** (Eqs. 14-19; added
  separately). A minimum-variance beamformer with the usual distortionless
  constraint *plus* a maximum-cross-correlation-to-template constraint that locks
  the output onto the known DR/IED morphology.

Localization follows the paper's criterion: the source is the grid location whose
beamformer output has the **maximum cross-correlation with the desired template**
:math:`u` at the best lag (not the output power, which is what LCMV maximizes).
The template is supplied by the caller -- in the paper, expert-annotated IED or DR
morphologies -- so ABMC can be steered to any known target waveform, not a fixed
shape. Both stages are provided here: :func:`sbl_covariance` (Stage 1) and
:func:`make_abmc` (Stage 2).
"""
# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

from dataclasses import dataclass

import numpy as np
from mne import Covariance
from mne.utils import _validate_type, logger, warn


def _aligned_leadfield_and_cov(info, forward, data_cov):
    """Return leadfield ``G`` and covariance ``C`` aligned on common channels.

    ``G`` is ``(n_channels, n_columns)`` -- one column per source for a
    fixed-orientation forward, three per source (x, y, z) for a free-orientation
    forward -- and ``C`` is the matching ``(n_channels, n_channels)`` block of
    ``data_cov``, both ordered by the returned ``ch_names``.
    """
    fwd_ch = list(forward["sol"]["row_names"])
    cov_ch = list(data_cov["names"])
    keep = set(cov_ch) & set(info["ch_names"]) & set(fwd_ch)
    ch_names = [c for c in fwd_ch if c in keep]
    if len(ch_names) < 2:
        raise ValueError(
            "fewer than 2 channels are common to info, forward and data_cov; "
            "cannot estimate a covariance."
        )
    g_idx = [fwd_ch.index(c) for c in ch_names]
    c_idx = [cov_ch.index(c) for c in ch_names]
    leadfield = np.asarray(forward["sol"]["data"], float)[g_idx]
    cov = np.asarray(data_cov.data, float)[np.ix_(c_idx, c_idx)]
    return leadfield, cov, ch_names


def _sbl_cost(cov, precision):
    r"""Type-II ML cost :math:`F=\mathrm{tr}(CR^{-1})+\log|R|` (Eq. 6)."""
    sign, logdet = np.linalg.slogdet(np.linalg.inv(precision))
    return float(np.trace(cov @ precision) + (logdet if sign > 0 else np.inf))


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

    Fits the generative model of Shirani et al. (2024), Eqs. 2-5,

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

    Parameters
    ----------
    info : mne.Info
        Measurement info; only the channel set is used, to align ``forward`` and
        ``data_cov``.
    forward : mne.Forward
        Forward solution. A fixed-orientation forward gives one source per grid
        point; a free-orientation forward gives three columns (x, y, z) per grid
        point, each treated as an independent source in the prior.
    data_cov : mne.Covariance
        Sensor data covariance :math:`C=\tfrac1T XX^\mathsf{T}`, e.g. from
        :func:`mne.compute_covariance` over the analysis segment.
    noise_cov : mne.Covariance | None
        If given, its diagonal initialises :math:`\Lambda`; otherwise
        :math:`\Lambda` is initialised isotropically at
        :math:`\mathrm{tr}(C)/M`. The noise variances are then learned either
        way (Eq. 10).
    max_iter : int
        Maximum number of update iterations (default 100).
    tol : float
        Stop when the relative change in the cost :math:`F` (Eq. 6) between
        iterations falls below this (default 1e-5).
    return_source_power : bool
        If ``True``, also return the fitted source-power vector :math:`\alpha`
        (one value per leadfield column). Note it carries the usual SBL depth
        bias, since the raw (un-normalised) leadfield is used.
    verbose : bool | str | int | None
        Passed through to the MNE logger.

    Returns
    -------
    cov : mne.Covariance
        The model covariance :math:`R` over the common channels.
    source_power : ndarray, shape (n_columns,)
        The fitted :math:`\alpha`; returned only if ``return_source_power`` is
        ``True``.

    Notes
    -----
    The updates are run in the sensor space with a per-channel diagonal noise
    model, matching the paper's single-sensor-type intracranial recordings. For
    data combining multiple sensor types with different units, whiten before
    calling so that a diagonal :math:`\Lambda` is meaningful.
    """
    _validate_type(data_cov, Covariance, "data_cov")
    _validate_type(noise_cov, (Covariance, None), "noise_cov")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}.")

    leadfield, cov, ch_names = _aligned_leadfield_and_cov(info, forward, data_cov)
    n_channels, n_columns = leadfield.shape

    # initialise the source variances and the diagonal noise
    alpha = np.ones(n_columns)
    if noise_cov is None:
        lam = np.full(n_channels, np.trace(cov) / n_channels)
    else:
        ncov_ch = list(noise_cov["names"])
        di = [ncov_ch.index(c) for c in ch_names]
        lam = np.array([noise_cov.data[k, k] for k in di])
        lam = np.clip(lam, np.finfo(float).eps, None)

    prev_cost = np.inf
    converged = False
    for iteration in range(max_iter):
        precision = np.linalg.inv((leadfield * alpha) @ leadfield.T + np.diag(lam))
        prec_c_prec = precision @ cov @ precision

        # alpha update (Eq. 9): needs diag(G^T (R^-1 C R^-1) G) and diag(G^T R^-1 G)
        num_alpha = np.einsum("mk,mk->k", leadfield, prec_c_prec @ leadfield)
        den_alpha = np.einsum("mk,mk->k", leadfield, precision @ leadfield)
        alpha = alpha * np.sqrt(
            np.clip(num_alpha, 0, None) / np.clip(den_alpha, np.finfo(float).eps, None)
        )

        # lambda update (Eq. 10): diagonals of R^-1 C R^-1 and R^-1
        num_lam = np.clip(np.diag(prec_c_prec), 0, None)
        den_lam = np.clip(np.diag(precision), np.finfo(float).eps, None)
        lam = lam * np.sqrt(num_lam / den_lam)

        cost = _sbl_cost(cov, precision)
        if np.isfinite(prev_cost) and abs(prev_cost - cost) <= tol * abs(prev_cost):
            converged = True
            logger.info(f"SBL covariance converged after {iteration + 1} iterations.")
            break
        prev_cost = cost

    if not converged:
        warn(
            f"SBL covariance did not converge in {max_iter} iterations "
            f"(relative cost change above tol={tol}); returning current estimate."
        )

    model = (leadfield * alpha) @ leadfield.T + np.diag(lam)
    model = 0.5 * (model + model.T)  # symmetrise against round-off
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

    Attributes
    ----------
    stc : mne.VolSourceEstimate
        The localization map -- the per-grid-point template match (see
        ``template_match``), as a single-time source estimate for plotting.
    template_match : ndarray, shape (n_sources,)
        Primary localizer: :math:`|\mathrm{corr}(W^\mathsf{T}X, u_{j^*})|`, the
        absolute correlation between the beamformer output and the lag-aligned
        template, maximised over orientation at each grid point. The peak is the
        estimated source.
    power : ndarray, shape (n_sources,)
        The beamformer output variance :math:`\tfrac12 W^\mathsf{T}RW` summed over
        orientations -- the filter's minimisation objective (what LCMV would
        localise on), retained here only to select the source orientation, not for
        the scan.
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
    """Return leadfield ``G`` and sensor data ``X`` aligned on common channels."""
    from mne import Evoked
    from mne.io import BaseRaw

    if isinstance(data, (Evoked, BaseRaw)):
        x_full, data_ch = np.asarray(data.data, float), list(data.ch_names)
    else:
        x_full = np.asarray(data, float)
        data_ch = list(info["ch_names"])
        if x_full.shape[0] != len(data_ch):
            raise ValueError(
                f"data has {x_full.shape[0]} rows but info has {len(data_ch)} "
                "channels; pass an Evoked/Raw or a matching array."
            )
    fwd_ch = list(forward["sol"]["row_names"])
    keep = set(data_ch) & set(info["ch_names"]) & set(fwd_ch)
    ch_names = [c for c in fwd_ch if c in keep]
    if len(ch_names) < 2:
        raise ValueError("fewer than 2 channels are common to info, forward and data.")
    g_idx = [fwd_ch.index(c) for c in ch_names]
    leadfield = np.asarray(forward["sol"]["data"], float)[g_idx]
    x = x_full[[data_ch.index(c) for c in ch_names]]
    return leadfield, x, ch_names


def _shift_template(u, lag):
    """Shift ``u`` by ``lag`` samples with zero padding (positive = delay)."""
    out = np.zeros_like(u)
    if lag >= 0:
        out[lag:] = u[: len(u) - lag] if lag < len(u) else 0.0
    else:
        out[: len(u) + lag] = u[-lag:]
    return out


def make_abmc(
    info,
    forward,
    data,
    template,
    *,
    cov=None,
    noise_cov=None,
    P=0.03,
    mu=None,
    reg=0.05,
    f=1.0,
    max_iter=60,
    tol=1e-4,
    max_lag=None,
    return_weights=False,
    verbose=None,
):
    r"""Localise a spike-like source with the ABMC beamformer (Shirani et al., 2024).

    Runs the template-constrained iterative beamformer of Eqs. 14-19 over the
    source grid. For each grid point and orientation the weight vector solves

    .. math::

        \min_W \tfrac12 W^\mathsf{T} R W \quad\text{s.t.}\quad
        G^\mathsf{T} W = f \;\text{ and }\; \max_W (W^\mathsf{T} X \cdot u),

    a distortionless minimum-variance beamformer with an added
    maximum-cross-correlation-to-template constraint. The unconstrained
    Lagrangian is descended (Eqs. 15-17) with :math:`\beta_1` eliminated via the
    gain constraint (Eq. 19) and :math:`\beta_2 = P\beta_1`. The template lag
    :math:`j^*` is fixed once per grid point (seeded from an initial LCMV output),
    per the confirmed reading of the paper.

    Following the paper, the source is localised by the **maximum cross-correlation
    between the beamformer output and the template**,
    :math:`|\mathrm{corr}(W^\mathsf{T}X, u_{j^*})|`, maximised over orientation --
    the grid location whose output best matches the desired morphology at the best
    lag. The output variance :math:`\tfrac12 W^\mathsf{T}RW` is the beamformer's
    minimisation *objective*, not the localiser; it is retained per grid point only
    to select the source orientation.

    Parameters
    ----------
    info : mne.Info
        Measurement info (channel set only).
    forward : mne.Forward
        Forward solution (volume source space). Fixed- or free-orientation.
    data : mne.Evoked | mne.io.Raw | ndarray, shape (n_channels, n_times)
        The sensor data segment :math:`X` to localise.
    template : ndarray, shape (n_times,)
        The desired-source waveform :math:`u` to localise -- the morphology of the
        target activity, supplied by the caller and the same length as ``data``. In
        the paper these are expert-annotated IED or DR templates, but any known
        target morphology may be passed; ABMC is steered to the location whose
        output best matches this template (at the best lag).
    cov : mne.Covariance | None
        Covariance :math:`R` for the beamformer. If ``None``, it is estimated
        from ``data`` by :func:`sbl_covariance` (the intended ABMC pipeline).
    noise_cov : mne.Covariance | None
        Passed to :func:`sbl_covariance` when ``cov`` is ``None``.
    P : float
        Ratio :math:`\beta_2/\beta_1` weighting the template constraint (default
        0.03). Larger emphasises the template but risks the non-convergence /
        weight-blow-up regime of the paper; keep it small.
    mu : float | None
        Gradient step size. If ``None``, set to :math:`1/\lambda_{\max}(R)`.
    reg : float
        Diagonal loading (fraction of :math:`\mathrm{tr}(R)/M`) for the inverse
        used to seed the lag (default 0.05).
    f : float
        Distortionless gain (default 1.0).
    max_iter : int
        Maximum weight-update iterations (default 60).
    tol : float
        Stop when the relative change in :math:`\|W\|` falls below this.
    max_lag : int | None
        Restrict the template lag search to :math:`|j|\le` ``max_lag`` samples.
        ``None`` searches all lags.
    return_weights : bool
        If ``True``, include the beamformer weights in the result.
    verbose : bool | str | int | None
        Passed to the MNE logger.

    Returns
    -------
    result : ABMCResult
        The localization map and diagnostics; see :class:`ABMCResult`.
    """
    from mne import Covariance, VolSourceEstimate

    if not 0 < P:
        raise ValueError(f"P must be > 0, got {P}.")
    leadfield, x, ch_names = _aligned_leadfield_and_data(info, forward, data)
    n_channels, n_columns = leadfield.shape
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
        r_cov = sbl_covariance(
            info, forward, data_cov, noise_cov=noise_cov, verbose=verbose
        )
        cov_mat = r_cov.data
    else:
        cov_ch = list(cov["names"])
        idx = [cov_ch.index(c) for c in ch_names]
        cov_mat = np.asarray(cov.data, float)[np.ix_(idx, idx)]

    r = 0.5 * (cov_mat + cov_mat.T)
    if mu is None:
        mu = 1.0 / np.linalg.eigvalsh(r).max()

    # seed the per-column lag from an initial (distortionless LCMV) output
    r_reg = r + reg * np.trace(r) / n_channels * np.eye(n_channels)
    r_inv = np.linalg.inv(r_reg)
    r_inv_g = r_inv @ leadfield
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

    # iterate the weights (Eqs. 17-19), vectorised over all columns
    gg = np.einsum("mk,mk->k", leadfield, leadfield)
    gc = np.einsum("mk,mk->k", leadfield, c)
    denom = mu * (gg + P * gc)
    w = np.zeros((n_channels, n_columns))
    converged = False
    n_run = 0
    for iteration in range(max_iter):
        rw = r @ w
        g_w = np.einsum("mk,mk->k", leadfield, w)
        g_rw = np.einsum("mk,mk->k", leadfield, rw)
        beta1 = (f - g_w + mu * g_rw) / denom
        beta2 = P * beta1
        step = rw - leadfield * beta1[None, :] - c * beta2[None, :]
        w_new = w - mu * step
        n_run = iteration + 1
        delta = np.linalg.norm(w_new - w) / max(
            np.linalg.norm(w_new), np.finfo(float).eps
        )
        w = w_new
        if np.all(np.isfinite(w)) and delta < tol:
            converged = True
            break
    if not np.all(np.isfinite(w)):
        warn("ABMC weights diverged; reduce P or mu.")

    # readouts per column
    out = np.einsum("mk,mt->kt", w, x)
    out_c = out - out.mean(1, keepdims=True)
    us_c = u_shift - u_shift.mean(1, keepdims=True)
    denom_corr = np.linalg.norm(out_c, axis=1) * np.linalg.norm(us_c, axis=1)
    num_corr = np.abs(np.einsum("kt,kt->k", out_c, us_c))
    tmatch = num_corr / np.clip(denom_corr, np.finfo(float).eps, None)
    power = 0.5 * np.einsum("mk,mk->k", w, r @ w)

    col_norm = np.linalg.norm(w, axis=0)
    blowup = float((col_norm > 10 * np.median(col_norm)).mean())
    if blowup > 0.05:
        warn(
            f"{blowup:.0%} of grid weights blew up; P={P} is likely too large "
            "for this segment (cf. the paper's non-convergence regime)."
        )

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

    vertices = [np.asarray(forward["src"][0]["vertno"])]
    stc = VolSourceEstimate(
        tmatch_grid[:, None], vertices=vertices, tmin=0.0, tstep=1.0, subject=None
    )
    return ABMCResult(
        stc=stc,
        template_match=tmatch_grid,
        power=power_grid,
        lag=lag_grid,
        orientation=orientation if n_ori > 1 else np.zeros(n_sources, int),
        weights=w if return_weights else None,
        n_iter=n_run,
        converged=converged,
        blowup_fraction=blowup,
    )
