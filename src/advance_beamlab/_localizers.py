r"""MCMV source localizers and data-driven orientation.

This module implements the four MCMV-based scanning localizers of Moiseev et
al. (2011), together with the closed-form optimal source orientation derived in
the same paper. The four are multi-source activity index (MAI), multi-source
pseudo-Z (MPZ), multi-source event-related (MER) and its reduced form (rMER).
These are the ingredients of the iterative source-search that turns MCMV from a
filter for *known* sources into a tool that *discovers* correlated sources.

The localizers are given by Table 1 and the matrix definitions by Table 2 of
:footcite:`Moiseev2011`; the orientation solution is Eqs. (13)-(14)
(Appendix B). All expressions are written verbatim in terms of an explicit noise
covariance :math:`\mathbf{N}`, and are invariant under whitening (for an
invertible whitener :math:`\mathbf{S},\mathbf{G},\mathbf{T},\mathbf{E}` are
unchanged), so callers may pass whitened quantities with :math:`\mathbf{N}=
\mathbf{I}`.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import warnings

import numpy as np
from mne import Covariance
from mne.beamformer._compute_beamformer import _reg_pinv
from mne.forward import is_fixed_orient
from mne.utils import _validate_type, logger, verbose
from scipy.linalg import eigh, solve

from ._mcmv import (
    _MIN_WHITENED_GAIN,
    _align_channels,
    _check_noise_cov_required,
    _compute_mcmv_weights,
    _cov_as_matrix,
    _intersect_noise_cov,
    _make_whitener,
    _split_rank,
    make_mcmv,
)

# The linear algebra behind a localizer evaluation is a symmetric-definite
# generalized eigenproblem (``scipy.linalg.eigh``) or a solve. At a numerically
# degenerate grid location the relevant matrix loses positive-definiteness, and
# LAPACK's failure surfaces as ``LinAlgError`` from ``solve``/``numpy`` but as a
# plain ``ValueError`` from ``eigh`` (its ``potrf`` step reports a non-positive
# leading minor). Both must be caught wherever such a location is to be skipped
# rather than aborting the scan.
_SINGULAR_ERRORS = (np.linalg.LinAlgError, ValueError)

# Each localizer is (A_key, B_key, subtract_n): its value is
# Tr(B_result A_result^-1) - (n if subtract_n else 0), where A is the
# denominator matrix and B the numerator matrix (Table 1). The orientation
# eigenproblem (Eqs. 13-14) uses the same A (in F) and B (in D).
_LOCALIZERS = {
    "mai": ("S", "G", True),  # Tr(G S^-1) - n
    "mpz": ("T", "S", True),  # Tr(S T^-1) - n
    "mer": ("T", "E", False),  # Tr(E T^-1)
    "rmer": ("S", "E", False),  # Tr(E S^-1)
}

_EVENT_RELATED = ("mer", "rmer")  # localizers that require the evoked covariance


def _check_localizer(name, evoked_cov):
    """Validate the localizer name and the presence of an evoked covariance."""
    if name not in _LOCALIZERS:
        raise ValueError(
            f"localizer must be one of {sorted(_LOCALIZERS)}, got {name!r}."
        )
    if name in _EVENT_RELATED and evoked_cov is None:
        raise ValueError(
            f"localizer {name!r} is event-related and requires ``evoked_cov`` "
            "(the covariance of the epoch-averaged field)."
        )


def _metric_matrices(R, N, evoked_cov=None, *, metrics=None):
    r"""Return the per-localizer metric matrices that sit between leadfields.

    Returns the matrices :math:`\mathbf{M}` such that the Table-2 quantities are
    :math:`\mathbf{S}=\mathbf{H}^{\mathsf T}\mathbf{M}_S\mathbf{H}` and so on:

    - ``S`` uses :math:`\mathbf{R}^{-1}`
    - ``G`` uses :math:`\mathbf{N}^{-1}`
    - ``T`` uses :math:`\mathbf{R}^{-1}\mathbf{N}\mathbf{R}^{-1}`
    - ``E`` uses :math:`\mathbf{R}^{-1}\bar{\mathbf{R}}\mathbf{R}^{-1}`
      (only when ``evoked_cov`` is supplied)

    The matrices depend only on the covariances, never on the leadfields, so a
    scan over a source grid computes them once and passes them back in through
    ``metrics``, which is then returned unchanged. That turns the two dense
    inversions below into a single one per scan. Otherwise they would be
    repeated at every grid point of every iteration.
    """
    if metrics is not None:
        return metrics
    Rinv = np.linalg.inv(R)
    metrics = {
        "S": Rinv,
        "G": np.linalg.inv(N),
        "T": Rinv @ N @ Rinv,
    }
    if evoked_cov is not None:
        metrics["E"] = Rinv @ np.asarray(evoked_cov, dtype=np.float64) @ Rinv
    return metrics


def _result_matrix(H, metric):
    """Return the n x n matrix ``H^T metric H`` (a Table-2 quantity)."""
    return H.T @ metric @ H


def localizer_value(name, H, R, N, *, evoked_cov=None, metrics=None):
    r"""Evaluate an MCMV localizer for a set of ``n`` constrained sources.

    Implements the Table-1 functions of :footcite:`Moiseev2011`:
    :math:`P_{\mathrm{MAI}}=\mathrm{Tr}(\mathbf{G}\mathbf{S}^{-1})-n`,
    :math:`P_{\mathrm{MPZ}}=\mathrm{Tr}(\mathbf{S}\mathbf{T}^{-1})-n`,
    :math:`P_{\mathrm{MER}}=\mathrm{Tr}(\mathbf{E}\mathbf{T}^{-1})` and
    :math:`P_{\mathrm{rMER}}=\mathrm{Tr}(\mathbf{E}\mathbf{S}^{-1})`.

    Parameters
    ----------
    name : 'mai' | 'mpz' | 'mer' | 'rmer'
        Which localizer to evaluate.
    H : ndarray, shape (n_channels, n_sources)
        The joint forward matrix of the constrained sources.
    R : ndarray, shape (n_channels, n_channels)
        The data covariance.
    N : ndarray, shape (n_channels, n_channels)
        The noise covariance (pass the identity when working in whitened space).
    evoked_cov : ndarray, shape (n_channels, n_channels) | None
        The covariance of the epoch-averaged field :math:`\bar{\mathbf{R}}`,
        required for the event-related localizers ``'mer'`` and ``'rmer'``.
    metrics : dict | None
        Pre-computed Table-2 metric matrices for these
        covariances. When given, ``R``, ``N`` and ``evoked_cov`` are not used to
        rebuild them; this is how a grid scan avoids re-inverting ``R`` at every
        location. Leave as ``None`` to compute them from the covariances.

    Returns
    -------
    value : float
        The localizer value; it peaks (globally) at the true source set.

    References
    ----------
    .. footbibliography::
    """
    _check_localizer(name, evoked_cov)
    a_key, b_key, subtract_n = _LOCALIZERS[name]
    metrics = _metric_matrices(R, N, evoked_cov, metrics=metrics)
    A = _result_matrix(H, metrics[a_key])
    B = _result_matrix(H, metrics[b_key])
    # Tr(B A^-1) = Tr(A^-1 B) = trace of the solution of A X = B, computed via a
    # solve rather than an explicit inverse for numerical stability.
    value = np.trace(np.linalg.solve(A, B))
    if subtract_n:
        value -= H.shape[1]
    return float(value)


def optimal_orientation(name, H_ref, H_loc, R, N, *, evoked_cov=None, metrics=None):
    r"""Data-driven orientation of a new source (Moiseev 2011, Eqs. 13-14).

    Given the leadfields ``H_ref`` of the sources already found (held fixed) and
    the ``(n_channels, 3)`` leadfield block ``H_loc`` of a candidate location,
    returns the unit orientation that maximises the chosen localizer. No search
    over orientations is needed: the orientation is the eigenvector of the
    largest eigenvalue of the :math:`3\times 3` generalized problem
    :math:`\mathbf{D}\mathbf{u}=\lambda\mathbf{F}\mathbf{u}` with

    .. math::
        \mathbf{F} &= \mathbf{A}_{kk}
            - \mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk}, \\
        \mathbf{D} &= \mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{B}_{RR}
            \mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk}
            - \mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{B}_{Rk}
            - \mathbf{B}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk}
            + \mathbf{B}_{kk},

    where the blocks are the Table-2 matrices built from ``H_ref`` and ``H_loc``,
    ``A`` is the localizer's denominator matrix and ``B`` its numerator (i.e.
    :math:`(\mathbf{A},\mathbf{B})` is ``(S, G)`` for MAI, ``(T, S)`` for MPZ,
    ``(T, E)`` for MER and ``(S, E)`` for rMER). With no references the problem
    reduces to :math:`\mathbf{B}_{kk}\mathbf{u}=\lambda\mathbf{A}_{kk}\mathbf{u}`.

    Parameters
    ----------
    name : 'mai' | 'mpz' | 'mer' | 'rmer'
        Which localizer to maximise.
    H_ref : ndarray, shape (n_channels, n_ref)
        Leadfields of the already-found sources; may have ``n_ref == 0``.
    H_loc : ndarray, shape (n_channels, 3)
        The three orthogonal leadfields of the candidate location.
    R, N : ndarray, shape (n_channels, n_channels)
        Data and noise covariances (pass the identity for ``N`` when whitened).
    evoked_cov : ndarray | None
        The averaged-field covariance, required for ``'mer'`` and ``'rmer'``.
    metrics : dict | None
        Pre-computed Table-2 metric matrices, as in
        :func:`localizer_value`; supply them to avoid re-inverting ``R`` at every
        location of a grid scan.

    Returns
    -------
    u : ndarray, shape (3,)
        The unit orientation maximising the localizer at this location.

    References
    ----------
    .. footbibliography::
    """
    _check_localizer(name, evoked_cov)
    if H_loc.shape[1] != 3:
        raise ValueError(
            f"H_loc must have 3 columns (x, y, z leadfields), got {H_loc.shape[1]}."
        )
    a_key, b_key, _ = _LOCALIZERS[name]
    metrics = _metric_matrices(R, N, evoked_cov, metrics=metrics)
    mA, mB = metrics[a_key], metrics[b_key]

    # kk blocks (3 x 3) always present.
    A_kk = H_loc.T @ mA @ H_loc
    B_kk = H_loc.T @ mB @ H_loc

    if H_ref.shape[1] == 0:
        # First source: no references, D = B_kk, F = A_kk (Eqs. 13-14 limit).
        D, F = B_kk, A_kk
    else:
        A_RR = H_ref.T @ mA @ H_ref
        A_Rk = H_ref.T @ mA @ H_loc
        B_RR = H_ref.T @ mB @ H_ref
        B_Rk = H_ref.T @ mB @ H_loc
        B_kR = B_Rk.T
        # P = A_kR A_RR^-1, reused across the D terms. A_RR is symmetric positive
        # definite (it is S or T over the reference leadfields), so it is formed
        # by a Cholesky solve rather than an explicit inverse: solving
        # A_RR X = A_Rk gives X = A_RR^-1 A_Rk, whose transpose is P.
        P = solve(A_RR, A_Rk, assume_a="pos").T
        F = A_kk - P @ A_Rk
        D = P @ B_RR @ P.T - P @ B_Rk - B_kR @ P.T + B_kk

    # Symmetrise to clean up round-off, then solve the symmetric-definite
    # generalized eigenproblem; eigenvalues come out in ascending order.
    D = 0.5 * (D + D.T)
    F = 0.5 * (F + F.T)
    eigvals, eigvecs = eigh(D, F)
    u = eigvecs[:, np.argmax(eigvals)]
    u = u / np.linalg.norm(u)
    # An eigenvector is defined only up to sign, and every localizer is even in
    # ``u``, so fix a deterministic convention: the largest-magnitude component
    # is positive. Without it the returned orientation (and hence the sign of
    # the reconstructed time course) flips arbitrarily between LAPACK builds.
    return u if u[np.argmax(np.abs(u))] >= 0 else -u


class MCMVScanResult(dict):
    r"""Result of a sequential MCMV source search (see :func:`scan_mcmv`).

    Behaves like a dictionary with the following keys.

    sources : list of int
        Grid indices (into ``forward``) of the discovered sources, in the order
        the greedy search found them. That order is not guaranteed to be
        monotone in source strength: each iteration maximises the *joint*
        localizer given the sources already fixed, so a source found later can
        carry the larger pseudo-Z.
    orientations : ndarray, shape (n_sources, 3) | None
        Unit orientation of each discovered source **in head coordinates** (the
        frame :func:`make_mcmv` takes), or ``None`` for a fixed-orientation
        forward.
    pseudo_z : ndarray, shape (n_sources,)
        Pseudo-Z :math:`\bar z_k = (\mathbf{w}_k^{\mathsf T}\mathbf{R}
        \mathbf{w}_k)/(\mathbf{w}_k^{\mathsf T}\mathbf{N}\mathbf{w}_k)` of the
        source added at iteration ``k``, evaluated on **that source's row of the
        joint ``k``-source MCMV filter**, not on a single-source (LCMV) filter
        at the same location. It is computed in the whitened space the scan works
        in, where :math:`\mathbf{N}=\mathbf{I}`, so the denominator is the squared
        Euclidean norm of the whitened filter; for a single sensor type that
        equals the sensor-space ratio above, and for a mixed array it equals it
        only up to the cross-type blocks of the whitened noise covariance (the
        same caveat that applies to ``weight_norm='unit-noise-gain'`` here and in
        :func:`mne.beamformer.make_lcmv`). Used to decide how many sources are
        real: iterate until it drops to a baseline (which, per Moiseev et al.
        2011, must be determined experimentally and is generally not one).
    filters : instance of MCMVBeamformer
        The jointly-optimal MCMV filters for the discovered source set.
    localizer : str
        The localizer that was used.
    maps : list of ndarray, shape (n_locations,)
        The localizer map over all grid locations at each iteration, with NaN at
        already-found or numerically invalid locations (for visualisation).
    """

    def __repr__(self):
        pz = np.array2string(np.asarray(self["pseudo_z"]), precision=2)
        return (
            f"<MCMVScanResult | {self['localizer'].upper()} | "
            f"{len(self['sources'])} source(s) {self['sources']} | pseudo-Z {pz}>"
        )


@verbose
def scan_mcmv(
    info,
    forward,
    data_cov,
    *,
    localizer="mai",
    n_sources=1,
    noise_cov=None,
    evoked_cov=None,
    reg=0.05,
    rank=None,
    verbose=None,
):
    r"""Discover correlated sources by iterative MCMV scanning.

    Implements the sequential source-search of :footcite:`Moiseev2011`: a
    single-source localizer scan finds the strongest source; that source is then
    held fixed and the multi-source localizer is re-scanned for the next source,
    and so on. Because each new source is added to a *joint* constraint, the
    source-cancellation that hides correlated activity from one-at-a-time LCMV
    scanning is progressively removed. At each step the source orientation is
    obtained in closed form (no orientation search; see
    :func:`optimal_orientation`), and all computation is performed in the
    noise-covariance-whitened space, so the search is valid for any sensor
    configuration.

    Parameters
    ----------
    info : instance of mne.Info
        Measurement info.
    forward : instance of mne.Forward
        The forward solution. A free-orientation forward is used to estimate
        each source's orientation; a fixed-orientation forward is also accepted
        (its baked-in orientations are used and no orientation is estimated).
    data_cov : instance of mne.Covariance
        The data covariance :math:`\mathbf{R}`.
    localizer : 'mai' | 'mpz' | 'mer' | 'rmer'
        Which Table-1 localizer to scan. ``'mai'`` and ``'mpz'`` are power-based;
        ``'mer'`` and ``'rmer'`` target evoked activity and need ``evoked_cov``.
    n_sources : int
        Number of sources to find (the beamformer order reached). Inspect
        ``pseudo_z`` in the result to judge how many are genuine.
    noise_cov : instance of mne.Covariance | None
        The noise covariance, used for whitening (see :func:`make_mcmv`). If
        ``None``, MNE's ad-hoc per-type model is used.
    evoked_cov : instance of mne.Covariance | None
        Covariance of the epoch-averaged field :math:`\bar{\mathbf{R}}`,
        required for ``localizer in {'mer', 'rmer'}``.
    reg : float
        Diagonal-loading regularisation applied to the (whitened) data
        covariance, as in :func:`make_mcmv`. Must be in ``[0, 1]``.
    rank : None | 'full' | 'info' | dict
        Rank handling for the whitener and the data-covariance inverse, in MNE's
        convention and applied exactly as in :func:`make_mcmv`. The default
        ``None`` auto-detects the rank per sensor type and drops the null space.
        This matters more here than for a single filter: the scan compares
        localizer values *across* the grid, and inverting the near-null
        directions of a rank-deficient covariance perturbs those values enough to
        change which locations win.
    %(verbose)s

    Returns
    -------
    result : instance of MCMVScanResult
        The discovered sources, their orientations and pseudo-Z, the localizer
        maps, and the jointly-optimal :class:`MCMVBeamformer`.

    Raises
    ------
    ValueError
        If ``reg`` is outside ``[0, 1]``, ``n_sources`` is below one, exceeds the
        number of grid locations, or exceeds the rank of the (whitened) data
        covariance.
    RuntimeError
        If no grid location could be evaluated at some iteration.

    Notes
    -----
    Already-found grid locations are excluded from subsequent scans. Locations
    that cannot be evaluated are skipped rather than raising; they appear as NaN
    in ``maps``. Such a location is one where a (numerically) collinear
    constraint makes the localizer singular, where the leadfield is non-finite,
    or where the leadfield is numerically silent in the whitened space, so that
    the array cannot see the location at all and any value computed for it would
    be a ratio of round-off.

    The sources are returned in the order the greedy search found them, which is
    not necessarily the order of decreasing strength (see
    :class:`MCMVScanResult`).

    References
    ----------
    .. footbibliography::
    """
    _validate_type(data_cov, Covariance, "data_cov")
    if noise_cov is not None:
        _validate_type(noise_cov, Covariance, "noise_cov")
    _check_localizer(localizer, evoked_cov)
    if n_sources < 1:
        raise ValueError(f"n_sources must be >= 1, got {n_sources}.")
    # Validate ``reg`` here rather than letting make_mcmv reject it at the very
    # end: by then the whole grid scan has already been paid for.
    if not (0.0 <= float(reg) <= 1.0):
        raise ValueError(f"reg must be in [0, 1], got {reg}.")
    fixed = is_fixed_orient(forward)

    # -- common channel space (shared helpers, so the scan, make_mcmv and the
    # filters returned below all select exactly the same channels) ---------- #
    common_ch, R = _align_channels(info, forward, data_cov)
    common_ch, R = _intersect_noise_cov(common_ch, R, noise_cov)
    _check_noise_cov_required(info, common_ch, noise_cov)

    n_loc = forward["nsource"]
    if n_sources > n_loc:
        raise ValueError(
            f"n_sources={n_sources} exceeds the number of grid locations ({n_loc})."
        )

    # -- whiten: in this space the noise covariance is the identity --------- #
    whitener_rank, pinv_rank = _split_rank(rank)
    whitener = _make_whitener(info, noise_cov, common_ch, whitener_rank)
    fwd_ch = list(forward["sol"]["row_names"])
    fwd_idx = [fwd_ch.index(ch) for ch in common_ch]
    G = np.asarray(forward["sol"]["data"], dtype=np.float64)[fwd_idx]
    G_w = whitener @ G  # whitened leadfield (n_white, n_cols)
    R_w = whitener @ R @ whitener.T
    # The congruence is symmetric in exact arithmetic; enforce it against
    # floating-point drift, which would otherwise trip the Hermitian check in
    # ``_reg_pinv`` (see make_mcmv).
    R_w = (R_w + R_w.T) / 2.0
    n_white = R_w.shape[0]

    # -- regularised inverse of the (whitened) data covariance -------------- #
    # This is the same inversion make_mcmv performs, with the same ``reg``
    # rescaling for our ``pca=True`` whitener, so the localizer maps are built
    # from exactly the covariance inverse that the returned filters use. The
    # rank truncation is not cosmetic here: on a covariance whose deficiency the
    # whitener cannot see (ICA/SSP-cleaned data whose projectors never reached
    # ``info['projs']``, too few samples, a mismatched noise_cov) a plain
    # inverse amplifies the near-null directions and the scan silently ranks the
    # wrong locations highest.
    reg_eff = float(reg) * n_white / len(common_ch)
    Rinv_w, _, rnk = _reg_pinv(R_w, reg=reg_eff, rank=pinv_rank)
    if rnk < n_white:
        warnings.warn(
            f"data_cov is rank-deficient (rank {rnk} < {n_white} whitened "
            "dimensions); a pseudo-inverse was used. Supply a shrinkage-"
            "regularised covariance (e.g. method='oas'/'shrunk' in "
            "mne.compute_covariance) or set ``reg`` > 0 for a stable inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_sources > rnk:
        raise ValueError(
            f"n_sources={n_sources} exceeds the data-covariance rank ({rnk}); "
            "fewer than n_sources independent spatial dimensions are available "
            "to satisfy the constraints."
        )

    Rbar_w = None
    if evoked_cov is not None:
        Rbar_w = whitener @ _cov_as_matrix(evoked_cov, common_ch) @ whitener.T

    # The Table-2 metric matrices depend only on the covariances, so they are
    # built once for the whole scan instead of at every grid point. In the
    # whitened space N is the identity, which removes both dense inversions:
    # G = N^-1 = I and T = R^-1 N R^-1 = R^-1 R^-1.
    # ``R_w``, ``N_w`` and ``Rbar_w`` are still handed to the localizers below so
    # that their argument checks see a consistent call, but with ``metrics``
    # supplied they are not used to rebuild anything.
    N_w = np.eye(n_white)  # whitened noise covariance
    metrics = {"S": Rinv_w, "G": N_w, "T": Rinv_w @ Rinv_w}
    if Rbar_w is not None:
        metrics["E"] = Rinv_w @ Rbar_w @ Rinv_w

    # -- locations the array cannot see are not candidates ------------------ #
    # A grid location whose whitened leadfield has collapsed to numerical
    # silence -- the centre of a spherical model, a radial source seen by MEG
    # only, a location whose sensors were all dropped as bad -- carries no
    # signal for any localizer to rank. Its Table-1 value is then a ratio of
    # round-off, which is as likely to come out large as small, and if such a
    # location wins, the source that is reported is noise with a name.
    # ``_MIN_WHITENED_GAIN`` is the same absolute threshold ``make_mcmv``
    # refuses a constraint at, so a location the scan would return is always one
    # a filter can be built for. A non-finite leadfield is deliberately left
    # out of this test: it is handled where it always has been, by the
    # per-location skip inside the loop.
    gains = np.linalg.norm(G_w, axis=0)
    if not fixed:  # three columns per location, silent only if all three are
        gains = np.linalg.norm(gains.reshape(-1, 3), axis=1)
    silent = gains <= _MIN_WHITENED_GAIN

    # -- sequential search -------------------------------------------------- #
    sources, orientations, pseudo_z, maps = [], [], [], []
    H_ref = np.empty((n_white, 0))  # whitened leadfields of found sources
    for k in range(n_sources):
        vals = np.full(n_loc, -np.inf)
        oris = [None] * n_loc
        n_skipped = 0
        for i in range(n_loc):
            if i in sources:
                continue  # never re-select a found location
            if silent[i]:
                n_skipped += 1
                continue  # the array cannot see this location
            if fixed:
                h, u = G_w[:, i], None
            else:
                H_loc = G_w[:, 3 * i : 3 * i + 3]
                try:
                    u = optimal_orientation(
                        localizer,
                        H_ref,
                        H_loc,
                        R_w,
                        N_w,
                        evoked_cov=Rbar_w,
                        metrics=metrics,
                    )
                except _SINGULAR_ERRORS:
                    n_skipped += 1
                    continue  # skip a numerically singular location
                h = H_loc @ u
            H_full = np.column_stack([H_ref, h])
            try:
                vals[i] = localizer_value(
                    localizer, H_full, R_w, N_w, evoked_cov=Rbar_w, metrics=metrics
                )
            except _SINGULAR_ERRORS:
                n_skipped += 1
                continue
            oris[i] = u

        # ``vals`` may hold -inf (skipped/already-found) and, if a localizer
        # evaluated to a non-finite number without raising, NaN or +inf; mask
        # everything non-finite out before taking the maximum.
        finite = np.isfinite(vals)
        if not finite.any():
            raise RuntimeError(
                "No valid source location found; the data covariance or forward "
                "may be degenerate, or too many sources were requested."
            )
        logger.debug(
            f"    Iteration {k + 1}/{n_sources}: {int(finite.sum())} of {n_loc} "
            f"locations evaluated ({n_skipped} skipped as singular or "
            "invisible to the array)."
        )
        best = int(np.nanargmax(np.where(finite, vals, np.nan)))
        sources.append(best)
        orientations.append(oris[best])
        maps.append(np.where(np.isfinite(vals), vals, np.nan))

        # Extend the reference set and record the new source's pseudo-Z.
        h_best = G_w[:, best] if fixed else G_w[:, 3 * best : 3 * best + 3] @ oris[best]
        H_ref = np.column_stack([H_ref, h_best])
        w_new = _compute_mcmv_weights(H_ref, Rinv_w)[-1]  # newest filter row
        pseudo_z.append(float((w_new @ R_w @ w_new) / (w_new @ w_new)))

    # -- jointly-optimal filters for the discovered set (sensor space) ------ #
    if fixed:
        ori_out = None
        filters = make_mcmv(
            info, forward, data_cov, sources, noise_cov=noise_cov, reg=reg, rank=rank
        )
    else:
        ori_out = np.array(orientations)
        # The orientations were estimated against the forward's gain columns,
        # which for a ``surf_ori=True`` forward are expressed in each source's
        # local surface frame. Rotate them back into head coordinates so that
        # the returned array means the same physical dipole whichever
        # representation of a forward was scanned. Head coordinates are the
        # frame ``forward['source_nn']`` uses and the one make_mcmv documents
        # for its ``orientations`` argument.
        if forward.get("surf_ori", False):
            nn = np.asarray(forward["source_nn"], dtype=np.float64)
            ori_out = np.array(
                [
                    nn[3 * s : 3 * s + 3].T @ u
                    for s, u in zip(sources, ori_out, strict=True)
                ]
            )
        filters = make_mcmv(
            info,
            forward,
            data_cov,
            sources,
            orientations=ori_out,
            noise_cov=noise_cov,
            reg=reg,
            rank=rank,
        )

    logger.info(f"    MCMV scan ({localizer}) found {n_sources} source(s): {sources}.")
    return MCMVScanResult(
        sources=sources,
        orientations=ori_out,
        pseudo_z=np.array(pseudo_z),
        filters=filters,
        localizer=localizer,
        maps=maps,
    )
