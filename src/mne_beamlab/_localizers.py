r"""MCMV source localizers and data-driven orientation.

This module implements the four MCMV-based scanning localizers of Moiseev et
al. (2011) -- multi-source activity index (MAI), multi-source pseudo-Z (MPZ),
multi-source event-related (MER) and its reduced form (rMER) -- together with
the closed-form optimal source orientation derived in the same paper. These are
the ingredients of the iterative source-search that turns MCMV from a filter
for *known* sources into a tool that *discovers* correlated sources.

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

# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import numpy as np
from mne.forward import is_fixed_orient
from mne.utils import logger
from scipy.linalg import eigh

from ._mcmv import (
    _align_channels,
    _compute_mcmv_weights,
    _make_whitener,
    make_mcmv,
)

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


def _metric_matrices(R, N, evoked_cov=None):
    r"""Return the per-localizer metric matrices that sit between leadfields.

    Returns the matrices :math:`\mathbf{M}` such that the Table-2 quantities are
    :math:`\mathbf{S}=\mathbf{H}^{\mathsf T}\mathbf{M}_S\mathbf{H}` and so on:

    - ``S`` uses :math:`\mathbf{R}^{-1}`
    - ``G`` uses :math:`\mathbf{N}^{-1}`
    - ``T`` uses :math:`\mathbf{R}^{-1}\mathbf{N}\mathbf{R}^{-1}`
    - ``E`` uses :math:`\mathbf{R}^{-1}\bar{\mathbf{R}}\mathbf{R}^{-1}`
      (only when ``evoked_cov`` is supplied)
    """
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


def localizer_value(name, H, R, N, *, evoked_cov=None):
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
    metrics = _metric_matrices(R, N, evoked_cov)
    A = _result_matrix(H, metrics[a_key])
    B = _result_matrix(H, metrics[b_key])
    # Tr(B A^-1) = Tr(A^-1 B) = trace of the solution of A X = B, computed via a
    # solve rather than an explicit inverse for numerical stability.
    value = np.trace(np.linalg.solve(A, B))
    if subtract_n:
        value -= H.shape[1]
    return float(value)


def optimal_orientation(name, H_ref, H_loc, R, N, *, evoked_cov=None):
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
            f"H_loc must have 3 columns (x, y, z leadfields), got "
            f"{H_loc.shape[1]}."
        )
    a_key, b_key, _ = _LOCALIZERS[name]
    metrics = _metric_matrices(R, N, evoked_cov)
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
        A_kR = A_Rk.T
        B_RR = H_ref.T @ mB @ H_ref
        B_Rk = H_ref.T @ mB @ H_loc
        B_kR = B_Rk.T
        # P = A_kR A_RR^-1 (solve for stability), reused across the D terms.
        P = A_kR @ np.linalg.inv(A_RR)
        F = A_kk - P @ A_Rk
        D = P @ B_RR @ P.T - P @ B_Rk - B_kR @ P.T + B_kk

    # Symmetrise to clean up round-off, then solve the symmetric-definite
    # generalized eigenproblem; eigenvalues come out in ascending order.
    D = 0.5 * (D + D.T)
    F = 0.5 * (F + F.T)
    eigvals, eigvecs = eigh(D, F)
    u = eigvecs[:, np.argmax(eigvals)]
    return u / np.linalg.norm(u)


def _cov_as_matrix(cov, ch_names):
    """Dense covariance matrix over ``ch_names`` (handles diagonal covs)."""
    cov_ch = list(cov.ch_names)
    idx = [cov_ch.index(ch) for ch in ch_names]
    data = np.asarray(cov.data, dtype=np.float64)
    if data.ndim == 1:  # a diagonal covariance stores only its diagonal
        return np.diag(data[idx])
    return data[np.ix_(idx, idx)]


class MCMVScanResult(dict):
    r"""Result of a sequential MCMV source search (see :func:`scan_mcmv`).

    Behaves like a dictionary with the following keys.

    sources : list of int
        Grid indices (into ``forward``) of the discovered sources, ordered as
        found (strongest first).
    orientations : ndarray, shape (n_sources, 3) | None
        Unit orientation of each discovered source; ``None`` for a
        fixed-orientation forward.
    pseudo_z : ndarray, shape (n_sources,)
        Single-source pseudo-Z :math:`\\bar z_k = (\\mathbf{w}_k^{\\mathsf T}
        \\mathbf{R}\\mathbf{w}_k)/(\\mathbf{w}_k^{\\mathsf T}\\mathbf{N}
        \\mathbf{w}_k)` of each source at the iteration it was found. Used to
        decide how many sources are real: iterate until it drops to a baseline
        (which, per Moiseev et al. 2011, must be determined experimentally and
        is generally not one).
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
        covariance, as in :func:`make_mcmv`.
    rank : int | None | 'full'
        Rank handling for the whitener and the covariance inverse, as in
        :func:`make_mcmv`. The default ``None`` auto-detects the rank and drops
        the null space.
    %(verbose)s

    Returns
    -------
    result : instance of MCMVScanResult
        The discovered sources, their orientations and pseudo-Z, the localizer
        maps, and the jointly-optimal :class:`MCMVBeamformer`.

    Notes
    -----
    Already-found grid locations are excluded from subsequent scans, and
    locations whose (numerically) collinear constraint makes the localizer
    singular are skipped rather than raising.

    References
    ----------
    .. footbibliography::
    """
    _check_localizer(localizer, evoked_cov)
    if n_sources < 1:
        raise ValueError(f"n_sources must be >= 1, got {n_sources}.")
    fixed = is_fixed_orient(forward)

    # -- common channel space (also intersect noise_cov, as make_mcmv does) - #
    common_ch, R = _align_channels(info, forward, data_cov)
    if noise_cov is not None:
        ncov_set = set(noise_cov.ch_names)
        keep = [i for i, ch in enumerate(common_ch) if ch in ncov_set]
        if len(keep) == 0:
            raise ValueError(
                "noise_cov shares no channels with info/forward/data_cov."
            )
        if len(keep) < len(common_ch):
            common_ch = [common_ch[i] for i in keep]
            R = R[np.ix_(keep, keep)]

    n_loc = forward["nsource"]
    if n_sources > n_loc:
        raise ValueError(
            f"n_sources={n_sources} exceeds the number of grid locations "
            f"({n_loc})."
        )

    # -- whiten: in this space the noise covariance is the identity --------- #
    whitener = _make_whitener(info, noise_cov, common_ch, rank)
    fwd_ch = list(forward["sol"]["row_names"])
    fwd_idx = [fwd_ch.index(ch) for ch in common_ch]
    G = np.asarray(forward["sol"]["data"], dtype=np.float64)[fwd_idx]
    G_w = whitener @ G  # whitened leadfield (n_white, n_cols)
    R_w = whitener @ R @ whitener.T
    n_white = R_w.shape[0]

    # Diagonal loading matching make_mcmv (a fraction of the mean eigenvalue).
    loading = reg * np.trace(R_w) / n_white
    R_w_reg = R_w + loading * np.eye(n_white)
    N_w = np.eye(n_white)  # whitened noise covariance
    Rinv_w = np.linalg.inv(R_w_reg)

    Rbar_w = None
    if evoked_cov is not None:
        Rbar_w = whitener @ _cov_as_matrix(evoked_cov, common_ch) @ whitener.T

    # -- sequential search -------------------------------------------------- #
    sources, orientations, pseudo_z, maps = [], [], [], []
    H_ref = np.empty((n_white, 0))  # whitened leadfields of found sources
    for _k in range(n_sources):
        vals = np.full(n_loc, -np.inf)
        oris = [None] * n_loc
        for i in range(n_loc):
            if i in sources:
                continue  # never re-select a found location
            if fixed:
                h, u = G_w[:, i], None
            else:
                H_loc = G_w[:, 3 * i : 3 * i + 3]
                try:
                    u = optimal_orientation(
                        localizer, H_ref, H_loc, R_w_reg, N_w, evoked_cov=Rbar_w
                    )
                except np.linalg.LinAlgError:
                    continue  # skip a numerically singular location
                h = H_loc @ u
            H_full = np.column_stack([H_ref, h])
            try:
                vals[i] = localizer_value(
                    localizer, H_full, R_w_reg, N_w, evoked_cov=Rbar_w
                )
            except np.linalg.LinAlgError:
                continue
            oris[i] = u

        if not np.any(np.isfinite(vals)):
            raise RuntimeError(
                "No valid source location found; the data covariance or forward "
                "may be degenerate, or too many sources were requested."
            )
        best = int(np.argmax(vals))
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
        filters = make_mcmv(
            info, forward, data_cov, sources, orientations=ori_out,
            noise_cov=noise_cov, reg=reg, rank=rank,
        )

    logger.info(
        f"    MCMV scan ({localizer}) found {n_sources} source(s): {sources}."
    )
    return MCMVScanResult(
        sources=sources,
        orientations=ori_out,
        pseudo_z=np.array(pseudo_z),
        filters=filters,
        localizer=localizer,
        maps=maps,
    )
