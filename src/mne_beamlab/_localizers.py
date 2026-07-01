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

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import numpy as np
from scipy.linalg import eigh

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
