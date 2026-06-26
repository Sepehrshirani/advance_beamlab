r"""ReciPSIICOS data-covariance modification for correlated-source beamforming.

This module implements the ReciPSIICOS and Whitened ReciPSIICOS sensor-space
covariance modification of Kuznetsova, Nurislamova and Ossadtchi (2021)
:footcite:`KuznetsovaEtAl2021`.

The starting point is the generative model of the sensor covariance. Writing
the (vectorised) sensor covariance as a vector in the :math:`M^2`-dimensional
space of matrices, it splits into a sum of *auto-products* of the source
topographies (carrying the source powers) and *cross-products* of pairs of
topographies (carrying the source couplings) (Eq. 8). The cross-products are
exactly what an LCMV beamformer exploits to cancel correlated sources. The
ReciPSIICOS procedure builds, from the forward model alone, a projector that
suppresses the cross-product subspace while sparing the auto-product subspace,
and applies it to the data covariance. The result approximates the covariance
that would have been measured had the same sources been uncorrelated, so a
standard LCMV beamformer built on it no longer suffers signal cancellation.

Because the method only modifies the data covariance, it does not define a new
spatial filter: the modified covariance is handed unchanged to
:func:`mne.beamformer.make_lcmv`. Two public functions are provided --
:func:`make_recipsiicos_cov`, which returns the modified
:class:`mne.Covariance`, and :func:`make_recipsiicos_lcmv`, a convenience
wrapper that performs the modification and the LCMV call in one step -- plus
:func:`recipsiicos_rank_curve` to guide the choice of the single free
parameter, the projection rank.

Notes
-----
Equation numbers in the comments refer to Kuznetsova et al. (2021).

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import warnings

import numpy as np
from mne import Covariance
from mne.forward import is_fixed_orient
from mne.utils import _check_option, _validate_type, logger, verbose

# Reuse the channel-alignment helper from the MCMV module so that both
# algorithms read the forward solution and the covariance in exactly the same
# (common, ordered) channel space.
from ._mcmv import _align_channels

_ALLOWED_METHOD = ("recipsiicos", "whitened")

# Fraction of the eigenvalue energy that the spectral-flip step is allowed to
# carry in the negative eigenvalues before a warning is raised (the authors
# recommend the modification stay below ~20%; their Eq. 24).
_NEG_ENERGY_LIMIT = 0.20


# --------------------------------------------------------------------------- #
# vec / unvec (column-stacking convention used throughout the paper)
# --------------------------------------------------------------------------- #
def _vec(matrix):
    """Column-stack a matrix into a vector (the vec() operator of Eq. 8)."""
    # order="F" stacks columns, matching the column-major vec() convention.
    return matrix.ravel(order="F")


def _unvec(vector, n):
    """Inverse of :func:`_vec` for an ``n x n`` matrix."""
    return vector.reshape(n, n, order="F")


# --------------------------------------------------------------------------- #
# Topographies of the forward model (handling source orientation)
# --------------------------------------------------------------------------- #
def _local_topographies(forward, common_ch):
    """Return one topography block per source location.

    For a fixed-orientation forward each location has a single topography, so
    the returned array has a trailing dimension of 1. For a free-orientation
    forward each location is reduced to the two dominant (tangential)
    topographies, obtained from the left singular vectors of the local
    ``(n_channels, 3)`` forward block scaled by their singular values -- the
    reduction used by Kuznetsova et al. (2021), Section 2.5, which discards the
    poorly observed radial direction.

    Parameters
    ----------
    forward : instance of mne.Forward
        The forward solution.
    common_ch : list of str
        Channels (in order) shared with the data covariance.

    Returns
    -------
    topos : ndarray, shape (n_locations, n_channels, n_ori)
        ``n_ori`` is 1 for a fixed-orientation forward and 2 otherwise.
    """
    fwd_ch = list(forward["sol"]["row_names"])
    idx = [fwd_ch.index(ch) for ch in common_ch]
    gain = np.asarray(forward["sol"]["data"], dtype=np.float64)[idx]
    n_channels = gain.shape[0]

    if is_fixed_orient(forward):
        n_loc = gain.shape[1]
        return gain.T.reshape(n_loc, n_channels, 1)

    # Free orientation: 3 leadfield columns per location.
    n_loc = gain.shape[1] // 3
    topos = np.empty((n_loc, n_channels, 2), dtype=np.float64)
    for i in range(n_loc):
        local = gain[:, 3 * i : 3 * i + 3]  # (n_channels, 3)
        # SVD: the first two left singular vectors, scaled by their singular
        # values, are the two tangential-plane topographies of this location.
        u, s, _ = np.linalg.svd(local, full_matrices=False)
        topos[i, :, 0] = u[:, 0] * s[0]
        topos[i, :, 1] = u[:, 1] * s[1]
    return topos


# --------------------------------------------------------------------------- #
# Power and correlation subspace columns (Eqs. 9, 13, 14, 22, 23)
# --------------------------------------------------------------------------- #
def _power_columns(topos):
    """Vectorised auto-products spanning the source-power subspace (G_pwr).

    Fixed orientation: one column ``vec(g g^T)`` per location (Eq. 9/14).
    Free orientation: the three columns of Eq. (22) per location, so that the
    auto-product of an arbitrarily oriented dipole is spanned.
    """
    n_loc, n_channels, n_ori = topos.shape
    cols = []
    for i in range(n_loc):
        a = topos[i, :, 0]
        if n_ori == 1:
            cols.append(_vec(np.outer(a, a)))
        else:
            b = topos[i, :, 1]
            cols.append(_vec(np.outer(a, a)))
            cols.append(_vec(np.outer(a, b) + np.outer(b, a)))
            cols.append(_vec(np.outer(b, b)))
    return np.column_stack(cols)


def _correlation_blocks(topos, block_pairs=20000):
    """Yield blocks of vectorised cross-products (columns of G_cor).

    Fixed orientation: one column ``vec(g_i g_j^T + g_j g_i^T)`` per ordered
    pair ``i < j`` (Eq. 13). Free orientation: the four columns of Eq. (23) per
    pair. The columns are produced in blocks so that the (potentially very
    large) matrix G_cor is never held in memory all at once -- the caller only
    needs the running Gram matrix ``G_cor G_cor^T``.
    """
    n_loc, n_channels, n_ori = topos.shape
    buffer = []
    for i in range(n_loc):
        for j in range(i + 1, n_loc):
            ai, aj = topos[i, :, 0], topos[j, :, 0]
            if n_ori == 1:
                buffer.append(_vec(np.outer(ai, aj) + np.outer(aj, ai)))
            else:
                bi, bj = topos[i, :, 1], topos[j, :, 1]
                buffer.append(_vec(np.outer(ai, aj) + np.outer(aj, ai)))
                buffer.append(_vec(np.outer(ai, bj) + np.outer(bj, ai)))
                buffer.append(_vec(np.outer(bi, aj) + np.outer(aj, bi)))
                buffer.append(_vec(np.outer(bi, bj) + np.outer(bj, bi)))
            if len(buffer) >= block_pairs:
                yield np.column_stack(buffer)
                buffer = []
    if buffer:
        yield np.column_stack(buffer)


def _correlation_gram(topos):
    """Accumulate the Gram matrix C_cor = G_cor G_cor^T without storing G_cor."""
    msq = topos.shape[1] ** 2
    gram = np.zeros((msq, msq), dtype=np.float64)
    for block in _correlation_blocks(topos):
        gram += block @ block.T
    return gram


# --------------------------------------------------------------------------- #
# Projector construction (Eqs. 10, 15-17)
# --------------------------------------------------------------------------- #
def _power_projector(g_pwr, rank):
    """ReciPSIICOS projector onto the K-dim principal power subspace (Eq. 10).

    Returns the orthogonal projector ``P = U_K U_K^T`` and the full singular
    value spectrum of G_pwr (used by the rank-selection helper).
    """
    u, s, _ = np.linalg.svd(g_pwr, full_matrices=False)
    rank = int(min(rank, u.shape[1]))
    u_k = u[:, :rank]
    return u_k @ u_k.T, s


def _whitening_pair(c_pwr, reg, rtol=1e-6):
    """Symmetric whitener for the power subspace and its inverse (Eq. 15).

    ``C_pwr`` is rank-limited (its rank cannot exceed the number of sources),
    so the inverse square root is taken only on its range: eigenvalues at or
    below ``rtol`` times the largest are treated as the null space and dropped,
    which prevents the whitener from amplifying directions that carry no source
    power. The retained eigenvalues are stabilised with a small ridge
    ``reg * mean(eigenvalue)``. Returns ``(W, W_inv)`` with
    ``W = E Lambda^{-1/2} E^T`` and ``W_inv = E Lambda^{1/2} E^T`` over the
    range of ``C_pwr``.
    """
    eigvals, eigvecs = np.linalg.eigh(c_pwr)
    eigvals = np.clip(eigvals, 0.0, None)
    keep = eigvals > rtol * eigvals.max()
    e = eigvecs[:, keep]
    lam = eigvals[keep]
    lam = lam + reg * lam.mean()
    w = e @ np.diag(lam ** -0.5) @ e.T
    w_inv = e @ np.diag(lam ** 0.5) @ e.T
    return w, w_inv


def _whitened_projector(g_pwr, c_cor, rank, reg):
    """Whitened ReciPSIICOS projector (Eqs. 15-17).

    Whitens with respect to the power subspace, removes the top-K directions of
    the correlation subspace in that whitened space, then unwhitens.
    """
    c_pwr = g_pwr @ g_pwr.T
    w, w_inv = _whitening_pair(c_pwr, reg)
    # Correlation Gram in the whitened space, and its principal eigenvectors.
    c_cor_w = w @ c_cor @ w.T
    eigvals, eigvecs = np.linalg.eigh(c_cor_w)
    # eigh returns ascending eigenvalues; take the K largest.
    rank = int(min(rank, eigvecs.shape[1]))
    e_k = eigvecs[:, eigvecs.shape[1] - rank :]
    msq = g_pwr.shape[0]
    projector = w_inv @ (np.eye(msq) - e_k @ e_k.T) @ w
    return projector, eigvals[::-1]


# --------------------------------------------------------------------------- #
# Applying the projector and restoring positive-definiteness (Eqs. 11, 12, 24)
# --------------------------------------------------------------------------- #
def _apply_projector(projector, cov):
    """Project a covariance through the M^2-space projector (Eq. 11)."""
    n = cov.shape[0]
    modified = _unvec(projector @ _vec(cov), n)
    # The projection returns a symmetric matrix up to numerical error.
    return 0.5 * (modified + modified.T)


def _spectral_flip(cov):
    """Restore positive-definiteness by flipping negative eigenvalues (Eq. 12).

    Returns the corrected covariance and the fraction of the total eigenvalue
    energy that was carried by the negative eigenvalues (Eq. 24), used to warn
    the user when the heuristic is being stretched too far.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    abs_eigvals = np.abs(eigvals)
    total = abs_eigvals.sum()
    neg_energy = abs_eigvals[eigvals < 0].sum() / total if total > 0 else 0.0
    corrected = eigvecs @ np.diag(abs_eigvals) @ eigvecs.T
    corrected = 0.5 * (corrected + corrected.T)
    return corrected, float(neg_energy)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@verbose
def make_recipsiicos_cov(
    data_cov,
    forward,
    *,
    rank,
    method="recipsiicos",
    info=None,
    reg=0.05,
    verbose=None,
):
    """Modify a data covariance to suppress correlated-source contributions.

    Implements the ReciPSIICOS (``method='recipsiicos'``) and Whitened
    ReciPSIICOS (``method='whitened'``) covariance modification of
    Kuznetsova et al. (2021). The returned covariance is intended to be passed
    directly to :func:`mne.beamformer.make_lcmv`.

    Parameters
    ----------
    data_cov : instance of mne.Covariance
        The sensor-space data covariance to modify.
    forward : instance of mne.Forward
        The forward solution. The projector is built from this alone, so it is
        independent of the data and can be reused across datasets that share
        the forward model.
    rank : int
        The projection rank ``K`` -- the dimension of the principal subspace
        used to build the projector. This is the only free parameter of the
        method; :func:`recipsiicos_rank_curve` helps choose it.
    method : 'recipsiicos' | 'whitened'
        Which projector to build. ``'recipsiicos'`` projects onto the principal
        power subspace (Eq. 10). ``'whitened'`` projects away from the principal
        correlation subspace in the power-whitened space (Eqs. 15-17), which
        spares the source-power terms more effectively.
    info : instance of mne.Info | None
        Measurement info used to determine the common channels. If ``None``,
        ``data_cov`` and ``forward`` are intersected directly.
    reg : float
        Ridge added to the power-subspace eigenvalues before whitening (used by
        ``method='whitened'`` only) as a fraction of their mean.
    %(verbose)s

    Returns
    -------
    cov : instance of mne.Covariance
        The modified, positive-definite data covariance over the common
        channels.

    Notes
    -----
    The modification is followed by a spectral-flip step that restores positive
    definiteness (Eq. 12). A warning is raised if the negative eigenvalues
    carried more than 20%% of the eigenvalue energy (Eq. 24), the threshold above
    which the authors caution against trusting the result.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(data_cov, Covariance, "data_cov")
    _check_option("method", method, _ALLOWED_METHOD)
    if int(rank) < 1:
        raise ValueError(f"rank must be a positive integer, got {rank}.")

    # Resolve the common channel space and the data covariance over it. When no
    # info is supplied, intersect the covariance and forward channel sets.
    if info is None:
        fwd_ch = set(forward["sol"]["row_names"])
        common_ch = [ch for ch in data_cov.ch_names if ch in fwd_ch]
        if len(common_ch) == 0:
            raise ValueError(
                "No common channels between data_cov and forward."
            )
        cov_idx = [list(data_cov.ch_names).index(ch) for ch in common_ch]
        cov = np.asarray(data_cov.data, dtype=np.float64)[np.ix_(cov_idx, cov_idx)]
    else:
        common_ch, cov = _align_channels(info, forward, data_cov)

    n_channels = len(common_ch)
    msq = n_channels * n_channels
    if int(rank) > msq:
        raise ValueError(
            f"rank={rank} exceeds the dimension M^2={msq} of the covariance "
            "vector space."
        )

    # Build the topographies and the projector.
    topos = _local_topographies(forward, common_ch)
    g_pwr = _power_columns(topos)
    if method == "recipsiicos":
        projector, _ = _power_projector(g_pwr, rank)
    else:
        c_cor = _correlation_gram(topos)
        projector, _ = _whitened_projector(g_pwr, c_cor, rank, reg)

    # Project the covariance and restore positive-definiteness.
    modified = _apply_projector(projector, cov)
    modified, neg_energy = _spectral_flip(modified)
    logger.info(
        f"    ReciPSIICOS ({method}): modified covariance over {n_channels} "
        f"channels at rank {rank}; negative-eigenvalue energy "
        f"{neg_energy * 100:.1f}%."
    )
    if neg_energy > _NEG_ENERGY_LIMIT:
        warnings.warn(
            f"The spectral-flip step carried {neg_energy * 100:.1f}% of the "
            f"eigenvalue energy in negative eigenvalues (> "
            f"{_NEG_ENERGY_LIMIT * 100:.0f}%); the modified covariance may be "
            "unreliable. Consider a different projection rank.",
            RuntimeWarning,
            stacklevel=2,
        )

    return Covariance(
        modified,
        common_ch,
        list(data_cov.get("bads", [])),
        list(data_cov.get("projs", [])),
        nfree=int(data_cov.get("nfree", 1)),
    )


@verbose
def make_recipsiicos_lcmv(
    info,
    forward,
    data_cov,
    *,
    rank,
    method="recipsiicos",
    noise_cov=None,
    reg=0.05,
    label=None,
    pick_ori=None,
    weight_norm="unit-noise-gain-invariant",
    rank_lcmv="info",
    reduce_rank=False,
    depth=None,
    inversion="matrix",
    verbose=None,
):
    """Build an LCMV beamformer on a ReciPSIICOS-modified data covariance.

    Convenience wrapper that modifies ``data_cov`` with
    :func:`make_recipsiicos_cov` and then calls
    :func:`mne.beamformer.make_lcmv` with the modified covariance. All LCMV
    options are passed straight through, so the beamformer is identical to a
    standard MNE LCMV except for the correlated-source-robust covariance.

    Parameters
    ----------
    info : instance of mne.Info
        The measurement info.
    forward : instance of mne.Forward
        The forward solution.
    data_cov : instance of mne.Covariance
        The (unmodified) data covariance.
    rank : int
        The ReciPSIICOS projection rank (see :func:`make_recipsiicos_cov`).
    method : 'recipsiicos' | 'whitened'
        Which ReciPSIICOS projector to use.
    noise_cov : instance of mne.Covariance | None
        Noise covariance, passed to ``make_lcmv``.
    reg : float
        Regularisation. Used both for the ReciPSIICOS whitening ridge and as the
        ``reg`` argument of ``make_lcmv``.
    label : instance of mne.Label | None
        Restrict the source space, passed to ``make_lcmv``.
    pick_ori : None | str
        Source-orientation handling, passed to ``make_lcmv``.
    weight_norm : str | None
        Weight normalisation, passed to ``make_lcmv``.
    rank_lcmv : None | int | 'full' | 'info'
        The ``rank`` argument of ``make_lcmv`` (named ``rank_lcmv`` here to
        avoid clashing with the ReciPSIICOS projection rank).
    reduce_rank : bool
        Passed to ``make_lcmv``.
    depth : None | float | dict
        Passed to ``make_lcmv``.
    inversion : 'matrix' | 'single'
        Passed to ``make_lcmv``.
    %(verbose)s

    Returns
    -------
    filters : instance of mne.beamformer.Beamformer
        The LCMV spatial filter built on the modified covariance.

    References
    ----------
    .. footbibliography::
    """
    # Imported here to keep the module import light and to make the dependency
    # on MNE's LCMV explicit at the point of use.
    from mne.beamformer import make_lcmv

    modified_cov = make_recipsiicos_cov(
        data_cov,
        forward,
        rank=rank,
        method=method,
        info=info,
        reg=reg,
    )
    return make_lcmv(
        info,
        forward,
        modified_cov,
        reg=reg,
        noise_cov=noise_cov,
        label=label,
        pick_ori=pick_ori,
        rank=rank_lcmv,
        weight_norm=weight_norm,
        reduce_rank=reduce_rank,
        depth=depth,
        inversion=inversion,
    )


def recipsiicos_rank_curve(forward, *, method="recipsiicos", info=None, data_cov=None):
    """Power-vs-correlation retention as a function of projection rank.

    Computes, for every projection rank ``k``, the fraction of the power- and
    correlation-subspace energy that survives the projection (Eqs. 20-21). The
    useful rank is the largest ``k`` for which the correlation subspace is
    depleted faster than the power subspace; plotting ``p_cor`` against
    ``p_pwr`` makes this trade-off visible (the authors pick the 45-degree
    point of the curve).

    Parameters
    ----------
    forward : instance of mne.Forward
        The forward solution.
    method : 'recipsiicos' | 'whitened'
        Which projector family to characterise.
    info : instance of mne.Info | None
        Used, with ``data_cov``, to determine the common channels. If ``None``
        all forward channels are used.
    data_cov : instance of mne.Covariance | None
        Only used together with ``info`` to intersect channels.

    Returns
    -------
    ranks : ndarray, shape (n_ranks,)
        The projection ranks evaluated (1 .. M^2).
    p_pwr : ndarray, shape (n_ranks,)
        Fraction of power-subspace energy retained at each rank (Eq. 20).
    p_cor : ndarray, shape (n_ranks,)
        Fraction of correlation-subspace energy retained at each rank (Eq. 21).
    """
    _check_option("method", method, _ALLOWED_METHOD)
    if info is not None and data_cov is not None:
        common_ch, _ = _align_channels(info, forward, data_cov)
    else:
        common_ch = list(forward["sol"]["row_names"])

    topos = _local_topographies(forward, common_ch)
    g_pwr = _power_columns(topos)
    c_pwr = g_pwr @ g_pwr.T
    c_cor = _correlation_gram(topos)
    tr_pwr, tr_cor = np.trace(c_pwr), np.trace(c_cor)

    msq = g_pwr.shape[0]
    ranks = np.arange(1, msq + 1)
    p_pwr = np.empty(ranks.size)
    p_cor = np.empty(ranks.size)
    for idx, k in enumerate(ranks):
        if method == "recipsiicos":
            projector, _ = _power_projector(g_pwr, k)
        else:
            projector, _ = _whitened_projector(g_pwr, c_cor, k, reg=0.05)
        p_pwr[idx] = np.trace(projector @ c_pwr @ projector.T) / tr_pwr
        p_cor[idx] = np.trace(projector @ c_cor @ projector.T) / tr_cor
    return ranks, p_pwr, p_cor
