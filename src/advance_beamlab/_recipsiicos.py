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

Working space and whitening
---------------------------
Two steps are taken *around* the projector, both of which are needed to make the
method device-agnostic. The first is ours; the second follows the paper:

* **Noise whitening.** The projector is built from and applied in the space
  whitened by the noise covariance (via :func:`mne.cov.compute_whitener`, the
  same routine :func:`mne.beamformer.make_lcmv` uses, with the rank resolved per
  sensor type). Kuznetsova et al. (2021, Section 2.6) discuss whitening only to
  note that it changes the forward operator and would therefore require
  recomputing the ReciPSIICOS projections; they did not apply it, and worked in
  raw sensor space throughout. Whitening here is an addition of ours, not a step
  the paper prescribes: it is what makes the method valid for arrays that mix
  magnetometers (T) and gradiometers (T/m), whose covariance is otherwise
  dominated by the larger-unit sensor type.

* **Virtual sensors.** Following the paper's preprocessing, the whitened
  leadfield is reduced by a truncated SVD to the principal directions capturing
  a chosen fraction of its variance (``pct_var``). At the default 0.99 a
  whole-head array reduces to roughly 30-80 virtual sensors (29 magnetometers,
  43 gradiometers and 76 for the two combined on the 306-channel ``sample``
  array). Everything -- the :math:`M^2`-space projector, the data covariance and
  the beamformer -- then lives in this small working space, which is what makes
  the :math:`M^2 \times M^2` correlation Gram tractable (a 306-channel array
  would otherwise need a 93k x 93k matrix).

Because whitening and the virtual-sensor reduction both change the space the
projector lives in, the cleaned covariance cannot simply be handed back to
:func:`mne.beamformer.make_lcmv` (which would whiten a second time and solve
against the wrong leadfield). The LCMV problem is therefore solved *in the
working space* by reusing MNE's own filter computation
(:func:`mne.beamformer._compute_beamformer._compute_beamformer`, which performs
the orientation selection, weight normalisation and rank handling), after which
the reduction operator is folded into the returned
:class:`~mne.beamformer.Beamformer` as its whitener so that
:func:`mne.beamformer.apply_lcmv` applies it to the sensor data unchanged.

Three public functions are provided: :func:`make_recipsiicos_lcmv`, which builds
the beamformer end to end; :func:`make_recipsiicos_cov`, which returns the
modified sensor-space :class:`mne.Covariance` (for inspection and interop); and
:func:`recipsiicos_rank_curve`, which characterises -- and can automatically
select -- the single free parameter, the projection rank.

Notes
-----
Equation numbers in the comments refer to Kuznetsova et al. (2021).

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import warnings

import numpy as np
from mne import Covariance, pick_channels_cov
from mne.beamformer import Beamformer
from mne.beamformer._compute_beamformer import _check_src_normal, _compute_beamformer
from mne.forward import is_fixed_orient
from mne.forward.forward import _subject_from_forward
from mne.source_estimate import _get_src_type
from mne.source_space._source_space import _get_vertno
from mne.utils import _check_option, _validate_type, logger, verbose

# Reuse the channel-alignment, noise-covariance and whitener helpers from the
# MCMV module so that both algorithms read the forward/covariance in the same
# channel space and whiten identically to MNE's make_lcmv.
from ._mcmv import (
    _align_channels,
    _check_eeg_reference,
    _check_noise_cov_required,
    _intersect_noise_cov,
    _make_whitener,
)

_ALLOWED_METHOD = ("recipsiicos", "whitened")

# Mirrors mne.beamformer._compute_beamformer._prepare_beamformer_input and
# _compute_beamformer, whose validation this module has to reproduce because it
# solves the LCMV in its own working space rather than going through
# _prepare_beamformer_input.
_ALLOWED_PICK_ORI = ("normal", "max-power", "vector", None)
_ALLOWED_WEIGHT_NORM = ("unit-noise-gain-invariant", "unit-noise-gain", "nai", None)

# Fraction of the eigenvalue energy that the spectral-flip step is allowed to
# carry in the negative eigenvalues before a warning is raised (the authors
# recommend the modification stay below ~20%; their Eq. 24).
_NEG_ENERGY_LIMIT = 0.20

# Default fraction of whitened-leadfield variance kept by the virtual-sensor
# reduction. Kuznetsova et al. (2021) keep 90% of the variance (95% in their
# real-data analysis); 0.99 is the more conservative default here, and still
# reduces a whole-head array to a few tens of virtual sensors.
_DEFAULT_PCT_VAR = 0.99

# Memory budget for one block of vectorised cross-products in
# :func:`_correlation_blocks`. Each column is q^2 long, so the column count per
# block has to be derived from q rather than fixed.
_CORR_BLOCK_BYTES = 64 * 1024**2


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
# Forward-model topographies (handling source orientation)
# --------------------------------------------------------------------------- #
def _forward_gain(forward, common_ch):
    """Return the leadfield over ``common_ch`` and whether it is fixed-oriented.

    Returns
    -------
    gain : ndarray, shape (n_channels, n_dipoles)
        ``n_dipoles`` is ``n_sources`` for a fixed-orientation forward and
        ``3 * n_sources`` for a free-orientation one (three columns per source).
    fixed : bool
        Whether the forward has fixed source orientation.
    """
    fwd_ch = list(forward["sol"]["row_names"])
    idx = [fwd_ch.index(ch) for ch in common_ch]
    gain = np.asarray(forward["sol"]["data"], dtype=np.float64)[idx]
    return gain, is_fixed_orient(forward)


def _tangential_topographies(gain, fixed):
    """Reduce a leadfield to one topography block per source location.

    For a fixed-orientation forward each location has its single topography, so
    the returned array has a trailing dimension of 1. For a free-orientation
    forward each location is reduced to the two dominant (tangential)
    topographies, obtained from the first two left singular vectors of the local
    ``(n_channels, 3)`` block scaled by their singular values -- the reduction
    of Kuznetsova et al. (2021), Section 2.5, which discards the poorly observed
    radial direction (Ahlfors et al., 2010: the third direction carries a median
    6% of the local source energy).

    Parameters
    ----------
    gain : ndarray, shape (n_channels, n_dipoles)
        The leadfield (already in whatever -- e.g. working -- space is desired).
    fixed : bool
        Whether the forward has fixed source orientation.

    Returns
    -------
    topos : ndarray, shape (n_locations, n_channels, n_ori)
        ``n_ori`` is 1 for a fixed-orientation forward and 2 otherwise.
    """
    n_channels = gain.shape[0]
    if fixed:
        n_loc = gain.shape[1]
        return gain.T.reshape(n_loc, n_channels, 1)

    n_loc = gain.shape[1] // 3
    topos = np.empty((n_loc, n_channels, 2), dtype=np.float64)
    for i in range(n_loc):
        local = gain[:, 3 * i : 3 * i + 3]  # (n_channels, 3)
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
    n_loc, _, n_ori = topos.shape
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


def _correlation_blocks(topos, max_block_bytes=_CORR_BLOCK_BYTES):
    """Yield blocks of vectorised cross-products (columns of G_cor).

    Fixed orientation: one column ``vec(g_i g_j^T + g_j g_i^T)`` per ordered
    pair ``i < j`` (Eq. 13). Free orientation: the four columns of Eq. (23) per
    pair. The columns are produced in blocks so that the (potentially very
    large) matrix G_cor is never held in memory all at once -- the caller only
    needs the running Gram matrix ``G_cor G_cor^T``.

    Parameters
    ----------
    topos : ndarray, shape (n_locations, q, n_ori)
        The working-space topographies.
    max_block_bytes : int
        Memory budget for one block. A block is a dense ``(q^2, n_cols)`` array,
        so ``n_cols`` is derived from ``q`` rather than fixed: a free-orientation
        forward emits four columns of length ``q^2`` per source *pair*, and at
        ``q = 60`` a thousand columns already cost 29 MB. The count is rounded
        down to a whole number of pairs so that a block never overshoots the
        budget. The block is materialised twice at the yield point (once as the
        list of columns, once as the stacked array), so the transient cost is
        about twice this budget. The caller's running Gram is a separate,
        unavoidable ``q^2 x q^2`` array.
    """
    n_loc, q, n_ori = topos.shape
    per_pair = 1 if n_ori == 1 else 4
    block_cols = max(1, int(max_block_bytes // (8 * q**2)) // per_pair) * per_pair
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
            if len(buffer) >= block_cols:
                yield np.column_stack(buffer)
                buffer = []
    if buffer:
        yield np.column_stack(buffer)


def _correlation_gram(topos):
    """Accumulate the Gram matrix C_cor = G_cor G_cor^T without storing G_cor.

    Peak memory is the ``q^2 x q^2`` Gram plus one block (see
    :func:`_correlation_blocks`); the rank-updating product is written into a
    reused buffer so that no second Gram-sized temporary is allocated per block.
    """
    msq = topos.shape[1] ** 2
    gram = np.zeros((msq, msq), dtype=np.float64)
    update = np.empty((msq, msq), dtype=np.float64)
    for block in _correlation_blocks(topos):
        np.matmul(block, block.T, out=update)
        gram += update
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


def _whitening_pair(c_pwr, reg, rtol=None):
    r"""Symmetric whitener for the power subspace and its inverse (Eq. 15).

    ``C_pwr`` is rank-limited -- its rank cannot exceed the number of independent
    source topographies, and never exceeds the symmetric dimension
    :math:`q(q+1)/2` -- so the inverse square root is taken only on its range.

    The cut is at machine precision by default, and that matters. Because the
    projector is :math:`P = \Pi - W^{-1}E_K E_K^{\mathsf T}W` with
    :math:`\Pi = W^{-1}W = EE^{\mathsf T}` the projector onto the retained range,
    everything this cut discards is lost from the covariance at *every* rank,
    including :math:`K=1`, before a single correlation direction has been
    removed. A loose threshold is therefore not a harmless safety margin: at
    ``rtol=1e-6`` it dropped roughly 63% of a real whitened ``sample`` covariance
    and moved the localised peak by 24 mm. Conditioning is the job of the ridge
    ``reg * mean(eigenvalue)`` below, which bounds :math:`\|W\|` however small the
    retained eigenvalues are; the cut's only job is to remove the exact numerical
    null space.

    Returns ``(W, W_inv, n_keep)`` with ``W = E Lambda^{-1/2} E^T`` and
    ``W_inv = E Lambda^{1/2} E^T`` over the range of ``C_pwr``, and ``n_keep`` the
    retained rank -- the rank at which the whitened projector annihilates the
    covariance entirely.
    """
    eigvals, eigvecs = np.linalg.eigh(c_pwr)
    eigvals = np.clip(eigvals, 0.0, None)
    if rtol is None:
        rtol = eigvals.size * np.finfo(float).eps
    keep = eigvals > rtol * eigvals.max()
    e = eigvecs[:, keep]
    lam = eigvals[keep]
    lam = lam + reg * lam.mean()
    w = e @ np.diag(lam**-0.5) @ e.T
    w_inv = e @ np.diag(lam**0.5) @ e.T
    return w, w_inv, int(keep.sum())


def _whitened_projector(g_pwr, c_cor, rank, reg):
    """Whitened ReciPSIICOS projector (Eqs. 15-17).

    Whitens with respect to the power subspace, removes the top-K directions of
    the correlation subspace in that whitened space, then unwhitens.
    """
    c_pwr = g_pwr @ g_pwr.T
    w, w_inv, n_keep = _whitening_pair(c_pwr, reg)
    # Correlation Gram in the whitened space, and its principal eigenvectors.
    c_cor_w = w @ c_cor @ w.T
    eigvals, eigvecs = np.linalg.eigh(c_cor_w)
    # eigh returns ascending eigenvalues; take the K largest.
    rank = int(min(rank, eigvecs.shape[1]))
    e_k = eigvecs[:, eigvecs.shape[1] - rank :]
    msq = g_pwr.shape[0]
    projector = w_inv @ (np.eye(msq) - e_k @ e_k.T) @ w
    return projector, eigvals[::-1], n_keep


# --------------------------------------------------------------------------- #
# Rank curves (Eqs. 20-21) -- closed-form over all ranks at once
# --------------------------------------------------------------------------- #
# The retained-energy fractions P_pwr(k) and P_cor(k) are needed for every rank
# k = 1 .. q^2. Building and applying the projector separately at each rank costs
# an SVD/eigendecomposition per rank (q^2 of them, each on a q^2 x q^2 matrix),
# which is intractable for realistic virtual-sensor counts. Both curves can be
# obtained from a single decomposition, exactly, as cumulative sums.
def _power_rank_curve(g_pwr, c_pwr, c_cor, tr_pwr, tr_cor):
    """Exact ReciPSIICOS retention curves for all ranks (Eq. 20-21).

    With ``P_k = U_k U_k^T`` the projector onto the ``k`` leading power
    directions, ``tr(P_k C P_k^T) = tr(U_k^T C U_k) = sum_{i<=k} u_i^T C u_i``,
    so each curve is the cumulative sum of the per-direction quadratic forms.
    """
    u, _, _ = np.linalg.svd(g_pwr, full_matrices=False)
    n_dir = u.shape[1]
    msq = g_pwr.shape[0]
    d_pwr = np.einsum("ij,ij->j", u, c_pwr @ u)  # u_i^T C_pwr u_i
    d_cor = np.einsum("ij,ij->j", u, c_cor @ u)  # u_i^T C_cor u_i
    cum_pwr = np.cumsum(d_pwr) / tr_pwr
    cum_cor = np.cumsum(d_cor) / tr_cor
    # For k > n_dir the projector is unchanged (all directions kept), so the
    # curve is flat at its final value.
    p_pwr = np.empty(msq)
    p_cor = np.empty(msq)
    p_pwr[:n_dir] = cum_pwr
    p_pwr[n_dir:] = cum_pwr[-1]
    p_cor[:n_dir] = cum_cor
    p_cor[n_dir:] = cum_cor[-1]
    return p_pwr, p_cor


def _whitened_rank_curve(g_pwr, c_pwr, c_cor, tr_pwr, tr_cor, reg):
    r"""Exact Whitened ReciPSIICOS retention curves for all ranks (Eq. 20-21).

    Here ``P_k = W^{-1}(I - E_k E_k^T) W`` with ``E`` the eigenvectors of the
    power-whitened correlation Gram (largest first). Writing ``C_w = W C W^T``,
    ``M = W^{-1 T} W^{-1}``, ``P = E^T C_w E`` and ``Q = E^T M E``,

    .. math::
        \mathrm{tr}(P_k C P_k^{\mathsf T}) =
        \mathrm{tr}(PQ) - \sum_{i\le k}(PQ)_{ii} - \sum_{i\le k}(QP)_{ii}
        + \sum_{i,j\le k} P_{ij} Q_{ji},

    every term of which is a cumulative sum over the ordered eigenvectors, so
    both curves follow from a single eigendecomposition.
    """
    w, w_inv, _ = _whitening_pair(c_pwr, reg)
    c_cor_w = w @ c_cor @ w.T
    eigvals, eigvecs = np.linalg.eigh(c_cor_w)
    order = np.argsort(eigvals)[::-1]  # largest first (as removed by the projector)
    e = eigvecs[:, order]
    metric = e.T @ (w_inv.T @ w_inv) @ e  # Q
    c_cor_e = e.T @ c_cor_w @ e  # P for the correlation Gram (diagonal = eigvals)
    c_pwr_e = e.T @ (w @ c_pwr @ w.T) @ e  # P for the power Gram

    def _curve(p_mat, trace):
        pq = p_mat @ metric
        qp = metric @ p_mat
        diag_pq = np.diag(pq)
        diag_qp = np.diag(qp)
        block = p_mat * metric.T  # (i, j) -> P_ij Q_ji
        block_cumsum = np.diag(np.cumsum(np.cumsum(block, axis=0), axis=1))
        f_k = diag_pq.sum() - np.cumsum(diag_pq) - np.cumsum(diag_qp) + block_cumsum
        return f_k / trace

    return _curve(c_pwr_e, tr_pwr), _curve(c_cor_e, tr_cor)


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
# Working space: noise whitening + virtual-sensor reduction
# --------------------------------------------------------------------------- #
def _virtual_sensor_count(sv, pct_var, n_virtual):
    """Return the number of virtual sensors to keep from a singular spectrum.

    Either an explicit count (``n_virtual``) or the smallest number of leading
    directions whose cumulative squared-singular-value energy reaches
    ``pct_var`` of the total.
    """
    n_max = sv.size
    if n_virtual is not None:
        return int(np.clip(n_virtual, 1, n_max))
    if sv[0] == 0:
        return 1
    cum = np.cumsum(sv**2)
    cum /= cum[-1]
    q = int(np.searchsorted(cum, pct_var) + 1)
    return int(np.clip(q, 1, n_max))


def _reduction_operator(
    info, forward, common_ch, *, noise_cov, whitener_rank, pct_var, n_virtual
):
    r"""Build the reduction operator ``B`` mapping sensors to the working space.

    ``B = U_q^T W`` where ``W`` whitens by the noise covariance (per-type rank,
    :func:`mne.cov.compute_whitener`) and ``U_q`` holds the ``q`` leading left
    singular vectors of the *whitened* leadfield -- the virtual sensors of
    Kuznetsova et al. (2021). Whitening makes the reduction device-agnostic
    (commensurable units across sensor types); the SVD truncation makes the
    subsequent :math:`M^2`-space computations tractable.

    Returns
    -------
    b_op : ndarray, shape (q, M)
        The reduction operator (working-space rows, sensor-space columns).
    n_ret : int
        The number of virtual sensors ``q`` retained.
    n_white : int
        The whitened rank ``r`` (upper bound on ``q``), for logging.
    """
    whitener = _make_whitener(info, noise_cov, common_ch, whitener_rank)  # (r, M)
    gain, _ = _forward_gain(forward, common_ch)  # (M, n_dipoles)
    gain_white = whitener @ gain  # (r, n_dipoles)
    u, sv, _ = np.linalg.svd(gain_white, full_matrices=False)  # u: (r, k)
    q = _virtual_sensor_count(sv, pct_var, n_virtual)
    u_q = u[:, :q]  # (r, q)
    b_op = u_q.T @ whitener  # (q, M)
    return b_op, q, whitener.shape[0]


def _build_projector(gain_work, fixed, method, rank, reg):
    """Build the ReciPSIICOS/Whitened projector from working-space topographies."""
    topos = _tangential_topographies(gain_work, fixed)
    g_pwr = _power_columns(topos)
    msq = gain_work.shape[0] ** 2
    if int(rank) < 1:
        raise ValueError(f"rank must be a positive integer, got {rank}.")
    if int(rank) > msq:
        raise ValueError(
            f"rank={rank} exceeds the dimension q^2={msq} of the working "
            "covariance vector space (q is the number of virtual sensors)."
        )
    if method == "recipsiicos":
        projector, _ = _power_projector(g_pwr, rank)
    else:
        # The whitened projector lives on the range of the power whitener, which
        # is contained in the q(q+1)/2-dimensional subspace of symmetric
        # matrices. Removing that many correlation directions removes all of it,
        # so the projector -- and hence the modified covariance -- is exactly
        # zero. The bound that actually bites is the retained rank of C_pwr,
        # which for a rank-deficient forward is reached well before q(q+1)/2, so
        # the warning is raised against that rather than against the symmetric
        # dimension. Warn rather than raise, so a caller sweeping ranks is not
        # stopped part way.
        c_cor = _correlation_gram(topos)
        projector, _, n_keep = _whitened_projector(g_pwr, c_cor, rank, reg)
        if int(rank) >= n_keep:
            warnings.warn(
                f"rank={rank} is at or above {n_keep}, the rank of the power "
                "Gram C_pwr (the number of independent source topographies, "
                "capped at q(q+1)/2), so the whitened projector annihilates the "
                "covariance and the modified covariance is exactly zero. Use a "
                "smaller rank (see recipsiicos_rank_curve).",
                RuntimeWarning,
                stacklevel=2,
            )
    return projector


def _recipsiicos_working(
    info,
    forward,
    data_cov,
    *,
    method,
    rank,
    noise_cov,
    whitener_rank,
    pct_var,
    n_virtual,
    reg,
):
    """Clean the data covariance in the whitened, reduced working space.

    Returns the reduction operator, the working-space leadfield, the cleaned
    working-space covariance and diagnostics, shared by both public entry
    points.
    """
    _validate_type(data_cov, Covariance, "data_cov")
    _check_option("method", method, _ALLOWED_METHOD)

    common_ch, cov = _align_channels(info, forward, data_cov)
    # A MEG-only empty-room noise covariance with a MEG+EEG forward is standard
    # practice, so drop whatever the noise covariance does not cover before it
    # reaches compute_whitener; and refuse the mixed-sensor-type case without a
    # noise covariance exactly as make_lcmv does.
    common_ch, cov = _intersect_noise_cov(common_ch, cov, noise_cov)
    _check_noise_cov_required(info, common_ch, noise_cov)
    _check_eeg_reference(info, common_ch)
    b_op, q, r = _reduction_operator(
        info,
        forward,
        common_ch,
        noise_cov=noise_cov,
        whitener_rank=whitener_rank,
        pct_var=pct_var,
        n_virtual=n_virtual,
    )
    gain, fixed = _forward_gain(forward, common_ch)
    gain_work = b_op @ gain  # (q, n_dipoles)
    cov_work = b_op @ cov @ b_op.T  # (q, q)

    projector = _build_projector(gain_work, fixed, method, rank, reg)
    modified = _apply_projector(projector, cov_work)
    cov_clean, neg_energy = _spectral_flip(modified)

    logger.info(
        f"    ReciPSIICOS ({method}): {r} whitened dims reduced to {q} virtual "
        f"sensors; projection rank {rank}; negative-eigenvalue energy "
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
    return common_ch, b_op, gain_work, fixed, cov_clean, neg_energy, q


def _check_pick_ori(forward, pick_ori):
    """Validate ``pick_ori`` against the forward exactly as MNE's LCMV does.

    :func:`mne.beamformer.make_lcmv` runs these four checks inside
    :func:`mne.beamformer._compute_beamformer._prepare_beamformer_input`, which
    this module bypasses because it solves the beamformer in its own working
    space. They are reproduced here -- in MNE's order and with MNE's wording --
    so that the two entry points refuse the same inputs in the same way.

    Without them ``pick_ori='normal'`` silently reaches the filter computation,
    which takes the third orientation column of each source. On the
    ``surf_ori=False`` forward that :func:`mne.read_forward_solution` returns by
    default that column is the head-coordinate +Z axis, and on a volume source
    space it is not a surface normal at all, so the result is a perfectly
    plausible :class:`~mne.SourceEstimate` for the wrong dipole orientation.
    """
    _check_option("pick_ori", pick_ori, _ALLOWED_PICK_ORI)
    if pick_ori is not None and is_fixed_orient(forward):
        raise ValueError(
            f"Normal or max-power orientation (got {pick_ori!r}) can only be picked "
            "when a forward operator with free orientation is used."
        )
    if pick_ori == "normal" and not forward["surf_ori"]:
        raise ValueError(
            "Normal orientation can only be picked when a forward operator oriented in "
            "surface coordinates is used."
        )
    _check_src_normal(pick_ori, forward["src"])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@verbose
def make_recipsiicos_cov(
    data_cov,
    forward,
    info,
    *,
    rank,
    method="recipsiicos",
    noise_cov=None,
    whitener_rank=None,
    pct_var=_DEFAULT_PCT_VAR,
    n_virtual=None,
    reg=0.05,
    verbose=None,
):
    r"""ReciPSIICOS modification of a sensor-space data covariance.

    Builds the ReciPSIICOS (``method='recipsiicos'``) or Whitened ReciPSIICOS
    (``method='whitened'``) projector from the forward model in the noise-
    whitened, virtual-sensor working space, applies it to the data covariance
    there, restores positive-definiteness by the spectral-flip step (Eq. 12) and
    maps the result back to sensor space.

    The returned covariance is provided for inspection and interoperability. To
    build a beamformer, prefer :func:`make_recipsiicos_lcmv`, which solves the
    LCMV problem directly in the working space; feeding the covariance returned
    here to :func:`mne.beamformer.make_lcmv` with the same ``noise_cov`` would
    whiten a second time.

    Parameters
    ----------
    data_cov : instance of mne.Covariance
        The sensor-space data covariance to modify.
    forward : instance of mne.Forward
        The forward solution. The projector is built from this alone.
    info : instance of mne.Info
        Measurement info, required to whiten per sensor type. Must cover the
        channels shared by ``data_cov`` and ``forward``.
    rank : int
        Projection rank ``K`` (Eq. 10 / Eq. 17), in the ``q^2``-dimensional
        working covariance vector space (``q`` = number of virtual sensors). Use
        :func:`recipsiicos_rank_curve` to choose it.
    method : 'recipsiicos' | 'whitened'
        Which projector to build. ``'recipsiicos'`` projects onto the principal
        power subspace (Eq. 10); ``'whitened'`` projects away from the principal
        correlation subspace in the power-whitened space (Eqs. 15-17).
    noise_cov : instance of mne.Covariance | None
        Noise covariance used to whiten. If ``None``, an ad-hoc per-type model
        (:func:`mne.make_ad_hoc_cov`) is used; for a single sensor type this is
        a global scaling that leaves the projector subspaces unchanged.
    whitener_rank : int | None | 'full'
        Rank handling passed to :func:`mne.cov.compute_whitener`. The
        default ``None`` auto-detects the covariance rank per sensor type
        and drops the null space -- required when SSP/SSS projectors make the
        covariance rank-deficient, since ``'full'`` would instead invert the
        near-null directions and yield an unstable whitener.
    pct_var : float
        Fraction of whitened-leadfield variance kept by the virtual-sensor
        reduction (Section 2.6). Ignored if ``n_virtual`` is given.
    n_virtual : int | None
        Explicit number of virtual sensors. Overrides ``pct_var``.
    reg : float
        Ridge added to the power-subspace eigenvalues before whitening (used by
        ``method='whitened'`` only) as a fraction of their mean.
    %(verbose)s

    Returns
    -------
    cov : instance of mne.Covariance
        The modified sensor-space covariance (rank ``q``), over the channels
        shared by ``data_cov``, ``forward`` and ``info``.

    Notes
    -----
    Equation numbers refer to Kuznetsova et al. (2021).

    References
    ----------
    .. footbibliography::
    """
    _validate_type(info, "info", "info")
    common_ch, b_op, _, _, cov_clean, _, _ = _recipsiicos_working(
        info,
        forward,
        data_cov,
        method=method,
        rank=rank,
        noise_cov=noise_cov,
        whitener_rank=whitener_rank,
        pct_var=pct_var,
        n_virtual=n_virtual,
        reg=reg,
    )
    # Map the cleaned working-space covariance back to sensor space through the
    # pseudo-inverse of the reduction operator. The result is the rank-q sensor-
    # space representation of the cleaned covariance.
    b_pinv = np.linalg.pinv(b_op)  # (M, q)
    cov_sensor = b_pinv @ cov_clean @ b_pinv.T
    cov_sensor = 0.5 * (cov_sensor + cov_sensor.T)

    # ``bads`` must only name channels this covariance actually carries: the
    # working space is built on ``common_ch``, which already excludes the bad
    # channels, so carrying the input's list forward would advertise channels
    # that are not in ``ch_names`` and confuse ``pick_channels_cov``.
    kept = set(common_ch)
    return Covariance(
        cov_sensor,
        common_ch,
        [ch for ch in data_cov.get("bads", []) if ch in kept],
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
    weight_norm="unit-noise-gain",
    reduce_rank=False,
    inversion="matrix",
    whitener_rank=None,
    pct_var=_DEFAULT_PCT_VAR,
    n_virtual=None,
    verbose=None,
):
    r"""LCMV beamformer on a ReciPSIICOS-modified data covariance.

    Cleans the data covariance with the ReciPSIICOS or Whitened ReciPSIICOS
    projector in the noise-whitened, virtual-sensor working space, then solves
    the LCMV problem *in that working space* by reusing MNE's own filter
    computation (orientation selection, weight normalisation and rank handling).
    The reduction operator is folded into the returned filters as their
    whitener, so :func:`mne.beamformer.apply_lcmv` applies the whole pipeline to
    sensor data without modification.

    Parameters
    ----------
    info : instance of mne.Info
        Measurement info. Used to whiten per sensor type and to align channels.
    forward : instance of mne.Forward
        The forward solution.
    data_cov : instance of mne.Covariance
        The sensor-space data covariance.
    rank : int
        Projection rank ``K`` in the ``q^2``-dimensional working covariance
        vector space (``q`` = number of virtual sensors). See
        :func:`recipsiicos_rank_curve`.
    method : 'recipsiicos' | 'whitened'
        Which ReciPSIICOS projector to use.
    noise_cov : instance of mne.Covariance | None
        Noise covariance used to whiten and to define the working-space noise
        model (which is white by construction). If ``None``, an ad-hoc per-type
        model is used.
    reg : float
        Tikhonov regularisation for the working-space covariance inversion, as a
        fraction of its mean eigenvalue (passed to MNE's filter computation).
        Also the ridge for the whitened-projector power subspace.
    label : instance of mne.Label | None
        Restrict the source space to this label before building the filters.
    pick_ori : None | 'normal' | 'max-power' | 'vector'
        Source orientation handling, passed to MNE's filter computation. Subject
        to the same restrictions as in :func:`mne.beamformer.make_lcmv`: anything
        other than ``None`` requires a free-orientation forward, and ``'normal'``
        additionally requires a surface-oriented forward (``surf_ori=True``,
        which :func:`mne.read_forward_solution` does *not* give by default) on a
        surface or discrete source space.
    weight_norm : None | 'unit-noise-gain' | 'unit-noise-gain-invariant' | 'nai'
        Weight normalisation, passed to MNE's filter computation. In the working
        space the noise is white, so unit-noise-gain normalises to unit norm
        there.
    reduce_rank : bool
        Whether to reduce the leadfield rank by one during the solve (MNE
        option). Required for free-orientation MEG: a three-orientation MEG
        leadfield is effectively rank two per location (the radial field is
        silent in a spherical conductor), so the solve is singular otherwise.
    inversion : 'matrix' | 'single'
        Covariance inversion scheme, passed to MNE's filter computation.
    whitener_rank : int | None | 'full'
        Rank handling passed to :func:`mne.cov.compute_whitener`. The
        default ``None`` auto-detects the covariance rank per sensor type
        and drops the null space -- required when SSP/SSS projectors make the
        covariance rank-deficient, since ``'full'`` would instead invert the
        near-null directions and yield an unstable whitener.
    pct_var : float
        Fraction of whitened-leadfield variance kept by the virtual-sensor
        reduction. Ignored if ``n_virtual`` is given.
    n_virtual : int | None
        Explicit number of virtual sensors. Overrides ``pct_var``.
    %(verbose)s

    Returns
    -------
    filters : instance of mne.beamformer.Beamformer
        The LCMV filters, ready for :func:`mne.beamformer.apply_lcmv`. Their
        ``weights`` act in the working space and their ``whitener`` is the
        reduction operator, so the combination acts on sensor data.

    Notes
    -----
    Equation numbers refer to Kuznetsova et al. (2021).

    For free-orientation MEG pass ``reduce_rank=True`` (see that parameter).
    The ``whitened`` projector builds a correlation Gram over every source
    *pair*, which is :math:`O(N^2)` in the number of sources; build the
    projector on a decimated source space or restrict it with ``label`` for
    real, high-resolution forwards. The ``recipsiicos`` projector uses only the
    auto-products and stays linear in :math:`N`.

    Memory, unlike runtime, is governed by ``q`` rather than :math:`N`: the
    cross-product columns are accumulated in bounded blocks, so the peak is the
    :math:`q^2 \times q^2` Gram (99 MiB at ``q = 60``, 1.5 GiB at ``q = 120``)
    plus one same-sized accumulator. Lower ``pct_var`` or set ``n_virtual`` if
    that is too much.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(info, "info", "info")
    if label is not None:
        from mne import Label
        from mne.forward import restrict_forward_to_label

        _validate_type(label, Label, "label")
        forward = restrict_forward_to_label(forward, label)

    # Refuse the same orientation/normalisation inputs make_lcmv refuses, before
    # spending any time on the projector.
    _check_pick_ori(forward, pick_ori)
    _check_option("weight_norm", weight_norm, _ALLOWED_WEIGHT_NORM)

    common_ch, b_op, gain_work, fixed, cov_clean, _, q = _recipsiicos_working(
        info,
        forward,
        data_cov,
        method=method,
        rank=rank,
        noise_cov=noise_cov,
        whitener_rank=whitener_rank,
        pct_var=pct_var,
        n_virtual=n_virtual,
        reg=reg,
    )

    n_orient = 1 if fixed else 3

    if q < n_orient:
        raise ValueError(
            f"The whitened, virtual-sensor-reduced forward has only q={q} working "
            f"dimensions, fewer than the n_orient={n_orient} source orientations, "
            f"so the working-space beamformer is underdetermined. This happens "
            f"with a low-rank forward -- e.g. a single-shell sphere model, whose "
            f"MEG leadfield is rank ~2 per location. Use a realistic BEM forward, "
            f"a fixed-orientation forward, or raise pct_var / n_virtual."
        )

    # Source normals and vertices, following MNE's convention: for a free
    # forward keep one normal per source; a surface-oriented forward uses the
    # local +Z direction.
    nn = forward["source_nn"]
    if not fixed:
        nn = nn[2::3]
    nn = nn.copy()
    if forward["surf_ori"]:
        nn[...] = [0.0, 0.0, 1.0]
    vertno = _get_vertno(forward["src"])
    orient_std = np.ones(gain_work.shape[1])

    # Solve the LCMV in the working space. The working-space noise is white, so
    # the identity is the correct whitener here; MNE's routine does the
    # orientation selection, weight normalisation and rank handling.
    weights, max_power_ori = _compute_beamformer(
        gain_work,
        cov_clean,
        reg,
        n_orient,
        weight_norm,
        pick_ori,
        reduce_rank,
        q,
        inversion=inversion,
        nn=nn,
        orient_std=orient_std,
        whitener=np.eye(q),
    )

    src_type = _get_src_type(forward["src"], vertno)
    subject = _subject_from_forward(forward)
    is_free_ori = (not fixed) if pick_ori in (None, "vector") else False

    # Record the projectors the filters were built under, exactly as make_lcmv
    # does. ``compute_whitener`` already folded them into ``b_op``, so
    # ``apply_lcmv`` will not re-apply them (it only does that when ``whitener``
    # is None); recording them re-enables MNE's safety check that the data a
    # filter is applied to carries the same projectors it was computed with.
    from mne._fiff.proj import make_projector

    proj, _, _ = make_projector(info["projs"], list(common_ch))
    is_ssp = bool(info["projs"])

    # Store the covariances the way make_lcmv does: restricted to the channels
    # actually used, and with the ``estimator`` entry that mne.compute_covariance
    # attaches removed. That entry is a scikit-learn object, which h5io cannot
    # serialise, so leaving it in makes ``filters.save()`` raise part way through
    # and leave a truncated file that ``read_beamformer`` then loads without
    # complaint -- surfacing much later as a KeyError inside apply_lcmv.
    data_cov = pick_channels_cov(data_cov, include=list(common_ch))
    data_cov.pop("estimator", None)
    if noise_cov is not None:
        noise_cov = noise_cov.copy()
        noise_cov.pop("estimator", None)

    filters = Beamformer(
        kind="LCMV",
        weights=weights,
        data_cov=data_cov,
        noise_cov=noise_cov,
        # Fold the reduction operator in as the whitener: apply_lcmv applies it
        # to the sensor data before the working-space weights.
        whitener=b_op,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        ch_names=list(common_ch),
        proj=proj,
        is_ssp=is_ssp,
        vertices=vertno,
        is_free_ori=is_free_ori,
        n_sources=forward["nsource"],
        src_type=src_type,
        source_nn=forward["source_nn"].copy(),
        subject=subject,
        rank=int(q),
        max_power_ori=max_power_ori,
        inversion=inversion,
    )
    return filters


def _optimal_rank(p_pwr, p_cor, method, floor=0.1):
    """Rank ``K*`` at the 45-degree point of the power-vs-correlation curve.

    Kuznetsova et al. (2021), Section 2.4: the optimal rank is where the marginal
    energy loss of the two subspaces balances -- the 45-degree tangent of the
    ``(P_pwr, P_cor)`` curve, i.e. the sign change of
    ``dP_cor/dk - dP_pwr/dk``.

    The two projectors traverse the rank axis in *opposite directions*, which is
    the whole subtlety of this criterion, and it is worth being explicit about it
    because the sign convention is easy to get backwards. Section 3.1 of the
    paper states that "increased projection rank for ReciPSIICOS corresponds to a
    less restrictive situation", and its Fig. 19 plots the two methods on a
    shared axis whose lower (ReciPSIICOS) scale "is arranged in the descending
    order". So:

    * ``whitened`` projects *away from* the correlation subspace. Its identity
      end is ``k = 0``, both curves fall with ``k``, and the axis is scanned
      *upward* from ``k = 1``: ``K*`` is the last rank before the difference
      turns positive.
    * ``recipsiicos`` projects *onto* the power subspace. Its identity end is
      ``k = q^2``, both curves rise with ``k``, and the axis is scanned
      *downward* from there: ``K*`` is the last rank before the difference turns
      negative *as ``k`` decreases*, i.e. one past the final negative difference
      when read in ascending order. Scanning it upward instead would stop at the
      first negative difference, which on any realistic forward occurs almost
      immediately -- the difference *starts* negative, because the leading power
      direction is also the direction the correlation subspace overlaps most --
      and would return ``K* = 1`` or ``2``, discarding nearly all of the
      source-power subspace.

    In each case a ``floor`` on the retained power guards against a degenerate
    curve (a rank-deficient forward, e.g. a single-shell sphere model, whose
    subspaces do not separate) returning a rank that (near-)empties the
    covariance; the back-off runs toward that method's identity end.
    """
    n = int(p_pwr.size)
    if n < 3:
        return n
    delta = np.diff(p_cor) - np.diff(p_pwr)  # delta[i] links rank i+1 and i+2
    if method == "recipsiicos":
        # Descending axis: the crossing is the *last* negative difference.
        hit = np.nonzero(delta < 0)[0]
        k = int(hit[-1] + 2) if hit.size else 1
        # Increasing curve: back off toward larger (near-identity) rank.
        while k < n and p_pwr[k - 1] < floor:
            k += 1
    else:
        hit = np.nonzero(delta > 0)[0]
        k = int(hit[0] + 1) if hit.size else 1
        # Decreasing curve: back off toward smaller (near-identity) rank so the
        # projector does not remove essentially all of the covariance.
        while k > 1 and p_pwr[k - 1] < floor:
            k -= 1
    return max(1, min(k, n))


@verbose
def recipsiicos_rank_curve(
    forward,
    info,
    *,
    method="recipsiicos",
    data_cov=None,
    noise_cov=None,
    whitener_rank=None,
    pct_var=_DEFAULT_PCT_VAR,
    n_virtual=None,
    reg=0.05,
    return_optimal=False,
    verbose=None,
):
    r"""Power-vs-correlation retention as a function of projection rank.

    Computes, for every projection rank ``k`` in the working space, the fraction
    of the power- and correlation-subspace energy that survives the projection
    (Eqs. 20-21). The useful rank is the one beyond which the correlation
    subspace stops being depleted faster than the power subspace; plotting
    ``p_cor`` against ``p_pwr`` makes this trade-off visible and its 45-degree
    point (Section 2.4) is the recommended rank.

    Note that the two methods traverse the rank axis in opposite directions --
    ``k = q^2`` is the identity for ``'recipsiicos'`` and ``k = 0`` is the
    identity for ``'whitened'`` -- so ``p_pwr`` and ``p_cor`` rise with ``k`` for
    the former and fall with it for the latter, and the returned ``K*`` is
    located accordingly (Fig. 19 of the paper puts the ReciPSIICOS scale in
    descending order for exactly this reason).

    Parameters
    ----------
    forward : instance of mne.Forward
        The forward solution.
    info : instance of mne.Info
        Measurement info, used to whiten and reduce to the working space so that
        the curve matches the space the beamformer is built in.
    method : 'recipsiicos' | 'whitened'
        Which projector family to characterise.
    data_cov : instance of mne.Covariance | None
        Only used to intersect channels; its values do not affect the curve.
    noise_cov : instance of mne.Covariance | None
        Noise covariance used to whiten (see :func:`make_recipsiicos_lcmv`).
    whitener_rank : int | None | 'full'
        Rank handling passed to :func:`mne.cov.compute_whitener`. The
        default ``None`` auto-detects the covariance rank per sensor type
        and drops the null space -- required when SSP/SSS projectors make the
        covariance rank-deficient, since ``'full'`` would instead invert the
        near-null directions and yield an unstable whitener.
    pct_var : float
        Fraction of whitened-leadfield variance kept by the virtual-sensor
        reduction. Ignored if ``n_virtual`` is given.
    n_virtual : int | None
        Explicit number of virtual sensors. Overrides ``pct_var``.
    reg : float
        Ridge for the whitened-projector power subspace.
    return_optimal : bool
        If ``True``, also return the automatically selected rank ``K*``.
    %(verbose)s

    Returns
    -------
    ranks : ndarray, shape (n_ranks,)
        The projection ranks evaluated (1 .. q^2).
    p_pwr : ndarray, shape (n_ranks,)
        Fraction of power-subspace energy retained at each rank (Eq. 20).
    p_cor : ndarray, shape (n_ranks,)
        Fraction of correlation-subspace energy retained at each rank (Eq. 21).
    optimal : int
        The selected rank ``K*`` (only if ``return_optimal=True``).

    Notes
    -----
    Equation numbers refer to Kuznetsova et al. (2021).

    The curve needs a forward with rich leadfield structure. On a single-shell
    sphere model the tangential leadfields are so low-rank that the power
    subspace collapses to a few directions and the curve degenerates (there is
    nothing to separate); a realistic BEM forward gives the smooth, separable
    curve the 45-degree criterion expects, and on a degenerate curve ``K*``
    falls back to a near-identity rank. Both curves build a correlation Gram
    over every source pair (:math:`O(N^2)` in runtime), so decimate the source
    space for real forwards; peak memory is instead the
    :math:`q^2 \times q^2` Gram, so keep ``q`` modest as well.

    References
    ----------
    .. footbibliography::
    """
    _check_option("method", method, _ALLOWED_METHOD)
    _validate_type(info, "info", "info")

    # Determine the common channels (values of data_cov are irrelevant here).
    if data_cov is not None:
        common_ch, _ = _align_channels(info, forward, data_cov)
    else:
        bads = set(info["bads"])
        fwd_set = set(forward["sol"]["row_names"])
        common_ch = [ch for ch in info["ch_names"] if ch in fwd_set and ch not in bads]
        if len(common_ch) == 0:
            raise ValueError("No common good channels between info and forward.")

    # Same channel selection as the beamformer entry points, so the curve
    # characterises the space the filters are actually built in. Only the
    # channel list matters here, hence the placeholder matrix.
    n_ch = len(common_ch)
    common_ch, _ = _intersect_noise_cov(common_ch, np.empty((n_ch, n_ch)), noise_cov)
    _check_noise_cov_required(info, common_ch, noise_cov)
    _check_eeg_reference(info, common_ch)

    b_op, _, _ = _reduction_operator(
        info,
        forward,
        common_ch,
        noise_cov=noise_cov,
        whitener_rank=whitener_rank,
        pct_var=pct_var,
        n_virtual=n_virtual,
    )
    gain, fixed = _forward_gain(forward, common_ch)
    gain_work = b_op @ gain

    topos = _tangential_topographies(gain_work, fixed)
    g_pwr = _power_columns(topos)
    c_pwr = g_pwr @ g_pwr.T
    c_cor = _correlation_gram(topos)
    tr_pwr, tr_cor = np.trace(c_pwr), np.trace(c_cor)

    msq = g_pwr.shape[0]
    ranks = np.arange(1, msq + 1)
    if method == "recipsiicos":
        p_pwr, p_cor = _power_rank_curve(g_pwr, c_pwr, c_cor, tr_pwr, tr_cor)
    else:
        p_pwr, p_cor = _whitened_rank_curve(g_pwr, c_pwr, c_cor, tr_pwr, tr_cor, reg)

    if return_optimal:
        return ranks, p_pwr, p_cor, _optimal_rank(p_pwr, p_cor, method)
    return ranks, p_pwr, p_cor
