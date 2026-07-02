"""Multiple Constrained Minimum Variance (MCMV) beamformer.

This module implements the multi-source MCMV spatial filter for MEG/EEG source
reconstruction, exactly following the formulation derived by
Moiseev et al. (2011) :footcite:`Moiseev2011`. The MCMV beamformer generalises
the single-source Linearly Constrained Minimum Variance (LCMV) beamformer
:footcite:`VanVeenEtAl1997` -- itself rooted in the linearly constrained
adaptive array of Frost (1972) :footcite:`Frost1972` -- to ``n`` sources that
are constrained *jointly*. By imposing unit gain on each source's own forward
field and *zero* gain on every other constrained source's field, the MCMV
filter is insensitive to correlations between the constrained sources and
therefore suppresses the source leakage and signal cancellation that bias
single-source LCMV when sources are correlated :footcite:`Moiseev2011`.

Notes
-----
The names and equation numbers used in the comments below refer to
Moiseev et al. (2011) :footcite:`Moiseev2011`.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>
# License: BSD-3-Clause

import warnings

import numpy as np
from mne import Covariance

# NOTE: we use the stdlib ``warnings.warn`` (RuntimeWarning) rather than
# ``mne.utils.warn`` so that warnings are emitted regardless of MNE's logging
# level and are reliably catchable by ``pytest.warns``. When upstreaming into
# MNE-Python these can be swapped for ``mne.utils.warn``, whose test harness
# handles the stack-rewriting behaviour.
# MNE's regularised (pseudo-)inverse of a covariance matrix. Re-using it here
# means the data-covariance inversion -- including the ``reg`` diagonal-loading
# convention and rank handling -- is numerically identical to that performed by
# ``mne.beamformer.make_lcmv``, so the n == 1 MCMV filter reduces *exactly* to
# the corresponding LCMV filter and every covariance estimator/shrinkage method
# available through ``mne.compute_covariance`` is inherited unchanged.
from mne.beamformer._compute_beamformer import _reg_pinv
from mne.forward import is_fixed_orient
from mne.utils import _check_option, _validate_type, logger, verbose

_ALLOWED_WEIGHT_NORM = ("unit-gain", "unit-noise-gain", None)


class MCMVBeamformer(dict):
    """Container for MCMV spatial-filter weights.

    This is a thin ``dict`` subclass that mirrors
    :class:`mne.beamformer.Beamformer` so that, when the algorithm is upstreamed
    into MNE-Python, the native container can be substituted with no change to
    calling code. It stores, among other metadata, the spatial-filter weight
    matrix under the key ``'weights'`` with shape ``(n_sources, n_channels)``
    (one filter -- one row -- per constrained source).
    """

    def __repr__(self):  # noqa: D105
        n = self.get("n_sources", "?")
        norm = self.get("weight_norm", "?")
        return f"<MCMVBeamformer | {n} source(s), weight_norm={norm!r}>"


# --------------------------------------------------------------------------- #
# Numerical core (pure NumPy, no MNE objects) -- unit-testable in isolation.
# --------------------------------------------------------------------------- #
def _compute_mcmv_weights(leadfield, cov_inv, *, cond_warn=1e8):
    r"""Compute the unit-gain MCMV weight matrix from the linear algebra inputs.

    Implements Moiseev et al. (2011), Eq. (5):

    .. math::  \mathbf{W} = \mathbf{R}^{-1}\mathbf{H}
               \left(\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}
               \right)^{-1}

    subject to the unit-gain / zero-gain constraint of Eq. (3),
    :math:`\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}_n`.

    Parameters
    ----------
    leadfield : ndarray, shape (n_channels, n_sources)
        The joint forward matrix :math:`\mathbf{H}` whose ``i``-th column is the
        forward field :math:`\mathbf{h}_i` of the ``i``-th constrained source.
    cov_inv : ndarray, shape (n_channels, n_channels)
        The (regularised) inverse data covariance :math:`\mathbf{R}^{-1}`.
    cond_warn : float
        Condition-number threshold of :math:`\mathbf{H}^{\mathsf T}
        \mathbf{R}^{-1}\mathbf{H}` above which a numerical-stability warning is
        emitted.

    Returns
    -------
    weights : ndarray, shape (n_sources, n_channels)
        The MCMV spatial filters, one per constrained source (row-wise).

    Raises
    ------
    RuntimeError
        If :math:`\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}` is singular,
        which happens when the constrained sources are (numerically) collinear
        -- e.g. coincident locations with identical orientation -- or when the
        beamformer order exceeds the rank of the data.
    """
    H = np.asarray(leadfield, dtype=np.float64)
    Rinv = np.asarray(cov_inv, dtype=np.float64)

    # A = R^-1 H  (n_channels x n_sources)
    A = Rinv @ H
    # B = H^T R^-1 H  (n_sources x n_sources) -- the matrix that is inverted in
    # Eq. (5); also equals S of Table 2 when R is the data covariance.
    B = H.T @ A

    # Guard the inversion of B explicitly so we can raise an informative error
    # rather than letting a silent near-singular inverse corrupt the weights.
    cond = np.linalg.cond(B)
    if not np.isfinite(cond) or cond > 1.0 / np.finfo(np.float64).eps:
        raise RuntimeError(
            "H^T R^-1 H is singular (condition number is non-finite): the "
            "constrained sources are collinear (coincident location and "
            "orientation) or the beamformer order exceeds the data rank. "
            "Reduce the number of sources or separate them."
        )
    if cond > cond_warn:
        warnings.warn(
            f"H^T R^-1 H is poorly conditioned (condition number {cond:.2e}); "
            "the constrained sources are nearly collinear and the MCMV weights "
            "may be numerically unstable. Consider regularisation (``reg``) or "
            "removing near-coincident sources.",
            RuntimeWarning,
            stacklevel=2,
        )

    # W = A B^-1, computed via a solve for numerical stability:
    #   W^T = B^-T A^T  ->  solve(B^T, A^T)
    weights = np.linalg.solve(B.T, A.T)  # (n_sources, n_channels)
    return weights


def _align_channels(info, forward, data_cov):
    """Return the common good-channel names and the data covariance over them."""
    bads = set(info["bads"]) | set(data_cov.get("bads", []))
    fwd_ch = list(forward["sol"]["row_names"])
    info_ch = list(info["ch_names"])
    cov_ch = list(data_cov.ch_names)
    fwd_set, cov_set = set(fwd_ch), set(cov_ch)

    # Preserve the channel order of ``info`` for determinism.
    common = [
        ch for ch in info_ch if ch in fwd_set and ch in cov_set and ch not in bads
    ]
    if len(common) == 0:
        raise ValueError(
            "No common good channels between info, forward and data_cov. "
            "Check that the three were computed on the same montage/sensors."
        )
    if len(common) < len(info_ch):
        dropped = len(info_ch) - len(common)
        logger.info(
            f"    Using {len(common)} channels common to info, forward and "
            f"data_cov ({dropped} dropped as missing or bad)."
        )

    cov_idx = [cov_ch.index(ch) for ch in common]
    R = np.asarray(data_cov.data, dtype=np.float64)[np.ix_(cov_idx, cov_idx)]
    return common, R


def _get_leadfield(forward, common_ch, sources, orientations):
    """Build the joint forward matrix H (n_channels x n_sources)."""
    fwd_ch = list(forward["sol"]["row_names"])
    fwd_idx = [fwd_ch.index(ch) for ch in common_ch]
    G = np.asarray(forward["sol"]["data"], dtype=np.float64)[fwd_idx]  # rows aligned
    fixed = is_fixed_orient(forward)

    if fixed:
        # One leadfield column per source; orientation is baked into the forward.
        H = G[:, sources]  # (n_channels, n_sources)
    else:
        # Free-orientation forward: 3 columns per source location. The forward
        # field of source i is h_i = L_i u_i, with L_i the (n_channels x 3)
        # leadfield block and u_i the orientation unit vector.
        H = np.empty((G.shape[0], len(sources)), dtype=np.float64)
        for col, (s, u) in enumerate(zip(sources, orientations, strict=True)):
            L = G[:, 3 * s : 3 * s + 3]  # (n_channels, 3)
            H[:, col] = L @ u
    return H


def _make_whitener(info, noise_cov, common_ch, rank):
    r"""Spatial whitener ``W`` with ``W C_n W^T = I`` over ``common_ch``.

    Built from the noise covariance via :func:`mne.cov.compute_whitener`, the
    same routine MNE's :func:`~mne.beamformer.make_lcmv` uses, so a single-source
    MCMV reproduces MNE's whitening exactly. Whitening re-expresses every channel
    in dimensionless, unit-noise units; this is what makes the beamformer valid
    when the array mixes sensor types with different physical units (e.g.
    magnetometers in T and gradiometers in T/m). In the original units the data
    covariance is dominated by whichever type has the larger numerical scale and
    its inverse is meaningless.

    The whitener is computed with ``pca=True``, which drops the null space of the
    noise covariance (its numerical rank is below the channel count after
    SSS/Maxwell filtering or ICA) and returns a possibly rectangular
    ``(n_white, n_channels)`` matrix. Crucially, the rank is resolved
    *per sensor type*: a single global eigenvalue threshold would discard the
    smaller-unit sensor type as if it were null, so MNE's per-type handling is
    required for genuinely mixed arrays -- not merely preferable.

    Parameters
    ----------
    info : instance of mne.Info
        Measurement info carrying the channel types used to scale and rank each
        sensor type.
    noise_cov : instance of mne.Covariance | None
        The noise covariance. If ``None``, an ad-hoc per-type model
        (:func:`mne.make_ad_hoc_cov`) is used; for a single sensor type that is
        a global scaling that cancels from the unit-gain filter.
    common_ch : list of str
        Channels (in order) the whitener columns must align to.
    rank : int | 'full'
        Rank handling passed through to :func:`mne.cov.compute_whitener`.
    """
    from mne import make_ad_hoc_cov
    from mne.cov import compute_whitener

    noise_used = make_ad_hoc_cov(info) if noise_cov is None else noise_cov
    picks = [info["ch_names"].index(ch) for ch in common_ch]
    whitener, wch = compute_whitener(
        noise_used, info, picks=picks, rank=rank, pca=True, verbose=False
    )
    # Guarantee the columns are in ``common_ch`` order so W lines up with R/H.
    if list(wch) != list(common_ch):
        order = [list(wch).index(ch) for ch in common_ch]
        whitener = whitener[:, order]
    return whitener


def _check_source_separation(forward, sources, min_dist=5e-3):
    """Warn if any pair of constrained sources is closer than ``min_dist`` (m)."""
    rr = forward.get("source_rr")
    if rr is None:
        return
    rr = np.asarray(rr)[sources]
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            d = np.linalg.norm(rr[i] - rr[j])
            if d < min_dist:
                warnings.warn(
                    f"Constrained sources {sources[i]} and {sources[j]} are "
                    f"{d * 1e3:.1f} mm apart (< {min_dist * 1e3:.0f} mm). MCMV "
                    "becomes ill-conditioned for near-coincident sources "
                    "(Moiseev et al., 2011); results may be unstable.",
                    RuntimeWarning,
                    stacklevel=2,
                )


@verbose
def make_mcmv(
    info,
    forward,
    data_cov,
    sources,
    *,
    noise_cov=None,
    reg=0.05,
    orientations=None,
    weight_norm="unit-gain",
    rank="full",
    verbose=None,
):
    r"""Compute a Multiple Constrained Minimum Variance (MCMV) beamformer.

    The MCMV beamformer :footcite:`Moiseev2011` reconstructs ``n`` jointly
    constrained sources with a single weight matrix :math:`\mathbf{W}` that
    satisfies, for every pair of constrained sources ``i`` and ``j`` (Eq. (3)):

    .. math::  \mathbf{w}_i^{\mathsf T}\mathbf{h}_i = 1, \qquad
               \mathbf{w}_i^{\mathsf T}\mathbf{h}_j = 0 \;\; (i \ne j).

    The zero-gain conditions make the filter insensitive to correlations between
    the constrained sources, removing the leakage and signal cancellation that
    bias single-source LCMV for correlated sources. The closed-form solution
    (Eq. (5)) is :math:`\mathbf{W} = \mathbf{R}^{-1}\mathbf{H}
    (\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H})^{-1}`, with
    :math:`\mathbf{R}` the data covariance and :math:`\mathbf{H}` the joint
    forward matrix.

    Parameters
    ----------
    info : instance of mne.Info
        The measurement info; defines the channels used.
    forward : instance of mne.Forward
        The forward solution providing the source leadfields. May be a
        fixed- or free-orientation forward (see ``orientations``).
    data_cov : instance of mne.Covariance
        The data covariance matrix :math:`\mathbf{R}`. Compute it with
        :func:`mne.compute_covariance` using any estimator/shrinkage method
        (``'empirical'``, ``'shrunk'``, ``'ledoit_wolf'``, ``'oas'``, ...); the
        choice is inherited here unchanged.
    sources : array-like of int, shape (n_sources,)
        Indices (into the in-use source space of ``forward``) of the sources to
        constrain jointly. Must be unique. This sets the beamformer order ``n``.
    noise_cov : instance of mne.Covariance | None
        The noise covariance :math:`\mathbf{C}_n`. Used to whiten the leadfield
        and data covariance before the solve, which is what makes the filter
        valid for arrays that mix sensor types (e.g. magnetometers and
        gradiometers). If ``None``, an ad-hoc per-sensor-type noise model is
        used (:func:`mne.make_ad_hoc_cov`); for a single sensor type this is a
        global scaling that leaves the unit-gain filter unchanged.
    reg : float
        Diagonal-loading regularisation of ``data_cov`` as a fraction of the
        mean eigenvalue, applied via MNE's :func:`~mne.beamformer` inversion
        machinery. ``reg=0`` uses the unregularised inverse of Eq. (5). This is
        a deliberate, documented departure from the bare equation -- never a
        silent fix -- and defaults to ``0.05`` to match ``make_lcmv``.
    orientations : ndarray, shape (n_sources, 3) | None
        Source orientation unit vectors, required when ``forward`` has free
        orientation and ignored (must be ``None``) when it is fixed-orientation.
        Data-driven orientation estimation (the generalised-eigenvalue
        localiser of Moiseev et al., 2011) is provided by a separate function.
    weight_norm : 'unit-gain' | 'unit-noise-gain' | None
        Output normalisation. ``'unit-gain'`` (and ``None``) returns the literal
        Eq. (5) filter (the unit-gain / zero-gain constraint
        :math:`\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}` holds on the raw
        leadfield). ``'unit-noise-gain'`` rescales each filter so that
        :math:`\mathbf{w}_i^{\mathsf T}\mathbf{C}_n\mathbf{w}_i = 1`
        :footcite:`SekiharaNagarajan2008`; because the solve is performed in the
        whitened space this is exactly unit Euclidean norm of the whitened
        filter, matching MNE's definition.
    rank : int | 'full'
        Rank handling, applied to both the noise-covariance whitener
        (:func:`mne.cov.compute_whitener`) and the data-covariance inverse. Use
        an integer for rank-reduced data (e.g. after SSP/ICA); ``'full'`` assumes
        full rank. For genuinely rank-deficient data prefer an explicit integer.
    %(verbose)s

    Returns
    -------
    filters : instance of MCMVBeamformer
        The MCMV spatial filter and metadata. The weights are under
        ``filters['weights']`` with shape ``(n_sources, n_channels)``.

    Raises
    ------
    ValueError
        If inputs are inconsistent or mathematically invalid: non-unique or
        out-of-range ``sources``; missing/forbidden ``orientations`` for the
        forward type; no common channels; non-finite covariance; a beamformer
        order exceeding the data rank.
    RuntimeError
        If the constrained-source system is singular (see
        :func:`_compute_mcmv_weights`).

    Notes
    -----
    With a single source (``n == 1``) and ``weight_norm`` in
    ``{'unit-gain', None}``, this reduces *exactly* to the unit-gain LCMV filter
    :math:`\mathbf{w} = \mathbf{R}^{-1}\mathbf{h} /
    (\mathbf{h}^{\mathsf T}\mathbf{R}^{-1}\mathbf{h})`.

    References
    ----------
    .. footbibliography::
    """
    # -- validate simple argument types/values ------------------------------ #
    _validate_type(data_cov, Covariance, "data_cov")
    if noise_cov is not None:
        _validate_type(noise_cov, Covariance, "noise_cov")
    _check_option("weight_norm", weight_norm, _ALLOWED_WEIGHT_NORM)
    if not (0.0 <= float(reg) <= 1.0):
        raise ValueError(f"reg must be in [0, 1], got {reg}.")

    sources = np.atleast_1d(np.asarray(sources, dtype=int)).ravel()
    if sources.size == 0:
        raise ValueError("`sources` must contain at least one source index.")
    if np.unique(sources).size != sources.size:
        raise ValueError(
            "`sources` must be unique; duplicate indices request coincident "
            "constraints, which makes the MCMV system singular."
        )
    n_src_space = int(forward["nsource"])
    if sources.min() < 0 or sources.max() >= n_src_space:
        raise ValueError(
            f"`sources` indices must be in [0, {n_src_space}); got "
            f"[{sources.min()}, {sources.max()}]."
        )

    # -- orientations vs forward type --------------------------------------- #
    fixed = is_fixed_orient(forward)
    if fixed:
        if orientations is not None:
            raise ValueError(
                "`orientations` must be None for a fixed-orientation forward; "
                "the orientation is already defined by the forward solution."
            )
    else:
        if orientations is None:
            raise ValueError(
                "`orientations` is required for a free-orientation forward. "
                "Provide an (n_sources, 3) array of unit vectors, or use the "
                "MCMV localiser to estimate orientations from the data."
            )
        orientations = np.asarray(orientations, dtype=np.float64)
        if orientations.shape != (sources.size, 3):
            raise ValueError(
                f"`orientations` must have shape ({sources.size}, 3), got "
                f"{orientations.shape}."
            )
        norms = np.linalg.norm(orientations, axis=1)
        if np.any(norms == 0):
            raise ValueError("`orientations` contains a zero vector.")
        if not np.allclose(norms, 1.0, atol=1e-6):
            warnings.warn(
                "`orientations` were not unit vectors; normalising them.",
                RuntimeWarning,
                stacklevel=2,
            )
            orientations = orientations / norms[:, None]

    # -- assemble R and H in a common, consistent channel space ------------- #
    common_ch, R = _align_channels(info, forward, data_cov)
    if not np.all(np.isfinite(R)):
        raise ValueError("data_cov contains non-finite values.")

    # If a noise covariance is supplied it must also cover the channels we use,
    # so intersect with it before building anything (the no-noise-cov path uses
    # an ad-hoc model defined on all of ``info`` and needs no intersection).
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

    n = sources.size
    if n > len(common_ch):
        raise ValueError(
            f"Beamformer order n={n} exceeds the number of channels "
            f"({len(common_ch)}); the constraint system is overdetermined."
        )

    H = _get_leadfield(forward, common_ch, sources, orientations)
    if not np.all(np.isfinite(H)):
        raise ValueError("The forward solution contains non-finite values.")
    _check_source_separation(forward, sources)

    # -- whiten so the beamformer is valid for any sensor configuration ----- #
    # The data covariance mixes sensor types with different physical units, so
    # we move into the noise-covariance-whitened space where every channel is
    # dimensionless with unit noise variance. There the types are commensurable
    # and the data covariance is well conditioned. With no noise covariance we
    # fall back to MNE's ad-hoc per-type model; for a single sensor type (e.g. a
    # CTF gradiometer array) that model is a global scalar that cancels from the
    # unit-gain filter, so single-type results are unchanged.
    adhoc_noise = noise_cov is None
    whitener = _make_whitener(info, noise_cov, common_ch, rank)
    H_w = whitener @ H  # whitened leadfield (n_white, n_sources)
    R_w = whitener @ R @ whitener.T  # whitened data covariance (n_white, n_white)
    # The congruence above is symmetric in exact arithmetic; enforce it against
    # floating-point drift, which is large for the wide dynamic range of a real
    # magnetometer+gradiometer covariance and would otherwise fail the strict
    # Hermitian check in ``_reg_pinv``.
    R_w = (R_w + R_w.T) / 2.0
    n_white = R_w.shape[0]

    # -- regularised inverse of the (whitened) data covariance -------------- #
    # Diagonal-loading convention identical to MNE's make_lcmv (_reg_pinv).
    Rinv_w, loading, rnk = _reg_pinv(R_w, reg=reg, rank=rank)
    if rnk < n_white:
        warnings.warn(
            f"data_cov is rank-deficient (rank {rnk} < {n_white} whitened "
            "dimensions); a pseudo-inverse was used. Supply a shrinkage-"
            "regularised covariance (e.g. method='oas'/'shrunk' in "
            "mne.compute_covariance) or set ``reg`` > 0 for a stable inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
    if n > rnk:
        raise ValueError(
            f"Beamformer order n={n} exceeds the data-covariance rank "
            f"({rnk}); fewer than n independent spatial dimensions are "
            "available to satisfy the constraints."
        )

    # -- the MCMV solve (Eq. (5)) in the whitened space --------------------- #
    weights_w = _compute_mcmv_weights(H_w, Rinv_w)  # (n_sources, n_white)

    # -- weight normalisation ----------------------------------------------- #
    # In whitened space the noise covariance is the identity, so unit-noise-gain
    # is exactly unit Euclidean norm of each whitened filter
    # :footcite:`SekiharaNagarajan2008` -- this matches MNE's definition.
    if weight_norm == "unit-noise-gain":
        if adhoc_noise:
            warnings.warn(
                "weight_norm='unit-noise-gain' without a noise_cov normalises "
                "against MNE's ad-hoc noise model; supply a measured noise_cov "
                "for a data-accurate unit-noise-gain.",
                RuntimeWarning,
                stacklevel=2,
            )
        weights_w = weights_w / np.linalg.norm(weights_w, axis=1, keepdims=True)

    # -- fold the whitener back so the filters act on raw sensor data ------- #
    # s_hat = weights_w (W x) = (weights_w W) x, and (weights_w W) H = I, so the
    # unit-gain / zero-gain constraint is preserved in the original sensor space
    # and apply_mcmv can be applied directly to unwhitened data.
    weights = weights_w @ whitener  # (n_sources, n_channels)

    logger.info(f"    Computed MCMV beamformer for {n} source(s).")
    return MCMVBeamformer(
        kind="MCMV",
        weights=weights,
        ch_names=common_ch,
        n_sources=n,
        sources=sources,
        orientations=None if fixed else orientations,
        is_fixed_orient=fixed,
        weight_norm=weight_norm,
        reg=float(reg),
        loading_factor=float(loading),
        rank=int(rnk),
        leadfield=H,
    )


def _pick_data(data, ch_names):
    """Return a (n_channels, n_times) array for the filter's channels."""
    # Accept either an MNE object exposing .get_data()/.ch_names or a raw array.
    if hasattr(data, "ch_names") and hasattr(data, "get_data"):
        idx = [data.ch_names.index(ch) for ch in ch_names]
        arr = np.asarray(data.get_data())
        return arr[..., idx, :]
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape[-2] != len(ch_names):
        raise ValueError(
            f"Array has {arr.shape[-2]} channels but the filter expects "
            f"{len(ch_names)}. Pass an MNE object or a correctly ordered array."
        )
    return arr


@verbose
def apply_mcmv(data, filters, *, verbose=None):
    r"""Apply MCMV spatial filters to sensor data.

    Computes the reconstructed source amplitudes
    :math:`\hat{s}_i(t) = \mathbf{w}_i^{\mathsf T}\mathbf{b}(t)` (Eq. (2)).

    Parameters
    ----------
    data : instance of mne.Evoked | mne.Epochs | mne.io.Raw | ndarray
        The sensor data. An ndarray must have shape ``(..., n_channels,
        n_times)`` with channels ordered as ``filters['ch_names']``.
    filters : instance of MCMVBeamformer
        The MCMV filters from :func:`make_mcmv`.
    %(verbose)s

    Returns
    -------
    source_data : ndarray, shape (..., n_sources, n_times)
        The reconstructed time course of each constrained source.
    """
    _validate_type(filters, MCMVBeamformer, "filters")
    W = filters["weights"]  # (n_sources, n_channels)
    b = _pick_data(data, filters["ch_names"])  # (..., n_channels, n_times)
    # s = W b, broadcast over any leading (e.g. epochs) dimension.
    return np.einsum("sc,...ct->...st", W, b)


def apply_mcmv_cov(data_cov, filters):
    r"""Reconstruct the source covariance from a sensor covariance.

    Returns :math:`\mathbf{W}^{\mathsf T}\mathbf{R}\mathbf{W}`, the
    ``n_sources x n_sources`` covariance of the reconstructed sources. Its
    diagonal gives the source powers and its off-diagonal entries the (leakage-
    suppressed) cross-source terms used in connectivity analyses.

    Parameters
    ----------
    data_cov : instance of mne.Covariance
        A sensor covariance over the same channels as ``filters``.
    filters : instance of MCMVBeamformer
        The MCMV filters from :func:`make_mcmv`.

    Returns
    -------
    source_cov : ndarray, shape (n_sources, n_sources)
        The reconstructed source covariance.
    """
    _validate_type(data_cov, Covariance, "data_cov")
    _validate_type(filters, MCMVBeamformer, "filters")
    ch = filters["ch_names"]
    cov_ch = list(data_cov.ch_names)
    idx = [cov_ch.index(c) for c in ch]
    R = np.asarray(data_cov.data)[np.ix_(idx, idx)]
    W = filters["weights"]  # (n_sources, n_channels)
    return W @ R @ W.T
