"""Multiple Constrained Minimum Variance (MCMV) beamformer.

This module implements the multi-source MCMV spatial filter for MEG/EEG source
reconstruction, exactly following the formulation derived by
Moiseev et al. (2011) :footcite:`Moiseev2011`. The MCMV beamformer generalises
the single-source Linearly Constrained Minimum Variance (LCMV) beamformer
:footcite:`VanVeenEtAl1997` to ``n`` sources that are constrained *jointly*.
LCMV is itself rooted in the linearly constrained adaptive array of Frost
(1972) :footcite:`Frost1972`. By imposing unit gain on each source's own forward
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

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import warnings
from copy import deepcopy

import numpy as np
from mne import Covariance

# NOTE: we use the stdlib ``warnings.warn`` (RuntimeWarning) rather than
# ``mne.utils.warn`` so that warnings are emitted regardless of MNE's logging
# level and are reliably catchable by ``pytest.warns``. When upstreaming into
# MNE-Python these can be swapped for ``mne.utils.warn``, whose test harness
# handles the stack-rewriting behaviour.
# MNE's regularised (pseudo-)inverse of a covariance matrix. We re-use it, with the
# ``reg`` rescaling applied in ``make_mcmv`` to compensate for our ``pca=True``
# whitener. The diagonal-loading convention is therefore identical to
# ``mne.beamformer.make_lcmv``'s, and the n == 1 MCMV filter reduces to the
# corresponding LCMV filter to machine precision on a well-conditioned data
# covariance (measured: 6e-15 relative, unit-gain and unit-noise-gain alike).
#
# The rank handling is deliberately *not* identical. ``make_lcmv`` passes
# ``sum(compute_rank(data_cov, rank=rank, info=info).values())`` to ``_reg_pinv``
# as an explicit integer; ``_split_rank`` below passes ``None`` unless the caller
# said ``'full'``, which makes ``_reg_pinv`` estimate the rank numerically from
# the unregularised whitened covariance. The two coincide whenever the data
# covariance's only deficiency is the one the whitener has already removed (SSP,
# average reference, SSS) -- the ordinary case. They diverge on a sample-starved
# covariance, where this package truncates to the smaller numerical rank while
# ``make_lcmv`` inverts the diagonally-loaded near-null space: on a 100-sample
# covariance the two filters differ by more than their own norm. Truncating is
# the safer choice, but it is a choice, so do not read the reduction above as
# holding on a rank-deficient covariance.
#
# Every covariance estimator/shrinkage method available through
# ``mne.compute_covariance`` is inherited unchanged.
from mne.beamformer._compute_beamformer import _reg_pinv
from mne.forward import is_fixed_orient
from mne.utils import _check_option, _validate_type, logger, verbose

_ALLOWED_WEIGHT_NORM = ("unit-gain", "array-gain", "unit-noise-gain", None)

# Smallest whitened leadfield norm a constrained source may have. Whitening is
# what makes an absolute threshold possible at all: every whitened channel
# carries unit noise variance, so the norm of a whitened leadfield column is the
# field a unit-strength source produces measured in noise standard deviations, a
# dimensionless number that does not move with the units the array happens to be
# recorded in. A leadfield in SI units against a measured noise covariance lands
# at 1e8 and above, so the threshold sits fourteen orders of magnitude below
# anything real and is reached only by a column that has collapsed: a source at
# the centre of a spherical model, a radial source seen by MEG alone, a location
# whose sensors were all dropped. Such a source is not weak but invisible, and
# the only filters satisfying its unit-gain constraint are enormous ones whose
# output is amplified noise carrying a source's name.
_MIN_WHITENED_GAIN = 1e-6


def _cov_as_matrix(cov, ch_names):
    """Return the dense covariance over ``ch_names`` as a square ndarray.

    A :class:`mne.Covariance` may store a *diagonal* covariance as a 1-D array
    (``cov['diag'] is True``). This is what :func:`mne.make_ad_hoc_cov` returns
    and what MNE writes for diagonal covariance files. ``Covariance._get_square``
    is the same accessor :func:`mne.beamformer.make_lcmv` uses, so diagonal and
    full covariances are handled identically here and in MNE.
    """
    cov_ch = list(cov.ch_names)
    idx = [cov_ch.index(ch) for ch in ch_names]
    return np.asarray(cov._get_square(), dtype=np.float64)[np.ix_(idx, idx)]


def _split_rank(rank):
    """Split ``rank`` into the whitener and the covariance-inverse conventions.

    :func:`mne.cov.compute_whitener` accepts ``None | 'full' | 'info' | dict``
    whereas :func:`mne.beamformer._compute_beamformer._reg_pinv` accepts only
    ``None | 'full' | int``. Passing one convention to the other raises
    ``TypeError``, so the caller-facing ``rank`` (MNE's convention) is translated
    here: the whitener receives it unchanged, and the covariance inverse receives
    ``'full'`` only when the caller explicitly asked for it, else ``None``, which
    makes ``_reg_pinv`` estimate the rank numerically from the unregularised
    whitened covariance. Note that ``make_lcmv`` does *not* do this -- it passes
    an explicit integer from :func:`mne.compute_rank`. See the module header for
    when the two agree and when they do not.
    """
    _validate_type(rank, (None, str, dict), "rank")
    if isinstance(rank, str) and rank not in ("full", "info"):
        raise ValueError(
            f"rank must be None, 'full', 'info' or a dict of per-channel-type "
            f"ranks, got {rank!r}. An explicit integer is not supported; use a "
            "dict such as {'meg': 60} to set the rank of a sensor type."
        )
    return rank, ("full" if rank == "full" else None)


class MCMVBeamformer(dict):
    """Container for MCMV spatial-filter weights.

    This is a thin ``dict`` subclass shaped after
    :class:`mne.beamformer.Beamformer`, so that upstreaming the algorithm into
    MNE-Python is a small step. It stores, among other metadata, the
    spatial-filter weight matrix under the key ``'weights'`` with shape
    ``(n_sources, n_channels)``, one row holding the filter for one constrained
    source.

    It is not a drop-in replacement for MNE's container, and the two are not
    interchangeable in either direction. MNE stores its weights in the
    *whitened* space and :func:`mne.beamformer.apply_lcmv` whitens the data
    before applying them; the weights here are in the original sensor space and
    :func:`~advance_beamlab.apply_mcmv` multiplies the data as it stands. The
    two differ by exactly the whitener, so swapping one container for the other
    would change the answer silently rather than raise. This container also
    carries only the keys MCMV needs, not MNE's full set.
    """

    def copy(self):
        """Return a deep copy of the beamformer.

        Overrides ``dict.copy`` (which is shallow) so that the semantics match
        :meth:`mne.beamformer.Beamformer.copy`: mutating the copy's arrays
        (e.g. ``filters.copy()['weights'] *= 2``) must not corrupt the original.

        Returns
        -------
        filters : instance of MCMVBeamformer
            A deep copy of this beamformer.
        """
        return deepcopy(self)

    def __repr__(self):  # noqa: D105
        n = self.get("n_sources", "?")
        norm = self.get("weight_norm", "?")
        return f"<MCMVBeamformer | {n} source(s), weight_norm={norm!r}>"


# --------------------------------------------------------------------------- #
# Numerical core (pure NumPy, no MNE objects). Unit-testable in isolation.
# --------------------------------------------------------------------------- #
def _compute_mcmv_weights(
    leadfield, cov_inv, *, cond_warn=1e8, gain_tol=_MIN_WHITENED_GAIN
):
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
        Both callers hand it over in the noise-whitened space, which is what
        gives ``gain_tol`` an absolute scale to be measured against.
    cov_inv : ndarray, shape (n_channels, n_channels)
        The (regularised) inverse data covariance :math:`\mathbf{R}^{-1}`.
    cond_warn : float
        Condition-number threshold of :math:`\mathbf{H}^{\mathsf T}
        \mathbf{R}^{-1}\mathbf{H}` above which a numerical-stability warning is
        emitted.
    gain_tol : float
        Smallest whitened leadfield norm a constrained source may have. At or
        below it the source is taken to be invisible to the array rather than
        merely weak, and the constraint is refused (see the comment in the body
        for why the condition number cannot see this).

    Returns
    -------
    weights : ndarray, shape (n_sources, n_channels)
        The MCMV spatial filters, one per constrained source (row-wise).

    Raises
    ------
    ValueError
        If a constrained source's whitened leadfield is numerically silent, so
        that no filter can estimate it.
    RuntimeError
        If :math:`\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}` is singular.
        This happens when the constrained sources are (numerically) collinear,
        for instance at coincident locations with identical orientation, or when
        the beamformer order exceeds the rank of the data.
    """
    H = np.asarray(leadfield, dtype=np.float64)
    Rinv = np.asarray(cov_inv, dtype=np.float64)

    # An absolute-scale guard on the constraints, which the condition number
    # below cannot supply: cond(B) is invariant to the scale of H, so scaling a
    # column into numerical silence leaves it at 1 when n == 1, and scaling every
    # column leaves it unchanged for any n. A silent column is not an exotic
    # input: in a spherical model the MEG leadfield vanishes exactly at the
    # centre and, to round-off, for a radial source, and a channel selection can
    # drop the only sensors that see a source. Left alone the solve returns
    # weights of order 1/||h|| and says nothing (see ``_MIN_WHITENED_GAIN``).
    gains = np.linalg.norm(H, axis=0)
    silent = np.nonzero(gains <= gain_tol)[0]
    if silent.size:
        raise ValueError(
            f"Constrained source(s) {silent.tolist()} (positions in the "
            f"requested set) have a numerically silent leadfield: whitened norm "
            f"{gains[silent].min():.3g} <= {gain_tol:g}. The array cannot see "
            "them, so the requested filter would be pure noise amplification. "
            "Check the source location and orientation -- a radial source, or "
            "one at the centre of a spherical model, is invisible to MEG -- and "
            "that the channels which do see it were not dropped as bad or "
            "excluded by the noise covariance."
        )

    # A = R^-1 H  (n_channels x n_sources)
    A = Rinv @ H
    # B = H^T R^-1 H  (n_sources x n_sources), the matrix that is inverted in
    # Eq. (5); also equals S of Table 2 when R is the data covariance.
    B = H.T @ A

    # Guard the inversion of B explicitly so we can raise an informative error
    # rather than letting a silent near-singular inverse corrupt the weights.
    cond = np.linalg.cond(B)
    if not np.isfinite(cond) or cond > 1.0 / np.finfo(np.float64).eps:
        raise RuntimeError(
            f"H^T R^-1 H is singular (condition number {cond:.3g}): the "
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

    R = _cov_as_matrix(data_cov, common)
    return common, R


def _get_leadfield(forward, common_ch, sources, orientations):
    """Build the joint forward matrix H (n_channels x n_sources).

    ``orientations`` are always interpreted in **head coordinates**. For a
    ``surf_ori=True`` free-orientation forward the three gain columns of a source
    are expressed in that source's *local surface* frame (two tangential
    directions and the normal) rather than in head x/y/z, so the orientation is
    rotated into the local frame before it is applied. Without this the same
    ``orientations`` array would mean two different physical dipoles depending on
    which representation of the same forward the caller happened to pass. The
    most natural user action, taking the cortical normal from
    ``forward['source_nn']`` (which is in head coordinates), would be the one
    that breaks.
    """
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
        surf_ori = bool(forward.get("surf_ori", False))
        nn = np.asarray(forward["source_nn"], dtype=np.float64)
        H = np.empty((G.shape[0], len(sources)), dtype=np.float64)
        for col, (s, u) in enumerate(zip(sources, orientations, strict=True)):
            L = G[:, 3 * s : 3 * s + 3]  # (n_channels, 3)
            if surf_ori:
                # ``source_nn`` holds, per source, the 3x3 rotation whose rows are
                # the local frame axes expressed in head coordinates. Its action
                # on a head-coordinate vector gives the local components.
                u = nn[3 * s : 3 * s + 3] @ u
            H[:, col] = L @ u
    return H


def _intersect_noise_cov(common_ch, cov_matrix, noise_cov):
    """Restrict ``common_ch``/``cov_matrix`` to the channels ``noise_cov`` covers.

    A MEG-only empty-room noise covariance combined with a MEG+EEG forward is
    standard practice, so the noise covariance is allowed to cover a subset of
    the channels; everything it does not cover is dropped, exactly as
    :func:`make_mcmv` has always done. Shared with the localizer scan and the
    ReciPSIICOS entry points so that all algorithms select channels identically.
    """
    if noise_cov is None:
        return common_ch, cov_matrix
    ncov_set = set(noise_cov.ch_names)
    keep = [i for i, ch in enumerate(common_ch) if ch in ncov_set]
    if len(keep) == 0:
        raise ValueError("noise_cov shares no channels with info/forward/data_cov.")
    if len(keep) < len(common_ch):
        logger.info(
            f"    Dropping {len(common_ch) - len(keep)} channel(s) not covered by "
            "noise_cov."
        )
        common_ch = [common_ch[i] for i in keep]
        cov_matrix = cov_matrix[np.ix_(keep, keep)]
    return common_ch, cov_matrix


def _check_eeg_reference(info, common_ch):
    """Require an average EEG reference projector, as MNE's inverses do.

    Beamforming EEG without an average reference is not well posed: the forward
    model is computed against an average reference, so a differently referenced
    recording is modelled with the wrong topographies.
    :func:`mne.beamformer.make_lcmv` raises for this (via ``_check_reference``),
    and so do we: the messages below are MNE's verbatim, so the contract is
    identical.
    """
    from mne import pick_info
    from mne._fiff.pick import _electrode_types
    from mne._fiff.proj import _needs_eeg_average_ref_proj

    picks = [info["ch_names"].index(ch) for ch in common_ch]
    info_pick = pick_info(info, picks, verbose=False)
    if _needs_eeg_average_ref_proj(info_pick):
        raise ValueError(
            "EEG average reference (using a projector) is mandatory for "
            "modeling, use the method set_eeg_reference(projection=True)"
        )
    if _electrode_types(info_pick) and info_pick.get("custom_ref_applied", False):
        raise ValueError("Custom EEG reference is not allowed for inverse modeling.")


def _check_noise_cov_required(info, common_ch, noise_cov):
    """Raise if several sensor types are present without a noise covariance.

    The ad-hoc fallback is only defensible for a single sensor type, where it is
    a global scaling that cancels from the unit-gain filter. With more than one
    type its arbitrary per-type variances set the relative weighting of
    magnetometers, gradiometers and EEG and the beamformer localises elsewhere.
    :func:`mne.beamformer.make_lcmv` refuses this case, and the error message
    below is MNE's verbatim so the contract is identical.
    """
    if noise_cov is not None:
        return
    from mne import pick_info
    from mne._fiff.pick import _contains_ch_type

    picks = [info["ch_names"].index(ch) for ch in common_ch]
    info_pick = pick_info(info, picks, verbose=False)
    n_types = sum(_contains_ch_type(info_pick, tt) for tt in ("mag", "grad", "eeg"))
    if n_types > 1:
        raise ValueError(
            "Source reconstruction with several sensor types"
            " requires a noise covariance matrix to be "
            "able to apply whitening."
        )


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
    smaller-unit sensor type as if it were null, so for genuinely mixed arrays
    MNE's per-type handling is not merely preferable but required.

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
    rank : None | 'full' | 'info' | dict
        Rank handling passed through to :func:`mne.cov.compute_whitener`, in
        MNE's own convention (the whitener half of what ``_split_rank``
        returns). A bare integer is deliberately absent: ``compute_whitener``
        does not accept one, because the rank has to be resolved per sensor
        type.
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
    rank=None,
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
        mean eigenvalue, applied via the same :mod:`mne.beamformer` inversion
        machinery. ``reg=0`` uses the unregularised inverse of Eq. (5). This is
        a deliberate, documented departure from the bare equation, never a
        silent fix. It defaults to ``0.05`` to match ``make_lcmv``.
    orientations : ndarray, shape (n_sources, 3) | None
        Source orientation unit vectors **in head coordinates**, required when
        ``forward`` has free orientation and ignored (must be ``None``) when it is
        fixed-orientation. A ``surf_ori=True`` forward stores its gain columns in
        each source's local surface frame; the rotation is applied internally, so
        the same head-coordinate vectors give the same physical dipole for either
        representation of a forward. Take them from the *source space* --
        ``forward['src'][hemi]['nn'][vertno]`` -- or from
        :func:`~advance_beamlab.optimal_orientation`. Do **not** reach for
        ``forward['source_nn'][2::3]``: that holds cortical normals only when
        ``forward['surf_ori']`` is ``True``, and both
        :func:`mne.read_forward_solution` and
        :func:`mne.make_forward_solution` hand back ``surf_ori=False``, where
        ``source_nn`` is a tiled identity and ``[2::3]`` is the head +z axis for
        every source. It is a unit vector of the right shape, so it is accepted
        without complaint, and every source is then constrained to a dipole
        pointing at the top of the head. Data-driven orientation
        estimation (the generalised-eigenvalue localiser of Moiseev et al., 2011)
        is provided by :func:`~advance_beamlab.scan_mcmv`.
    weight_norm : 'unit-gain' | 'array-gain' | 'unit-noise-gain' | None
        Output normalisation. ``'unit-gain'`` (and ``None``) returns the literal
        Eq. (5) filter (the unit-gain / zero-gain constraint
        :math:`\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}` holds on the raw
        leadfield). ``'array-gain'`` imposes
        :math:`\mathbf{w}_i^{\mathsf T}\mathbf{h}_i =
        \lVert\tilde{\mathbf{h}}_i\rVert` instead, with
        :math:`\tilde{\mathbf{h}}_i` the *whitened* leadfield, by normalising
        each leadfield before the constraint: the output is the source's field
        across the array in units of the noise rather than a dipole moment, and
        it does not inherit unit-gain's bias towards deep sources, which arises
        because a deep leadfield is small and the filter must amplify to reach
        unit gain :footcite:`SekiharaNagarajan2008`. The norm has to be the
        whitened one. Summed in the recorded units, squared leadfield entries
        add volts to teslas to teslas per metre whenever the array mixes sensor
        types, so the normalisation is not a physical quantity at all: it is
        fixed by whichever type carries the larger numbers -- gradiometers over
        magnetometers, EEG over MEG by orders of magnitude -- and it moves if
        the same recording is expressed in T/cm instead of T/m. Whitening puts
        every channel on one dimensionless unit-noise scale, the only scale on
        which the types are commensurable, which is why MNE's own depth
        weighting offers ``limit_depth_chs='whiten'`` and advises against the
        raw-unit norm (:func:`mne.forward.compute_depth_prior`). The price is
        that the amplitudes carry the noise level with them, so they are
        comparable across analyses only while the noise covariance is. It
        remains the depth-neutral choice when no noise covariance was measured:
        the ad-hoc model then merely sets that scale, and for a single sensor
        type it is one number common to every source.
        ``'unit-noise-gain'`` rescales each filter to unit Euclidean
        norm *in the whitened space* :footcite:`SekiharaNagarajan2008`, which is
        MNE's definition and is used by :func:`mne.beamformer.make_lcmv`. That
        equals :math:`\mathbf{w}_i^{\mathsf T}\mathbf{C}_n\mathbf{w}_i = 1`
        exactly when the whitener is a single block. :func:`mne.cov.compute_whitener`
        eigendecomposes the noise covariance separately per sensor-type group
        (MEG together, EEG separately), so for a combined MEG+EEG array the
        whitened noise covariance is the identity only up to its cross-type
        blocks and the realised noise gain departs from 1 (by of order one tenth
        in power on the MNE ``sample`` data). This matches ``make_lcmv`` exactly.
    rank : None | 'full' | 'info' | dict
        Rank handling, applied to both the noise-covariance whitener
        (:func:`mne.cov.compute_whitener`) and the data-covariance inverse, using
        MNE's own convention. The default ``None`` auto-detects the rank per
        sensor type and drops the null space. That is the correct choice for
        rank-deficient data (e.g. after SSP/ICA), where ``'full'`` would instead
        invert the near-null directions and give an unstable whitener. Use a dict
        such as ``{'meg': 60}`` to set the rank of a sensor type explicitly; a
        bare integer is not accepted, because the rank must be resolved per
        sensor type. Note :func:`mne.beamformer.make_lcmv` defaults to ``'info'``
        rather than ``None``.
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
        order exceeding the data rank; a constrained source whose whitened
        leadfield is numerically silent, so that the array cannot see it.
    RuntimeError
        If :math:`\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}` is singular:
        the constrained sources are numerically collinear (coincident location
        and orientation), or the beamformer order exceeds the rank of the data.

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
    common_ch, R = _intersect_noise_cov(common_ch, R, noise_cov)
    _check_noise_cov_required(info, common_ch, noise_cov)
    _check_eeg_reference(info, common_ch)

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
    whitener_rank, pinv_rank = _split_rank(rank)
    whitener = _make_whitener(info, noise_cov, common_ch, whitener_rank)
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
    #
    # ``_reg_pinv`` loads by ``reg * mean(singular values of the matrix it is
    # given``. ``make_lcmv`` whitens with ``pca=False``, so its whitened
    # covariance is ``n_channels x n_channels`` of rank ``n_white`` and the mean
    # is taken over ``n_channels`` values, including the ``n_channels -
    # n_white`` structural zeros. We whiten with ``pca=True`` (which is what lets
    # a rank-deficient noise covariance be handled per sensor type), so our
    # matrix is ``n_white x n_white`` and the mean is over ``n_white`` values.
    # Without the rescaling below our absolute loading would be larger by exactly
    # ``n_channels / n_white`` and the n == 1 filter would *not* reduce to
    # ``make_lcmv`` on rank-deficient data (SSP, average reference, SSS/ICA).
    reg_eff = float(reg) * n_white / len(common_ch)
    Rinv_w, loading, rnk = _reg_pinv(R_w, reg=reg_eff, rank=pinv_rank)
    # ``_reg_pinv`` returns the rank it measured *before* diagonal loading, but
    # with ``rank='full'`` it masks with the rank *after* loading -- which is
    # full, because the loading lifts the null space. So with ``rank='full'`` and
    # ``reg > 0`` nothing is truncated and every whitened direction is inverted,
    # even though the returned number is the smaller pre-loading rank. Reporting
    # that number would warn about a pseudo-inverse that was not taken and could
    # refuse an order the inverse can support.
    if pinv_rank == "full" and reg_eff > 0:
        rnk = n_white
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
    # :footcite:`SekiharaNagarajan2008`. This matches MNE's definition.
    if weight_norm == "array-gain":
        # w'l = ||l||, i.e. the leadfield is normalised before the constraint is
        # imposed. Unit-gain returns a dipole moment, which is biased towards
        # deep sources because a deep leadfield is small and the filter has to
        # amplify to reach unit gain; array-gain divides that amplification back
        # out :footcite:`SekiharaNagarajan2008`.
        #
        # The norm is taken on the *whitened* leadfield. A norm of the raw one
        # sums squared entries across the whole array, and as soon as the array
        # mixes sensor types those entries carry different units: the sum is
        # V^2 + T^2 + (T/m)^2, which is not a physical quantity, and it is
        # dominated by whichever type happens to have the larger numbers. On
        # MNE's ``sample`` MEG+EEG array the 59 electrodes account for very
        # nearly all of the raw norm and the 305 MEG channels for almost none of
        # it, so what the caller asked to be a depth normalisation for the array
        # is one for a single sensor type -- and it would change if the same
        # recording were expressed in T/cm rather than T/m. Nor is the
        # difference a common rescaling that would cancel from comparisons: the
        # raw and whitened norms disagree by up to a factor of twenty across
        # that source space, which moves sources relative to one another.
        # Whitening re-expresses every channel in unit-noise units, the one
        # scale on which the types are commensurable, and the normalisation it
        # gives is invariant to those unit conventions. It is the same reasoning
        # behind MNE's ``limit_depth_chs='whiten'`` depth weighting, whose
        # documentation warns that the raw units bias the weighting towards
        # whichever channel type has the largest values in SI. The cost, stated
        # plainly in the parameter documentation, is that the amplitudes are in
        # units of the noise rather than of the array.
        scale = np.linalg.norm(H_w, axis=0)  # one norm per constrained source
        weights_w = weights_w * scale[:, None]

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
    # s_hat = weights_w (W x) = (weights_w W) x, so the constraint the whitened
    # solve imposed is preserved in the original sensor space and apply_mcmv can
    # be applied directly to unwhitened data. The zero-gain half, (w_i^T g_j) = 0
    # for i != j, holds for every weight_norm. The unit-gain half holds only for
    # weight_norm='unit-gain'; 'array-gain' and 'unit-noise-gain' have rescaled
    # each row above, so their diagonal is that row's scale factor rather than 1.
    weights = weights_w @ whitener  # (n_sources, n_channels)

    # Record the projectors the filters were built under, exactly as make_lcmv
    # and make_recipsiicos_lcmv do. ``compute_whitener`` has already folded them
    # into the whitener, so applying the filter must not re-apply them; what
    # storing them buys is MNE's safety check that the data a filter is used on
    # carries the same projectors it was computed with. Without it, adding SSP
    # to the data after building the filter changes the source estimate silently.
    from mne._fiff.proj import make_projector

    proj, _, _ = make_projector(info["projs"], list(common_ch))

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
        proj=proj,
        is_ssp=bool(info["projs"]),
    )


def _check_proj_match(data, filters):
    """Refuse data whose projectors differ from the ones the filter was built on.

    The whitener folded ``info['projs']`` into the weights, so a filter is only
    valid for data carrying those same projectors. Applying it to data that has
    since had, say, EOG or ECG SSP applied changes the answer by tens of per
    cent with nothing to show for it. :func:`mne.beamformer.apply_lcmv` refuses
    that case, and the message below is MNE's verbatim so the contract matches.
    """
    if filters.get("proj") is None:  # a filter from before this was recorded
        return
    from mne._fiff.proj import make_projector

    proj_data, _, _ = make_projector(data.info["projs"], list(filters["ch_names"]))
    if not np.allclose(
        proj_data, filters["proj"], atol=np.finfo(float).eps, rtol=1e-13
    ):
        raise ValueError(
            "The SSP projections present in the data "
            "do not match the projections used when "
            "calculating the spatial filter."
        )


def _pick_data(data, ch_names, start=None, stop=None):
    """Return a (n_channels, n_times) array for the filter's channels."""
    # Accept either an MNE object exposing .get_data()/.ch_names or a raw array.
    if hasattr(data, "ch_names") and hasattr(data, "get_data"):
        missing = [ch for ch in ch_names if ch not in data.ch_names]
        if missing:
            # MNE's wording: a bare ``list.index`` ValueError names the channel
            # but says nothing about spatial filters or what to do next.
            raise ValueError(
                f"The spatial filter was computed with channel {missing[0]} "
                "which is not present in the data. You should compute a new "
                "spatial filter restricted to the good data channels."
            )
        idx = [data.ch_names.index(ch) for ch in ch_names]
        # Pick inside ``get_data`` rather than slicing afterwards. Reading the
        # whole object first materialises every channel the filter never uses --
        # stim, EOG, the other sensor type -- and, for a Raw, every sample: on an
        # hour of 306-channel data that is several gigabytes to reconstruct two
        # sources. MNE returns the channels in the order asked for, which is the
        # order the weights are in.
        kwargs = {}
        if start is not None or stop is not None:
            if not hasattr(data, "n_times") or not hasattr(data, "first_samp"):
                raise TypeError(
                    "start/stop are only supported for Raw data, got "
                    f"{type(data).__name__}. Crop the object instead."
                )
            kwargs = dict(start=start, stop=stop)
        return np.asarray(data.get_data(picks=idx, **kwargs))
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape[-2] != len(ch_names):
        raise ValueError(
            f"Array has {arr.shape[-2]} channels but the filter expects "
            f"{len(ch_names)}. The filter is built on the good channels shared by "
            "info, forward and data_cov, so it excludes info['bads']; pass an MNE "
            "object (Raw/Epochs/Evoked), whose channels are matched by name, or "
            "index the array down to filters['ch_names'] first."
        )
    return arr


@verbose
def apply_mcmv(data, filters, *, start=None, stop=None, verbose=None):
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
    start, stop : int | None
        First and last sample to reconstruct, for :class:`mne.io.Raw` only,
        mirroring :func:`mne.beamformer.apply_lcmv_raw`. ``None`` reconstructs
        the whole recording. Use these rather than cropping a copy: only the
        requested samples are read from disk.
    %(verbose)s

    Returns
    -------
    source_data : ndarray, shape (..., n_sources, n_times)
        The reconstructed time course of each constrained source.
    """
    _validate_type(filters, MCMVBeamformer, "filters")
    if hasattr(data, "info"):
        _check_proj_match(data, filters)
    W = filters["weights"]  # (n_sources, n_channels)
    b = _pick_data(data, filters["ch_names"], start, stop)  # (..., n_ch, n_times)
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
    # A covariance carries its own projectors, and apply_lcmv_cov checks them
    # against the filter's for the same reason apply_lcmv checks the data's.
    if filters.get("proj") is not None:
        from mne._fiff.proj import make_projector

        proj_cov, _, _ = make_projector(data_cov["projs"], list(filters["ch_names"]))
        if not np.allclose(
            proj_cov, filters["proj"], atol=np.finfo(float).eps, rtol=1e-13
        ):
            raise ValueError(
                "The SSP projections present in the data "
                "do not match the projections used when "
                "calculating the spatial filter."
            )
    R = _cov_as_matrix(data_cov, filters["ch_names"])
    W = filters["weights"]  # (n_sources, n_channels)
    return W @ R @ W.T
