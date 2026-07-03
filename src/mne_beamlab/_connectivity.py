r"""Pairwise and augmented-pairwise MCMV for leakage-free connectivity.

This module implements the connectivity estimators of Nunes et al. (2020),
NeuroImage 208:116386, built on top of the multi-source MCMV beamformer
(:func:`~mne_beamlab.make_mcmv`).

The idea is that connectivity between two regions is confounded by the spatial
spread ("signal leakage") and coherent-source cancellation of the inverse
operator. A single-source LCMV reconstructs each region with an independent
filter, so the estimate :math:`\hat s_a` of region :math:`a` contains a linear
mixture of every other active region; any region that is genuinely coupled to
:math:`b` therefore injects a spurious :math:`a`--:math:`b` edge. An MCMV filter
built jointly on a set of regions instead satisfies the zero-gain condition
:math:`w_a^\mathsf{T} h_c = 0` for every *other* constrained region :math:`c`
(Moiseev et al., 2011, Eq. 4), so those regions cannot leak into
:math:`\hat s_a` at all.

Two estimators are provided:

- **PW-MCMV** (pairwise MCMV): every pair :math:`(a, b)` is reconstructed with a
  2-source MCMV constraining exactly :math:`\{a, b\}`. This removes *direct*
  leakage between the pair. Connectivity is then computed between the two
  leakage-corrected time courses.
- **APW-MCMV** (augmented pairwise MCMV, Nunes et al. 2020, Sec. 2.4): PW-MCMV
  removes direct leakage but not *indirect* leakage -- if region :math:`a`
  leaks into a neighbour :math:`k` that is coupled to :math:`b`, a spurious
  :math:`a`--:math:`b` edge remains. APW-MCMV suppresses this by adding, for
  every statistically significant pair, up to two neighbouring regions of each
  source to the beamformer (order 2 to 6), which places explicit nulls on the
  "conductor" regions.

The two design constraints stressed in the paper are honoured by the caller
rather than hidden here: (i) beamformer weights should be built from a
*band-limited* covariance matching the analysis band -- broadband weights
become mis-tuned and manufacture spurious high-frequency connectivity (Nunes
et al. 2020, Discussion), so ``data`` and ``data_cov`` should be band-filtered;
(ii) beamformer order is kept small (APW-MCMV caps at 6) because an
:math:`n`-source filter spends :math:`n` of its :math:`M` degrees of freedom on
the constraints, degrading SNR.

Connectivity metrics themselves are delegated to :mod:`mne_connectivity`
(``envelope_correlation`` for resting-state amplitude coupling;
``spectral_connectivity_epochs`` for coherence / PLV), so nothing already
implemented there is duplicated.
"""
# Authors: Sepehr Shirani <sepehrshirani@gmail.com>
# License: BSD-3-Clause

from itertools import combinations

import numpy as np
from mne.utils import _validate_type, logger, warn

from ._mcmv import apply_mcmv, make_mcmv

# Metrics understood by :func:`_pair_connectivity`. "envelope" uses
# mne_connectivity.envelope_correlation; the rest are passed through to
# mne_connectivity.spectral_connectivity_epochs.
_SPECTRAL_METHODS = ("coh", "cohy", "imcoh", "plv", "ciplv", "ppc", "pli", "wpli")
_CONN_METHODS = ("envelope",) + _SPECTRAL_METHODS


def _as_pairs(n_sources):
    """Return the list of upper-triangular index pairs for ``n_sources`` ROIs."""
    if n_sources < 2:
        raise ValueError(f"need at least 2 sources for connectivity, got {n_sources}.")
    return list(combinations(range(n_sources), 2))


def _validate_conn_params(method, sfreq, fmin, fmax):
    """Validate the connectivity metric and its band arguments up front."""
    _validate_type(method, str, "method")
    if method not in _CONN_METHODS:
        raise ValueError(f"method must be one of {_CONN_METHODS}, got {method!r}.")
    if method in _SPECTRAL_METHODS:
        if sfreq is None:
            raise ValueError(f"method={method!r} requires ``sfreq`` to be given.")
        if fmin is None or fmax is None:
            raise ValueError(
                f"method={method!r} requires a frequency band ``fmin`` and ``fmax``."
            )


def _epoched(time_courses):
    """Return ``(n_epochs, n_signals, n_times)`` from an :func:`apply_mcmv` output.

    ``apply_mcmv`` returns ``(n_sources, n_times)`` for continuous input and
    ``(n_epochs, n_sources, n_times)`` for epoched input; ``mne_connectivity``
    always wants a leading epoch axis, so a single continuous segment becomes a
    one-epoch array.
    """
    time_courses = np.asarray(time_courses)
    if time_courses.ndim == 2:
        return time_courses[np.newaxis]
    if time_courses.ndim == 3:
        return time_courses
    raise ValueError(
        "reconstructed time courses must be 2D (n_sources, n_times) or 3D "
        f"(n_epochs, n_sources, n_times), got {time_courses.ndim}D."
    )


def _pair_connectivity(
    pair_tc, method, *, sfreq, fmin, fmax, orthogonalize, absolute, mt_bandwidth
):
    """Return connectivity between the two reconstructed rows of a pair.

    ``pair_tc`` holds the two leakage-corrected time courses of a single pair
    (row 0 and row 1). The value is delegated to :mod:`mne_connectivity`.
    """
    data = _epoched(pair_tc)  # (n_epochs, 2, n_times)
    if method == "envelope":
        from mne_connectivity import envelope_correlation

        # Delegated to mne-connectivity: the Hilbert amplitude envelopes are
        # correlated directly. The paper additionally low-pass filters each
        # envelope to 0.5 Hz before correlating (a noise-reduction step from
        # Colclough et al., 2015); that smoothing is not applied here, as it is
        # not exposed by ``envelope_correlation`` and does not affect the
        # leakage suppression that is this module's contribution.
        conn = envelope_correlation(
            data, orthogonalize=orthogonalize, absolute=absolute
        )
        # (n_epochs, 2, 2, 1) -> average the off-diagonal over epochs (a single
        # continuous segment is one epoch and returns one value).
        values = conn.get_data(output="dense")[:, 0, 1, 0]
        return float(np.mean(values))

    from mne_connectivity import spectral_connectivity_epochs

    if sfreq is None:
        raise ValueError(f"method={method!r} requires ``sfreq`` to be given.")
    if fmin is None or fmax is None:
        # Without a band the estimate is per-frequency; a single connectivity
        # value would silently take only the first frequency bin. Require an
        # explicit band so the value is the band-averaged connectivity.
        raise ValueError(
            f"method={method!r} requires a frequency band ``fmin`` and ``fmax``."
        )
    if data.shape[0] < 2:
        warn(
            f"spectral connectivity (method={method!r}) is being estimated from a "
            "single epoch; provide epoched data for a meaningful spectral estimate."
        )
    conn = spectral_connectivity_epochs(
        data,
        method=method,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_bandwidth=mt_bandwidth,
        verbose=False,
    )
    # spectral output is (2, 2, 1) after band averaging; [1, 0] is the pair.
    return float(conn.get_data(output="dense")[1, 0, 0])


def _orientations_for(orientations, indices):
    """Select the orientation rows for ``indices`` (or ``None`` for fixed forward)."""
    if orientations is None:
        return None
    return np.asarray(orientations, float)[list(indices)]


def reconstruct_pairwise_mcmv(
    data,
    info,
    forward,
    data_cov,
    sources,
    *,
    orientations=None,
    noise_cov=None,
    reg=0.05,
    weight_norm="unit-gain",
    rank=None,
):
    r"""Reconstruct every ROI pair with a 2-source (pairwise) MCMV.

    For each unordered pair :math:`(a, b)` of the requested ``sources`` a
    2-source MCMV beamformer constraining :math:`\{a, b\}` is built and applied
    to ``data``. Because the MCMV weights satisfy :math:`w_a^\mathsf{T} h_b = 0`
    exactly (Moiseev et al., 2011, Eq. 4), the two reconstructed time courses
    carry no *direct* leakage of one source into the other -- the precondition
    for unbiased pairwise connectivity (Nunes et al., 2020).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        Sensor data to reconstruct. For connectivity in a frequency band this
        should be band-filtered, consistently with ``data_cov`` (see module
        docstring).
    info : mne.Info
        Measurement info describing the ``n_channels`` sensors of ``data``.
    forward : mne.Forward
        Forward solution covering the ``sources``.
    data_cov : mne.Covariance
        Sensor data covariance :math:`R`, band-limited to the analysis band.
    sources : sequence of int
        Indices of the regions of interest into the forward source space.
    orientations : ndarray, shape (n_sources, 3) | None
        Fixed source orientations for a free-orientation forward, in the same
        order as ``sources``. Must be ``None`` for a fixed-orientation forward.
    noise_cov : mne.Covariance | None
        Noise covariance used for whitening. For resting-state data a diagonal
        (ad-hoc) covariance is customary, as no baseline exists (Nunes et al.,
        2020, Sec. 2.3).
    reg : float
        Regularisation passed through to :func:`~mne_beamlab.make_mcmv`.
    weight_norm : 'unit-gain' | 'unit-noise-gain' | None
        Weight normalisation passed through to :func:`~mne_beamlab.make_mcmv`.
        ``'unit-gain'`` reconstructs physical source amplitude.
    rank : None | int | 'full'
        Rank handling passed through to :func:`~mne_beamlab.make_mcmv`.

    Returns
    -------
    pairs : list of tuple of int
        The ``(i, j)`` index pairs into ``sources`` (i.e. positions, not vertex
        numbers), in upper-triangular order.
    time_courses : list of ndarray
        ``time_courses[k]`` is the reconstruction for ``pairs[k]``; row 0 is the
        first source of the pair and row 1 the second. Its shape is
        ``(2, n_times)`` for continuous ``data`` and
        ``(n_epochs, 2, n_times)`` for epoched ``data``.

    Notes
    -----
    A separate 2-source beamformer is built for every pair, all sharing the same
    data covariance ``R`` but using the pair-specific leadfields ``H = [h_a,
    h_b]``. The reconstruction of a given source therefore differs from pair to
    pair -- this is intrinsic to pairwise MCMV.
    """
    sources = list(sources)
    pairs = _as_pairs(len(sources))
    time_courses = []
    for i, j in pairs:
        filters = make_mcmv(
            info,
            forward,
            data_cov,
            sources=[sources[i], sources[j]],
            orientations=_orientations_for(orientations, (i, j)),
            noise_cov=noise_cov,
            reg=reg,
            weight_norm=weight_norm,
            rank=rank,
        )
        time_courses.append(apply_mcmv(data, filters))
    return pairs, time_courses


def pairwise_mcmv_connectivity(
    data,
    info,
    forward,
    data_cov,
    sources,
    *,
    method="envelope",
    sfreq=None,
    fmin=None,
    fmax=None,
    orientations=None,
    noise_cov=None,
    reg=0.05,
    weight_norm="unit-gain",
    rank=None,
    orthogonalize=False,
    absolute=False,
    mt_bandwidth=None,
):
    r"""Pairwise-MCMV connectivity matrix (PW-MCMV, Nunes et al., 2020).

    Every pair of ``sources`` is reconstructed with a 2-source MCMV
    (:func:`reconstruct_pairwise_mcmv`) and connectivity between the two
    leakage-corrected time courses is computed with :mod:`mne_connectivity`.

    Parameters
    ----------
    data, info, forward, data_cov, sources
        See :func:`reconstruct_pairwise_mcmv`.
    orientations, noise_cov, reg, weight_norm, rank
        See :func:`reconstruct_pairwise_mcmv`.
    method : 'envelope' | 'coh' | 'imcoh' | 'plv' | 'ciplv' | 'pli' | 'wpli' | ...
        Connectivity metric. ``'envelope'`` uses
        :func:`mne_connectivity.envelope_correlation` (resting-state amplitude
        coupling); every other value is forwarded to
        :func:`mne_connectivity.spectral_connectivity_epochs` (task coherence /
        phase measures) and requires ``sfreq``.
    sfreq : float | None
        Sampling frequency; required for the spectral methods.
    fmin, fmax : float | None
        Frequency band for the spectral methods; the metric is averaged over the
        band. Both are required for the spectral methods (the value is otherwise
        per-frequency and ill-defined for a single edge).
    orthogonalize : bool | 'pairwise'
        Passed to :func:`mne_connectivity.envelope_correlation`. The default
        ``False`` gives *plain* envelope correlation, as MCMV already removes
        leakage; leakage-orthogonalisation (the symmetric-orthogonalisation
        baseline of Nunes et al., 2020) would double-correct and discard genuine
        zero-lag coupling.
    absolute : bool
        Passed to :func:`mne_connectivity.envelope_correlation`. The default
        ``False`` returns the *signed* Pearson correlation of the amplitude
        envelopes, as in Nunes et al. (2020); ``True`` returns its magnitude.
    mt_bandwidth : float | None
        Multitaper frequency smoothing (Hz) for the spectral methods (the paper
        uses 2 Hz for its task coherence/PLV analyses).

    Returns
    -------
    conn : ndarray, shape (n_sources, n_sources)
        Symmetric connectivity matrix; ``conn[i, j]`` is the PW-MCMV
        connectivity between ``sources[i]`` and ``sources[j]``. The diagonal is
        zero.
    """
    _validate_conn_params(method, sfreq, fmin, fmax)
    sources = list(sources)
    n = len(sources)
    pairs, time_courses = reconstruct_pairwise_mcmv(
        data,
        info,
        forward,
        data_cov,
        sources,
        orientations=orientations,
        noise_cov=noise_cov,
        reg=reg,
        weight_norm=weight_norm,
        rank=rank,
    )
    conn = np.zeros((n, n))
    for (i, j), pair_tc in zip(pairs, time_courses, strict=True):
        value = _pair_connectivity(
            pair_tc,
            method,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            orthogonalize=orthogonalize,
            absolute=absolute,
            mt_bandwidth=mt_bandwidth,
        )
        conn[i, j] = conn[j, i] = value
    return conn


def _select_neighbours(i, j, positions, significance, degree, radius, max_neighbours):
    r"""Select augmenting neighbours for a pair (Nunes et al., 2020, Sec. 2.4).

    For each source of the pair, candidate neighbours are the *other* regions
    within ``radius`` of it that themselves carry at least one significant
    connection (``degree >= 1``). Up to ``max_neighbours`` candidates per source
    are kept, ranked by their number of significant connections ("maximal total
    number of connections"); ties are broken deterministically by proximity
    (nearer first), which the paper leaves unspecified. A region qualifying for
    both sources, or already chosen, is added only once.
    """
    chosen = []
    for src in (i, j):
        dist = np.linalg.norm(positions - positions[src], axis=1)
        candidates = [
            k
            for k in range(len(positions))
            if k not in (i, j)
            and k not in chosen
            and dist[k] <= radius
            and degree[k] >= 1
        ]
        candidates.sort(key=lambda k: (-int(degree[k]), float(dist[k])))
        chosen.extend(candidates[:max_neighbours])
    return chosen


def augmented_pairwise_mcmv_connectivity(
    data,
    info,
    forward,
    data_cov,
    sources,
    connectivity,
    significance,
    *,
    positions=None,
    method="envelope",
    radius=0.04,
    max_neighbours=2,
    sfreq=None,
    fmin=None,
    fmax=None,
    orientations=None,
    noise_cov=None,
    reg=0.05,
    weight_norm="unit-gain",
    rank=None,
    orthogonalize=False,
    absolute=False,
    mt_bandwidth=None,
):
    r"""Augmented pairwise-MCMV connectivity (APW-MCMV, Nunes et al., 2020, Sec. 2.4).

    Refines an existing pairwise-MCMV connectivity matrix by re-estimating each
    statistically significant pair with a higher-order MCMV that also nulls the
    pair's neighbouring regions, suppressing *indirect* leakage. The procedure
    follows the paper exactly:

    1. Start from the PW-MCMV ``connectivity`` and the ``significance`` mask of
       its edges (both supplied by the caller; the mask is obtained e.g. from
       :func:`ar1_surrogate_significance`).
    2. For every significant pair, add up to ``max_neighbours`` neighbours of
       *each* source within ``radius`` that carry significant connections,
       chosen by their number of significant connections (see
       :func:`_select_neighbours`). This yields a beamformer of order 2 to
       ``2 + 2 * max_neighbours`` (2 to 6 with the defaults).
    3. Rebuild the MCMV on the augmented source set, reconstruct the two time
       courses of the pair, and recompute their connectivity.

    Non-significant pairs keep their PW-MCMV value (they are treated as absent
    connections downstream); only significant pairs are re-estimated.

    Parameters
    ----------
    data, info, forward, data_cov, sources
        See :func:`reconstruct_pairwise_mcmv`.
    orientations, noise_cov, reg, weight_norm, rank
        See :func:`reconstruct_pairwise_mcmv`.
    connectivity : ndarray, shape (n_sources, n_sources)
        The PW-MCMV connectivity matrix to refine (from
        :func:`pairwise_mcmv_connectivity`).
    significance : ndarray of bool, shape (n_sources, n_sources)
        Symmetric mask of statistically significant PW-MCMV edges.
    positions : ndarray, shape (n_sources, 3) | None
        Source positions (metres) used for the neighbour search. Defaults to the
        forward positions of ``sources`` (``forward['source_rr'][sources]``).
    method, sfreq, fmin, fmax, orthogonalize, absolute, mt_bandwidth
        Connectivity settings, as in :func:`pairwise_mcmv_connectivity`; use the
        same values that produced ``connectivity``.
    radius : float
        Neighbour search radius in metres (default 0.04 m = 4 cm, per the paper).
    max_neighbours : int
        Maximum neighbours added per source of the pair (default 2, per the
        paper), capping the beamformer order at ``2 + 2 * max_neighbours``.

    Returns
    -------
    conn : ndarray, shape (n_sources, n_sources)
        Connectivity matrix with the significant edges re-estimated under
        augmentation and the non-significant edges left at their PW-MCMV value.
    """
    _validate_conn_params(method, sfreq, fmin, fmax)
    sources = list(sources)
    n = len(sources)
    connectivity = np.array(connectivity, dtype=float)
    if connectivity.shape != (n, n):
        raise ValueError(
            f"connectivity must be shape ({n}, {n}), got {connectivity.shape}."
        )
    significance = np.asarray(significance, dtype=bool)
    if significance.shape != (n, n):
        raise ValueError(
            f"significance must be shape ({n}, {n}), got {significance.shape}."
        )
    if positions is None:
        positions = np.asarray(forward["source_rr"], float)[sources]
    positions = np.asarray(positions, float)
    if positions.shape != (n, 3):
        raise ValueError(
            f"positions must be ({n}, 3) for {n} sources, got {positions.shape}."
        )

    degree = significance.sum(axis=1)  # significant connections per region
    max_order = 2 + 2 * max_neighbours
    conn = connectivity.copy()
    for i, j in _as_pairs(n):
        if not significance[i, j]:
            continue
        neighbours = _select_neighbours(
            i, j, positions, significance, degree, radius, max_neighbours
        )
        order = 2 + len(neighbours)
        logger.info(
            f"APW-MCMV: pair ({sources[i]}, {sources[j]}) augmented with "
            f"{len(neighbours)} neighbour(s), beamformer order {order}."
        )
        if order > max_order:  # defensive; _select_neighbours already caps this
            warn(f"APW-MCMV beamformer order {order} exceeds the cap {max_order}.")
        aug_indices = [i, j, *neighbours]
        aug_sources = [sources[k] for k in aug_indices]
        filters = make_mcmv(
            info,
            forward,
            data_cov,
            sources=aug_sources,
            orientations=_orientations_for(orientations, aug_indices),
            noise_cov=noise_cov,
            reg=reg,
            weight_norm=weight_norm,
            rank=rank,
        )
        reconstructed = apply_mcmv(data, filters)
        # rows 0 and 1 are the pair (i, j); neighbours occupy the remaining rows.
        if reconstructed.ndim == 3:
            pair_tc = reconstructed[:, :2, :]
        else:
            pair_tc = reconstructed[:2]
        value = _pair_connectivity(
            pair_tc,
            method,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            orthogonalize=orthogonalize,
            absolute=absolute,
            mt_bandwidth=mt_bandwidth,
        )
        conn[i, j] = conn[j, i] = value
    return conn


def ar1_surrogate_significance(
    connectivity,
    reference_time_courses,
    *,
    method="envelope",
    n_surrogates=200,
    alpha=0.05,
    sfreq=None,
    fmin=None,
    fmax=None,
    orthogonalize=False,
    absolute=False,
    mt_bandwidth=None,
    random_state=None,
):
    r"""Significance mask for a connectivity matrix via an AR(1) surrogate null.

    Implements the significance procedure of Nunes et al. (2020), Sec. 2.6-2.7,
    for pairwise connectivity:

    1. Fit an order-one autoregressive model
       :math:`x_t = \varphi x_{t-1} + \varepsilon_t` to each region's
       ``reference_time_courses``, capturing its temporal smoothness.
    2. Generate ``n_surrogates`` sets of *independent* Gaussian AR(1) source
       signals with those per-region coefficients -- surrogates with realistic
       temporal structure but no genuine pairwise coupling.
    3. Compute the same connectivity ``method`` for every surrogate pair,
       apply the Fisher :math:`z`-transform, and take the null standard
       deviation.
    4. Convert the Fisher-transformed real connectivity to :math:`z`-scores
       using that null standard deviation, then to two-sided :math:`p`-values
       under Gaussianity, and threshold with Benjamini-Hochberg FDR at
       ``alpha``.

    Parameters
    ----------
    connectivity : ndarray, shape (n_sources, n_sources)
        The connectivity matrix to test (from
        :func:`pairwise_mcmv_connectivity`).
    reference_time_courses : ndarray, shape (n_sources, n_times)
        One representative reconstructed time course per region, used to fit the
        AR(1) coefficients. For epoched data, concatenate epochs first.
    method, sfreq, fmin, fmax, orthogonalize, absolute, mt_bandwidth
        Connectivity settings; must match those used for ``connectivity``.
    n_surrogates : int
        Number of surrogate datasets used to estimate the null (default 200).
    alpha : float
        FDR level (default 0.05, per the paper).
    random_state : None | int | numpy.random.Generator
        Seed / generator for surrogate generation.

    Returns
    -------
    significance : ndarray of bool, shape (n_sources, n_sources)
        Symmetric mask of significant edges (diagonal ``False``).

    Notes
    -----
    The paper fits the AR model to "the reconstructed time courses of real
    data"; since a pairwise reconstruction is pair-specific, a single
    representative course per region (for example a single-source LCMV, or the
    region's course from any one pair) is used to characterise its temporal
    smoothness. The surrogate connectivity is computed directly on the surrogate
    source signals, which by construction contain no leakage.
    """
    _validate_conn_params(method, sfreq, fmin, fmax)

    rng = np.random.default_rng(random_state)
    connectivity = np.asarray(connectivity, float)
    ref = np.asarray(reference_time_courses, float)
    n, n_times = ref.shape
    if connectivity.shape != (n, n):
        raise ValueError(
            f"connectivity {connectivity.shape} is inconsistent with "
            f"{n} reference time courses."
        )

    # AR(1) fit per region: phi = <x_t x_{t-1}> / <x_{t-1}^2>, innovation std.
    phis = np.empty(n)
    sigmas = np.empty(n)
    for k in range(n):
        x = ref[k] - ref[k].mean()
        denom = float(x[:-1] @ x[:-1])
        phi = 0.0 if denom == 0 else float(x[1:] @ x[:-1] / denom)
        phi = float(np.clip(phi, -0.999, 0.999))  # keep the process stationary
        resid = x[1:] - phi * x[:-1]
        phis[k] = phi
        sigmas[k] = resid.std() if resid.size else 1.0

    pairs = _as_pairs(n)
    null = np.empty((n_surrogates, len(pairs)))
    for s in range(n_surrogates):
        surr = np.empty((n, n_times))
        noise = rng.standard_normal((n, n_times))
        for k in range(n):
            x = np.empty(n_times)
            x[0] = noise[k, 0] * sigmas[k]
            for t in range(1, n_times):
                x[t] = phis[k] * x[t - 1] + sigmas[k] * noise[k, t]
            surr[k] = x
        for p, (i, j) in enumerate(pairs):
            null[s, p] = _pair_connectivity(
                np.stack([surr[i], surr[j]]),
                method,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                orthogonalize=orthogonalize,
                absolute=absolute,
                mt_bandwidth=mt_bandwidth,
            )

    # Fisher z-transform, standardise by the null mean and std (so the z-scores
    # have zero mean and unit variance under the null, per Colclough et al.,
    # 2015), two-sided p, then FDR. Subtracting the null mean matters when the
    # metric is non-negative (e.g. ``absolute=True``); for signed correlation the
    # null mean is ~0 and this reduces to dividing by the null std.
    def _fisher(r):
        return np.arctanh(np.clip(r, -0.999999, 0.999999))

    from scipy.stats import norm

    null_z = _fisher(null)
    null_mean = null_z.mean(axis=0)
    null_std = null_z.std(axis=0, ddof=1)
    null_std[null_std == 0] = np.finfo(float).eps

    real = np.array([connectivity[i, j] for i, j in pairs])
    zscores = (_fisher(real) - null_mean) / null_std
    pvals = 2.0 * norm.sf(np.abs(zscores))

    # Benjamini-Hochberg FDR at ``alpha``.
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed))
        keep[order[: cutoff + 1]] = True

    significance = np.zeros((n, n), dtype=bool)
    for p, (i, j) in enumerate(pairs):
        significance[i, j] = significance[j, i] = keep[p]
    return significanceq