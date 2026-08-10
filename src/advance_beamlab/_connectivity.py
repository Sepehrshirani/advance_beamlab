r"""Pairwise and augmented-pairwise MCMV for leakage-free connectivity.

This module implements the connectivity estimators of Nunes et al. (2020),
NeuroImage 208:116386, built on top of the multi-source MCMV beamformer
(:func:`~advance_beamlab.make_mcmv`).

The idea is that connectivity between two regions is confounded by the spatial
spread ("signal leakage") and coherent-source cancellation of the inverse
operator. A single-source LCMV reconstructs each region with an independent
filter, so the estimate :math:`\hat s_a` of region :math:`a` contains a linear
mixture of every other active region; any region that is genuinely coupled to
:math:`b` therefore injects a spurious edge between :math:`a` and :math:`b`. An
MCMV filter built jointly on a set of regions instead satisfies the zero-gain
condition :math:`w_a^\mathsf{T} h_c = 0` for every *other* constrained region
:math:`c` (Moiseev et al., 2011, Eq. 4), so those regions cannot leak into
:math:`\hat s_a` at all.

Two estimators are provided:

- **PW-MCMV** (pairwise MCMV): every pair :math:`(a, b)` is reconstructed with a
  2-source MCMV constraining exactly :math:`\{a, b\}`. This removes *direct*
  leakage between the pair. Connectivity is then computed between the two
  leakage-corrected time courses.
- **APW-MCMV** (augmented pairwise MCMV, Nunes et al. 2020, Sec. 2.4): PW-MCMV
  removes direct leakage but not *indirect* leakage. If region :math:`a` leaks
  into a neighbour :math:`k` that is coupled to :math:`b`, a spurious edge
  between :math:`a` and :math:`b` remains. APW-MCMV suppresses this by adding,
  for every statistically significant pair, up to two neighbouring regions of
  each source to the beamformer (order 2 to 6), which places explicit nulls on
  the "conductor" regions.

The two design constraints stressed in the paper are honoured by the caller
rather than hidden here. (i) Beamformer weights should be built from a
*band-limited* covariance matching the analysis band. Broadband weights
become mis-tuned and manufacture spurious high-frequency connectivity (Nunes
et al. 2020, Discussion), so ``data`` and ``data_cov`` should be band-filtered.
(ii) Beamformer order is kept small (APW-MCMV caps at 6) because an
:math:`n`-source filter spends :math:`n` of its :math:`M` degrees of freedom on
the constraints, degrading SNR.

Coherence and the phase measures are delegated to
:func:`mne_connectivity.spectral_connectivity_epochs`. The amplitude-envelope
metric is computed here rather than delegated, because the paper's definition
has a step :func:`mne_connectivity.envelope_correlation` does not expose: "the
envelopes of the signals were computed by taking the absolute values of the
analytic Hilbert transform of the signals and then low-pass filtering to 0.5 Hz"
(Nunes et al. 2020, Sec. 2.5), and the correlations are computed on the
downsampled envelopes. The low-pass cannot be applied from outside because
``envelope_correlation`` takes the Hilbert transform internally, and it is not
cosmetic: it is what makes the AR(1) surrogate null of
:func:`ar1_surrogate_significance` correctly sized, and that null is what gates
APW-MCMV. With ``envelope_lowpass=None`` the estimator reduces exactly to
``envelope_correlation``.
"""
# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import warnings
from itertools import combinations

import numpy as np
from mne.filter import filter_data, next_fast_len, resample
from mne.utils import _validate_type, logger
from scipy.signal import hilbert, lfilter

from ._mcmv import apply_mcmv, make_mcmv

# Metrics understood by :func:`_pair_connectivity`. "envelope" is computed here
# (see the module docstring); the rest are passed through to
# mne_connectivity.spectral_connectivity_epochs.
_SPECTRAL_METHODS = ("coh", "imcoh", "plv", "ciplv", "ppc", "pli", "wpli")
_CONN_METHODS = ("envelope",) + _SPECTRAL_METHODS
# Metrics whose mne-connectivity output is complex. The connectivity matrices
# returned by this module are real, so these are rejected rather than silently
# truncated to their real part.
_COMPLEX_METHODS = ("cohy",)


def _as_pairs(n_sources):
    """Return the list of upper-triangular index pairs for ``n_sources`` ROIs."""
    if n_sources < 2:
        raise ValueError(f"need at least 2 sources for connectivity, got {n_sources}.")
    return list(combinations(range(n_sources), 2))


def _validate_conn_params(
    method, sfreq, fmin, fmax, envelope_lowpass=None, envelope_resample=None
):
    """Validate the connectivity metric and its band / envelope arguments."""
    _validate_type(method, str, "method")
    if method in _COMPLEX_METHODS:
        raise ValueError(
            f"method={method!r} returns complex coherency, which cannot be "
            "represented in the real connectivity matrix returned here. Use "
            "method='coh' for its magnitude or method='imcoh' for its "
            "imaginary part."
        )
    if method not in _CONN_METHODS:
        raise ValueError(f"method must be one of {_CONN_METHODS}, got {method!r}.")
    if method in _SPECTRAL_METHODS:
        if sfreq is None:
            raise ValueError(f"method={method!r} requires ``sfreq`` to be given.")
        if fmin is None or fmax is None:
            raise ValueError(
                f"method={method!r} requires a frequency band ``fmin`` and ``fmax``."
            )
    elif sfreq is None and (
        envelope_lowpass is not None or envelope_resample is not None
    ):
        raise ValueError(
            "``envelope_lowpass`` and ``envelope_resample`` need ``sfreq`` to be "
            "given; pass ``sfreq``, or set both to None to correlate the "
            "unsmoothed envelopes."
        )


def _epoched(time_courses):
    """Return ``(n_epochs, n_signals, n_times)`` from an :func:`apply_mcmv` output.

    ``apply_mcmv`` returns ``(n_sources, n_times)`` for continuous input and
    ``(n_epochs, n_sources, n_times)`` for epoched input; a connectivity
    estimate always wants a leading epoch axis, so a single continuous segment
    becomes a one-epoch array.
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


def _analytic_signal(data):
    """Analytic signal of ``data`` along its last axis (zero-padded FFT)."""
    n_times = data.shape[-1]
    return hilbert(data, N=next_fast_len(n_times), axis=-1)[..., :n_times]


def _prepare_envelope(magnitude, sfreq, envelope_lowpass, envelope_resample):
    """Low-pass (and optionally downsample) an amplitude envelope.

    Nunes et al. (2020), Sec. 2.5: the envelopes are low-pass filtered to
    0.5 Hz and the correlations taken on the downsampled envelopes.
    """
    env = np.asarray(magnitude, float)
    if envelope_lowpass is not None:
        env = filter_data(env, sfreq, None, envelope_lowpass, verbose=False)
    if envelope_resample is not None:
        env = resample(
            env, up=1.0, down=sfreq / envelope_resample, npad="auto", verbose=False
        )
    return env


def _rowwise_pearson(x, y):
    """Pearson correlation of ``x[i]`` with ``y[i]`` along the last axis."""
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    nx = np.linalg.norm(x, axis=-1)
    ny = np.linalg.norm(y, axis=-1)
    nx[nx == 0] = 1.0
    ny[ny == 0] = 1.0
    return np.sum(x * y, axis=-1) / nx / ny


def _envelope_corr_matrix(
    epoch, *, sfreq, orthogonalize, absolute, envelope_lowpass, envelope_resample
):
    """Envelope-correlation matrix of one ``(n_signals, n_times)`` segment.

    With ``envelope_lowpass=None`` and ``envelope_resample=None`` this
    reproduces :func:`mne_connectivity.envelope_correlation` exactly (for a
    single epoch), including its pairwise-orthogonalisation variant; the
    envelope smoothing of Nunes et al. (2020) is inserted between taking the
    Hilbert magnitude and correlating.
    """
    if orthogonalize not in (False, "pairwise"):
        raise ValueError(
            f"orthogonalize must be False or 'pairwise', got {orthogonalize!r}."
        )
    analytic = _analytic_signal(np.asarray(epoch, float))
    mag = _prepare_envelope(
        np.abs(analytic), sfreq, envelope_lowpass, envelope_resample
    )
    n_signals = len(analytic)
    if orthogonalize is False:
        centred = mag - mag.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(centred, axis=-1)
        norms[norms == 0] = 1.0
        centred /= norms[:, np.newaxis]
        corr = centred @ centred.T
        return np.abs(corr) if absolute else corr
    # Pairwise orthogonalisation (Hipp et al., 2012): project the analytic
    # signal of one region out of the other before enveloping, then correlate
    # and symmetrise, as in mne_connectivity.envelope_correlation.
    unit = analytic.conj() / np.abs(analytic)
    corr = np.empty((n_signals, n_signals))
    for k in range(n_signals):
        orth = np.abs((analytic[k] * unit).imag)
        orth[k] = 1.0  # self term: constant, contributes zero correlation
        orth = _prepare_envelope(orth, sfreq, envelope_lowpass, envelope_resample)
        corr[k] = _rowwise_pearson(orth, mag)
    if absolute:
        corr = np.abs(corr)
    return (corr + corr.T) / 2.0


def _pair_connectivity(
    pair_tc,
    method,
    *,
    sfreq,
    fmin,
    fmax,
    orthogonalize,
    absolute,
    mt_bandwidth,
    envelope_lowpass=0.5,
    envelope_resample=None,
):
    """Return connectivity between the two reconstructed rows of a pair.

    ``pair_tc`` holds the two leakage-corrected time courses of a single pair
    (row 0 and row 1).
    """
    data = _epoched(pair_tc)  # (n_epochs, 2, n_times)
    if method == "envelope":
        values = [
            _envelope_corr_matrix(
                epoch,
                sfreq=sfreq,
                orthogonalize=orthogonalize,
                absolute=absolute,
                envelope_lowpass=envelope_lowpass,
                envelope_resample=envelope_resample,
            )[0, 1]
            for epoch in data
        ]
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
        warnings.warn(
            f"spectral connectivity (method={method!r}) is being estimated from a "
            "single epoch; provide epoched data for a meaningful spectral estimate.",
            RuntimeWarning,
            stacklevel=2,
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
    carry no *direct* leakage of one source into the other. That is the
    precondition for unbiased pairwise connectivity (Nunes et al., 2020).

    Parameters
    ----------
    data : mne.Evoked | mne.Epochs | mne.io.Raw | ndarray
        Sensor data to reconstruct. For connectivity in a frequency band this
        should be band-filtered, consistently with ``data_cov`` (see module
        docstring). An MNE object is preferred, as its channels are matched by
        name. A plain ndarray of shape ``(n_channels, n_times)`` or
        ``(n_epochs, n_channels, n_times)`` is taken to be in the channel order
        of the beamformer and must therefore *already be restricted to the good
        channels*: :func:`~advance_beamlab.make_mcmv` drops ``info['bads']``, so
        with bad channels present an array holding all of ``info['ch_names']``
        has the wrong number of rows and is rejected.
    info : mne.Info
        Measurement info describing the sensors of ``data``.
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
        Regularisation passed through to :func:`~advance_beamlab.make_mcmv`.
    weight_norm : 'unit-gain' | 'unit-noise-gain' | None
        Weight normalisation passed through to :func:`~advance_beamlab.make_mcmv`.
        ``'unit-gain'`` reconstructs physical source amplitude.
    rank : None | 'full' | 'info' | dict
        Rank handling passed through to :func:`~advance_beamlab.make_mcmv`.

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
    pair. This is intrinsic to pairwise MCMV.
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
    envelope_lowpass=0.5,
    envelope_resample=None,
):
    r"""Pairwise-MCMV connectivity matrix (PW-MCMV, Nunes et al., 2020).

    Every pair of ``sources`` is reconstructed with a 2-source MCMV
    (:func:`reconstruct_pairwise_mcmv`) and connectivity between the two
    leakage-corrected time courses is computed.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        Sensor data to reconstruct, band-filtered to the analysis band. An array
        must already be restricted to the good channels; see
        :func:`reconstruct_pairwise_mcmv`.
    info : instance of mne.Info
        Measurement info describing the sensors of ``data``.
    forward : instance of mne.Forward
        Forward solution covering the ``sources``.
    data_cov : instance of mne.Covariance
        Sensor data covariance, band-limited to the analysis band.
    sources : sequence of int
        Indices of the regions of interest into the forward source space.
    method : 'envelope' | 'coh' | 'imcoh' | 'plv' | 'ciplv' | 'ppc' | 'pli' | 'wpli'
        Connectivity metric. ``'envelope'`` is the amplitude-envelope
        correlation of Nunes et al. (2020), Sec. 2.5 (resting-state amplitude
        coupling); every other value is forwarded to
        :func:`mne_connectivity.spectral_connectivity_epochs` (task coherence /
        phase measures) and requires ``sfreq``, ``fmin`` and ``fmax``.
        ``'cohy'`` is not accepted because coherency is complex and this
        function returns a real matrix. Use ``'coh'`` or ``'imcoh'``.
    sfreq : float | None
        Sampling frequency. Required for the spectral methods; for
        ``method='envelope'`` it is needed by ``envelope_lowpass`` /
        ``envelope_resample`` and defaults to ``info['sfreq']``.
    fmin, fmax : float | None
        Frequency band for the spectral methods; the metric is averaged over the
        band. Both are required for the spectral methods (the value is otherwise
        per-frequency and ill-defined for a single edge).
    orientations : ndarray, shape (n_sources, 3) | None
        Head-coordinate source orientations for a free-orientation forward, in
        the same order as ``sources``; ``None`` for a fixed-orientation forward.
    noise_cov : instance of mne.Covariance | None
        Noise covariance used for whitening. For resting-state data a diagonal
        (ad-hoc) covariance is customary, as no baseline exists.
    reg : float
        Regularisation passed through to :func:`~advance_beamlab.make_mcmv`.
    weight_norm : 'unit-gain' | 'unit-noise-gain' | None
        Weight normalisation passed through to :func:`~advance_beamlab.make_mcmv`.
        ``'unit-gain'`` reconstructs physical source amplitude.
    rank : None | 'full' | 'info' | dict
        Rank handling passed through to :func:`~advance_beamlab.make_mcmv`.
    orthogonalize : bool | 'pairwise'
        Envelope leakage-orthogonalisation (Hipp et al., 2012). The default
        ``False`` gives *plain* envelope correlation, as MCMV already removes
        leakage; leakage-orthogonalisation (the symmetric-orthogonalisation
        baseline of Nunes et al., 2020) would double-correct and discard genuine
        zero-lag coupling.
    absolute : bool
        Whether to take the magnitude of the envelope correlation. The default
        ``False`` returns the *signed* Pearson correlation of the amplitude
        envelopes, as in Nunes et al. (2020); ``True`` returns its magnitude.
        Unlike :func:`mne_connectivity.envelope_correlation`, which ignores this
        flag unless ``orthogonalize='pairwise'``, it is honoured for both
        ``orthogonalize`` settings.
    mt_bandwidth : float | None
        Multitaper frequency smoothing (Hz) for the spectral methods (the paper
        uses 2 Hz for its task coherence/PLV analyses).
    envelope_lowpass : float | None
        Cut-off (Hz) of the low-pass filter applied to each amplitude envelope
        before correlating, per Nunes et al. (2020), Sec. 2.5 ("low-pass
        filtering to 0.5 Hz"). ``None`` correlates the unsmoothed envelopes,
        reproducing :func:`mne_connectivity.envelope_correlation`; that variant
        leaves far more sub-band ripple in the envelopes than the AR(1)
        surrogate null of :func:`ar1_surrogate_significance` models, and is
        anticonservative. Only used for ``method='envelope'``.
    envelope_resample : float | None
        If given, the sampling frequency (Hz) the low-passed envelopes are
        downsampled to before correlating (the paper uses "downsampled envelope
        correlations"). Downsampling does not change the expected correlation,
        only the cost. Only used for ``method='envelope'``.

    Returns
    -------
    conn : ndarray, shape (n_sources, n_sources)
        Symmetric connectivity matrix; ``conn[i, j]`` is the PW-MCMV
        connectivity between ``sources[i]`` and ``sources[j]``. The diagonal is
        zero.
    """
    if sfreq is None and method == "envelope":
        sfreq = float(info["sfreq"])
    _validate_conn_params(
        method, sfreq, fmin, fmax, envelope_lowpass, envelope_resample
    )
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
            envelope_lowpass=envelope_lowpass,
            envelope_resample=envelope_resample,
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
    envelope_lowpass=0.5,
    envelope_resample=None,
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
       the neighbour-selection rule below). This yields a beamformer of order 2 to
       ``2 + 2 * max_neighbours`` (2 to 6 with the defaults).
    3. Rebuild the MCMV on the augmented source set, reconstruct the two time
       courses of the pair, and recompute their connectivity.

    Non-significant pairs keep their PW-MCMV value (they are treated as absent
    connections downstream); only significant pairs are re-estimated.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        Sensor data to reconstruct, band-filtered to the analysis band. An array
        must already be restricted to the good channels; see
        :func:`reconstruct_pairwise_mcmv`.
    info : instance of mne.Info
        Measurement info describing the sensors of ``data``.
    forward : instance of mne.Forward
        Forward solution covering the ``sources``.
    data_cov : instance of mne.Covariance
        Sensor data covariance, band-limited to the analysis band.
    sources : sequence of int
        Indices of the regions of interest into the forward source space.
    connectivity : ndarray, shape (n_sources, n_sources)
        The PW-MCMV connectivity matrix to refine (from
        :func:`pairwise_mcmv_connectivity`).
    significance : ndarray of bool, shape (n_sources, n_sources)
        Symmetric mask of statistically significant PW-MCMV edges.
    positions : ndarray, shape (n_sources, 3) | None
        Source positions (metres) used for the neighbour search. Defaults to the
        forward positions of ``sources`` (``forward['source_rr'][sources]``).
    method : 'envelope' | 'coh' | 'imcoh' | 'plv' | 'ciplv' | 'ppc' | 'pli' | 'wpli'
        Connectivity metric, as in :func:`pairwise_mcmv_connectivity`; use the
        same value that produced ``connectivity``.
    radius : float
        Neighbour search radius in metres (default 0.04 m = 4 cm, per the paper).
    max_neighbours : int
        Maximum neighbours added per source of the pair (default 2, per the
        paper), capping the beamformer order at ``2 + 2 * max_neighbours``.
    sfreq : float | None
        Sampling frequency, as in :func:`pairwise_mcmv_connectivity`.
    fmin, fmax : float | None
        Frequency band for the spectral methods, as in
        :func:`pairwise_mcmv_connectivity`.
    orientations : ndarray, shape (n_sources, 3) | None
        Head-coordinate source orientations for a free-orientation forward, in
        the same order as ``sources``; ``None`` for a fixed-orientation forward.
    noise_cov : instance of mne.Covariance | None
        Noise covariance used for whitening.
    reg : float
        Regularisation passed through to :func:`~advance_beamlab.make_mcmv`.
    weight_norm : 'unit-gain' | 'unit-noise-gain' | None
        Weight normalisation passed through to :func:`~advance_beamlab.make_mcmv`.
    rank : None | 'full' | 'info' | dict
        Rank handling passed through to :func:`~advance_beamlab.make_mcmv`.
    orthogonalize : bool | 'pairwise'
        Envelope leakage-orthogonalisation, as in
        :func:`pairwise_mcmv_connectivity`.
    absolute : bool
        Whether to take the magnitude of the envelope correlation, as in
        :func:`pairwise_mcmv_connectivity`.
    mt_bandwidth : float | None
        Multitaper frequency smoothing (Hz) for the spectral methods.
    envelope_lowpass : float | None
        Envelope low-pass cut-off (Hz), as in
        :func:`pairwise_mcmv_connectivity`; use the same value that produced
        ``connectivity``.
    envelope_resample : float | None
        Envelope resampling frequency (Hz), as in
        :func:`pairwise_mcmv_connectivity`; use the same value that produced
        ``connectivity``.

    Returns
    -------
    conn : ndarray, shape (n_sources, n_sources)
        Connectivity matrix with the significant edges re-estimated under
        augmentation and the non-significant edges left at their PW-MCMV value.
    """
    if sfreq is None and method == "envelope":
        sfreq = float(info["sfreq"])
    _validate_conn_params(
        method, sfreq, fmin, fmax, envelope_lowpass, envelope_resample
    )
    sources = list(sources)
    n = len(sources)
    connectivity = np.array(connectivity, dtype=float)
    if connectivity.shape != (n, n):
        raise ValueError(
            f"connectivity must be shape ({n}, {n}), got {connectivity.shape}."
        )
    if int(max_neighbours) < 0:
        raise ValueError(f"max_neighbours must be >= 0, got {max_neighbours}.")
    if float(radius) < 0:
        raise ValueError(f"radius must be >= 0 metres, got {radius}.")
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
            warnings.warn(
                f"APW-MCMV beamformer order {order} exceeds the cap {max_order}.",
                RuntimeWarning,
                stacklevel=2,
            )
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
            envelope_lowpass=envelope_lowpass,
            envelope_resample=envelope_resample,
        )
        conn[i, j] = conn[j, i] = value
    return conn


def _fisher_z(r):
    """Fisher :math:`z`-transform, clipped away from the +/-1 singularities."""
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def _benjamini_hochberg(pvals, alpha):
    """Benjamini-Hochberg step-up rejections at level ``alpha``.

    Equivalent to ``scipy.stats.false_discovery_control(pvals) <= alpha``.
    Non-finite p-values are treated as 1 (not rejected).
    """
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = int(np.max(np.nonzero(passed)))
        keep[order[: cutoff + 1]] = True
    return keep


def _ar1_fit(ref):
    """Per-row AR(1) coefficient and innovation standard deviation."""
    n = len(ref)
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
    return phis, sigmas


def _ar1_surrogate(phis, sigmas, n_times, rng):
    r"""Draw one ``(n_sources, n_times)`` set of independent AR(1) surrogates.

    The recursion :math:`x_t = \varphi x_{t-1} + \sigma \varepsilon_t` with
    :math:`x_0 = \sigma \varepsilon_0` is the all-pole filter
    :math:`1 / (1 - \varphi z^{-1})` at rest, so it is generated with
    :func:`scipy.signal.lfilter` instead of a per-sample Python loop.
    """
    noise = rng.standard_normal((len(phis), n_times))
    surr = np.empty((len(phis), n_times))
    for k in range(len(phis)):
        surr[k] = lfilter([1.0], [1.0, -phis[k]], sigmas[k] * noise[k])
    return surr


def ar1_surrogate_significance(
    connectivity,
    reference_time_courses,
    *,
    method="envelope",
    n_surrogates=200,
    alpha=0.05,
    sfreq=None,
    orthogonalize=False,
    absolute=False,
    envelope_lowpass=0.5,
    envelope_resample=None,
    random_state=None,
):
    r"""Significance mask for a connectivity matrix via an AR(1) surrogate null.

    Implements the significance procedure of Nunes et al. (2020), Sec. 2.6-2.7,
    for pairwise envelope connectivity:

    1. Fit an order-one autoregressive model
       :math:`x_t = \varphi x_{t-1} + \varepsilon_t` to each region's
       ``reference_time_courses``, capturing its temporal smoothness.
    2. Generate ``n_surrogates`` sets of *independent* Gaussian AR(1) source
       signals with those per-region coefficients. These surrogates have
       realistic temporal structure but no genuine pairwise coupling.
    3. Compute the same envelope connectivity for every surrogate pair, apply
       the Fisher :math:`z`-transform, and take the null mean and standard
       deviation.
    4. Convert the Fisher-transformed real connectivity to :math:`z`-scores
       using that null, then to two-sided :math:`p`-values under Gaussianity,
       and threshold with Benjamini-Hochberg FDR at ``alpha``.

    Parameters
    ----------
    connectivity : ndarray, shape (n_sources, n_sources)
        The connectivity matrix to test (from
        :func:`pairwise_mcmv_connectivity`).
    reference_time_courses : ndarray, shape (n_sources, n_times)
        One representative reconstructed time course per region, used to fit the
        AR(1) coefficients and to set the surrogate length. For epoched data,
        concatenate epochs first.
    method : 'envelope'
        Only ``'envelope'`` is accepted. The paper prescribes this source-level
        AR(1) surrogate for the resting-state amplitude-envelope correlations
        only, and it is specific to them: a surrogate is a single continuous
        segment, from which coherence and the phase measures cannot be estimated
        (with one segment they are identically one), and the Fisher
        :math:`z`-transform of step 3 is a variance-stabiliser for a correlation
        coefficient, not for a phase-locking value. Task coherence/PLV in the
        paper is tested with a different (trial-based) procedure.
    n_surrogates : int
        Number of surrogate datasets used to estimate the null (default 200).
        Must be at least 2.
    alpha : float
        FDR level (default 0.05, per the paper).
    sfreq : float | None
        Sampling frequency of ``reference_time_courses``. Required unless both
        ``envelope_lowpass`` and ``envelope_resample`` are ``None``.
    orthogonalize : bool | 'pairwise'
        Envelope leakage-orthogonalisation; must match the value used for
        ``connectivity``, or the null will not be the null of the tested
        statistic.
    absolute : bool
        Whether the envelope correlation is taken in magnitude; must match the
        value used for ``connectivity``.
    envelope_lowpass : float | None
        Envelope low-pass cut-off (Hz); must match the value used for
        ``connectivity``.
    envelope_resample : float | None
        Envelope resampling frequency (Hz); must match the value used for
        ``connectivity``.
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

    An edge whose surrogate null degenerates (zero or non-finite standard
    deviation, for instance because the surrogates are too short for the
    envelope filter) is reported as *not* significant, with a warning; a
    degenerate null must never read as evidence of an effect.
    """
    if method != "envelope":
        raise ValueError(
            "ar1_surrogate_significance is defined for method='envelope' only "
            f"(got {method!r}). Nunes et al. (2020) apply the AR(1) source-level "
            "surrogate to the resting-state amplitude-envelope correlations; a "
            "single continuous surrogate segment carries no usable coherence or "
            "phase-locking estimate, and the Fisher z-transform used here "
            "stabilises the variance of a correlation coefficient only."
        )
    _validate_conn_params(
        method, sfreq, None, None, envelope_lowpass, envelope_resample
    )
    n_surrogates = int(n_surrogates)
    if n_surrogates < 2:
        raise ValueError(
            f"n_surrogates must be at least 2 to estimate a null standard "
            f"deviation, got {n_surrogates}."
        )

    rng = np.random.default_rng(random_state)
    connectivity = np.asarray(connectivity, float)
    ref = np.asarray(reference_time_courses, float)
    n, n_times = ref.shape
    if connectivity.shape != (n, n):
        raise ValueError(
            f"connectivity {connectivity.shape} is inconsistent with "
            f"{n} reference time courses."
        )

    phis, sigmas = _ar1_fit(ref)

    pairs = _as_pairs(n)
    rows = np.array([i for i, _ in pairs])
    cols = np.array([j for _, j in pairs])
    null = np.empty((n_surrogates, len(pairs)))
    for s in range(n_surrogates):
        surr = _ar1_surrogate(phis, sigmas, n_times, rng)
        # One Hilbert transform and one envelope filter for the whole surrogate
        # set, then read off every pair from the resulting matrix.
        corr = _envelope_corr_matrix(
            surr,
            sfreq=sfreq,
            orthogonalize=orthogonalize,
            absolute=absolute,
            envelope_lowpass=envelope_lowpass,
            envelope_resample=envelope_resample,
        )
        null[s] = corr[rows, cols]

    # Fisher z-transform, standardise by the null mean and std (so the z-scores
    # have zero mean and unit variance under the null, per Colclough et al.,
    # 2015), two-sided p, then FDR. Subtracting the null mean matters when the
    # metric is non-negative (e.g. ``absolute=True``); for signed correlation the
    # null mean is ~0 and this reduces to dividing by the null std.
    from scipy.stats import norm

    null_z = _fisher_z(null)
    with np.errstate(invalid="ignore"):
        null_mean = null_z.mean(axis=0)
        null_std = null_z.std(axis=0, ddof=1)
    degenerate = ~np.isfinite(null_mean) | ~np.isfinite(null_std) | (null_std <= 0)
    if degenerate.any():
        warnings.warn(
            f"{int(degenerate.sum())} of {len(pairs)} edges have a degenerate "
            "AR(1) surrogate null (zero or non-finite standard deviation); they "
            "are reported as non-significant. Check that "
            "``reference_time_courses`` are long enough for the envelope filter.",
            RuntimeWarning,
            stacklevel=2,
        )
        null_mean = np.where(degenerate, 0.0, null_mean)
        null_std = np.where(degenerate, np.inf, null_std)

    real = connectivity[rows, cols]
    zscores = (_fisher_z(real) - null_mean) / null_std
    pvals = 2.0 * norm.sf(np.abs(zscores))
    keep = _benjamini_hochberg(pvals, alpha)

    significance = np.zeros((n, n), dtype=bool)
    significance[rows, cols] = keep
    significance[cols, rows] = keep
    return significance
