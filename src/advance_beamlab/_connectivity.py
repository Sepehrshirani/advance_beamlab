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
APW-MCMV. With ``envelope_lowpass=None`` the estimator reduces to
``envelope_correlation``, exactly so for three of the four
``(orthogonalize, absolute)`` combinations. The exception is
``orthogonalize=False, absolute=True``: this estimator honours ``absolute`` for
both ``orthogonalize`` settings, whereas ``envelope_correlation`` applies it
only when orthogonalising.
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
# Metrics that are magnitudes, so that only an unusually *high* value is
# evidence of coupling and the surrogate test is one-sided. The signed metrics
# ("imcoh", "ppc" and the signed envelope correlation) are tested two-sided.
_NONNEGATIVE_METHODS = ("coh", "plv", "ciplv", "pli", "wpli")
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
    method,
    sfreq,
    fmin,
    fmax,
    envelope_lowpass=None,
    envelope_resample=None,
    orthogonalize=False,
):
    """Validate the connectivity metric and its band / envelope arguments.

    Every public entry point calls this before it builds anything, so that an
    argument the estimator cannot honour costs a moment rather than the whole
    pairwise sweep of beamformers that would otherwise be solved first.
    """
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
    if orthogonalize not in (False, "pairwise"):
        # ``'pairwise'`` is the only orthogonalisation implemented, so a bare
        # ``True`` cannot be honoured; guessing at it would silently change
        # which estimator the caller believes produced their matrix.
        raise ValueError(
            f"orthogonalize must be False or 'pairwise', got {orthogonalize!r}."
        )
    if method in _SPECTRAL_METHODS:
        if sfreq is None:
            raise ValueError(f"method={method!r} requires ``sfreq`` to be given.")
        if fmin is None or fmax is None:
            raise ValueError(
                f"method={method!r} requires a frequency band ``fmin`` and ``fmax``."
            )
        for name, edge in (("fmin", fmin), ("fmax", fmax)):
            if np.ndim(edge) != 0:
                raise ValueError(
                    f"method={method!r} estimates one band per call, so ``{name}`` "
                    f"must be a scalar, got {edge!r}. "
                    "``spectral_connectivity_epochs`` would accept the sequence "
                    "and return one matrix per band, but a connectivity matrix "
                    "holds a single value per edge, so every band after the first "
                    "would be dropped without a word. Call once per band and keep "
                    "the matrices apart."
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

    ``orthogonalize`` is ``False`` or ``'pairwise'``. The check for that lives
    in :func:`_validate_conn_params` rather than here, so that it fires at the
    public entry point instead of after a whole sweep of pairwise beamformers
    has been solved for a run that was never going to finish.
    """
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


def _spectral_conn_matrix(data, method, *, sfreq, fmin, fmax, mt_bandwidth):
    """Band-averaged spectral connectivity of an ``(n_epochs, n_signals, n_times)`` set.

    Returns the dense ``(n_signals, n_signals)`` matrix as
    :func:`mne_connectivity.spectral_connectivity_epochs` fills it: only the
    lower triangle is populated, and entry ``[j, i]`` with :math:`j > i` holds
    the connectivity of the *ordered* pair ``(j, i)``. The order is not
    cosmetic, as ``imcoh`` changes sign with it.

    All signals go in one call rather than pair by pair. Each metric is a
    function of one pair's cross-spectrum alone, so the value read out of the
    joint call is the value the two-signal call returns, to round-off (at most
    a few units in the last place apart over the supported metrics -- about
    3e-16 near a value of one).
    """
    from mne_connectivity import spectral_connectivity_epochs

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
    return conn.get_data(output="dense")[..., 0]


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
    matrix = _spectral_conn_matrix(
        data, method, sfreq=sfreq, fmin=fmin, fmax=fmax, mt_bandwidth=mt_bandwidth
    )
    # the dense output is lower-triangular; [1, 0] is the pair.
    return float(matrix[1, 0])


def _check_orientations(orientations, n_sources):
    """Validate ``orientations`` against ``sources``, returning a float array.

    The rows are picked out *by position* in ``sources`` (see
    :func:`_orientations_for`), so an array with the wrong number of rows is
    not merely rejected late: a short one raises an opaque ``IndexError`` from
    inside the pair loop, and one that is merely too long, or that was built
    for a different source list, quietly orients a region along another
    region's dipole and returns a connectivity matrix that looks entirely
    ordinary. Only a check against ``sources``, before anything is built, makes
    that visible.
    """
    if orientations is None:
        return None
    orientations = np.asarray(orientations, float)
    if orientations.shape != (n_sources, 3):
        raise ValueError(
            f"orientations must be ({n_sources}, 3) for {n_sources} sources, got "
            f"{orientations.shape}."
        )
    return orientations


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
    orientations = _check_orientations(orientations, len(sources))
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
        per-frequency and ill-defined for a single edge), and both must be
        scalars: one band per call. Unlike
        :func:`mne_connectivity.spectral_connectivity_epochs` these do not take
        a sequence of band edges, because a connectivity matrix has room for a
        single value per edge; a sequence is rejected rather than quietly
        reduced to its first band.
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
    orthogonalize : False | 'pairwise'
        Envelope leakage-orthogonalisation (Hipp et al., 2012). The default
        ``False`` gives *plain* envelope correlation, as MCMV already removes
        leakage; leakage-orthogonalisation (the symmetric-orthogonalisation
        baseline of Nunes et al., 2020) would double-correct and discard genuine
        zero-lag coupling. ``'pairwise'`` is the only orthogonalisation
        implemented, so a bare ``True`` is rejected rather than read as a
        request for it.
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
        correlations"). This is an FFT resample, so it band-limits the envelope
        to ``envelope_resample / 2`` Hz on the way down: a second smoothing
        stage, not a free saving. It leaves the correlation alone only when that
        limit sits well above the ``envelope_lowpass`` stopband; bring it close
        and the correlation moves. Only used for ``method='envelope'``.

    Returns
    -------
    conn : ndarray, shape (n_sources, n_sources)
        Connectivity matrix; ``conn[i, j]`` is the PW-MCMV connectivity of the
        ordered pair ``(sources[i], sources[j])``. The diagonal is zero. The
        matrix is symmetric for every metric except ``'imcoh'``, which is
        antisymmetric (``conn[j, i] == -conn[i, j]``) because the imaginary part
        of coherency changes sign with the order of the pair; take
        ``np.abs(conn)`` if you want a magnitude.

        Mind the transpose when handing an ``'imcoh'`` matrix to
        :mod:`mne_connectivity`. Its dense output puts the ordered pair
        ``(i, j)`` at ``[j, i]``, so this matrix is that one's transpose, and
        ``plot_connectivity_circle`` reads only ``np.tril_indices(n, -1)``.
        Passing this matrix unchanged therefore draws every edge with the
        opposite lag direction. Pass ``conn.T``, or ``np.abs(conn)`` if the
        direction does not matter. Symmetric metrics are unaffected.
    """
    if sfreq is None and method == "envelope":
        sfreq = float(info["sfreq"])
    _validate_conn_params(
        method, sfreq, fmin, fmax, envelope_lowpass, envelope_resample, orthogonalize
    )
    sources = list(sources)
    n = len(sources)
    orientations = _check_orientations(orientations, n)
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
        # ``imcoh`` is antisymmetric -- ImCoh(a, b) = -ImCoh(b, a) -- so writing
        # one scalar into both triangles would report the same lag direction for
        # both orders of the pair and half the matrix would carry the wrong sign.
        # Every other supported metric is symmetric in its two arguments.
        conn[i, j] = value
        conn[j, i] = -value if method == "imcoh" else value
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
    orthogonalize : False | 'pairwise'
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
        It follows the same convention as
        :func:`~advance_beamlab.pairwise_mcmv_connectivity`: symmetric for every
        metric except ``'imcoh'``, which is antisymmetric in the pair order.
    """
    if sfreq is None and method == "envelope":
        sfreq = float(info["sfreq"])
    _validate_conn_params(
        method, sfreq, fmin, fmax, envelope_lowpass, envelope_resample, orthogonalize
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
    # Keep what was coerced, rather than validating a coerced copy and passing
    # the raw value on: ``max_neighbours`` becomes a slice bound inside
    # :func:`_select_neighbours` and enters the order cap below, and ``radius``
    # is compared against distances, so a count written as a float or a bound
    # read from a text configuration would pass the check here and then fail
    # deep in the neighbour search, long after the caller could tell why.
    max_neighbours = int(max_neighbours)
    radius = float(radius)
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
    orientations = _check_orientations(orientations, n)

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
        # ``imcoh`` is antisymmetric -- ImCoh(a, b) = -ImCoh(b, a) -- so writing
        # one scalar into both triangles would report the same lag direction for
        # both orders of the pair and half the matrix would carry the wrong sign.
        # Every other supported metric is symmetric in its two arguments.
        conn[i, j] = value
        conn[j, i] = -value if method == "imcoh" else value
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
    fmin=None,
    fmax=None,
    orthogonalize=False,
    absolute=False,
    mt_bandwidth=None,
    envelope_lowpass=0.5,
    envelope_resample=None,
    random_state=None,
):
    r"""Significance mask for a connectivity matrix via an AR(1) surrogate null.

    Implements the significance procedure of Nunes et al. (2020), Sec. 2.6-2.7:

    1. Fit an order-one autoregressive model
       :math:`x_t = \varphi x_{t-1} + \varepsilon_t` to each region's
       ``reference_time_courses``, capturing its temporal smoothness.
    2. Generate ``n_surrogates`` sets of *independent* Gaussian AR(1) source
       signals with those per-region coefficients, cut into the epoch geometry
       of the reference (a no-op for the single continuous segment of the
       paper). These surrogates have realistic temporal structure but no
       genuine pairwise coupling.
    3. Compute the same connectivity metric for every surrogate pair, apply the
       Fisher :math:`z`-transform, and take the null mean and standard
       deviation.
    4. Convert the Fisher-transformed real connectivity to :math:`z`-scores
       using that null, then to :math:`p`-values under Gaussianity, and
       threshold with Benjamini-Hochberg FDR at ``alpha``.

    Nunes et al. (2020) prescribe this null for their resting-state
    amplitude-envelope correlations. ``method='envelope'`` is therefore the case
    the paper covers and the case this null is calibrated for. The spectral
    metrics are accepted here as an extension made by this package, on the
    measured behaviour below rather than on the authority of the paper.

    .. warning::
        For the spectral metrics this null is anticonservative. Under a complete
        null (8 mutually independent 9-11 Hz sources, 60 epochs of 2 s at
        200 Hz, band-averaged 8-12 Hz multitaper, 200 surrogates, 20
        realisations, 560 true-negative edges) the uncorrected per-edge
        rejection rate at ``alpha=0.05`` was 0.168 (coh), 0.114 (imcoh), 0.100
        (plv), 0.098 (wpli) and 0.077 (ppc), that is 1.5 to 3.4 times nominal,
        and 0.071, 0.004, 0.023, 0.018 and 0.029 of those true-null edges were
        still flagged after Benjamini-Hochberg. Repeating the simulation with
        sources that really are AR(1) (:math:`\varphi = 0.95`) gives 0.084,
        0.045, 0.079, 0.071 and 0.052 uncorrected: the gap from 0.05 up to
        those is the Gaussian tail approximation of step 4, and the gap from
        those up to the narrow-band rates is the AR(1) model itself (both are
        quantified in Notes). Envelope correlation, the case the paper
        prescribes, is not affected: on the same sources it rejects 0.045
        uncorrected and 0.007 after Benjamini-Hochberg.
        A false edge here does more than waste an APW-MCMV refit, because
        :func:`augmented_pairwise_mcmv_connectivity` ranks augmenting neighbours
        by their number of significant edges, so it also perturbs the
        augmentation of the genuine pairs. Read a spectral screen as a liberal
        filter, and where the exact rate matters prefer a surrogate that
        preserves each region's spectrum (Fourier phase randomisation of the
        reference courses, which this function does not implement).

    Parameters
    ----------
    connectivity : ndarray, shape (n_sources, n_sources)
        The connectivity matrix to test (from
        :func:`pairwise_mcmv_connectivity`).
    reference_time_courses : ndarray
        One representative reconstructed time course per region, of shape
        ``(n_sources, n_times)`` or ``(n_epochs, n_sources, n_times)``, used to
        fit the AR(1) coefficients and to set the surrogate geometry. The AR(1)
        fit is taken on the epochs concatenated, and the surrogates are then cut
        into the same geometry, which must be the geometry ``connectivity`` was
        computed from: the null of a spectral metric moves with the number of
        epochs (for independent signals, band-averaged coherence falls from 0.27
        at 2 epochs to 0.06 at 60). The spectral metrics therefore require a 3D
        reference with at least two epochs. For ``method='envelope'`` a 2D
        reference is the paper's case; a 3D one is enveloped and correlated
        epoch by epoch and averaged, as :func:`pairwise_mcmv_connectivity` does.
    method : 'envelope' | 'coh' | 'imcoh' | 'plv' | 'ciplv' | 'ppc' | 'pli' | 'wpli'
        Connectivity metric that ``connectivity`` holds. It must match the
        metric, the band and the envelope arguments used to compute
        ``connectivity``, or the null is not the null of the tested statistic.
        ``'envelope'`` is the default and the case the paper covers; see the
        warning above before using any of the spectral metrics.
    n_surrogates : int
        Number of surrogate datasets used to estimate the null (default 200).
        Must be at least 2.
    alpha : float
        FDR level (default 0.05, per the paper).
    sfreq : float | None
        Sampling frequency of ``reference_time_courses``. Required for the
        spectral metrics, and for ``method='envelope'`` unless both
        ``envelope_lowpass`` and ``envelope_resample`` are ``None``.
    fmin, fmax : float | None
        Frequency band of the spectral metric, averaged over as in
        :func:`pairwise_mcmv_connectivity`; both are required for the spectral
        metrics and unused for ``method='envelope'``.
    orthogonalize : False | 'pairwise'
        Envelope leakage-orthogonalisation; must match the value used for
        ``connectivity``, or the null will not be the null of the tested
        statistic. Only used for ``method='envelope'``.
    absolute : bool
        Whether the envelope correlation is taken in magnitude; must match the
        value used for ``connectivity``. Only used for ``method='envelope'``.
    mt_bandwidth : float | None
        Multitaper frequency smoothing (Hz) of the spectral metric; must match
        the value used for ``connectivity``. Only used for the spectral metrics.
    envelope_lowpass : float | None
        Envelope low-pass cut-off (Hz); must match the value used for
        ``connectivity``. Only used for ``method='envelope'``.
    envelope_resample : float | None
        Envelope resampling frequency (Hz); must match the value used for
        ``connectivity``. Only used for ``method='envelope'``.
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

    **What the null does and does not control for.** It is the distribution of
    the metric between *independent* regions that have the fitted lag-1
    autocorrelation and variance, at the given recording length and epoch
    geometry, so it controls for the connectivity that such smoothness and such
    a finite sample produce by chance. It is a null of no coupling, not a null
    of no leakage: the surrogates are generated at source level and never pass
    through the beamformer, so leakage that survives PW-MCMV (including the
    indirect leakage APW-MCMV exists to remove) is absent from the null and is
    not controlled for. Nothing about a region beyond its lag-1 autocorrelation
    and variance enters the null, so non-stationarity, spectral shape and
    non-Gaussianity are not controlled for either. Multiplicity across edges is
    controlled only in the Benjamini-Hochberg sense, on the p-values as
    computed.

    **Why the spectral extension is weaker, part one: the model.** An AR(1)
    process has a single real pole, so it is a one-pole low-pass: it can match
    the lag-1 autocorrelation of a narrow-band region and still put almost none
    of its power in the analysis band. Fitted to 9-11 Hz sources it gives
    :math:`\varphi = 0.949`, against the exact
    :math:`\cos(2 \pi \cdot 10 / 200) = 0.951`, yet only 4.6 per cent of its
    power falls in 8-12 Hz against 92.3 per cent for the sources, and its
    spectral peak sits at 0.4 Hz rather than 9.4 Hz. Averaging over the band
    then buys the surrogate 2.51 effectively independent frequency bins where
    the data buy 1.46, because the data's band edges carry only filter leakage
    from the passband and are redundant; the null comes out correctly located
    but 15 to 26 per cent too narrow, and it is the width that sets the
    p-value. Envelope correlation escapes this because the 0.5 Hz low-pass of
    Sec. 2.5 leaves only the slow amplitude fluctuation, which AR(1) does
    reproduce. Matching the band to the signal helps unevenly: narrowing 8-12 Hz
    to 9-11 Hz in the simulation above took coh from 0.168 to 0.116 and imcoh
    from 0.114 to 0.059, but left plv at 0.105.

    **Part two: the tail approximation.** Step 4 reads the p-value off a
    Gaussian, but the Fisher-transformed null of a magnitude that sits close to
    zero is right-skewed: measured over 500 surrogates on the AR(1) control its
    skewness is +0.72 for coh and +0.77 for plv, and the standardised null
    exceeds 1.645 with probability 0.068 rather than the Gaussian 0.05, so the
    upper tail is understated. On the same surrogates an empirical rank p (the
    fraction of surrogates at or above the real value) rejects 0.057 (coh) and
    0.054 (plv) on that control where the Gaussian route rejects 0.084 and
    0.079. The rank p is
    nevertheless not used, because it cannot fall below
    ``1 / (n_surrogates + 1)`` while Benjamini-Hochberg needs a p below
    ``alpha / n_edges`` to retain an isolated edge: at the default 200
    surrogates no single edge among 28 could ever survive. Raising
    ``n_surrogates`` sharpens the null estimate but does not remove this
    approximation.

    The test is one-sided for the non-negative metrics (``'coh'``, ``'plv'``,
    ``'ciplv'``, ``'pli'``, ``'wpli'``), for which an unusually *low* value is
    no evidence of coupling, and two-sided for the signed ones (``'imcoh'``,
    ``'ppc'``, and the envelope correlation, which is signed unless
    ``absolute=True``). One-sidedness is what makes the tail approximation
    visible: it puts the whole of ``alpha`` in the tail the approximation
    understates, where the two-sided test spends half of it on the lower tail,
    which for a magnitude cannot be evidence of anything. The Fisher
    :math:`z`-transform of step 3 is a variance-stabiliser for a correlation
    coefficient, and by extension for coherence magnitude; for the phase metrics
    it is applied only as a monotone rescaling before the standardisation, with
    no distributional claim attached.

    An edge whose surrogate null degenerates (zero or non-finite standard
    deviation, for instance because the surrogates are too short for the
    envelope filter) is reported as *not* significant, with a warning; a
    degenerate null must never read as evidence of an effect.

    Each surrogate set costs one connectivity call for *all* regions rather than
    one per pair, since both metrics are functions of a single pair's data. The
    default 200 surrogates over 8 regions and 60 epochs of 2 s at 200 Hz take
    about 2.4 s for coherence and 1.8 s for the envelope.
    """
    _validate_conn_params(
        method, sfreq, fmin, fmax, envelope_lowpass, envelope_resample, orthogonalize
    )
    n_surrogates = int(n_surrogates)
    if n_surrogates < 2:
        raise ValueError(
            f"n_surrogates must be at least 2 to estimate a null standard "
            f"deviation, got {n_surrogates}."
        )

    rng = np.random.default_rng(random_state)
    connectivity = np.asarray(connectivity, float)
    ref = _epoched(np.asarray(reference_time_courses, float))
    n_epochs, n, n_times = ref.shape
    if connectivity.shape != (n, n):
        raise ValueError(
            f"connectivity {connectivity.shape} is inconsistent with "
            f"{n} reference time courses."
        )
    if method != "envelope" and n_epochs < 2:
        raise ValueError(
            f"method={method!r} needs an epoched ``reference_time_courses`` of "
            "shape (n_epochs, n_sources, n_times) with at least 2 epochs, got a "
            "single segment. On one segment plv, pli and wpli are identically 1 "
            "and ppc is undefined, so the null would carry no information. Pass "
            "the reference in the epoch geometry ``connectivity`` was computed "
            "from."
        )

    # The AR(1) coefficient describes the region's temporal smoothness, a
    # property of the recording rather than of the epoching, so it is fitted on
    # the epochs concatenated (for 2D input this is the course itself).
    phis, sigmas = _ar1_fit(ref.transpose(1, 0, 2).reshape(n, n_epochs * n_times))

    pairs = _as_pairs(n)
    rows = np.array([i for i, _ in pairs])
    cols = np.array([j for _, j in pairs])
    null = np.empty((n_surrogates, len(pairs)))
    for s in range(n_surrogates):
        # Draw one continuous block per region, then cut it into the epoch
        # geometry of the data: a spectral null depends on the epoch count.
        surr = _ar1_surrogate(phis, sigmas, n_epochs * n_times, rng)
        surr = surr.reshape(n, n_epochs, n_times).transpose(1, 0, 2)
        if method == "envelope":
            # One Hilbert transform and one envelope filter per epoch for the
            # whole surrogate set, then read off every pair from the matrix.
            matrix = np.mean(
                [
                    _envelope_corr_matrix(
                        epoch,
                        sfreq=sfreq,
                        orthogonalize=orthogonalize,
                        absolute=absolute,
                        envelope_lowpass=envelope_lowpass,
                        envelope_resample=envelope_resample,
                    )
                    for epoch in surr
                ],
                axis=0,
            )
            null[s] = matrix[rows, cols]
        else:
            # Likewise one spectral call for the whole surrogate set. The dense
            # output is lower-triangular and ordered (j, i), which is the order
            # _pair_connectivity reads and therefore the sign convention of
            # ``connectivity`` for the signed metrics.
            matrix = _spectral_conn_matrix(
                surr,
                method,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                mt_bandwidth=mt_bandwidth,
            )
            null[s] = matrix[cols, rows]

    # Fisher z-transform, standardise by the null mean and std (so the z-scores
    # have zero mean and unit variance under the null, per Colclough et al.,
    # 2015), then p and FDR. Subtracting the null mean matters when the metric
    # is non-negative (e.g. ``absolute=True``); for signed correlation the null
    # mean is ~0 and this reduces to dividing by the null std.
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
    if method in _NONNEGATIVE_METHODS:
        # A magnitude below its null is no evidence of coupling, so testing both
        # tails would spend half of alpha on edges that are unusually *weakly*
        # coherent.
        pvals = norm.sf(zscores)
    else:
        pvals = 2.0 * norm.sf(np.abs(zscores))
    keep = _benjamini_hochberg(pvals, alpha)

    significance = np.zeros((n, n), dtype=bool)
    significance[rows, cols] = keep
    significance[cols, rows] = keep
    return significance
