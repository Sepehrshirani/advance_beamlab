r"""What a beamformer constraint actually does, made measurable.

Every method in this package differs from an ordinary LCMV in its *constraint*,
and the constraint is the part that is hardest to picture from the algebra. This
module turns it into a number anyone can read.

The quantity that matters is the filter's gain at each source,
:math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j`: what the filter for source
:math:`i` passes of source :math:`j`. Written out for a two-source scene it is a
2x2 table, and the four methods differ in it exactly as their equations say they
should:

* **LCMV** fixes the diagonal, :math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_i = 1`,
  and leaves the off-diagonal to whatever minimises output power. When the two
  sources are correlated, minimising output power means using one to cancel the
  other, so the off-diagonal is driven away from zero and the recovered
  amplitude collapses. The table shows the cancellation happening.
* **MCMV** :footcite:`Moiseev2011` fixes the whole table to the identity. The
  off-diagonal is zero by construction rather than by luck, so there is nothing
  for the filter to cancel with.
* **ReciPSIICOS** :footcite:`KuznetsovaEtAl2021` leaves the constraint exactly as
  LCMV has it and edits the data covariance instead, removing the cross-source
  structure that made cancelling profitable. Its table therefore looks like
  LCMV's in form, with an off-diagonal that no longer runs away.
* **ABMC** :footcite:`Shirani2024` trades the distortionless constraint against a
  template-match term as ``P`` rises, so its diagonal is not pinned at one.

Reading the table through the public ``apply`` functions rather than out of each
method's weight array is deliberate. The methods store their weights in
different spaces (MCMV folds the whitener in, ReciPSIICOS works in a reduced
virtual-sensor space), so the arrays are not comparable, while the response to a
known input is. Feeding in a scene that contains only source :math:`j` and
reading what comes out of filter :math:`i` measures
:math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j` whatever the method did internally.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

from dataclasses import dataclass, field

import numpy as np
from mne.utils import _check_option, logger, verbose

_METHODS = ("lcmv", "mcmv", "recipsiicos", "abmc")


@dataclass
class ConstraintDemo:
    r"""One simulated scene, reconstructed by one method.

    Attributes
    ----------
    method : str
        The method used.
    sources : list of int
        Grid indices of the simulated sources.
    positions : ndarray, shape (n_sources, 3)
        Their positions in head coordinates, in metres.
    separation : float
        Distance between the two simulated sources, in metres.
    correlation : float
        Correlation actually achieved between the simulated time courses, which
        is what should be quoted rather than the requested value.
    times : ndarray, shape (n_times,)
        Time axis in seconds.
    true_tcs : ndarray, shape (n_sources, n_times)
        The simulated source waveforms, in ampere-metres.
    sensor_data : ndarray, shape (n_channels, n_times)
        The simulated sensor recording, noise included.
    gains : ndarray, shape (n_sources, n_sources)
        The constraint table. ``gains[i, j]`` is
        :math:`\mathbf{w}_i^{\\mathsf T}\mathbf{g}_j`, the gain of the filter
        for source ``i`` at source ``j``, measured by passing a scene containing
        only source ``j`` through the filter. The diagonal is what the
        distortionless constraint pins; the off-diagonal is what the method does
        or does not control.
    reconstructed : ndarray, shape (n_sources, n_times)
        Recovered waveforms, in ampere-metres.
    amplitude_ratio : ndarray, shape (n_sources,)
        Recovered peak over true peak. One means the amplitude survived.
    power_map : ndarray, shape (n_grid,)
        The localiser map over the whole source grid, for display.
    peak_errors : ndarray, shape (n_sources,)
        Distance from each true source to the nearest of the map's strongest
        peaks, in metres.
    """

    method: str
    sources: list
    positions: np.ndarray
    separation: float
    correlation: float
    times: np.ndarray
    true_tcs: np.ndarray
    sensor_data: np.ndarray
    gains: np.ndarray
    reconstructed: np.ndarray
    amplitude_ratio: np.ndarray
    power_map: np.ndarray
    peak_errors: np.ndarray
    extra: dict = field(default_factory=dict)

    def __repr__(self):  # pragma: no cover - display only
        off = self.gains[0, 1] if self.gains.shape[0] > 1 else np.nan
        return (
            f"<ConstraintDemo | {self.method} | r={self.correlation:.2f} | "
            f"off-diagonal gain {off:+.3f} | "
            f"amplitude {np.array2string(self.amplitude_ratio, precision=2)}>"
        )


def _pick_pair(forward, separation, seed=0):
    """Two grid indices about ``separation`` metres apart, deterministically."""
    rr = forward["source_rr"]
    rng = np.random.default_rng(seed)
    # Start from a source near the centre of the grid so that both members of
    # the pair stay inside it whatever the separation asked for.
    centre = int(np.argmin(np.linalg.norm(rr - rr.mean(0), axis=1)))
    distances = np.linalg.norm(rr - rr[centre], axis=1)
    partner = int(np.argmin(np.abs(distances - separation)))
    if partner == centre:  # pragma: no cover - only if the grid is degenerate
        partner = int(rng.integers(len(rr)))
    return [centre, partner]


def _simulate(gain, sources, correlation, snr, n_times, sfreq, seed):
    """Two correlated sources at ``correlation``, plus sensor noise at ``snr``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq
    # Two 10 Hz oscillations phase shifted by arccos(rho) correlate at exactly
    # rho, which is what makes the requested correlation reproducible.
    phi = np.arccos(np.clip(correlation, -1.0, 1.0))
    amplitude = 20e-9  # 20 nA m, a normal cortical source
    true_tcs = np.stack(
        [
            np.sin(2 * np.pi * 10 * t) * amplitude,
            np.sin(2 * np.pi * 10 * t + phi) * amplitude,
        ]
    )
    clean = gain[:, sources] @ true_tcs
    noise = rng.standard_normal(clean.shape)
    # Scale the noise to the requested ratio of standard deviations, which is a
    # sensor-level SNR rather than a source-level one.
    noise *= clean.std() / (snr * noise.std())
    return t, true_tcs, clean + noise, noise


def _measure_gains(apply_fn, gain, sources, n_times):
    r"""Gain of every filter at every source, via the public apply path.

    Feeds a scene containing only source ``j`` and reads what filter ``i``
    returns. That is :math:`\mathbf{w}_i^{\\mathsf T}\mathbf{g}_j` by
    definition, and unlike the stored weight arrays it is directly comparable
    across methods, which keep their weights in different spaces.
    """
    n = len(sources)
    table = np.empty((n, n))
    probe = np.zeros(n_times)
    probe[n_times // 2] = 1.0
    for j, src in enumerate(sources):
        out = apply_fn(np.outer(gain[:, src], probe))
        table[:, j] = out[:, n_times // 2]
    return table


@verbose
def constraint_demo(
    info,
    forward,
    *,
    method="lcmv",
    separation=0.04,
    correlation=0.9,
    snr=3.0,
    n_times=200,
    sfreq=200.0,
    reg=0.05,
    seed=0,
    template_p=0.03,
    recipsiicos_rank=None,
    verbose=None,
):
    """Simulate a two-source scene and reconstruct it with one method.

    Everything the interactive panel shows comes from here, so that the panel
    and the local explorer cannot drift apart.

    Parameters
    ----------
    info : mne.Info
        Measurement info matching ``forward``.
    forward : mne.Forward
        Fixed-orientation forward solution.
    method : str
        ``'lcmv'``, ``'mcmv'``, ``'recipsiicos'`` or ``'abmc'``.
    separation : float
        Requested distance between the two sources, in metres. The nearest
        available pair on the grid is used, and the achieved distance is
        reported.
    correlation : float
        Requested correlation between the two source waveforms, achieved
        exactly by a phase shift of ``arccos(correlation)``.
    snr : float
        Ratio of clean sensor standard deviation to noise standard deviation.
    n_times : int
        Samples to simulate.
    sfreq : float
        Sampling frequency in Hz.
    reg : float
        Covariance regularisation passed to the beamformer.
    seed : int
        Seed for the sensor noise and the pair choice.
    template_p : float
        ``P`` for ABMC. Ignored by the other methods.
    recipsiicos_rank : int | None
        Projection rank for ReciPSIICOS. ``None`` selects it from the rank
        curve. Ignored by the other methods.
    %(verbose)s

    Returns
    -------
    demo : instance of ConstraintDemo
        The scene, the constraint table and the reconstruction.

    Notes
    -----
    The constraint table is measured through the public apply path rather than
    read out of the stored weights, because the methods keep their weights in
    different spaces and the arrays are not comparable. See the module
    docstring.

    MCMV is given the true source locations, as its definition requires, so its
    ``power_map`` is the sequential search's localiser rather than a map the
    filter itself produces. That difference is the point rather than an
    inconsistency: MCMV reconstructs sources it is told about, and
    :func:`~advance_beamlab.scan_mcmv` is what finds them.
    """
    import mne
    from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv

    from ._abmc import make_abmc
    from ._localizers import scan_mcmv
    from ._mcmv import apply_mcmv, make_mcmv
    from ._recipsiicos import make_recipsiicos_lcmv, recipsiicos_rank_curve

    _check_option("method", method, _METHODS)
    gain = forward["sol"]["data"]
    rr = forward["source_rr"]
    sources = _pick_pair(forward, separation, seed=seed)
    achieved_sep = float(np.linalg.norm(rr[sources[0]] - rr[sources[1]]))

    times, true_tcs, sensor, noise = _simulate(
        gain, sources, correlation, snr, n_times, sfreq, seed
    )
    achieved_r = float(np.corrcoef(true_tcs)[0, 1])
    logger.info(
        f"    constraint_demo: {method}, separation {achieved_sep * 100:.1f} cm, "
        f"r = {achieved_r:.3f}, SNR {snr:g}."
    )

    def _cov(x):
        ep = mne.EpochsArray(x[None], info, tmin=0.0, verbose=False)
        return mne.compute_covariance(ep, method="empirical", verbose=False)

    data_cov, noise_cov = _cov(sensor), _cov(noise)

    if method in ("lcmv", "recipsiicos"):
        if method == "lcmv":
            filters = make_lcmv(
                info,
                forward,
                data_cov,
                reg=reg,
                noise_cov=noise_cov,
                weight_norm=None,
                verbose=False,
            )
        else:
            rank = (
                recipsiicos_rank
                if recipsiicos_rank is not None
                else int(
                    recipsiicos_rank_curve(
                        forward,
                        info,
                        data_cov=data_cov,
                        noise_cov=noise_cov,
                        return_optimal=True,
                        verbose=False,
                    )[3]
                )
            )
            filters = make_recipsiicos_lcmv(
                info,
                forward,
                data_cov,
                rank=rank,
                noise_cov=noise_cov,
                reg=reg,
                weight_norm=None,
                verbose=False,
            )

        def apply_fn(x):
            ev = mne.EvokedArray(x, info, tmin=0.0, verbose=False)
            return np.asarray(apply_lcmv(ev, filters, verbose=False).data)[sources]

        power_map = np.asarray(
            apply_lcmv_cov(data_cov, filters, verbose=False).data
        ).ravel()

    elif method == "mcmv":
        bf = make_mcmv(
            info,
            forward,
            data_cov,
            sources,
            noise_cov=noise_cov,
            reg=reg,
            weight_norm="unit-gain",
            verbose=False,
        )

        def apply_fn(x):
            ev = mne.EvokedArray(x, info, tmin=0.0, verbose=False)
            return np.asarray(apply_mcmv(ev, bf, verbose=False).data)

        scan = scan_mcmv(
            info,
            forward,
            data_cov,
            n_sources=1,
            noise_cov=noise_cov,
            reg=reg,
            verbose=False,
        )
        power_map = np.nan_to_num(np.asarray(scan["maps"][0]))

    else:  # abmc
        template = true_tcs[0] / np.abs(true_tcs[0]).max()
        result = make_abmc(
            info,
            forward,
            sensor,
            template,
            noise_cov=noise_cov,
            P=template_p,
            return_weights=True,
            verbose=False,
        )
        w = result.weights[:, sources]

        def apply_fn(x):
            return w.T @ x

        power_map = np.abs(np.asarray(result.template_match))

    gains = _measure_gains(apply_fn, gain, sources, n_times)
    reconstructed = apply_fn(sensor)
    ratio = np.abs(reconstructed).max(axis=1) / np.abs(true_tcs).max(axis=1)

    peaks = np.argsort(power_map)[::-1][: len(sources)]
    peak_errors = np.array(
        [float(np.linalg.norm(rr[peaks] - rr[s], axis=1).min()) for s in sources]
    )

    return ConstraintDemo(
        method=method,
        sources=list(sources),
        positions=rr[sources],
        separation=achieved_sep,
        correlation=achieved_r,
        times=times,
        true_tcs=true_tcs,
        sensor_data=sensor,
        gains=gains,
        reconstructed=reconstructed,
        amplitude_ratio=ratio,
        power_map=power_map,
        peak_errors=peak_errors,
    )
