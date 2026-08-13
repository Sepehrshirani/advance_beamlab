r"""What a beamformer constraint actually does, made measurable.

Every method in this package differs from an ordinary LCMV in its *constraint*,
and the constraint is the part that is hardest to picture from the algebra. This
module turns it into a number anyone can read.

The quantity that matters is the filter's gain at each source,
:math:`\mathbf{w}_i^{\mathsf T}\mathbf{g}_j`: what the filter for source
:math:`i` passes of source :math:`j`. Written out for an ``n``-source scene it is
an ``n`` by ``n`` table, and the four methods differ in it exactly as their
equations say they should:

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
# Rhythms people actually study, plus the transient case ABMC is aimed at.
_MORPHOLOGIES = ("theta", "alpha", "beta", "transient")
_BAND = {"theta": 6.0, "alpha": 10.0, "beta": 20.0}
# How much of the interference is ongoing brain activity rather than sensor
# noise. Real localisation error is dominated by the former, and a simulation
# with only white sensor noise puts the peak exactly on the true grid node
# essentially every time.
_BRAIN_FRACTION = 0.75
_N_BACKGROUND = 300
# How different the true source is from the nearest point the beamformer can
# scan, as the correlation between their leadfields.
#
# Distance is the wrong knob here. On a folded cortex with fixed orientation two
# vertices a millimetre apart can sit on opposite walls of a sulcus with nearly
# orthogonal normals, so selecting by distance alone gave a "1 mm" offset that
# cost a single source 89 per cent of its amplitude. Leadfield correlation is the
# quantity that actually governs how much a filter pointed at the node still
# passes, and it is bounded away from 1 so the source is never exactly
# representable.
_LEADFIELD_MATCH = (0.95, 0.995)
_OFFSET_SEARCH = 0.012


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
        Mean distance between the simulated sources, in metres. Zero for a
        single source.
    correlation : float
        Mean pairwise correlation actually achieved between the simulated time
        courses, which is what should be quoted rather than the requested
        value. Zero for a single source, where it has no meaning.
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
        Recovered root mean square over true root mean square. One means the
        amplitude survived. Root mean square rather than peak because the
        reconstruction carries filtered sensor noise.
    power_map : ndarray, shape (n_grid,)
        The localiser map over the whole source grid, for display.
    peak_errors : ndarray, shape (n_sources,)
        Distance from each true source to the nearest of the map's strongest
        peaks, in metres.
    extra : dict
        ``'morphology'`` and ``'n_sources'`` as requested, the ``'leadfield'``
        columns of the simulated sources and the ``'noise_scale'`` applied to
        the shared noise field. The last two are what let a viewer rebuild
        ``sensor_data`` without storing it.
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


def _pick_sources(forward, n_sources, separation, seed=0):
    """``n_sources`` grid indices, mutually about ``separation`` metres apart.

    The first sits near the middle of the grid so the rest stay inside it
    whatever separation is asked for. Each further source is the grid point at
    about ``separation`` from the centre that is also not too close to anything
    already placed, which keeps three sources spread out instead of stacked.
    """
    rr = forward["source_rr"]
    centre = int(np.argmin(np.linalg.norm(rr - rr.mean(0), axis=1)))
    chosen = [centre]
    for _ in range(1, int(n_sources)):
        score = np.abs(np.linalg.norm(rr - rr[centre], axis=1) - separation)
        for placed in chosen[1:]:
            score = score + np.maximum(
                0.0, separation - np.linalg.norm(rr - rr[placed], axis=1)
            )
        score[chosen] = np.inf
        chosen.append(int(np.argmin(score)))
    return chosen


def _smooth_noise(rng, n_times, sfreq, width):
    """Smooth zero-mean unit-variance noise, for amplitude envelopes.

    The kernel is clamped to stay shorter than the signal. ``np.convolve`` with
    ``'same'`` returns the length of the *longer* input, so a smoothing width
    that exceeds the segment silently returns an array of the wrong length.
    """
    n = int(max(1, min(round(width * sfreq), max(1, (n_times - 1) // 6))))
    kernel = np.exp(-0.5 * (np.arange(-3 * n, 3 * n + 1) / n) ** 2)
    kernel /= kernel.sum()
    x = np.convolve(rng.standard_normal(n_times), kernel, "same")
    return (x - x.mean()) / (x.std() or 1.0)


def _pink(rng, n, n_times):
    """``n`` unit-variance signals with a 1/f spectrum, as ongoing activity has."""
    spectrum = np.fft.rfft(rng.standard_normal((n, n_times)), axis=1)
    freq = np.arange(spectrum.shape[1], dtype=float)
    freq[0] = 1.0
    out = np.fft.irfft(spectrum / np.sqrt(freq), n=n_times, axis=1)
    return out / (out.std(axis=1, keepdims=True) + 1e-30)


def _background(gain, rng, n_times, exclude):
    """Ongoing cortical activity everywhere else on the grid.

    This is what a beamformer is really working against. With only white sensor
    noise the peak of the localiser sits exactly on the true grid node in almost
    every configuration, which flatters every method equally and teaches the
    reader nothing about localisation error.
    """
    n_grid = gain.shape[1]
    pool = np.setdiff1d(np.arange(n_grid), np.asarray(exclude))
    picked = rng.choice(pool, size=min(_N_BACKGROUND, pool.size), replace=False)
    return gain[:, picked] @ _pink(rng, picked.size, n_times)


def _morphology_waveform(morphology, rng, n_times, sfreq):
    """One waveform of the requested kind, with independent randomness."""
    t = np.arange(n_times) / sfreq
    if morphology in _BAND:
        envelope = 1.0 + 0.6 * _smooth_noise(rng, n_times, sfreq, 0.5)
        freq = _BAND[morphology]
        return np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)) * envelope
    # A train of short bursts: the transient, spike-like regime, which is what
    # ABMC is aimed at and where an oscillation is a poor model of the data.
    x = np.zeros(n_times)
    width = max(2.0, 0.02 * sfreq)
    for onset in rng.uniform(0, n_times, max(3, int(n_times / sfreq * 1.5))):
        x += np.exp(-0.5 * ((np.arange(n_times) - onset) / width) ** 2)
    return x - x.mean()


def _simulate(
    true_gain,
    correlation,
    snr,
    n_times,
    sfreq,
    seed,
    morphology="alpha",
    background_gain=None,
):
    """``len(sources)`` sources at ``correlation``, plus sensor noise at ``snr``.

    One shared factor and one private factor per source, orthonormalised and
    mixed as sqrt(r) and sqrt(1 - r), give every pair exactly ``correlation``
    for any number of sources and any morphology. The phase shift this used to
    rely on is exact for two sinusoids and has no counterpart beyond that: three
    waveforms cannot be pairwise correlated at an arbitrary r by phase alone.
    """
    rng = np.random.default_rng(seed)
    n = true_gain.shape[1]
    basis = np.stack(
        [_morphology_waveform(morphology, rng, n_times, sfreq) for _ in range(n + 1)]
    )
    basis -= basis.mean(axis=1, keepdims=True)
    orthonormal = np.linalg.qr(basis.T)[0].T
    r = float(np.clip(correlation, 0.0, 1.0))
    true_tcs = np.sqrt(r) * orthonormal[0] + np.sqrt(1.0 - r) * orthonormal[1 : n + 1]
    amplitude = 20e-9  # 20 nA m, a normal cortical source
    true_tcs = true_tcs / (np.abs(true_tcs).max() or 1.0) * amplitude

    clean = true_gain @ true_tcs
    if background_gain is None:
        background_gain = true_gain
    # A separate generator for the interference, seeded independently of how much
    # randomness the waveforms happened to consume. The realisation is then the
    # same in every scene, so comparisons across correlation, separation and
    # morphology are paired rather than confounded by a different draw, and the
    # panel can reconstruct the recording exactly from one stored field.
    #
    # The interference is mostly other brain activity rather than sensor noise,
    # which is both realistic and the thing that makes localisation error behave
    # like it does on real data.
    interference_rng = np.random.default_rng(seed + 10007)
    brain = _background(background_gain, interference_rng, n_times, ())
    white = interference_rng.standard_normal(clean.shape)
    brain /= brain.std() or 1.0
    white /= white.std() or 1.0
    noise = _BRAIN_FRACTION * brain + (1.0 - _BRAIN_FRACTION) * white
    # Scale to the requested ratio of standard deviations, which is a
    # sensor-level SNR rather than a source-level one.
    scale = clean.std() / (snr * (noise.std() or 1.0))
    times = np.arange(n_times) / sfreq
    return times, true_tcs, clean + scale * noise, scale * noise, scale


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
    n_sources=2,
    sources=None,
    morphology="alpha",
    separation=0.04,
    correlation=0.9,
    snr=3.0,
    n_times=1000,
    sfreq=200.0,
    reg=0.05,
    seed=0,
    true_forward=None,
    off_grid_sources=True,
    template_p=0.03,
    recipsiicos_rank=None,
    verbose=None,
):
    """Simulate a multi-source scene and reconstruct it with one method.

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
    sources : array-like of int | None
        Grid indices to place the sources at, overriding ``n_sources`` and
        ``separation``. Use it to put them in named anatomical regions.
    n_sources : int
        How many sources to simulate, ignored when ``sources`` is given. One has
        nothing to cancel against and is
        the control case; two is the classic correlated pair; three shows what
        happens when a beamformer is given fewer constraints than there are
        active sources.
    morphology : str
        ``'theta'``, ``'alpha'`` or ``'beta'`` for a modulated rhythm at 6, 10 or
        20 Hz, or ``'transient'`` for a train of short bursts, which is the
        regime ABMC is aimed at.
    separation : float
        Requested distance between the sources, in metres. The nearest
        available pair on the grid is used, and the achieved distance is
        reported.
    correlation : float
        Requested correlation between the two source waveforms, achieved
        exactly by a phase shift of ``arccos(correlation)``.
    snr : float
        Ratio of clean sensor standard deviation to noise standard deviation.
    n_times : int
        Samples to simulate. The covariance is estimated from these, so it wants
        to be several times the channel count: below about five samples per
        channel the estimate is rank deficient, MNE says so, and the numbers
        move with it.
    sfreq : float
        Sampling frequency in Hz.
    reg : float
        Covariance regularisation passed to the beamformer.
    seed : int
        Seed for the interference and the source choice.
    true_forward : mne.Forward | None
        A finer forward the simulated sources are taken from, while ``forward``
        stays the grid the beamformer scans. Without it the sources sit exactly
        on scanned grid nodes and the same leadfield generates and inverts the
        data, which is an inverse crime: the localiser then peaks on the true
        node almost every time and reports zero error. Supplying the
        undecimated forward puts the sources where no method can scan them,
        which is the situation on real data. It is also the grid the ongoing
        background activity is drawn from, whether or not the sources move.
    off_grid_sources : bool
        Whether ``true_forward`` moves the sources as well as supplying the
        background. Setting it ``False`` keeps the sources on scanned grid
        points, which is the only way to see what the constraint itself does,
        while leaving the background identical so the two cases stay comparable.
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
    _check_option("morphology", morphology, _MORPHOLOGIES)
    if int(n_sources) < 1:
        raise ValueError(f"n_sources must be at least 1, got {n_sources}")
    gain = forward["sol"]["data"]
    rr = forward["source_rr"]
    if sources is None:
        sources = _pick_sources(forward, int(n_sources), separation, seed=seed)
    else:
        sources = [int(v) for v in sources]
    if len(sources) > 1:
        pairs = [
            float(np.linalg.norm(rr[a] - rr[b]))
            for i, a in enumerate(sources)
            for b in sources[i + 1 :]
        ]
        achieved_sep = float(np.mean(pairs))
    else:
        achieved_sep = 0.0

    # Where the sources really are. Taken from a finer forward when one is
    # supplied, so they sit between the points the beamformer scans rather than
    # exactly on them. Without this the localiser peaks on the true node in
    # essentially every configuration and reports 0 mm, which is an artefact of
    # the simulation rather than a property of any method.
    if true_forward is not None:
        fine_rr = true_forward["source_rr"]
        fine_gain = true_forward["sol"]["data"]
        # The background comes from the fine grid either way. Drawing it from
        # different grids in the two cases would give them different
        # interference, and the comparison between them is the whole point.
        background_gain = fine_gain
    else:
        background_gain = gain

    if true_forward is not None and off_grid_sources:
        place = np.random.default_rng(seed + 991)
        true_idx = []
        for s in sources:
            offsets = np.linalg.norm(fine_rr - rr[s], axis=1)
            near = np.flatnonzero((offsets > 1e-6) & (offsets < _OFFSET_SEARCH))
            column = gain[:, s]
            column = column / (np.linalg.norm(column) or 1.0)
            others = fine_gain[:, near]
            others = others / (np.linalg.norm(others, axis=0) + 1e-30)
            match = np.abs(column @ others)
            ok = near[(match >= _LEADFIELD_MATCH[0]) & (match <= _LEADFIELD_MATCH[1])]
            if ok.size == 0:  # pragma: no cover - only on a very coarse mesh
                ok = near[np.argsort(np.abs(match - np.mean(_LEADFIELD_MATCH)))[:1]]
            true_idx.append(int(place.choice(ok)))
        true_gain = fine_gain[:, true_idx]
        true_positions = fine_rr[true_idx]
    else:
        true_gain = gain[:, sources]
        true_positions = rr[sources]

    times, true_tcs, sensor, noise, noise_scale = _simulate(
        true_gain,
        correlation,
        snr,
        n_times,
        sfreq,
        seed,
        morphology,
        background_gain,
    )
    if len(sources) > 1:
        off = np.corrcoef(true_tcs)[np.triu_indices(len(sources), k=1)]
        achieved_r = float(np.mean(off))
    else:
        achieved_r = 0.0
    logger.info(
        f"    constraint_demo: {method}, {len(sources)} {morphology} source(s), "
        f"separation {achieved_sep * 100:.1f} cm, r = {achieved_r:.3f}, SNR {snr:g}."
    )

    def _cov(x):
        # ``.copy()`` is load bearing. EpochsArray applies the info's projectors
        # to the array it is handed, in place and without copying first, so
        # without this the simulated recording is average referenced behind the
        # caller's back and no longer equals its own leadfield times its own
        # source waveforms plus noise. The covariance is unaffected either way,
        # since EpochsArray projects its own copy regardless.
        ep = mne.EpochsArray(x[None].copy(), info, tmin=0.0, verbose=False)
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

        # Two filters, deliberately. The unit-gain filter above reads out the
        # true source amplitude, which is what the reconstruction and the
        # constraint table need, but its output power grows with depth, so a map
        # of it is dominated by one deep source and everything else normalises
        # to nothing. The display map therefore comes from a unit-noise-gain
        # filter, which is depth normalised, exactly as MNE's own examples do.
        #
        # It has to be the *same method's* filter, though. Building this one with
        # plain ``make_lcmv`` for both branches, which is what this did at first,
        # gave ReciPSIICOS a localiser identical to LCMV's: correlation 1.000 and
        # the same twenty strongest grid points, because the projected covariance
        # never entered the map at all.
        if method == "lcmv":
            display = make_lcmv(
                info,
                forward,
                data_cov,
                reg=reg,
                noise_cov=noise_cov,
                weight_norm="unit-noise-gain",
                verbose=False,
            )
        else:
            display = make_recipsiicos_lcmv(
                info,
                forward,
                data_cov,
                rank=rank,
                noise_cov=noise_cov,
                reg=reg,
                weight_norm="unit-noise-gain",
                verbose=False,
            )
        power_map = np.asarray(
            apply_lcmv_cov(data_cov, display, verbose=False).data
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
    # Root mean square rather than peak: the reconstruction carries filtered
    # sensor noise, and the peak of a noisy trace measures the noise as much as
    # the signal.
    ratio = np.sqrt((reconstructed**2).mean(axis=1) / (true_tcs**2).mean(axis=1))

    peaks = np.argsort(power_map)[::-1][: len(sources)]
    # Measured from where the source actually is, not from the grid node that
    # stands in for it.
    peak_errors = np.array(
        [float(np.linalg.norm(rr[peaks] - p, axis=1).min()) for p in true_positions]
    )

    return ConstraintDemo(
        method=method,
        sources=list(sources),
        positions=true_positions,
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
        extra=dict(
            morphology=morphology,
            n_sources=len(sources),
            # Enough to rebuild the recording exactly: the leadfield columns of
            # the simulated sources, and the factor the shared noise field was
            # scaled by. The web panel uses these instead of storing 203
            # channels per scene, which would be megabytes of incompressible
            # noise.
            leadfield=true_gain,
            true_positions=true_positions,
            noise_scale=float(noise_scale),
        ),
    )


def _draw(axes, demo, forward):
    """Redraw the four panels for one scene. Shared by the explorer and export."""
    rr = forward["source_rr"]
    ax_map, ax_sensor, ax_tc, ax_gain = axes

    # 1. Where the method points. Axial projection of the whole grid, coloured
    #    by the localiser, with the simulated sources marked.
    ax_map.clear()
    # Normalised to its own maximum and drawn with a colormap that is visible at
    # the low end. Clipping at a high percentile with a black-at-zero map, which
    # is the obvious thing to do, paints most of the grid black and hides the
    # result the panel exists to show.
    # Two traps here, both met on the way to this line. Clipping at a high
    # percentile with a black-at-zero colormap paints most of the grid black and
    # hides the result. Spanning the full range instead paints it all one colour,
    # because a depth-normalised localiser is flat: on this fixture its median is
    # 0.42 of the maximum and its 90th percentile only 0.52. Spanning the upper
    # half shows the peak while leaving the rest legible.
    # Displayed by rank rather than by value. The four localisers have very
    # different distributions, and any single value-based scaling that suits one
    # hides another: LCMV's unit-gain power is dominated by one deep source,
    # while a depth-normalised map is nearly flat, with a median of 0.42 of its
    # maximum. Rank is legible for all of them, at the cost of showing order
    # rather than magnitude, which is what the colour bar says.
    order = np.argsort(np.argsort(demo.power_map))
    power = order / (len(order) - 1)
    scatter = ax_map.scatter(
        rr[:, 0] * 100,
        rr[:, 1] * 100,
        c=power,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=46,
        linewidths=0,
    )
    if getattr(ax_map, "_advance_beamlab_cbar", None) is None:
        ax_map._advance_beamlab_cbar = ax_map.figure.colorbar(
            scatter, ax=ax_map, fraction=0.046, pad=0.03
        )
        ax_map._advance_beamlab_cbar.set_label("localiser rank", fontsize=8)
    # Ring the simulated sources rather than covering them: a peaked localiser
    # puts almost all of its energy on those two points, so a filled marker
    # hides the only part of the map worth seeing.
    ax_map.scatter(
        demo.positions[:, 0] * 100,
        demo.positions[:, 1] * 100,
        marker="o",
        s=260,
        facecolors="none",
        edgecolors="#D55E00",
        linewidths=2.2,
        label="simulated",
    )
    label = "scan_mcmv search map" if demo.method == "mcmv" else "localiser"
    ax_map.set(xlabel="x (cm)", ylabel="y (cm)", title=f"Where it points: {label}")
    ax_map.legend(loc="upper right", fontsize=8)
    ax_map.set_aspect("equal")

    # 2. What the sensors record.
    ax_sensor.clear()
    ax_sensor.plot(demo.times, demo.sensor_data[::8].T * 1e6, lw=0.6, color="0.45")
    ax_sensor.set(xlabel="time (s)", ylabel="sensor (uV)", title="What the sensors see")

    # 3. What comes back out.
    ax_tc.clear()
    palette = ("C0", "C3", "C2", "C1", "C4")
    for i in range(demo.true_tcs.shape[0]):
        colour = palette[i % len(palette)]
        ax_tc.plot(
            demo.times,
            demo.true_tcs[i] * 1e9,
            color=colour,
            lw=1.2,
            label=f"source {i + 1}, true",
        )
        ax_tc.plot(
            demo.times,
            demo.reconstructed[i] * 1e9,
            color=colour,
            lw=1.6,
            ls="--",
            label=f"source {i + 1}, recovered",
        )
    ax_tc.set(
        xlabel="time (s)",
        ylabel="amplitude (nA m)",
        title=(
            "Recovered "
            + " and ".join(f"{v:.2f}" for v in demo.amplitude_ratio)
            + " of the truth"
        ),
    )
    ax_tc.legend(fontsize=7, ncol=2)

    # 4. The constraint itself.
    ax_gain.clear()
    g = demo.gains
    ax_gain.imshow(g, cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax_gain.text(
                j,
                i,
                f"{g[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if abs(g[i, j]) > 0.6 else "black",
            )
    ax_gain.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["source 1", "source 2"],
        yticklabels=["filter 1", "filter 2"],
        title="The constraint: gain of each filter at each source",
    )


@verbose
def constraint_explorer(
    info,
    forward,
    *,
    method="lcmv",
    correlation=0.9,
    separation=0.04,
    snr=3.0,
    verbose=None,
):
    """Interactive panel showing what a beamformer constraint does.

    Drag the correlation and watch two things move together: the off-diagonal of
    the constraint table, which is the gain the filter has chosen at the *other*
    source, and the recovered amplitude. For LCMV they move in opposite
    directions, which is signal cancellation caught in the act. Switch to MCMV
    and the off-diagonal is pinned at zero and the amplitude stops moving.

    Requires a live matplotlib backend; in a script call
    ``matplotlib.pyplot.show()`` afterwards. The documentation carries a
    precomputed version of the same panel for readers who have not installed
    the package.

    Parameters
    ----------
    info : mne.Info
        Measurement info matching ``forward``.
    forward : mne.Forward
        Fixed-orientation forward solution.
    method : str
        Method selected when the panel opens.
    correlation : float
        Source correlation when the panel opens.
    separation : float
        Source separation in metres when the panel opens.
    snr : float
        Sensor signal-to-noise ratio when the panel opens.
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The panel. Keep a reference to it, or the widgets stop responding.

    See Also
    --------
    constraint_demo
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(
        2, 2, left=0.30, right=0.97, top=0.94, bottom=0.10, hspace=0.42, wspace=0.28
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    state = dict(method=method, correlation=correlation, separation=separation, snr=snr)

    def refresh(_=None):
        demo = constraint_demo(
            info,
            forward,
            method=state["method"],
            correlation=state["correlation"],
            separation=state["separation"],
            snr=state["snr"],
            verbose=False,
        )
        _draw(axes, demo, forward)
        fig.canvas.draw_idle()

    ax_method = fig.add_axes([0.02, 0.62, 0.22, 0.26])
    ax_method.set_title("method", fontsize=10)
    radio = RadioButtons(ax_method, _METHODS, active=_METHODS.index(method))
    s_corr = Slider(
        fig.add_axes([0.12, 0.50, 0.16, 0.03]),
        "correlation",
        0.0,
        0.99,
        valinit=correlation,
    )
    s_sep = Slider(
        fig.add_axes([0.12, 0.43, 0.16, 0.03]),
        "separation (cm)",
        1.0,
        9.0,
        valinit=separation * 100,
    )
    s_snr = Slider(
        fig.add_axes([0.12, 0.36, 0.16, 0.03]), "SNR", 0.5, 10.0, valinit=snr
    )

    def on_method(label):
        state["method"] = label
        refresh()

    def on_slider(_):
        state["correlation"] = float(s_corr.val)
        state["separation"] = float(s_sep.val) / 100.0
        state["snr"] = float(s_snr.val)
        refresh()

    radio.on_clicked(on_method)
    for slider in (s_corr, s_sep, s_snr):
        slider.on_changed(on_slider)
    # Keep the widgets alive: matplotlib drops them when they are garbage
    # collected, and the panel then silently stops responding.
    fig._advance_beamlab_widgets = (radio, s_corr, s_sep, s_snr)
    refresh()
    return fig
