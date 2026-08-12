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
        Recovered root mean square over true root mean square. One means the
        amplitude survived. Root mean square rather than peak because the
        reconstruction carries filtered sensor noise.
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

        # Two filters, deliberately. The unit-gain filter above reads out the
        # true source amplitude, which is what the reconstruction and the
        # constraint table need, but its output power grows with depth, so a map
        # of it is dominated by one deep source and everything else normalises
        # to nothing. The display map therefore comes from a unit-noise-gain
        # filter, which is depth normalised, exactly as MNE's own examples do.
        display = make_lcmv(
            info,
            forward,
            data_cov,
            reg=reg,
            noise_cov=noise_cov,
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
    for i, colour in enumerate(("C0", "C3")):
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
            f"Recovered {demo.amplitude_ratio[0]:.2f} and "
            f"{demo.amplitude_ratio[1]:.2f} of the truth"
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
