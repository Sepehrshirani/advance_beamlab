"""Normalised power images from a set of beamformer weights.

A beamformer's raw output power is not directly interpretable and not comparable
between locations: it grows with depth, with the filter's noise gain, and with
whatever the covariance happened to contain. The images defined here are the
normalisations that make it interpretable, and they are the form nearly every
beamformer result in the literature is actually reported in.

Three of them, all built from the same two ingredients -- the power a filter
passes from a covariance, and the power it passes from noise
:footcite:`VrbaRobinson2001`:

* **pseudo-Z** normalises one window by the projected noise. It answers "how far
  above the noise floor is this location", and it is the image to use when there
  is no control window to compare against.
* **pseudo-T** is the *differential* image: active minus control, over the noise
  both windows carry. It is what an evoked or induced response is normally
  reported as, because subtracting a control window removes the static
  background that a single window cannot separate from the response.
* **pseudo-F** is the ratio of the two windows rather than their difference. It
  is the natural choice when the effect is multiplicative -- a change in
  oscillatory power rather than an added response -- and it is reported as a
  log ratio by default here, because a ratio image is otherwise wildly
  asymmetric about no-change.

The power comes from each method's own public apply path rather than from its
stored weights. That is not fussiness: the methods keep their weights in
different spaces -- MNE's are in the whitened space, MCMV folds the whitener in
-- so the stored arrays are not comparable and combining them directly gives a
number that means nothing. See the note in :mod:`advance_beamlab._mcmv`.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
from mne.utils import _check_option, _validate_type, logger, verbose

# Below this, a projected-noise power is treated as numerically zero rather than
# divided by. A filter whose noise gain underflows is degenerate, and the image
# value there is not small, it is undefined.
_NOISE_FLOOR = 1e-300

_KINDS = ("pseudo-z", "pseudo-t", "pseudo-f")


def _power(filters, cov):
    """Diagonal of W^T C W, through whichever public apply path fits.

    Both entry points return the full source-by-source matrix; only its diagonal
    is a power, the off-diagonal being the cross-terms the constraint controls.
    """
    from mne.beamformer import apply_lcmv_cov

    from ._mcmv import MCMVBeamformer, apply_mcmv_cov

    if isinstance(filters, MCMVBeamformer):
        out = np.asarray(apply_mcmv_cov(cov, filters), float)
        return np.diag(out) if out.ndim == 2 else np.asarray(out, float).ravel()
    # Silenced deliberately: the apply path reports the rank and whitening of
    # each covariance it is handed, which here is an intermediate step the
    # caller did not ask for and would see two or three times over for a single
    # image. What the image itself came to is logged by the caller instead.
    stc = apply_lcmv_cov(cov, filters, verbose=False)
    return np.asarray(stc.data, float).ravel()


@verbose
def power_image(
    filters,
    active_cov,
    *,
    baseline_cov=None,
    noise_cov=None,
    kind="pseudo-z",
    log_ratio=True,
    verbose=None,
):
    r"""Normalised power image from a set of beamformer weights.

    Parameters
    ----------
    filters : instance of mne.beamformer.Beamformer | instance of MCMVBeamformer
        The spatial filters. Anything :func:`mne.beamformer.apply_lcmv_cov`
        accepts works, which includes the filters
        :func:`~advance_beamlab.make_recipsiicos_lcmv` returns, as does an
        :class:`~advance_beamlab.MCMVBeamformer`.
    active_cov : instance of mne.Covariance
        Covariance of the window being imaged.
    baseline_cov : instance of mne.Covariance | None
        Covariance of the control window. Required for ``'pseudo-t'`` and
        ``'pseudo-f'``, ignored by ``'pseudo-z'``.
    noise_cov : instance of mne.Covariance | None
        Noise covariance, used to project the noise each filter passes. Required
        for ``'pseudo-z'`` and ``'pseudo-t'``. If ``None`` for ``'pseudo-f'``,
        which does not need one, it is not used.
    kind : 'pseudo-z' | 'pseudo-t' | 'pseudo-f'
        Which image to form. See the module docstring for what each is for.
    log_ratio : bool
        For ``'pseudo-f'`` only: return the natural log of the ratio rather than
        the ratio itself. On by default, because a raw ratio image is asymmetric
        about no-change -- a halving is 0.5 and a doubling is 2 -- so a
        symmetric colour scale misrepresents it, and the log is what makes a
        decrease and an increase of the same size look the same size.
    %(verbose)s

    Returns
    -------
    image : ndarray, shape (n_sources,)
        One value per source the filters cover. Unitless in every case: the
        normalisation is what removes the units.

    Notes
    -----
    With :math:`S_a`, :math:`S_c` the power passed from the active and control
    windows and :math:`N` the power passed from the noise covariance,

    .. math::
        Z^2 = \frac{S_a}{N}, \qquad
        T = \frac{S_a - S_c}{2N}, \qquad
        F = \log\frac{S_a}{S_c}

    The square on the left is not a slip, and it is the one place the naming
    here can mislead. ``kind='pseudo-z'`` returns a ratio of *powers*, while the
    pseudo-Z of :footcite:`VrbaRobinson2001` is a ratio of amplitudes -- source
    strength over the standard deviation of the projected noise -- so what comes
    back is the square of the published quantity. Squaring is monotone, so no
    peak moves and no ranking changes, but the numbers are not the same ones: a
    location at three times the noise amplitude reads as 9 here rather than 3,
    and a threshold quoted from a paper has to be squared before it is applied
    to this image. Take the square root if you need the published scale. The
    power ratio is what is returned because it is the quantity the other two
    images are built from -- pseudo-T differences these powers, not their roots
    -- and because it stays linear in the active covariance, so scaling a window
    scales the image by the same factor.

    The factor of two in :math:`T` is the noise entering through both windows
    :footcite:`VrbaRobinson2001`. It is a constant, so it changes the scale of
    the image and never the location of its peaks; it is kept because the
    quantity is conventionally defined with it.

    A location whose projected noise underflows is returned as ``nan`` rather
    than as a very large number. That case is a degenerate filter, and the image
    value there is undefined rather than significant -- reporting it as a huge
    pseudo-Z would put the brightest voxel in the image where the beamformer
    failed most completely. ``'pseudo-f'`` can fail at either end of its ratio:
    a control window the filter passes no power from is the underflow just
    described, and an *active* window it passes no power from sends the ratio to
    zero, whose log is :math:`-\infty`. Both come back as ``nan``, because an
    infinitely negative voxel is no more meaningful than an infinitely positive
    one, and either would take the colour scale of the whole image with it.

    The same normalisation applies to any of the methods in this package. It is
    worth remembering what it does *not* fix: a differential image removes the
    background common to both windows, not the cancellation a correlated
    neighbour causes, which is a property of the filter rather than of the
    window it is applied to.

    Examples
    --------
    >>> from advance_beamlab import power_image  # doctest: +SKIP
    >>> img = power_image(  # doctest: +SKIP
    ...     filters, active_cov, baseline_cov=base_cov,
    ...     noise_cov=noise_cov, kind="pseudo-t",
    ... )

    References
    ----------
    .. footbibliography::
    """
    _check_option("kind", kind, _KINDS)
    _validate_type(log_ratio, bool, "log_ratio")

    if kind in ("pseudo-t", "pseudo-f") and baseline_cov is None:
        raise ValueError(f"kind={kind!r} needs a baseline_cov to contrast against")
    if kind in ("pseudo-z", "pseudo-t") and noise_cov is None:
        raise ValueError(
            f"kind={kind!r} is normalised by the projected noise, so it needs a "
            "noise_cov. Use kind='pseudo-f' for a ratio of two windows, which "
            "does not."
        )

    active = _power(filters, active_cov)
    if kind == "pseudo-f":
        base = _power(filters, baseline_cov)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                base > _NOISE_FLOOR, active / np.maximum(base, _NOISE_FLOOR), np.nan
            )
            # The log stays inside the errstate, and behind a guard of its own:
            # a location the filter passes no power from at all has a ratio of
            # zero, and log(0) is -inf rather than a very large decrease. That
            # is the same degenerate filter the noise floor catches on the
            # other side of the division, so it is reported the same way.
            out = np.where(ratio > 0.0, np.log(ratio), np.nan) if log_ratio else ratio
    else:
        noise = _power(filters, noise_cov)
        usable = noise > _NOISE_FLOOR
        out = np.full(active.shape, np.nan, float)
        if kind == "pseudo-z":
            out[usable] = active[usable] / noise[usable]
        else:
            base = _power(filters, baseline_cov)
            out[usable] = (active[usable] - base[usable]) / (2.0 * noise[usable])

    # The count of undefined locations is the part worth saying out loud: it is
    # invisible in a thresholded image, and an image that is largely nan is
    # reporting a source space the covariances could not support rather than a
    # weak effect.
    n_undefined = int(np.count_nonzero(~np.isfinite(out)))
    peak = np.nanmax(out) if n_undefined < out.size else np.nan
    logger.info(
        f"    {kind} image over {out.size} source(s)"
        f"{' as a log ratio' if kind == 'pseudo-f' and log_ratio else ''}: "
        f"peak {peak:.4g}, {n_undefined} undefined"
    )
    return out
