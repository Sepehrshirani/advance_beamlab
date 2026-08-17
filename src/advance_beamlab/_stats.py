"""Non-parametric significance for beamformer images.

A beamformer image is not something a parametric test can be pointed at. Its
values are ratios of quadratic forms in the data, its noise is not Gaussian by
the time it has been through an adaptive filter, and -- the part that catches
people -- its spatial smoothness is neither uniform nor known in advance,
because the filter's resolution varies with depth, with the local geometry and
with the source strength itself. A threshold derived from a nominal distribution
does not mean what it says on such an image.

Permutation removes the need for one. The exchangeability is the experiment's,
not the model's: if the condition labels are arbitrary under the null, then any
relabelling gives an image that could equally have been observed, and the
observed image is judged against the distribution its own relabellings produce
:footcite:`NicholsHolmes2002`.

Two corrections are provided, and the difference between them matters.
Correcting each location against its own null controls the error rate *at that
location*, which is not what you want across thousands of them; the maximum
statistic across the image controls the family-wise rate instead, and it does so
without assuming anything about how the image is smoothed
:footcite:`NicholsHolmes2002`. The second is the default here, because the first
is only ever appropriate at a location chosen in advance.

This is deliberately the simplest form that is correct: a one-sample test over
subjects or trials on images that have already been contrasted. It does not
cluster, and does not try to be a general linear model.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
from mne.utils import _check_option, _validate_type, logger, verbose

_CORRECTIONS = ("maximum", "none")


@verbose
def permutation_image_test(
    images,
    *,
    n_permutations=1024,
    correction="maximum",
    tail=0,
    seed=None,
    verbose=None,
):
    r"""Sign-flip permutation test on a stack of beamformer images.

    Parameters
    ----------
    images : array-like, shape (n_observations, n_sources)
        One already-contrasted image per subject or per trial. Contrasted
        matters: the test asks whether the mean of these is further from zero
        than relabelling would produce, so a stack of raw power images with a
        large positive mean is significant everywhere and says nothing.
    n_permutations : int
        How many sign flips to draw. With ``n`` observations there are
        :math:`2^n` distinct flips; when that is not more than
        ``n_permutations`` every one is used and the test is exact, and the
        count is reported.
    correction : 'maximum' | 'none'
        ``'maximum'`` compares each location against the distribution of the
        *largest* statistic anywhere in the image, which controls the
        family-wise error rate over the whole image. ``'none'`` compares each
        location against its own distribution, which controls only the rate at
        that location and is appropriate solely where the location was chosen in
        advance.
    tail : int
        ``0`` for two-sided, ``1`` for greater than zero, ``-1`` for less.
    seed : int | None
        Seed for the permutation draw when it is random rather than exhaustive.
    %(verbose)s

    Returns
    -------
    observed : ndarray, shape (n_sources,)
        The observed statistic, the mean over observations.
    p_values : ndarray, shape (n_sources,)
        One p-value per source. Never zero: with ``n`` draws the smallest
        attainable value is ``1 / (n + 1)``, because the observed arrangement is
        one of the arrangements the null admits and counting it is what keeps
        the test valid :footcite:`NicholsHolmes2002`.
    null : ndarray, shape (n_permutations,) or (n_permutations, n_sources)
        The null distribution actually used -- one maximum per permutation under
        ``'maximum'``, or the full surface under ``'none'``. Returned because a
        p-value from a distribution nobody looked at is worth very little.

    Notes
    -----
    Sign flipping, not label shuffling. For a one-sample test on contrasted
    images the null is that each image's sign is arbitrary, and flipping signs
    is the relabelling that expresses it. This assumes the contrast is symmetric
    under the null, which is what a within-subject difference image is.

    The smallest attainable p-value is set by ``n_permutations``: no number of
    sources can make a result more significant than the resolution of the null
    supports. If the smallest p-value in the image equals ``1 / (n + 1)``, the
    test has run out of resolution and more permutations, not more
    interpretation, is the answer.

    Examples
    --------
    >>> from advance_beamlab import permutation_image_test  # doctest: +SKIP
    >>> obs, p, null = permutation_image_test(subject_images)  # doctest: +SKIP

    References
    ----------
    .. footbibliography::
    """
    _check_option("correction", correction, _CORRECTIONS)
    _check_option("tail", tail, (-1, 0, 1))
    _validate_type(n_permutations, "int", "n_permutations")

    data = np.asarray(images, float)
    if data.ndim != 2:
        raise ValueError(
            f"images must be (n_observations, n_sources), got shape {data.shape}"
        )
    n_obs, n_src = data.shape
    if n_obs < 2:
        raise ValueError(f"need at least two observations, got {n_obs}")
    if not np.all(np.isfinite(data)):
        raise ValueError(
            "images contains non-finite values; a beamformer image can carry "
            "nan where a filter was degenerate, and those locations have to be "
            "removed or filled before testing rather than propagated"
        )
    if int(n_permutations) < 1:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}")

    observed = data.mean(axis=0)

    # Exhaustive when the sign-flip space is small enough, which also makes the
    # test exact rather than approximate. 2**n_obs grows fast, so this only
    # applies to small studies -- exactly where a sampled null is least reliable.
    exhaustive = n_obs <= 20 and 2**n_obs <= int(n_permutations)
    if exhaustive:
        n_draws = 2**n_obs
        bits = np.arange(n_draws)[:, None] >> np.arange(n_obs)[None, :]
        signs = 1.0 - 2.0 * (bits & 1).astype(float)
    else:
        n_draws = int(n_permutations)
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_draws, n_obs))
        # The observed arrangement is one the null admits, so it belongs in the
        # distribution it is being compared against. Without this the smallest
        # p-value is zero, which no finite permutation test can justify.
        signs[0] = 1.0

    means = (signs @ data) / n_obs  # (n_draws, n_sources)

    if tail == 0:
        stat, null_stat = np.abs(observed), np.abs(means)
    elif tail == 1:
        stat, null_stat = observed, means
    else:
        stat, null_stat = -observed, -means

    if correction == "maximum":
        null = null_stat.max(axis=1)  # (n_draws,)
        p_values = (null[None, :] >= stat[:, None]).sum(axis=1) / n_draws
    else:
        null = null_stat
        p_values = (null_stat >= stat[None, :]).sum(axis=0) / n_draws

    p_values = np.clip(p_values, 1.0 / n_draws, 1.0)
    logger.info(
        f"    Permutation test: {n_obs} observations, {n_draws} sign flips"
        f"{' (exhaustive)' if exhaustive else ''}, correction={correction!r}; "
        f"smallest p = {p_values.min():.4g}"
    )
    return observed, p_values, null
