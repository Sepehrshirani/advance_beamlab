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

# How much of the permutation surface to hold at once, in doubles, when only
# the maximum over it survives. Four mebibytes is small enough to be invisible
# on any machine that can hold the images themselves, and large enough that
# blocking the matrix product costs no measurable time.
_MAX_BLOCK_ELEMENTS = 1 << 19


def _fold_to_tail(values, tail):
    """Orient a statistic onto the side the tail is testing, in place.

    This rewrites its argument, so it must only ever be handed a temporary: the
    two-sided fold is exactly the step that would otherwise double the memory a
    permutation surface costs, and copying it back would defeat the point.
    """
    if tail == 0:
        return np.abs(values, out=values)
    if tail == -1:
        return np.negative(values, out=values)
    return values


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
        One p-value per source. Never zero: the observed arrangement is one of
        the arrangements the null admits, and counting it is what keeps the test
        valid :footcite:`NicholsHolmes2002`. It is counted *among* the draws
        rather than alongside them -- a sampled null uses the observed
        arrangement as its first draw, an exhaustive one contains it by
        construction -- so with ``n`` draws the smallest attainable value is
        ``1 / n``, with one exception. An *exhaustive* two-sided test
        (``tail=0``, taken when ``2 ** n_observations <= n_permutations``)
        enumerates every sign pattern together with its negation, and the fold
        to ``|statistic|`` makes those two draws identical, so its maximum is
        always attained at least twice and its floor is ``2 / n``. One-sided
        exhaustive tests and all sampled tests reach ``1 / n``.
    null : ndarray, shape (n_permutations,) or (n_permutations, n_sources)
        The null distribution actually used -- one maximum per permutation under
        ``'maximum'``, or the full surface under ``'none'``. Returned because a
        p-value from a distribution nobody looked at is worth very little. Mind
        the second shape: ``'none'`` holds a whole image per permutation, which
        on a whole-brain source space is far larger than the data it came from.

    Notes
    -----
    Sign flipping, not label shuffling. For a one-sample test on contrasted
    images the null is that each image's sign is arbitrary, and flipping signs
    is the relabelling that expresses it. This assumes the contrast is symmetric
    under the null, which is what a within-subject difference image is.

    The smallest attainable p-value is set by the number of draws: no number of
    sources can make a result more significant than the resolution of the null
    supports. With ``n`` draws that floor is ``1 / n``, the observed arrangement
    being one of the ``n``. Accounts that quote ``1 / (n + 1)`` instead are
    adding the observed arrangement to ``n`` further ones; the difference is a
    single draw's worth of resolution rather than one of principle, but it is
    worth knowing which convention a number was produced under before comparing
    two of them. If the smallest p-value in the image sits on the floor, the
    test has run out of resolution, and more permutations -- not more
    interpretation -- is the answer.

    The statistic is the raw mean over observations, which is a deliberate
    choice with a cost worth stating. The single-threshold maximum-statistic
    test controls the family-wise rate *exactly* when the statistic is pivotal,
    that is, when its null distribution is the same at every location
    :footcite:`NicholsHolmes2002`. A mean is not: a source whose variance across
    observations is large has a wider null than a quiet one, so the single
    threshold read off the maximum is set by the noisiest locations and the test
    is conservative at all the others. Where that variance is roughly uniform
    across the image this costs little; where it is not -- deep sources against
    superficial ones, or an image spanning regions the filter resolves very
    differently -- a real effect at a quiet location can be missed entirely.
    The remedy is available without leaving this function: divide each source's
    column by its root-mean-square over observations before passing the images
    in. Sign flipping leaves that denominator untouched, so the test stays
    exact, and the resulting statistic is a monotone function of the one-sample
    t at that location, which is pivotal. The mean is what is computed by
    default because it is the quantity the image is already in, and because a
    per-source variance estimated from a handful of subjects is itself unstable
    enough to make the studentised version the worse choice in a small study.

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

    # Out of place, deliberately: `observed` is returned to the caller as the
    # image's own statistic, so it must survive the fold unchanged.
    if tail == 0:
        stat = np.abs(observed)
    elif tail == 1:
        stat = observed
    else:
        stat = -observed

    if correction == "maximum":
        # Only one number per draw survives the maximum, so the surface is
        # built a block of draws at a time and reduced as it goes. Held whole
        # it is an (n_draws, n_sources) array of doubles, and the two-sided
        # fold would cost that again; on a whole-brain image at a thousand
        # permutations that is the difference between a test that runs and one
        # that cannot be started. Blocking splits the product by row, and each
        # entry's sum runs over observations alone, so no sum is ever divided
        # between blocks: every entry is formed from exactly the terms it would
        # have been, up to the order BLAS chooses to add them in.
        block = max(1, min(n_draws, _MAX_BLOCK_ELEMENTS // max(n_src, 1)))
        null = np.empty(n_draws)
        for start in range(0, n_draws, block):
            stop = min(start + block, n_draws)
            part = (signs[start:stop] @ data) / n_obs
            null[start:stop] = _fold_to_tail(part, tail).max(axis=1)
        # Counting by comparison would build an (n_sources, n_draws) boolean
        # array and give the saving straight back. A sorted null gives the same
        # count exactly: the number of draws at or above a value is n_draws
        # less the leftmost position it could be inserted at, which counts the
        # ties that `>=` counts and that the test's validity rests on.
        ordered = np.sort(null)
        p_values = (n_draws - np.searchsorted(ordered, stat, side="left")) / n_draws
    else:
        # Uncorrected, each source is judged against its own column, so the
        # whole surface is needed here -- and is what gets returned. The fold
        # is in place because the surface is a temporary nobody else holds.
        null = _fold_to_tail((signs @ data) / n_obs, tail)
        p_values = (null >= stat[None, :]).sum(axis=0) / n_draws

    p_values = np.clip(p_values, 1.0 / n_draws, 1.0)
    logger.info(
        f"    Permutation test: {n_obs} observations, {n_draws} sign flips"
        f"{' (exhaustive)' if exhaustive else ''}, correction={correction!r}; "
        f"smallest p = {p_values.min():.4g}"
    )
    return observed, p_values, null
