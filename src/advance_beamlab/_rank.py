"""Finding the usable rank of a covariance from its own eigenspectrum.

Every beamformer here inverts a covariance, and every one of them behaves badly
if that covariance is rank deficient and the inverse is not told. Real data is
rank deficient far more often than not: an average EEG reference costs exactly
one dimension, SSP costs one per projector, ICA one per removed component, and
signal-space separation typically leaves 64 to 80 of 306 channels' worth of
independent dimensions.

The usual advice is to pass the rank in. That works when you know it, and the
point of this module is the case where you do not -- someone else's data, a
pipeline you did not run, or a combination of steps whose costs are not
separately recorded. Both estimators here read the rank off the eigenvalue
spectrum, which carries it whether or not anything wrote it down.

Neither replaces :func:`mne.compute_rank`, which should be preferred when the
provenance is available: it reads the actual SSS and SSP bookkeeping out of
``info`` rather than inferring anything. These are for when it is not, and for
checking it when it is.

The two disagree in a useful way. ``'cliff'`` looks for the largest jump in the
log spectrum, which is sharp and unambiguous when a projection has zeroed
dimensions outright. ``'variance'`` keeps the smallest number of components
reaching a fraction of the total, which degrades gracefully when the drop is a
slope rather than a cliff. When they disagree, the spectrum is worth looking at
by eye, and :func:`rank_spectrum` returns it for exactly that.
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
from mne import Covariance
from mne.utils import _check_option, _validate_type, logger, verbose

_METHODS = ("cliff", "variance")

# A jump this many robust deviations above the typical one counts as the cliff.
# Five is the threshold the diagnostic is conventionally used with; it is
# exposed as ``threshold`` because a borderline spectrum should be argued about
# rather than silently resolved.
_CLIFF_Z = 5.0

# And it must be a fall of at least this many decades. The standardised test on
# its own is not enough: the top eigenvalue of any sample covariance sits a
# little apart from the rest, so on a smooth full-rank spectrum the very first
# drop clears five deviations and the estimate comes back as rank 1. Measured,
# the separation is not close -- a projection cliff is about 15 decades, and the
# largest drop on a well-conditioned full-rank covariance is 0.04 -- so a floor
# of one decade separates them with three orders of magnitude to spare.
_CLIFF_DECADES = 1.0
# What a projection leaves below its cliff, relative to the largest eigenvalue.
# Measured on the ``sample`` recording, the residual below the steepest drop of
# a FULL-RANK covariance is 6.7e-2 (EEG), 1.1e-1 (magnetometers) and 4.0e-1
# (gradiometers) -- real signal, whole percent of the largest eigenvalue. A
# genuine projection leaves round-off, some 1e-16. Six orders of magnitude
# separate the two, so the line is drawn in the middle of that gap rather than
# at either edge of it.
_CLIFF_RESIDUAL = 1e-6


def _eigenvalues(cov):
    """Descending eigenvalues of whatever the caller passed."""
    if isinstance(cov, Covariance):
        data = np.asarray(cov.data, float)
    else:
        data = np.asarray(cov, float)
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(
                f"cov must be square or an mne.Covariance, got shape {data.shape}"
            )
    values = np.linalg.eigvalsh(0.5 * (data + data.T))[::-1]
    return np.maximum(values, 0.0)


@verbose
def rank_spectrum(cov, *, verbose=None):
    """Eigenvalues of a covariance, largest first, for inspection.

    Parameters
    ----------
    cov : instance of mne.Covariance | ndarray, shape (n_channels, n_channels)
        The covariance to inspect.
    %(verbose)s

    Returns
    -------
    eigenvalues : ndarray, shape (n_channels,)
        Descending, clipped at zero. Numerically negative eigenvalues of a
        symmetric positive-semidefinite matrix are round-off and are reported as
        zero rather than as small negative numbers.

    Notes
    -----
    Plotting ``np.log10`` of this is the single most useful diagnostic before
    beamforming. A cliff says a projection has removed dimensions outright; a
    smooth decay says the covariance is merely ill-conditioned, which is a
    different problem with a different fix. Two cliffs usually mean two sensor
    types with different units, which should be whitened rather than truncated.
    """
    values = _eigenvalues(cov)
    logger.info(
        f"    Covariance spectrum: {values.size} values, "
        f"condition number {values[0] / max(values[-1], 1e-300):.3g}"
    )
    return values


@verbose
def estimate_rank(cov, *, method="cliff", threshold=None, pct_var=0.999, verbose=None):
    r"""Estimate the usable rank of a covariance from its eigenspectrum.

    Parameters
    ----------
    cov : instance of mne.Covariance | ndarray, shape (n_channels, n_channels)
        The covariance whose rank is wanted.
    method : 'cliff' | 'variance'
        ``'cliff'`` finds the largest drop in the log spectrum: the point where
        the eigenvalues fall off, which is what a projection leaves behind.
        ``'variance'`` keeps the fewest components whose cumulative eigenvalue
        mass reaches ``pct_var``.
    threshold : float | None
        For ``'cliff'``: how many robust deviations above the typical step a
        drop must be to count. ``None`` uses 5. Lower it if a known truncation
        is being missed, raise it if noise in the tail is being called a cliff.
    pct_var : float
        For ``'variance'``: the fraction of total eigenvalue mass to keep.
    %(verbose)s

    Returns
    -------
    rank : int
        The estimated rank, at least 1 and at most ``n_channels``.

    Notes
    -----
    Prefer :func:`mne.compute_rank` when the provenance is available: it reads
    the SSS and SSP bookkeeping out of ``info`` instead of inferring anything,
    and a recorded fact beats an estimate. This is for data whose history you do
    not have, and for checking a recorded rank against what the numbers say.

    ``'cliff'`` reports the first drop in :math:`-\Delta\log_{10}\lambda`
    that is both unusual for the spectrum -- more than ``threshold`` deviations
    above its median -- and an absolute fall of at least one decade. Both
    conditions are needed. The standardised test alone calls the top eigenvalue
    of any sample covariance a cliff, because it always sits slightly apart from
    the rest, and returns a rank of 1 on perfectly good data. The two regimes
    are far apart once measured: a projection cliff is a fall of some fifteen
    decades, and the largest drop on a well-conditioned full-rank covariance is
    about four hundredths of one.

    On a covariance of full rank ``'cliff'`` returns ``n_channels``.
    ``'variance'`` does so only when the spectrum is nearly flat: it answers
    "how many dimensions hold ``pct_var`` of the variance", and a real M/EEG
    spectrum is coloured, so at the default 0.999 a genuinely full-rank
    magnetometer covariance from the ``sample`` recording comes back as 63 of
    102. That is a variance-truncation point, not a rank, and it is the right
    number for a different question. Where the two disagree, plot
    :func:`rank_spectrum` and decide by eye.

    A caution that matters more than the choice of method: a rank estimate from
    a covariance mixing sensor types with different units is close to
    meaningless, because the spectrum then has one cliff per type and the
    largest is an artefact of the units. Whiten first, or estimate per type.
    """
    _check_option("method", method, _METHODS)
    _validate_type(pct_var, "numeric", "pct_var")
    # Strictly below one. "All of the variance" is not a question the spectrum
    # can answer: the cumulative sum reaches one only at the last eigenvalue, so
    # where the search lands is decided by floating-point accumulation rather
    # than by the data, and the same request returned the true rank on one
    # covariance and the full channel count on the next. Someone who wants every
    # usable dimension wants method='cliff'.
    if not 0 < float(pct_var) < 1:
        raise ValueError(
            f"pct_var must be in (0, 1), got {pct_var}. For every usable "
            f"dimension rather than a variance fraction, use method='cliff'."
        )

    values = _eigenvalues(cov)
    n = values.size
    if n == 0:
        raise ValueError("cov has no channels")
    positive = values > 0
    if positive.sum() <= 1:
        return int(max(1, positive.sum()))

    if method == "variance":
        # Only the strictly positive part, as the cliff branch already does.
        # The cumulative sum reaches 1 only at the last eigenvalue, and on a
        # rank-deficient covariance that last eigenvalue is a zero: searching
        # for pct_var=1.0 over the whole spectrum then lands past the end and
        # returns n_channels, restoring exactly the rank the estimate exists to
        # deny. Whether it did so depended on where the floating-point sum
        # happened to land, so the same request answered truthfully on one
        # matrix and not on the next.
        live = values[positive]
        total = live.sum()
        if total <= 0:
            return 1
        keep = int(np.searchsorted(np.cumsum(live) / total, float(pct_var)) + 1)
        rank = int(np.clip(keep, 1, live.size))
    else:
        z = _CLIFF_Z if threshold is None else float(threshold)
        # Work on the strictly positive part: a zeroed tail has no log.
        live = values[positive]
        if live.size < 3:
            return int(live.size)
        drops = -np.diff(np.log10(live))
        # Centred on the median before standardising. Dividing by the spread
        # alone asks "is this drop large relative to the scatter", which is not
        # the question: on a smooth full-rank spectrum every drop is a similar
        # positive number, the scatter is tiny, and the very first one clears
        # any threshold. Measured on a well-conditioned 60-channel covariance,
        # that reported a rank of 1. Centring asks the intended question --
        # is this drop unusual *for this spectrum* -- and leaves a smooth
        # spectrum with nothing above the threshold.
        centre = np.median(drops)
        spread = np.std(drops)
        if spread <= 0:
            rank = int(live.size)
        else:
            standardised = (drops - centre) / spread
            # What settles it is not how far the spectrum falls but what is left
            # underneath. A projection leaves its discarded directions at
            # round-off: everything below the cliff sits at 1e-15 or so of the
            # largest eigenvalue. A full-rank covariance that merely starts with
            # a steep step -- which real M/EEG does, its first eigenvalue being a
            # common-mode or reference component -- still has real signal below
            # that step, whole percent of the largest, not round-off.
            #
            # Judged on the drop alone the two are indistinguishable, and the
            # decade floor was set from white synthetic covariances where the
            # largest full-rank drop is 0.04 decades. Real EEG falls 1.17
            # decades at the very first step, clearing that floor, so the
            # estimator called the first step the cliff and returned a rank of
            # ONE on an ordinary full-rank recording -- which, handed to a
            # beamformer as ``rank``, destroys the inverse in silence.
            residual = live[1:] / live[0]
            hit = np.nonzero(
                (standardised > z)
                & (drops > _CLIFF_DECADES)
                & (residual < _CLIFF_RESIDUAL)
            )[0]
            # The first qualifying drop. In practice there is only ever one --
            # a single cliff inflates the spread enough that a second, smaller
            # collapse no longer clears the standardised test, which was checked
            # on a deliberately three-plateau spectrum -- but where several do
            # qualify the rank is the first genuine fall, not the deepest.
            rank = int(hit[0] + 1) if hit.size else int(live.size)
        # Anything the spectrum already zeroed is not usable whatever the cliff
        # test says.
        rank = int(min(rank, live.size))

    logger.info(f"    Estimated rank {rank} of {n} by method={method!r}.")
    return int(np.clip(rank, 1, n))
