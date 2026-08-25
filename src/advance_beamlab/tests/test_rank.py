# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from advance_beamlab import estimate_rank, rank_spectrum  # noqa: E402


def _projected(n, rank, seed=0, n_samples=600):
    """A covariance of exactly ``rank``, the way a projection leaves one."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((n, n)))[0][:, :rank]
    samples = rng.standard_normal((rank, n_samples))
    return basis @ (samples @ samples.T / n_samples) @ basis.T


def _full(n, seed=0, n_samples=600):
    """A well-conditioned covariance of full rank."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n, n_samples))
    return samples @ samples.T / n_samples


@pytest.mark.parametrize("rank", [55, 40, 20, 5, 1])
@pytest.mark.parametrize("method", ["cliff", "variance"])
def test_a_projected_covariance_reports_the_rank_it_was_given(method, rank):
    """The case the estimators exist for, and they get it exactly."""
    assert estimate_rank(_projected(60, rank), method=method) == rank


@pytest.mark.parametrize("method", ["cliff", "variance"])
def test_a_full_rank_covariance_is_not_truncated(method):
    """Reporting a deficiency that is not there is the more damaging error.

    'cliff' used to fail here, and badly: it standardised each drop by the
    spread without centring, so on a smooth spectrum -- where every drop is a
    similar small number and the scatter is tiny -- the very first one cleared
    the threshold and the estimate came back as 1 out of 60. The top eigenvalue
    of any sample covariance sits slightly apart from the rest, so this was not
    an edge case but the ordinary one.
    """
    assert estimate_rank(_full(60), method=method) == 60


def test_the_cliff_needs_to_be_a_cliff_and_not_merely_unusual():
    """Both conditions matter, so check the one that is easy to drop.

    A drop is only a cliff if it is unusual *and* an absolute fall of at least a
    decade. The two regimes are nowhere near each other: measured here, a
    projection cliff is about fifteen decades and the largest drop on a
    full-rank covariance is a few hundredths of one.
    """
    live = np.linalg.eigvalsh(_full(60))[::-1]
    biggest_smooth = float(np.max(-np.diff(np.log10(live))))
    cov = _projected(60, 40)
    values = np.linalg.eigvalsh(cov)[::-1]
    values = values[values > 0]
    biggest_cliff = float(np.max(-np.diff(np.log10(values))))
    assert biggest_smooth < 0.5
    assert biggest_cliff > 5.0


def test_a_covariance_object_and_a_bare_array_agree():
    """Callers have both; they must not get different answers."""
    data = _projected(32, 20)
    names = [f"EEG {i:03d}" for i in range(32)]
    info = mne.create_info(names, 200.0, "eeg", verbose=False)
    cov = mne.Covariance(data, names, bads=[], projs=[], nfree=600, verbose=False)
    del info
    assert estimate_rank(cov) == estimate_rank(data) == 20


def test_the_spectrum_is_descending_and_never_negative():
    """Round-off makes eigenvalues of a singular matrix slightly negative."""
    values = rank_spectrum(_projected(40, 12))
    assert values.shape == (40,)
    assert np.all(np.diff(values) <= 0)
    assert np.all(values >= 0)


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="square"):
        estimate_rank(np.zeros((4, 7)))
    with pytest.raises(ValueError, match="pct_var"):
        estimate_rank(_full(8), method="variance", pct_var=1.5)
    with pytest.raises(ValueError, match="method"):
        estimate_rank(_full(8), method="minka")


def test_a_threshold_can_be_argued_with():
    """A borderline spectrum should be settable, not silently resolved."""
    cov = _projected(60, 30)
    assert estimate_rank(cov, method="cliff", threshold=5.0) == 30
    # An absurd threshold finds no cliff at all and keeps everything positive.
    loose = estimate_rank(cov, method="cliff", threshold=1e9)
    assert loose >= 30


def test_a_multi_plateau_spectrum_stops_at_the_first_collapse():
    """Where a spectrum falls in stages, the rank is the first fall.

    Three plateaus nine orders apart: the usable rank is the top one, not the
    point where the numbers finally reach round-off. Only one drop clears the
    standardised test here, because a cliff inflates the spread enough to mask
    any later one, which is worth knowing when reading the estimate.
    """
    rng = np.random.default_rng(9)
    n = 60
    basis = np.linalg.qr(rng.standard_normal((n, n)))[0]
    scale = np.concatenate(
        [
            np.full(15, 1.0) * (1 + 0.01 * rng.standard_normal(15)),
            np.full(15, 1e-9) * (1 + 0.01 * rng.standard_normal(15)),
            np.full(30, 1e-18) * (1 + 0.01 * rng.standard_normal(30)),
        ]
    )
    cov = (basis * scale) @ basis.T
    cov = 0.5 * (cov + cov.T)
    assert estimate_rank(cov, method="cliff") == 15


def test_a_correlated_full_rank_spectrum_is_not_mistaken_for_a_projection():
    """A steep first step is not a cliff when there is real signal beneath it.

    The spectrum of a real M/EEG covariance is not white. Its largest eigenvalue
    is a common-mode or reference component and stands well clear of the rest, so
    the very first step of the spectrum is a large one -- on the ``sample``
    recording it is 1.17 decades for EEG and 0.95 for magnetometers. Judged on
    the size of the step alone that looks exactly like the collapse a projection
    leaves behind, and the estimator returned a rank of **one** on an ordinary,
    entirely full-rank recording. Handed to a beamformer as ``rank`` that
    silently destroys the inverse, and nothing in the call says so.

    What separates the two cases is not the fall but what lies under it. A
    projection leaves its discarded directions at round-off, some 1e-16 of the
    largest eigenvalue. A steep first step in a full-rank covariance still has
    whole percent of the largest eigenvalue below it -- 6.7e-2 for that EEG
    recording -- and that is real signal.

    The spectrum here reproduces the shape without needing the dataset: one
    dominant component, then a decade of ordinary correlated signal.
    """
    rng = np.random.default_rng(4)
    n = 40
    basis = np.linalg.qr(rng.standard_normal((n, n)))[0]
    # A first eigenvalue 15x the next -- a 1.2-decade step, as on real EEG --
    # and a decade of genuine signal spread beneath it.
    scale = np.concatenate([[15.0], np.logspace(0, -1, n - 1)])
    cov = basis * scale @ basis.T
    cov = 0.5 * (cov + cov.T)

    drops = -np.diff(np.log10(np.linalg.eigvalsh(cov)[::-1]))
    assert drops[0] > 1.0  # the premise: the first step really is that steep
    assert estimate_rank(cov, method="cliff") == n

    # And a genuine projection of the same spectrum is still found.
    projected = basis[:, :25] * scale[:25] @ basis[:, :25].T
    projected = 0.5 * (projected + projected.T)
    assert estimate_rank(projected, method="cliff") == 25
