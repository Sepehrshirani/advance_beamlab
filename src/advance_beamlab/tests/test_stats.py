# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
import pytest

from advance_beamlab import permutation_image_test


def test_the_family_wise_correction_controls_the_error_rate():
    """Under a complete null, about five images in a hundred should reject.

    This is the only property that makes the test worth having, and it is
    checked rather than asserted from the theory: 300 images of pure noise, 400
    sources each, and the fraction with any location at p <= 0.05 has to land
    near five per cent rather than near one hundred.
    """
    rng = np.random.default_rng(0)
    runs, n_sub, n_src = 300, 16, 400
    rejected = 0
    for run in range(runs):
        images = rng.standard_normal((n_sub, n_src))
        _, p, _ = permutation_image_test(
            images, n_permutations=256, correction="maximum", seed=run, verbose=False
        )
        rejected += int((p <= 0.05).any())
    rate = rejected / runs
    assert 0.01 < rate < 0.11, f"family-wise rate {rate:.3f} is not near 0.05"


def test_uncorrected_testing_fails_the_same_check():
    """The reason the correction is the default, demonstrated rather than said.

    With four hundred locations, testing each against its own null puts a false
    positive somewhere in essentially every image. Anyone reading an
    uncorrected beamformer map is reading that.
    """
    rng = np.random.default_rng(1)
    hits = 0
    for run in range(40):
        images = rng.standard_normal((12, 400))
        _, p, _ = permutation_image_test(
            images, n_permutations=256, correction="none", seed=run, verbose=False
        )
        hits += int((p <= 0.05).any())
    assert hits / 40 > 0.8


def test_a_real_effect_is_found():
    """Controlling the error rate is worthless if nothing survives it."""
    rng = np.random.default_rng(2)
    images = rng.standard_normal((16, 400))
    images[:, 100:110] += 1.2
    observed, p, _ = permutation_image_test(
        images, n_permutations=1024, seed=0, verbose=False
    )
    found = set(np.nonzero(p <= 0.05)[0].tolist())
    assert found, "the effect was not detected at all"
    assert found <= set(range(100, 110)), f"detections outside the effect: {found}"
    assert len(found) >= 5
    assert np.argmax(observed) in range(100, 110)


def test_no_p_value_is_ever_zero():
    """A finite permutation test cannot justify zero.

    The observed arrangement is one the null admits, so it belongs in the
    distribution it is compared against; leaving it out is what produces a
    p-value of zero and an overstated result.
    """
    rng = np.random.default_rng(3)
    images = rng.standard_normal((10, 50)) + 8.0  # a huge effect
    _, p, _ = permutation_image_test(images, n_permutations=64, seed=0, verbose=False)
    assert p.min() > 0
    assert p.min() >= 1.0 / 64

    # And the floor is attained where the observed really is the extreme of its
    # own null: identical positive images, tested one-sided, so no sign flip
    # can beat leaving every sign alone.
    same = np.ones((8, 5))
    _, exact, _ = permutation_image_test(
        same, n_permutations=1024, tail=1, seed=0, verbose=False
    )
    assert exact.min() == pytest.approx(1.0 / 2**8, rel=1e-9)


def test_a_small_study_is_tested_exhaustively():
    """With few observations every sign flip is available, so use them all.

    A sampled null is least reliable exactly where the exhaustive one is
    affordable, so the choice is not a performance detail.
    """
    rng = np.random.default_rng(4)
    images = rng.standard_normal((6, 20))
    _, p, null = permutation_image_test(
        images, n_permutations=1024, seed=0, verbose=False
    )
    assert null.shape == (2**6,)
    # Exhaustive, so the resolution is exactly one part in 2**n_obs and no
    # p-value can fall below it.
    assert p.min() >= 1.0 / 2**6
    assert np.all(np.isin(np.round(p * 2**6), np.arange(2**6 + 1)))


def test_the_tails_do_what_they_say():
    """A one-sided test must not find an effect pointing the other way."""
    rng = np.random.default_rng(5)
    images = rng.standard_normal((16, 200))
    images[:, 50:60] += 1.5
    _, greater, _ = permutation_image_test(
        images, n_permutations=512, tail=1, seed=0, verbose=False
    )
    _, less, _ = permutation_image_test(
        images, n_permutations=512, tail=-1, seed=0, verbose=False
    )
    assert (greater[50:60] <= 0.05).any()
    assert not (less[50:60] <= 0.05).any()


def test_the_null_is_returned_for_inspection():
    """A p-value from a distribution nobody looked at is worth little."""
    rng = np.random.default_rng(6)
    images = rng.standard_normal((12, 30))
    _, _, null = permutation_image_test(
        images, n_permutations=200, correction="maximum", seed=0, verbose=False
    )
    assert null.shape == (200,)
    assert np.all(null >= 0)  # a maximum of absolute values


def test_bad_input_is_refused():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="n_observations"):
        permutation_image_test(rng.standard_normal(20))
    with pytest.raises(ValueError, match="two observations"):
        permutation_image_test(rng.standard_normal((1, 20)))
    with pytest.raises(ValueError, match="correction"):
        permutation_image_test(rng.standard_normal((8, 20)), correction="fdr")
    with pytest.raises(ValueError, match="non-finite"):
        bad = rng.standard_normal((8, 20))
        bad[2, 3] = np.nan
        permutation_image_test(bad)
