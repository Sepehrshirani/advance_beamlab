# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
import pytest

from advance_beamlab import permutation_image_test
from advance_beamlab._stats import _MAX_BLOCK_ELEMENTS


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


def test_a_negative_effect_is_found_by_the_two_sided_test():
    """Two-sided means both sides, and only the observed side was abs'd.

    Dropping the absolute value from the observed statistic while keeping it on
    the null leaves every existing test green, because they all plant positive
    effects. A downward effect then cannot be found at all.
    """
    rng = np.random.default_rng(11)
    images = rng.standard_normal((16, 300))
    images[:, 40:50] -= 1.5
    _, p, _ = permutation_image_test(
        images, n_permutations=512, tail=0, seed=0, verbose=False
    )
    found = set(np.nonzero(p <= 0.05)[0].tolist())
    assert found, "a negative effect was not found by a two-sided test"
    assert found <= set(range(40, 50))


def test_the_observed_arrangement_is_inside_its_own_null():
    """Otherwise the p-value floor is a fiction propped up by the clip.

    The returned p-values are clipped at 1/n, so dropping the observed
    arrangement from the null -- by removing it from the draw, or by comparing
    with > instead of >= -- produces a raw count of zero that the clip hides.
    Check the null itself rather than the number that survives clipping.
    """
    rng = np.random.default_rng(12)
    images = rng.standard_normal((14, 200)) + 3.0
    observed, _, null = permutation_image_test(
        images, n_permutations=256, correction="maximum", seed=0, verbose=False
    )
    peak = float(np.abs(observed).max())
    assert null.max() >= peak - 1e-12, (
        "no permutation reaches the observed statistic, so the observed "
        "arrangement is not in the distribution it is judged against"
    )
    assert np.isclose(null, peak, rtol=0, atol=1e-12).any()


def test_ties_with_the_observed_statistic_are_counted():
    """The comparison has to be >=, and the clip hides it if it is not.

    Relaxing >= to > drops the observed arrangement from the distribution it is
    judged against. Usually that shows up only as a count of zero, which the
    clip at 1/n turns back into 1/n, so the error is invisible in the output.
    Enumerate the flips instead: a two-sided exhaustive test contains both an
    arrangement and its mirror, so exactly two of them tie the observed
    statistic and the smallest p-value must be 2/n rather than 1/n.
    """
    same = np.ones((7, 4))  # 2**7 = 128 flips, all of them enumerated
    _, p, null = permutation_image_test(
        same, n_permutations=1024, tail=0, seed=0, verbose=False
    )
    assert null.size == 2**7
    ties = int(np.isclose(null, np.abs(same.mean(axis=0)).max()).sum())
    assert ties == 2, f"expected an arrangement and its mirror to tie, got {ties}"
    assert p.min() == pytest.approx(2.0 / 2**7, rel=1e-9)


def test_the_floor_is_one_over_the_number_of_draws():
    """The docstring names a floor, so the floor is worth pinning.

    Two conventions are in circulation: counting the observed arrangement among
    the ``n`` draws, which floors the p-value at ``1 / n``, and adding it to
    ``n`` further ones, which floors it at ``1 / (n + 1)``. This function does
    the first, and a reader comparing its numbers with a published threshold
    needs the documentation to say which.
    """
    # Sampled: the observed arrangement is the first draw, and an effect this
    # large means nothing else reaches it.
    rng = np.random.default_rng(13)
    images = rng.standard_normal((11, 40)) + 9.0
    _, sampled, null = permutation_image_test(
        images, n_permutations=100, tail=1, seed=0, verbose=False
    )
    assert null.size == 100
    assert sampled.min() == pytest.approx(1.0 / 100, rel=1e-12)

    # Exhaustive: the same floor, over the flips that were actually enumerated
    # rather than over n_permutations.
    _, exact, null = permutation_image_test(
        np.ones((9, 3)), n_permutations=1024, tail=1, seed=0, verbose=False
    )
    assert null.size == 2**9
    assert exact.min() == pytest.approx(1.0 / 2**9, rel=1e-12)


def test_the_blocked_maximum_is_the_maximum_of_the_whole_surface():
    """Reducing the surface a block of draws at a time must change nothing.

    Under ``'maximum'`` the permutation surface is never held whole -- one
    number per draw survives it, and holding the rest is what makes a
    whole-brain test run out of memory. The blocking is only defensible if it
    reduces to the same numbers, so compare it against the surface itself,
    which ``'none'`` returns for the same draws.
    """
    n_src = 4000
    assert _MAX_BLOCK_ELEMENTS // n_src < 512, "the draws must span several blocks"
    rng = np.random.default_rng(14)
    images = rng.standard_normal((10, n_src))
    images[:, 500:505] += 1.4
    obs, p, blocked = permutation_image_test(
        images, n_permutations=512, correction="maximum", seed=8, verbose=False
    )
    _, _, surface = permutation_image_test(
        images, n_permutations=512, correction="none", seed=8, verbose=False
    )
    # Draw for draw, to floating point rather than bit for bit: the two runs
    # contract the same terms, but one forms a (block, n_obs) product and the
    # other an (n_draws, n_obs) one, and BLAS may sum those in a different
    # order. A block boundary in the wrong place would move a draw's maximum
    # by far more than this tolerance allows.
    np.testing.assert_allclose(blocked, surface.max(axis=1), rtol=1e-12, atol=0)

    # And the counting: a sorted null is searched rather than compared against
    # every source, which would rebuild an array of the size just avoided. The
    # count has to match the comparison exactly, ties included.
    counted = (blocked[None, :] >= np.abs(obs)[:, None]).sum(axis=1) / blocked.size
    np.testing.assert_array_equal(p, np.clip(counted, 1.0 / blocked.size, 1.0))
