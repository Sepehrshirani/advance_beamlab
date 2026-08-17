# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from advance_beamlab import make_mcmv, power_image  # noqa: E402


@pytest.fixture(scope="module")
def sphere():
    """A small EEG sphere model and a fixed-orientation forward on it."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = list(dict.fromkeys(montage.ch_names))[:32]
    info = mne.create_info(ch_names, 200.0, "eeg", verbose=False)
    info.set_montage(montage, verbose=False)
    tmp = mne.EvokedArray(np.zeros((len(ch_names), 1)), info, verbose=False)
    tmp.set_eeg_reference("average", projection=True, verbose=False)
    info = tmp.info
    bem = mne.make_sphere_model("auto", "auto", info, verbose=False)
    src = mne.setup_volume_source_space(sphere=bem, pos=25.0, verbose=False)
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=bem, eeg=True, meg=False, verbose=False
    )
    fwd = mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=False, verbose=False
    )
    return info, fwd


def _covs(info, fwd, gain_active=4.0, seed=0):
    """Noise, a control window, and an active window with one source added."""
    rng = np.random.default_rng(seed)
    n_ch = len(info["ch_names"])
    n_times = 4000
    noise = rng.standard_normal((n_ch, n_times)) * 1e-8
    lead = fwd["sol"]["data"][:, 7]
    source = rng.standard_normal(n_times)
    active = noise + gain_active * 1e-8 * np.outer(lead / np.linalg.norm(lead), source)

    def cov(x):
        ep = mne.EpochsArray(x[None].copy(), info, tmin=0.0, verbose=False)
        return mne.compute_covariance(ep, method="empirical", verbose=False)

    return cov(noise), cov(noise), cov(active)


def test_pseudo_z_is_a_ratio_to_the_noise_floor(sphere):
    """One is the value at a location carrying only noise."""
    info, fwd = sphere
    noise_cov, _, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    z = power_image(filters, active_cov, noise_cov=noise_cov, kind="pseudo-z")
    assert z.shape == (fwd["sol"]["data"].shape[1],)
    assert np.all(z[np.isfinite(z)] > 0)
    # Imaging the noise against itself has to give exactly one everywhere: the
    # same covariance in the numerator and the denominator.
    unity = power_image(filters, noise_cov, noise_cov=noise_cov, kind="pseudo-z")
    np.testing.assert_allclose(unity[np.isfinite(unity)], 1.0, rtol=1e-10)


def test_pseudo_t_is_zero_when_the_windows_match(sphere):
    """A differential image of a window against itself is flat zero.

    This is the property that makes it a *differential* image: whatever the
    filter passes from the shared background cancels, whether or not that
    background is small.
    """
    info, fwd = sphere
    noise_cov, base_cov, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    flat = power_image(
        filters, base_cov, baseline_cov=base_cov, noise_cov=noise_cov, kind="pseudo-t"
    )
    np.testing.assert_allclose(flat[np.isfinite(flat)], 0.0, atol=1e-12)

    # And with a source in the active window it is positive where the source is.
    t = power_image(
        filters, active_cov, baseline_cov=base_cov, noise_cov=noise_cov, kind="pseudo-t"
    )
    assert np.nanmax(t) > 0
    assert int(np.nanargmax(t)) == 7


def test_pseudo_f_is_a_log_ratio_and_can_be_turned_off(sphere):
    """The log is a presentation choice, so it has to be reversible."""
    info, fwd = sphere
    noise_cov, base_cov, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    logged = power_image(
        filters, active_cov, baseline_cov=base_cov, kind="pseudo-f", log_ratio=True
    )
    raw = power_image(
        filters, active_cov, baseline_cov=base_cov, kind="pseudo-f", log_ratio=False
    )
    np.testing.assert_allclose(np.exp(logged), raw, rtol=1e-10)
    # No change means zero on the log scale, which is the reason for the default.
    same = power_image(
        filters, base_cov, baseline_cov=base_cov, kind="pseudo-f", log_ratio=True
    )
    np.testing.assert_allclose(same[np.isfinite(same)], 0.0, atol=1e-12)


def test_the_image_locates_the_source_it_was_given(sphere):
    """All three kinds have to peak at the source, or none of them is an image."""
    info, fwd = sphere
    noise_cov, base_cov, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    for kind in ("pseudo-z", "pseudo-t", "pseudo-f"):
        img = power_image(
            filters,
            active_cov,
            baseline_cov=base_cov,
            noise_cov=noise_cov,
            kind=kind,
        )
        assert int(np.nanargmax(img)) == 7, kind


def test_it_works_on_mcmv_filters_too(sphere):
    """The whole point of routing through the public apply path.

    MCMV keeps its weights in a different space from MNE's, so an image built
    from the stored arrays would be meaningless. Built from the apply path, the
    same three kinds work unchanged.
    """
    info, fwd = sphere
    noise_cov, base_cov, active_cov = _covs(info, fwd)
    sources = [7, 20]
    bf = make_mcmv(
        info, fwd, active_cov, sources, noise_cov=noise_cov, reg=0.05, verbose=False
    )
    z = power_image(bf, active_cov, noise_cov=noise_cov, kind="pseudo-z")
    assert z.shape == (len(sources),)
    assert np.all(z > 0)
    t = power_image(
        bf, active_cov, baseline_cov=base_cov, noise_cov=noise_cov, kind="pseudo-t"
    )
    # The constrained source that carries the simulated activity is the larger.
    assert t[0] > t[1]


def test_a_missing_ingredient_is_refused_rather_than_assumed(sphere):
    """Silently imaging something else would be worse than an error."""
    info, fwd = sphere
    noise_cov, base_cov, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    with pytest.raises(ValueError, match="baseline_cov"):
        power_image(filters, active_cov, noise_cov=noise_cov, kind="pseudo-t")
    with pytest.raises(ValueError, match="noise_cov"):
        power_image(filters, active_cov, baseline_cov=base_cov, kind="pseudo-z")
    with pytest.raises(ValueError, match="kind"):
        power_image(filters, active_cov, noise_cov=noise_cov, kind="pseudo-q")


def test_a_degenerate_filter_is_nan_rather_than_enormous(sphere):
    """A filter with no noise gain has an undefined image value, not a huge one.

    Dividing by an underflowed noise power would put the brightest voxel in the
    image exactly where the beamformer failed most completely.
    """
    info, fwd = sphere
    noise_cov, _, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    zeroed = noise_cov.copy()
    zeroed["data"] = np.zeros_like(zeroed["data"])
    z = power_image(filters, active_cov, noise_cov=zeroed, kind="pseudo-z")
    assert np.all(np.isnan(z))


def test_pseudo_z_is_the_ratio_itself_and_not_a_function_of_it(sphere):
    """Scaling the active covariance scales the image by the same factor.

    Checking only that noise against itself gives one leaves a square root --
    or any other monotone function of the ratio -- undetectable, since it fixes
    the value at one as well.
    """
    info, fwd = sphere
    noise_cov, _, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    louder = active_cov.copy()
    louder["data"] = louder["data"] * 4.0
    one = power_image(filters, active_cov, noise_cov=noise_cov, kind="pseudo-z")
    four = power_image(filters, louder, noise_cov=noise_cov, kind="pseudo-z")
    np.testing.assert_allclose(four, 4.0 * one, rtol=1e-8)


def test_pseudo_t_carries_the_factor_of_two_it_is_defined_with(sphere):
    """The value, not just the sign and the peak.

    The denominator is twice the projected noise, for the noise entering
    through both windows. Dropping the two doubles every value in the image,
    which no test that only looks at where the peak is would notice.
    """
    info, fwd = sphere
    noise_cov, base_cov, active_cov = _covs(info, fwd)
    filters = mne.beamformer.make_lcmv(
        info, fwd, active_cov, reg=0.05, noise_cov=noise_cov, verbose=False
    )
    t = power_image(
        filters, active_cov, baseline_cov=base_cov, noise_cov=noise_cov, kind="pseudo-t"
    )
    # Rebuild it from the pieces the function is defined in terms of.
    z_active = power_image(filters, active_cov, noise_cov=noise_cov, kind="pseudo-z")
    z_base = power_image(filters, base_cov, noise_cov=noise_cov, kind="pseudo-z")
    np.testing.assert_allclose(t, (z_active - z_base) / 2.0, rtol=1e-8)
