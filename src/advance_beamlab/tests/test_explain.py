# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from advance_beamlab._explain import constraint_demo  # noqa: E402


@pytest.fixture(scope="module")
def sphere():
    """A small EEG sphere model, enough to show the constraint at work."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = list(dict.fromkeys(montage.ch_names))
    info = mne.create_info(ch_names, 200.0, "eeg", verbose=False)
    info.set_montage(montage, verbose=False)
    tmp = mne.EvokedArray(np.zeros((len(ch_names), 1)), info, verbose=False)
    tmp.set_eeg_reference("average", projection=True, verbose=False)
    info = tmp.info
    bem = mne.make_sphere_model("auto", "auto", info, verbose=False)
    src = mne.setup_volume_source_space(sphere=bem, pos=15.0, verbose=False)
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=bem, eeg=True, meg=False, verbose=False
    )
    return info, mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=False, verbose=False
    )


def test_requested_scene_is_achieved(sphere):
    """The reported correlation and separation are the achieved ones."""
    info, fwd = sphere
    demo = constraint_demo(info, fwd, correlation=0.9, separation=0.04, snr=5.0)
    assert abs(demo.correlation - 0.9) < 0.01
    assert 0.02 < demo.separation < 0.06
    assert demo.true_tcs.shape == demo.reconstructed.shape


def test_distortionless_constraint_holds_where_it_is_claimed(sphere):
    """LCMV and MCMV both pin their own gain to one; that is the constraint."""
    info, fwd = sphere
    for method in ("lcmv", "mcmv"):
        demo = constraint_demo(info, fwd, method=method, correlation=0.9, snr=5.0)
        np.testing.assert_allclose(np.diag(demo.gains), 1.0, rtol=1e-6)


def test_lcmv_cancels_and_mcmv_does_not(sphere):
    """The whole point, asserted: correlation drives LCMV's off-diagonal.

    LCMV is free to choose its gain at the partner source, and when the two are
    correlated the choice that minimises output power is a large negative one,
    which cancels the target along with the partner. MCMV forbids exactly that.
    """
    info, fwd = sphere
    off, ratio = {}, {}
    for method in ("lcmv", "mcmv"):
        for r in (0.0, 0.99):
            demo = constraint_demo(info, fwd, method=method, correlation=r, snr=5.0)
            off[method, r] = abs(demo.gains[0, 1])
            ratio[method, r] = demo.amplitude_ratio[0]

    # LCMV: near zero when there is nothing to cancel, large when there is.
    assert off["lcmv", 0.0] < 0.2
    assert off["lcmv", 0.99] > 0.7
    # MCMV: zero either way, by construction rather than by luck.
    assert off["mcmv", 0.0] < 1e-6
    assert off["mcmv", 0.99] < 1e-6
    # And the amplitude follows: LCMV loses most of it, MCMV keeps it.
    assert ratio["lcmv", 0.99] < 0.6
    assert ratio["mcmv", 0.99] > 0.9
    assert ratio["lcmv", 0.99] < ratio["mcmv", 0.99]


@pytest.mark.parametrize("method", ["lcmv", "mcmv", "recipsiicos", "abmc"])
def test_every_method_runs_and_returns_a_full_scene(sphere, method):
    info, fwd = sphere
    demo = constraint_demo(info, fwd, method=method, correlation=0.9, snr=5.0)
    n_grid = fwd["sol"]["data"].shape[1]
    assert demo.gains.shape == (2, 2)
    assert demo.power_map.shape == (n_grid,)
    assert np.all(np.isfinite(demo.power_map))
    assert demo.peak_errors.shape == (2,)
    assert demo.reconstructed.shape[0] == 2


def test_unknown_method_raises(sphere):
    info, fwd = sphere
    with pytest.raises(ValueError, match="method"):
        constraint_demo(info, fwd, method="wiener")
