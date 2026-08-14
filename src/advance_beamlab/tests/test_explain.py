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
    # And the amplitude follows. Measured on this fixture at signal-to-noise 5,
    # the amplitude the filter delivers runs 1.000 / 0.270 / 0.113 for LCMV at
    # correlation 0 / 0.9 / 0.99, and 1.000 at every correlation for MCMV, which
    # its constraint guarantees exactly rather than approximately.
    assert ratio["lcmv", 0.99] < 0.2
    assert ratio["mcmv", 0.99] == pytest.approx(1.0, abs=1e-6)
    assert ratio["mcmv", 0.0] == pytest.approx(1.0, abs=1e-6)
    assert ratio["lcmv", 0.0] - ratio["lcmv", 0.99] > 0.7


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


@pytest.mark.parametrize("n_sources", [1, 2, 3])
def test_source_count_is_respected(sphere, n_sources):
    """The constraint table grows with the number of sources."""
    info, fwd = sphere
    demo = constraint_demo(
        info, fwd, method="mcmv", n_sources=n_sources, correlation=0.9, snr=5.0
    )
    assert len(demo.sources) == n_sources
    assert demo.gains.shape == (n_sources, n_sources)
    assert demo.true_tcs.shape[0] == n_sources
    assert demo.reconstructed.shape[0] == n_sources
    # MCMV constrains the whole table whatever its size, which is the point.
    np.testing.assert_allclose(demo.gains, np.eye(n_sources), atol=1e-6)


def test_requested_correlation_holds_for_three_sources(sphere):
    """The phase shift could only do two; the shared-factor mixture does any n."""
    info, fwd = sphere
    for r in (0.0, 0.5, 0.9):
        demo = constraint_demo(
            info, fwd, method="lcmv", n_sources=3, correlation=r, snr=5.0
        )
        off = np.corrcoef(demo.true_tcs)[np.triu_indices(3, k=1)]
        np.testing.assert_allclose(off, r, atol=0.02)


def test_transient_morphology_is_actually_transient(sphere):
    """A burst train is peaky where a rhythm is not."""
    info, fwd = sphere
    kurtosis = {}
    for morphology in ("alpha", "transient"):
        demo = constraint_demo(
            info, fwd, method="lcmv", morphology=morphology, correlation=0.5, snr=5.0
        )
        x = demo.true_tcs[0]
        x = (x - x.mean()) / x.std()
        kurtosis[morphology] = float((x**4).mean())
        assert demo.extra["morphology"] == morphology
    assert kurtosis["transient"] > 2 * kurtosis["alpha"]


def test_unknown_morphology_raises(sphere):
    info, fwd = sphere
    with pytest.raises(ValueError, match="morphology"):
        constraint_demo(info, fwd, morphology="sawtooth")


def test_the_noise_field_is_shared_between_scenes(sphere):
    """What lets the panel rebuild a recording from a single stored field.

    The noise is drawn from a generator seeded independently of the waveforms,
    so every scene sees the same realisation scaled differently. Two scenes must
    therefore have noise that agrees exactly once each is divided by its own
    recorded scale.
    """
    info, fwd = sphere
    fields = []
    for correlation, snr in ((0.2, 2.0), (0.95, 8.0)):
        demo = constraint_demo(
            info, fwd, method="lcmv", correlation=correlation, snr=snr
        )
        clean = demo.extra["leadfield"] @ demo.true_tcs
        fields.append((demo.sensor_data - clean) / demo.extra["noise_scale"])
    np.testing.assert_allclose(fields[0], fields[1], atol=1e-9)


def test_sources_off_the_scan_grid_give_a_real_localisation_error(sphere):
    """Without this the localiser reports 0 mm almost everywhere.

    A source sitting exactly on a scanned grid node, generated with the very
    leadfield being inverted, is an inverse crime: the matched filter peaks on
    that node whatever the noise. Taking the sources from a finer forward puts
    them where no method can scan them, which is the situation on real data.
    """
    info, fwd = sphere
    fine_src = mne.setup_volume_source_space(
        sphere=mne.make_sphere_model("auto", "auto", info, verbose=False),
        pos=8.0,
        verbose=False,
    )
    fine = mne.make_forward_solution(
        info,
        trans=None,
        src=fine_src,
        bem=mne.make_sphere_model("auto", "auto", info, verbose=False),
        eeg=True,
        meg=False,
        verbose=False,
    )
    fine = mne.convert_forward_solution(
        fine, force_fixed=True, use_cps=False, verbose=False
    )

    on_grid = constraint_demo(info, fwd, method="lcmv", n_sources=1, snr=5.0)
    off_grid = constraint_demo(
        info, fwd, method="lcmv", n_sources=1, snr=5.0, true_forward=fine
    )
    assert on_grid.peak_errors[0] == 0.0
    assert off_grid.peak_errors[0] > 0.001
    # And the reported position is the true one, not the grid node.
    assert not np.allclose(off_grid.positions[0], fwd["source_rr"][off_grid.sources[0]])


def test_the_amplitude_measure_survives_a_single_noisy_trial(sphere):
    """The defect a reader caught: the old measure reported 750 per cent.

    ``amplitude_ratio`` used to be the output's root mean square over the
    truth's. The output is the source plus whatever interference survives the
    filter, so at a single trial it was mostly noise: it read far above one and,
    worse, LCMV and MCMV became indistinguishable, which is the opposite of what
    the panel exists to show. The measure now carries no noise at all.
    """
    info, fwd = sphere
    for snr in (0.2, 5.0):
        lcmv = constraint_demo(info, fwd, method="lcmv", correlation=0.99, snr=snr)
        mcmv = constraint_demo(info, fwd, method="mcmv", correlation=0.99, snr=snr)
        # MCMV's constraint fixes the whole table, so it delivers the amplitude
        # exactly whatever the noise is doing.
        assert mcmv.amplitude_ratio[0] == pytest.approx(1.0, abs=1e-6)
        # And it is bounded: the old measure ran to 7.8 here.
        assert abs(lcmv.amplitude_ratio[0]) < 2.5
        # The delivered amplitude has to agree with the constraint table it is
        # derived from: one plus the off-diagonal weighted by the correlation.
        r = float(np.corrcoef(lcmv.true_tcs)[0, 1])
        expected = lcmv.gains[0, 0] + r * lcmv.gains[0, 1]
        assert lcmv.amplitude_ratio[0] == pytest.approx(expected, abs=0.02)


def test_the_two_empirical_amplitude_measures_are_kept_for_reference(sphere):
    """Both alternatives stay available, and behave as described."""
    info, fwd = sphere
    noisy = constraint_demo(info, fwd, method="lcmv", correlation=0.99, snr=0.2)
    clean = constraint_demo(info, fwd, method="lcmv", correlation=0.99, snr=5.0)
    # The root-mean-square measure is dominated by interference at low SNR.
    assert noisy.extra["output_rms_ratio"][0] > 3.0
    assert clean.extra["output_rms_ratio"][0] < 1.5
    # The empirical projection tracks the noiseless one, loosely at one trial
    # and closely once there is signal to measure with.
    assert abs(clean.extra["projected_ratio"][0] - clean.amplitude_ratio[0]) < 0.05


def test_the_projection_rank_is_chosen_in_the_space_it_is_applied_in(sphere):
    """A rank means nothing without the q it was drawn from.

    Both the rank curve and the filter reduce the array to q virtual sensors
    before anything else happens, and q comes from a truncated SVD of the
    *whitened* leadfield. Selecting the rank from a curve built without the
    noise covariance the filter uses therefore picks it in a different space: on
    the panel's forward that was q = 49 against the filter's q = 78, so K* = 80
    was chosen as 80 out of 2401 and then applied as 80 out of 6084, and the
    projector kept a quite different part of the power subspace from the one the
    selection criterion had asked for.
    """
    from advance_beamlab import _recipsiicos

    info, fwd = sphere
    seen = []
    original = _recipsiicos._reduction_operator

    def spy(*args, **kwargs):
        b_op, q, n_white = original(*args, **kwargs)
        seen.append(q)
        return b_op, q, n_white

    _recipsiicos._reduction_operator = spy
    try:
        demo = constraint_demo(
            info, fwd, method="recipsiicos", correlation=0.9, snr=5.0
        )
    finally:
        _recipsiicos._reduction_operator = original

    assert len(seen) > 1, "expected the curve and the filter to reduce separately"
    assert len(set(seen)) == 1, (
        f"the rank was chosen for q={seen[0]}, applied for {seen[1:]}"
    )
    assert demo.extra["recipsiicos_virtual"] == seen[0]
    assert 0 < demo.extra["recipsiicos_rank"] <= seen[0] ** 2


def test_the_amplitude_is_measured_at_the_true_source_not_the_scan_node(sphere):
    """Two gain tables, and only one of them can see a wrong head model.

    ``gains`` is probed at the grid nodes the filters were constrained at, so
    its diagonal is pinned to one whatever the model does. ``delivered`` is
    probed at the positions actually simulated. Under a matched model they are
    the same table; under a mismatched one they are not, and building
    ``amplitude_ratio`` from the first reports a filter delivering all of an
    amplitude whose reconstruction holds almost none of it. Nothing used to
    notice the swap.
    """
    info, fwd = sphere
    fine = mne.make_forward_solution(
        info,
        trans=None,
        src=mne.setup_volume_source_space(
            sphere=mne.make_sphere_model("auto", "auto", info, verbose=False),
            pos=10.0,
            verbose=False,
        ),
        bem=mne.make_sphere_model("auto", "auto", info, verbose=False),
        eeg=True,
        meg=False,
        verbose=False,
    )
    fine = mne.convert_forward_solution(
        fine, force_fixed=True, use_cps=False, verbose=False
    )
    demo = constraint_demo(
        info,
        fwd,
        method="lcmv",
        correlation=0.9,
        snr=5.0,
        true_forward=fine,
        off_grid_sources=True,
    )
    # The constraint table still says the constraint held.
    np.testing.assert_allclose(np.diag(demo.gains), 1.0, rtol=1e-6)
    # The delivered amplitude must not simply repeat it, or the mismatch the
    # realistic head model exists to show would be invisible.
    from_gains = (
        demo.gains
        * (
            (demo.true_tcs @ demo.true_tcs.T).T
            / np.maximum(np.diag(demo.true_tcs @ demo.true_tcs.T), 1e-300)[:, None]
        )
    ).sum(axis=1)
    assert not np.allclose(demo.amplitude_ratio, from_gains, atol=0.02), (
        "amplitude_ratio was built from the scan-node table, so a wrong head "
        "model cannot show up in it"
    )
