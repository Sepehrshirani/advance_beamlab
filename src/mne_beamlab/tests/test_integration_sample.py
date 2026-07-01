# Authors: the mne-beamlab contributors
# License: BSD-3-Clause
"""Integration tests on the MNE ``sample`` dataset (a real MEG recording).

The ``sample`` dataset ships a real BEM forward and the bilateral auditory
response -- two temporally correlated sources -- which is precisely the regime
MCMV and ReciPSIICOS target. These tests exercise the whole pipeline on real
data: real leadfields, a real data covariance, whitening across magnetometers
and gradiometers, the virtual-sensor reduction, projector construction, the
working-space LCMV solve, and the MCMV joint constraint.

The dataset is not bundled with MNE; it is fetched once with
``mne.datasets.sample.data_path()``. When it is absent every test here is
skipped, so the suite still runs offline -- the algorithmic paths themselves
are covered by ``test_recipsiicos.py`` and ``test_mcmv.py``. The forward is
restricted to the bilateral superior-temporal labels so the O(N^2) correlation
Gram of the whitened projector stays small (this is also the physiologically
relevant region for the auditory response).
"""

import mne
import numpy as np
import pytest
from mne.beamformer import apply_lcmv_cov

from mne_beamlab import (
    apply_mcmv_cov,
    make_mcmv,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
)

# ---------------------------------------------------------------------------
# dataset guard: skip cleanly when ``sample`` is not downloaded
# ---------------------------------------------------------------------------
try:
    _DATA_PATH = mne.datasets.sample.data_path(download=False)
    _SAMPLE = _DATA_PATH / "MEG" / "sample"
    _SUBJECTS = _DATA_PATH / "subjects"
    _FWD = _SAMPLE / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    _HAVE_SAMPLE = _SAMPLE.is_dir() and _FWD.is_file()
except Exception:  # pragma: no cover - depends on local dataset state
    _HAVE_SAMPLE = False

pytestmark = pytest.mark.skipif(
    not _HAVE_SAMPLE, reason="MNE sample dataset not downloaded"
)


@pytest.fixture(scope="module")
def auditory():
    """Real MEG data + a real BEM forward restricted to auditory cortex."""
    raw = mne.io.read_raw_fif(
        _SAMPLE / "sample_audvis_filt-0-40_raw.fif", preload=True, verbose=False
    )
    events = mne.read_events(_SAMPLE / "sample_audvis_filt-0-40_raw-eve.fif")
    raw.pick("meg")  # magnetometers + gradiometers -> exercises whitening
    epochs = mne.Epochs(
        raw,
        events,
        {"Auditory/Left": 1, "Auditory/Right": 2},
        tmin=-0.2,
        tmax=0.25,
        baseline=(None, 0.0),
        preload=True,
        verbose=False,
    )
    data_cov = mne.compute_covariance(
        epochs, tmin=0.05, tmax=0.2, method="empirical", verbose=False
    )
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="empirical", verbose=False
    )

    fwd = mne.read_forward_solution(_FWD, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
    labels = mne.read_labels_from_annot(
        "sample",
        "aparc",
        regexp="superiortemporal",
        subjects_dir=_SUBJECTS,
        verbose=False,
    )
    fwd = mne.forward.restrict_forward_to_label(fwd, labels)
    return epochs.info, fwd, data_cov, noise_cov


@pytest.mark.parametrize("method", ["recipsiicos", "whitened"])
def test_recipsiicos_pipeline_on_real_meg(auditory, method):
    """Both projectors build a valid beamformer and localise finite power."""
    info, fwd, data_cov, noise_cov = auditory

    # The rank curve is well-formed on a real BEM forward, and K* is a usable
    # interior rank (not the degenerate q^2 seen on a sphere model).
    ranks, p_pwr, p_cor, kstar = recipsiicos_rank_curve(
        fwd, info, method=method, noise_cov=noise_cov, return_optimal=True
    )
    assert ranks[0] == 1 and ranks[-1] == len(ranks)
    assert np.all((p_pwr >= -1e-9) & (p_pwr <= 1 + 1e-9))
    assert np.all((p_cor >= -1e-9) & (p_cor <= 1 + 1e-9))
    assert 1 <= kstar <= len(ranks)

    filters = make_recipsiicos_lcmv(
        info,
        fwd,
        data_cov,
        rank=kstar,
        method=method,
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        reduce_rank=True,  # free-orientation MEG
    )
    stc = apply_lcmv_cov(data_cov, filters)
    assert stc.data.shape[0] == fwd["nsource"]
    assert np.all(np.isfinite(stc.data))
    assert np.any(stc.data > 0)


def test_recipsiicos_recovers_bilateral_auditory(auditory):
    """ReciPSIICOS recovers power in both hemispheres for the auditory response.

    Correlated bilateral sources are exactly what a plain LCMV tends to cancel;
    the cleaned covariance should retain both. This is a soft check: it asserts
    that each hemisphere carries a non-trivial share of the peak power, not a
    precise localisation.
    """
    info, fwd, data_cov, noise_cov = auditory
    ranks, _, _, kstar = recipsiicos_rank_curve(
        fwd, info, method="whitened", noise_cov=noise_cov, return_optimal=True
    )
    filters = make_recipsiicos_lcmv(
        info, fwd, data_cov, rank=kstar, method="whitened", noise_cov=noise_cov,
        pick_ori="max-power", weight_norm="unit-noise-gain", reduce_rank=True,
    )
    stc = apply_lcmv_cov(data_cov, filters)
    n_lh = len(stc.vertices[0])
    lh_peak = stc.data[:n_lh].max()
    rh_peak = stc.data[n_lh:].max()
    both = max(lh_peak, rh_peak)
    assert min(lh_peak, rh_peak) > 0.05 * both  # both hemispheres active


def test_mcmv_on_real_meg(auditory):
    """MCMV builds a finite joint filter on two separated real sources."""
    info, fwd, data_cov, noise_cov = auditory
    fwd_fixed = mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=True, verbose=False
    )
    # One source in each hemisphere (first and last vertices are opposite sides).
    sources = [0, fwd_fixed["nsource"] - 1]
    mcmv = make_mcmv(
        info, fwd_fixed, data_cov, sources=sources, noise_cov=noise_cov
    )
    assert np.all(np.isfinite(mcmv["weights"]))
    stc = apply_mcmv_cov(data_cov, mcmv)
    assert stc.data.shape[0] == len(sources)
    assert np.all(np.isfinite(stc.data))
