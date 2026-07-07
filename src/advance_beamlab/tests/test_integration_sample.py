# Authors: Sepehr Shirani and Muzhi Wang
# License: BSD-3-Clause
"""Integration tests on the MNE ``sample`` dataset (a real MEG recording).

The ``sample`` dataset ships a real BEM forward and the bilateral auditory
response -- two temporally correlated sources -- which is precisely the regime
MCMV and ReciPSIICOS target. These tests exercise the whole pipeline on real
data: real leadfields, a real data covariance, whitening across magnetometers
and gradiometers, the virtual-sensor reduction, projector construction, the
working-space LCMV solve, and the MCMV joint constraint.

Two practical choices make this a faithful test rather than a misleading one:

* The forward is decimated to a coarse *whole-brain* grid, not restricted to a
  region. The ReciPSIICOS projector is built from the forward and must span
  where the data covariance's energy actually lives (whole-brain activity plus
  noise); a region-restricted forward would send most of that energy negative in
  the spectral-flip step. Decimation keeps the O(N^2) correlation Gram of the
  ``whitened`` projector tractable while preserving whole-brain coverage.
* The covariances are shrinkage-regularised, the correct estimator for a real
  magnetometer+gradiometer covariance (an empirical estimate is wildly
  ill-conditioned across the two unit scales).

The dataset is not bundled with MNE; it is fetched once with
``mne.datasets.sample.data_path()``. When it is absent every test here is
skipped, so the suite still runs offline -- the algorithmic paths themselves are
covered by ``test_recipsiicos.py`` and ``test_mcmv.py``.
"""

import mne
import numpy as np
import pytest
from mne.beamformer import apply_lcmv_cov

from advance_beamlab import (
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
    _FWD = _SAMPLE / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    _HAVE_SAMPLE = _SAMPLE.is_dir() and _FWD.is_file()
except Exception:  # pragma: no cover - depends on local dataset state
    _HAVE_SAMPLE = False

pytestmark = pytest.mark.skipif(
    not _HAVE_SAMPLE, reason="MNE sample dataset not downloaded"
)


def _decimate_forward(fwd, step):
    """Coarse *whole-brain* forward: keep every ``step``-th vertex per hemisphere.

    Uses the forward's own vertices, so no BEM/trans/recompute is needed. Keeps
    both hemispheres so the source grid still spans the head.
    """
    from mne import Label
    from mne.forward import restrict_forward_to_label

    labels = []
    for hemi_idx, hemi in enumerate(("lh", "rh")):
        vertno = fwd["src"][hemi_idx]["vertno"][::step]
        labels.append(Label(vertno, hemi=hemi, subject="sample"))
    return restrict_forward_to_label(fwd, labels)


@pytest.fixture(scope="module")
def auditory():
    """Real MEG data + a coarse whole-brain BEM forward + shrunk covariances."""
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
    # Shrinkage covariances: the correct estimator across mag/grad unit scales.
    data_cov = mne.compute_covariance(
        epochs, tmin=0.05, tmax=0.2, method="shrunk", verbose=False
    )
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="shrunk", verbose=False
    )

    fwd = mne.read_forward_solution(_FWD, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
    fwd = _decimate_forward(fwd, step=16)  # ~450 whole-brain sources
    return epochs.info, fwd, data_cov, noise_cov


@pytest.mark.parametrize("method", ["recipsiicos", "whitened"])
def test_recipsiicos_pipeline_on_real_meg(auditory, method):
    """Both projectors build a valid beamformer and localise finite power.

    On a whole-brain forward the cleaned covariance stays (numerically) positive
    semi-definite, so the localised power must be non-negative -- the property
    violated by a region-restricted forward.
    """
    info, fwd, data_cov, noise_cov = auditory

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
    # power is non-negative up to numerical tolerance, and there is signal
    assert stc.data.min() > -1e-6 * stc.data.max()
    assert stc.data.max() > 0


def test_recipsiicos_covers_both_hemispheres(auditory):
    """The bilateral auditory response leaves power in both hemispheres.

    A soft check: each hemisphere carries a non-trivial share of the peak, not a
    precise localisation.
    """
    info, fwd, data_cov, noise_cov = auditory
    _, _, _, kstar = recipsiicos_rank_curve(
        fwd, info, method="whitened", noise_cov=noise_cov, return_optimal=True
    )
    filters = make_recipsiicos_lcmv(
        info,
        fwd,
        data_cov,
        rank=kstar,
        method="whitened",
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        reduce_rank=True,
    )
    stc = apply_lcmv_cov(data_cov, filters)
    n_lh = len(stc.vertices[0])
    lh_peak = stc.data[:n_lh].max()
    rh_peak = stc.data[n_lh:].max()
    assert min(lh_peak, rh_peak) > 0.02 * max(lh_peak, rh_peak)


def test_mcmv_on_real_meg(auditory):
    """MCMV builds a finite joint filter on two separated real sources."""
    info, fwd, data_cov, noise_cov = auditory
    fwd_fixed = mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=True, verbose=False
    )
    # One source in each hemisphere (first and last vertices are opposite sides).
    sources = [0, fwd_fixed["nsource"] - 1]
    mcmv = make_mcmv(info, fwd_fixed, data_cov, sources=sources, noise_cov=noise_cov)
    assert np.all(np.isfinite(mcmv["weights"]))
    src_cov = apply_mcmv_cov(data_cov, mcmv)
    assert src_cov.shape == (len(sources), len(sources))
    assert np.all(np.isfinite(src_cov))
