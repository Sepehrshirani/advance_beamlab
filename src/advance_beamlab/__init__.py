"""advance_beamlab: advanced minimum-variance beamformers for MNE-Python.

A staging package of peer-reviewed beamforming algorithms that extend the
adaptive spatial filters in :mod:`mne.beamformer`, written to MNE-Python's
contribution standards so that each algorithm can be upstreamed.

Currently implemented:

- Multiple Constrained Minimum Variance (MCMV) beamformer
  (:func:`~advance_beamlab.make_mcmv`), after Moiseev et al. (2011).
- ReciPSIICOS data-covariance modification for correlated-source robustness
  (:func:`~advance_beamlab.make_recipsiicos_cov` and
  :func:`~advance_beamlab.make_recipsiicos_lcmv`), after Kuznetsova et al. (2021).
- Pairwise and augmented-pairwise MCMV for leakage-free connectivity
  (:func:`~advance_beamlab.pairwise_mcmv_connectivity` and
  :func:`~advance_beamlab.augmented_pairwise_mcmv_connectivity`), after Nunes et
  al. (2020).
- Adaptive Bayesian beamformer with multiple constraints (ABMC) for localising
  low-power spike-like transients such as epileptic IEDs and delayed responses
  (:func:`~advance_beamlab.make_abmc`, with the sparse Bayesian learning
  covariance :func:`~advance_beamlab.sbl_covariance`), after Shirani et al. (2024).
"""

from ._abmc import (
    ABMCResult,
    make_abmc,
    sbl_covariance,
)
from ._connectivity import (
    ar1_surrogate_significance,
    augmented_pairwise_mcmv_connectivity,
    pairwise_mcmv_connectivity,
    reconstruct_pairwise_mcmv,
)
from ._localizers import (
    MCMVScanResult,
    localizer_value,
    optimal_orientation,
    scan_mcmv,
)
from ._mcmv import (
    MCMVBeamformer,
    apply_mcmv,
    apply_mcmv_cov,
    make_mcmv,
)
from ._recipsiicos import (
    make_recipsiicos_cov,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
)

__all__ = [
    "make_mcmv",
    "apply_mcmv",
    "apply_mcmv_cov",
    "MCMVBeamformer",
    "scan_mcmv",
    "MCMVScanResult",
    "localizer_value",
    "optimal_orientation",
    "make_recipsiicos_cov",
    "make_recipsiicos_lcmv",
    "recipsiicos_rank_curve",
    "reconstruct_pairwise_mcmv",
    "pairwise_mcmv_connectivity",
    "augmented_pairwise_mcmv_connectivity",
    "ar1_surrogate_significance",
    "sbl_covariance",
    "make_abmc",
    "ABMCResult",
]

__version__ = "0.1.0.dev0"
