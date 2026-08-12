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
- A finite-element head model for EEG (:func:`~advance_beamlab.read_ny_head_forward`),
  wrapping the precomputed New York Head lead field of Huang et al. (2016) as an
  :class:`mne.Forward` so that any beamformer here or in :mod:`mne.beamformer` can
  be run on a FEM forward, which MNE-Python cannot otherwise compute.
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

from ._abmc import (
    ABMCResult,
    abmc_stability_curve,
    make_abmc,
    make_abmc_dictionary,
    sbl_covariance,
)
from ._connectivity import (
    ar1_surrogate_significance,
    augmented_pairwise_mcmv_connectivity,
    pairwise_mcmv_connectivity,
    reconstruct_pairwise_mcmv,
)
from ._explain import ConstraintDemo, constraint_demo, constraint_explorer
from ._fem import (
    fetch_ny_head,
    make_ny_head_info,
    ny_head_montage,
    ny_head_picks,
    ny_head_plot_indices,
    ny_head_scalp,
    read_ny_head_forward,
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
    "make_abmc_dictionary",
    "abmc_stability_curve",
    "ABMCResult",
    "fetch_ny_head",
    "read_ny_head_forward",
    "ny_head_montage",
    "make_ny_head_info",
    "ny_head_scalp",
    "ny_head_plot_indices",
    "ny_head_picks",
    "constraint_demo",
    "ConstraintDemo",
    "constraint_explorer",
]

__version__ = "0.1.0.dev0"
