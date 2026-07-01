"""mne-beamlab: advanced minimum-variance beamformers for MNE-Python.

A staging package of peer-reviewed beamforming algorithms that extend the
adaptive spatial filters in :mod:`mne.beamformer`, written to MNE-Python's
contribution standards so that each algorithm can be upstreamed.

Currently implemented:

- Multiple Constrained Minimum Variance (MCMV) beamformer
  (:func:`~mne_beamlab.make_mcmv`), after Moiseev et al. (2011).
- ReciPSIICOS data-covariance modification for correlated-source robustness
  (:func:`~mne_beamlab.make_recipsiicos_cov` and
  :func:`~mne_beamlab.make_recipsiicos_lcmv`), after Kuznetsova et al. (2021).
"""

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
]

__version__ = "0.1.0.dev0"