"""mne-beamformers: advanced minimum-variance beamformers for MNE-Python.

A staging package of peer-reviewed beamforming algorithms that extend the
adaptive spatial filters in :mod:`mne.beamformer`, written to MNE-Python's
contribution standards so that each algorithm can be upstreamed.

Currently implemented:

- Multiple Constrained Minimum Variance (MCMV) beamformer
  (:func:`~mne_beamformers.make_mcmv`), after Moiseev et al. (2011).
"""

from ._mcmv import (
    MCMVBeamformer,
    apply_mcmv,
    apply_mcmv_cov,
    make_mcmv,
)

__all__ = [
    "make_mcmv",
    "apply_mcmv",
    "apply_mcmv_cov",
    "MCMVBeamformer",
]

__version__ = "0.1.0.dev0"
