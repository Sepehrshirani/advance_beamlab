API reference
=============

All public functions and classes, documented from their in-code docstrings
(which carry the equation references used throughout the
:doc:`mathematical background <index>`).

.. currentmodule:: mne_beamlab

MCMV beamformer
---------------

.. autosummary::
   :toctree: generated/

   make_mcmv
   apply_mcmv
   apply_mcmv_cov
   MCMVBeamformer

Source discovery (localizers and search)
----------------------------------------

.. autosummary::
   :toctree: generated/

   localizer_value
   optimal_orientation
   scan_mcmv
   MCMVScanResult

ReciPSIICOS covariance modification
-----------------------------------

.. autosummary::
   :toctree: generated/

   make_recipsiicos_cov
   make_recipsiicos_lcmv
   recipsiicos_rank_curve

References
----------

.. footbibliography::
