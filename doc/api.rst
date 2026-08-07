API reference
=============

All public functions and classes, documented from their in-code docstrings
(which carry the equation references used throughout the
:doc:`mathematical background <index>`).

.. currentmodule:: advance_beamlab

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

Connectivity (pairwise and augmented-pairwise MCMV)
---------------------------------------------------

.. autosummary::
   :toctree: generated/

   reconstruct_pairwise_mcmv
   pairwise_mcmv_connectivity
   augmented_pairwise_mcmv_connectivity
   ar1_surrogate_significance

ABMC (adaptive Bayesian beamformer with multiple constraints)
-------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   sbl_covariance
   make_abmc
   make_abmc_dictionary
   abmc_stability_curve
   ABMCResult

References
----------

.. footbibliography::
