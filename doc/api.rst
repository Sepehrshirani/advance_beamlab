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

Head modelling (finite element, EEG)
------------------------------------

MNE-Python computes forward solutions with the boundary element method. These
functions supply a *finite element* alternative for EEG by wrapping the
precomputed New York Head lead field as an ordinary :class:`mne.Forward`, which
every beamformer here and in :mod:`mne.beamformer` accepts unchanged. The model
is downloaded on demand and is not redistributed with this package.

.. autosummary::
   :toctree: generated/

   fetch_ny_head
   read_ny_head_forward
   ny_head_montage
   make_ny_head_info
   ny_head_scalp
   ny_head_plot_indices
   ny_head_picks

Understanding the constraint
-----------------------------

The methods here differ from an ordinary LCMV in their *constraint*, which is
the part hardest to picture from the algebra. :func:`constraint_demo` turns it
into a number: the gain of each filter at each source, measured rather than read
out of the weights, so that the four methods are directly comparable.

.. autosummary::
   :toctree: generated/

   constraint_demo
   ConstraintDemo

References
----------

.. footbibliography::
