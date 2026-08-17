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

Reporting a result
------------------

A beamformer's raw output power is not comparable between locations: it grows
with depth, with the filter's noise gain, and with whatever the covariance
happened to contain. :func:`power_image` applies the normalisations that make it
interpretable, and that nearly every beamformer result in the literature is
reported in -- pseudo-Z against the noise floor, the differential pseudo-T
against a control window, and the ratio pseudo-F for multiplicative changes.
They work on any of the filter sets in this package, because the power is taken
from each method's public apply path rather than from its stored weights.

.. autosummary::
   :toctree: generated/

   power_image

Knowing the rank
----------------

Every method here inverts a covariance, and every one behaves badly if that
covariance is rank deficient and the inverse is not told. Prefer
:func:`mne.compute_rank` when the data's provenance is available, since it reads
the recorded bookkeeping rather than inferring anything. These are for the
common case where it is not -- someone else's data, or a pipeline whose steps
are not separately recorded -- and for checking a recorded rank against what the
spectrum actually says.

.. autosummary::
   :toctree: generated/

   estimate_rank
   rank_spectrum

Deciding whether it is real
---------------------------

A beamformer image is not something a parametric test can be pointed at: its
values are ratios of quadratic forms, and its spatial smoothness is neither
uniform nor known in advance, because the filter's resolution varies with depth,
geometry and source strength. :func:`permutation_image_test` judges an image
against the distribution its own relabellings produce, and corrects across the
image by the maximum statistic, which controls the family-wise error rate
without assuming anything about the smoothing.

.. autosummary::
   :toctree: generated/

   permutation_image_test

Understanding the constraint
-----------------------------

The methods here differ from an ordinary LCMV in their *constraint*, which is
the part hardest to picture from the algebra. :func:`constraint_demo` turns it
into a number: the gain of each filter at each source, measured rather than read
out of the weights, so that the four methods are directly comparable.
:func:`evoked_demo` does the same for a recording, reporting only what needs no
truth, and :func:`resolve_template` builds the waveform ABMC is asked to look
for when you would rather name a band than supply an array.

.. autosummary::
   :toctree: generated/

   constraint_demo
   constraint_explorer
   ConstraintDemo
   evoked_demo
   evoked_sources
   EvokedDemo
   resolve_template

References
----------

.. footbibliography::
