.. _examples-gallery:

Examples
========

Worked examples demonstrating the advance_beamlab beamformers end to end. Several run
on the MNE ``sample`` dataset, downloaded on first run via
:func:`mne.datasets.sample.data_path`, and show a beamformer on real MEG data.
Two are self-contained simulations that need no download and isolate, in
controlled settings, the correlated-source cancellation that MCMV overcomes and
the signal leakage that pairwise and augmented MCMV remove from connectivity
estimates.
