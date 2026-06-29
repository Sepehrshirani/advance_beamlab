# MNE-beamlab

Advance beamformers for MEG/EEG source reconstruction,
built to be fully compatible with [MNE-Python](https://mne.tools) and to
MNE-Python's contribution standards, so that each algorithm can be upstreamed
into `mne.beamformer`.

This package is a staging ground: every algorithm is implemented exactly from
its peer-reviewed source, cites the official paper, uses the algorithm's exact
published name, and mirrors the `mne.beamformer` API.

## Implemented algorithms

### 1- Multiple Constrained Minimum Variance (MCMV) beamformer

The **Multiple Constrained Minimum Variance (MCMV) beamformer** reconstructs a
set of `n` sources that are constrained *jointly*, imposing unit gain on each
source's own forward field and **zero** gain on every other constrained
source's field. The zero-gain constraints make the filter insensitive to
correlations between the constrained sources, removing the source leakage and
signal cancellation that bias single-source LCMV when sources are correlated.

The implementation follows the derivation of **Moiseev et al. (2011)**, *Application of multi-source minimum variance beamformers for reconstruction of correlated neural activity*, NeuroImage 58(2):481-496 ([doi:10.1016/j.neuroimage.2011.05.081](https://doi.org/10.1016/j.neuroimage.2011.05.081)). The closed-form weight matrix is Eq. (5) of that paper,

```
W = R^-1 H (H^T R^-1 H)^-1
```

with `R` the data covariance and `H` the joint forward matrix. MCMV generalises
the single-source LCMV beamformer (Van Veen et al., 1997), itself rooted in the
linearly constrained adaptive array of Frost (1972). The connectivity
application and the augmented pairwise variant (APW-MCMV) come from Nunes et
al. (2020), NeuroImage 208:116386.

### 2- ReciPSIICOS covariance modification

The **ReciPSIICOS** method makes the *base* LCMV beamformer robust to
correlated sources by cleaning the data covariance before the beamformer is
built -- it is a covariance transform for the LCMV, not a new spatial filter. Treating the
`M x M` sensor covariance as a vector in the `M^2`-dimensional space of
matrices, the sensor covariance decomposes (Eq. 8) into *auto-products* of the
source topographies, which carry the source powers, and *cross-products* of
pairs of topographies, which carry the source couplings. The cross-products are
exactly what an LCMV beamformer exploits to cancel correlated sources.
ReciPSIICOS builds, **from the forward model alone**, a projector that
suppresses the cross-product subspace while sparing the auto-product subspace,
applies it to the data covariance, and restores positive-definiteness. A
standard `make_lcmv` built on the modified covariance no longer cancels
correlated sources.

The implementation follows **Kuznetsova, Nurislamova & Ossadtchi (2021)**, *Modified covariance beamformer for solving MEG inverse problem in the environment with correlated sources*, NeuroImage 228:117677 ([doi:10.1016/j.neuroimage.2020.117677](https://doi.org/10.1016/j.neuroimage.2020.117677)), and provides both projector variants from that paper:

- **`recipsiicos`** -- project the vectorised covariance *onto* the principal
  power subspace (Eq. 10).
- **`whitened`** -- project *away from* the principal correlation subspace, in a
  space whitened with respect to the power subspace (Eqs. 15-17), which spares
  the source-power terms more effectively.

Both depend only on the forward model, so the projector is built once and
reused across datasets that share it. The single free parameter is the
projection rank `K`; `recipsiicos_rank_curve` reports how much power- and
correlation-subspace energy each rank retains (Eqs. 20-21) to guide the choice.
Because the method modifies only the covariance, it is handed straight to MNE's
own `make_lcmv`; `make_recipsiicos_lcmv` does the modify-then-LCMV step in one
call.

## Installation

```bash
pip install -e .            # from a clone
pip install -e ".[dev]"     # with test + doc + lint tooling
```

Requires Python >= 3.10, `mne >= 1.10`, NumPy and SciPy.

## Usage

### MCMV

```python
import numpy as np
import mne
from mne_beamlab import (
    make_mcmv, 
    apply_mcmv, 
    apply_mcmv_cov,
)

# Any covariance estimator / shrinkage method supported by MNE is inherited
# unchanged -- the choice is made here, upstream of the beamformer.
data_cov = mne.compute_covariance(epochs, method="shrunk")  # or "oas", "empirical", ...

# Constrain two sources jointly. `sources` are indices into the forward's
# source space; `orientations` are required for a free-orientation forward
# (omit them for a fixed-orientation forward).
sources = [1234, 5678]
orientations = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

filters = make_mcmv(
    epochs.info, forward, data_cov, sources,
    orientations=orientations, reg=0.05, weight_norm="unit-gain",
)

# Reconstruct the joint source time courses -> ndarray (n_sources, n_times).
source_waveforms = apply_mcmv(evoked, filters)

# Reconstructed source covariance (n_sources x n_sources) for connectivity.
source_cov = apply_mcmv_cov(data_cov, filters)
```

### ReciPSIICOS

```python
import mne
from mne.beamformer import make_lcmv, apply_lcmv
from mne_beamlab import (
    make_recipsiicos_cov,
    make_recipsiicos_lcmv,
    recipsiicos_rank_curve,
)

data_cov = mne.compute_covariance(epochs, method="empirical")

# Optional: inspect the power/correlation retention curve to choose the rank.
ranks, p_pwr, p_cor = recipsiicos_rank_curve(forward, method="whitened")

# Either modify the covariance and hand it to MNE's make_lcmv yourself ...
clean_cov = make_recipsiicos_cov(
    data_cov, forward, rank=200, method="whitened", info=epochs.info,
)
filters = make_lcmv(epochs.info, forward, clean_cov, reg=0.05)
stc = apply_lcmv(evoked, filters)

# ... or do the modify-then-LCMV in one step.
filters = make_recipsiicos_lcmv(
    epochs.info, forward, data_cov, rank=200, method="whitened",
)
stc = apply_lcmv(evoked, filters)
```

The projection rank depends on the forward model and the sensor array. 
The ranks used in the paper are on the order of a few hundred for arrays reduced to 40-80 virtual sensors; `recipsiicos_rank_curve`
adapts the choice to your own forward model.

## Design notes

**Covariance estimation and shrinkage.** Both algorithms consume
`mne.Covariance` objects, so every estimator offered by
`mne.compute_covariance` -- `empirical`, `diagonal_fixed`, `shrunk`,
`ledoit_wolf`, `oas` (oracle approximating shrinkage), `factor_analysis`, and
`auto` -- is available with no extra work; the estimator is chosen when the
covariance is built. MCMV inverts the data covariance with MNE's own
`_reg_pinv`, so the `reg` diagonal-loading convention and rank handling are
numerically identical to `make_lcmv`, and the single-source (`n == 1`) MCMV
filter reduces **exactly** to the corresponding LCMV filter. ReciPSIICOS hands
its modified covariance to `make_lcmv` unchanged, so the beamformer that
consumes it is MNE's, with all of its options.

**Errors vs. warnings.** Mathematically invalid requests raise (non-unique or
out-of-range sources, missing/forbidden orientations for the forward type, no
common channels, a non-finite or singular covariance, a beamformer order
exceeding the data rank, an out-of-range ReciPSIICOS rank, an unknown method).
Valid-but-limited situations warn, always disclosing what was done
(rank-deficient covariance handled by a pseudo-inverse, near-coincident
sources, an ill-conditioned constraint system, unit-noise-gain requested
without a noise covariance). For ReciPSIICOS specifically, the spectral-flip
step that restores positive-definiteness warns when the negative eigenvalues
carry more than 20% of the eigenvalue energy (Eq. 24), the threshold above
which the authors caution against trusting the result.

## Roadmap

The next modules follow the same pattern (exact equations, validation/warning
contract, paper-faithful tests):

1. The four MCMV localizers -- MAI, MPZ, MER, and the noise-covariance-free
   reduced MER (rMER) -- Moiseev et al. (2011), Table 1.
2. Data-driven orientation estimation (generalised eigenvalue problem) and the
   sequential source-search procedure, for full data-driven localization.
3. Augmented Pairwise MCMV (APW-MCMV) for connectivity -- Nunes et al. (2020).

## License

BSD-3-Clause.

## Citing

If you use this package, please cite the original method papers -- Moiseev et
al. (2011) and Nunes et al. (2020) for MCMV, and Kuznetsova et al. (2021) for
ReciPSIICOS -- listed in `doc/references.bib`.
