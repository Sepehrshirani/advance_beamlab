# mne-beamformers

Advanced minimum-variance beamformers for MEG/EEG source reconstruction,
built to be fully compatible with [MNE-Python](https://mne.tools) and to
MNE-Python's contribution standards, so that each algorithm can be upstreamed
into `mne.beamformer`.

This package is a staging ground: every algorithm is implemented exactly from
its peer-reviewed source, cites the official paper, uses the algorithm's exact
published name, and mirrors the `mne.beamformer` API.

## Implemented algorithms

### Multiple Constrained Minimum Variance (MCMV) beamformer

The **Multiple Constrained Minimum Variance (MCMV) beamformer** reconstructs a
set of `n` sources that are constrained *jointly*, imposing unit gain on each
source's own forward field and **zero** gain on every other constrained
source's field. The zero-gain constraints make the filter insensitive to
correlations between the constrained sources, removing the source leakage and
signal cancellation that bias single-source LCMV when sources are correlated.

The implementation follows the derivation of **Moiseev et al. (2011)**, *Application of multi-source minimum variance beamformers for reconstruction of correlated neural activity*, NeuroImage 58(2):481–496 ([doi:10.1016/j.neuroimage.2011.05.081](https://doi.org/10.1016/j.neuroimage.2011.05.081)). The closed-form weight matrix is Eq. (5) of that paper,

```
W = R⁻¹ H (Hᵀ R⁻¹ H)⁻¹
```

with `R` the data covariance and `H` the joint forward matrix. MCMV generalises
the single-source LCMV beamformer (Van Veen et al., 1997), itself rooted in the
linearly constrained adaptive array of Frost (1972). The connectivity
application and the augmented pairwise variant (APW-MCMV) come from Nunes et
al. (2020), NeuroImage 208:116386.

## Installation

```bash
pip install -e .            # from a clone
pip install -e ".[dev]"     # with test + doc + lint tooling
```

Requires Python ≥ 3.10, `mne >= 1.10`, NumPy and SciPy.

## Usage

```python
import numpy as np
import mne
from mne_beamformers import make_mcmv, apply_mcmv, apply_mcmv_cov

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

## Design notes

**Covariance estimation and shrinkage.** `make_mcmv` consumes `mne.Covariance`
objects for both the data and noise covariances, so every estimator offered by
`mne.compute_covariance` — `empirical`, `diagonal_fixed`, `shrunk`,
`ledoit_wolf`, `oas` (oracle approximating shrinkage), `factor_analysis`, and
`auto` — is available with no extra work; the estimator is chosen when the
covariance is built. The data covariance is inverted with MNE's own
`_reg_pinv`, so the `reg` diagonal-loading convention and rank handling are
numerically identical to `make_lcmv`, and the single-source (`n == 1`) MCMV
filter reduces **exactly** to the corresponding LCMV filter.

**Errors vs. warnings.** Mathematically invalid requests raise (non-unique or
out-of-range sources, missing/forbidden orientations for the forward type, no
common channels, a non-finite or singular covariance, a beamformer order
exceeding the data rank). Valid-but-limited situations warn, always disclosing
what was done (rank-deficient covariance handled by a pseudo-inverse,
near-coincident sources, an ill-conditioned constraint system, unit-noise-gain
requested without a noise covariance).

## Roadmap

The next modules follow the same pattern (exact equations, validation/warning
contract, paper-faithful tests):

1. The four MCMV localizers — MAI, MPZ, MER, and the noise-covariance-free
   reduced MER (rMER) — Moiseev et al. (2011), Table 1.
2. Data-driven orientation estimation (generalised eigenvalue problem) and the
   sequential source-search procedure, for full data-driven localization.
3. Augmented Pairwise MCMV (APW-MCMV) for connectivity — Nunes et al. (2020).

## License
BSD-3-Clause (it's the one MNE uses)

## Citing

Please cite the original method papers (Moiseev et
al., 2011; Nunes et al., 2020) listed in `doc/references.bib`.
