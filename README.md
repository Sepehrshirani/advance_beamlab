# Advance Beam Lab

`advance_beamlab` — advanced minimum-variance beamformers for MEG/EEG source reconstruction, built
to be fully compatible with [MNE-Python](https://mne.tools) and to its
contribution standards, so that each algorithm can be upstreamed into
`mne.beamformer`.

Every algorithm is implemented exactly from its peer-reviewed source, cites the
official paper, uses the algorithm's published name, and mirrors the
`mne.beamformer` API. This README is written to be **self-contained**: the
[Mathematical background](https://sepehrshirani.github.io/advance_beamlab-docs/background.html)
section below derives every equation from first principles, so you can
understand and tune each method without reading the papers.

## Implemented algorithms

- **Multiple Constrained Minimum Variance (MCMV)** is a multi-source beamformer
  that constrains several sources *jointly*, which removes the signal
  cancellation that biases single-source LCMV when sources are correlated. It
  comes with the four **scanning localizers** (MAI, MPZ, MER, rMER) and the
  **sequential source search** that turn it into a discovery tool. After
  Moiseev et al. (2011).
- **ReciPSIICOS** makes an *ordinary* LCMV beamformer robust to correlated
  sources by cleaning the data covariance before the beamformer is built, with
  noise-whitening and virtual-sensor reduction so it applies to real,
  mixed-sensor MEG arrays. After Kuznetsova, Nurislamova & Ossadtchi (2021).
- **Pairwise & Augmented Pairwise MCMV (PW-/APW-MCMV) connectivity** provides
  leakage-free functional connectivity built on MCMV: each region pair is
  reconstructed with a 2-source MCMV (removing *direct* leakage between them),
  and every statistically significant pair is re-estimated with neighbouring
  regions added to the beamformer (suppressing *indirect* leakage through them).
  Coherence and the phase measures are delegated to `mne-connectivity`; the
  amplitude-envelope metric follows the paper's own definition, including the
  0.5 Hz envelope low-pass that `mne-connectivity` does not expose. After
  Nunes et al. (2020).
- **Adaptive Bayesian beamformer with multiple constraints (ABMC)** localises
  low-power, spike-like transients (epileptic IEDs and delayed responses to
  single-pulse electrical stimulation) that ordinary LCMV localises poorly. It
  pairs a sparse Bayesian learning (Champagne) covariance that strips
  correlated-source structure with a template-constrained beamformer
  that locks the output onto the known spike morphology. After Shirani et al.
  (2024).
- **A finite-element head model for EEG**. MNE-Python computes forward
  solutions with the boundary element method only, which cannot represent the
  cerebrospinal fluid, the layered skull, or the openings at the orbits and the
  auditory meatus; for EEG those omissions bias the forward field.
  `read_ny_head_forward` wraps the precomputed six-tissue FEM lead field of the
  New York Head (Huang, Parra & Haufe, 2016) as an ordinary `mne.Forward`, so
  `mne.beamformer.make_lcmv` and every method above run on a FEM forward
  unchanged. EEG only (see the caveat below). The model is downloaded on demand
  and is **not** redistributed with this package (its licence is GPL v3).

## Installation

```bash
pip install -e ".[dev]"     # from a clone, for development (tests + docs + lint)
# or the runtime install only:
pip install -e .
```

Requires Python ≥ 3.10 and MNE-Python ≥ 1.10. The FEM head model additionally
needs `h5py` (`pip install -e ".[fem]"`), because the model ships as a MATLAB
v7.3/HDF5 file; everything else works without it.

### The New York Head FEM forward model

```python
from advance_beamlab import read_ny_head_forward, make_ny_head_info

fwd = read_ny_head_forward(resolution="10K")  # fetches 678 MB on first use
info = make_ny_head_info(sfreq=250.0)  # 231 electrodes, average reference
```

`fwd` is an ordinary `mne.Forward`: pass it to `make_lcmv`, `make_mcmv`,
`make_recipsiicos_lcmv` or `make_abmc` exactly as you would a BEM forward. Two
things are worth knowing. The lead field is supplied in **common average
reference**, so it is rank deficient by exactly one (230 of 231) and the data
must carry an average-reference projector. `make_ny_head_info` builds an info
that does. And the approach is **EEG-only**, not by omission: EEG electrodes
follow a standardised layout, so a template head can be solved once and reused,
whereas MEG sensor positions relative to the brain differ every session
(`info['dev_head_t']`), so no template MEG array exists to solve for. This costs
little, because the tissues a BEM misrepresents are the ones MEG is least
sensitive to. BEM is a reasonable model for MEG, and the weak link for EEG.

## Quick start

**Discover correlated sources with MCMV** (no need to know where they are):

```python
import mne
from advance_beamlab import scan_mcmv, apply_mcmv

# info, forward (free orientation), data_cov, noise_cov as usual in MNE
result = scan_mcmv(
    info, forward, data_cov, noise_cov=noise_cov, localizer="mpz", n_sources=2
)

print(result["sources"])  # discovered grid indices
print(result["pseudo_z"])  # per-source pseudo-Z (judge how many are real)
stc_time_courses = apply_mcmv(epochs, result["filters"])  # jointly-optimal filters
```

**Make LCMV robust to correlation with ReciPSIICOS**:

```python
from advance_beamlab import make_recipsiicos_lcmv, recipsiicos_rank_curve
from mne.beamformer import apply_lcmv

# The projector is built from the forward model alone; K lives in the
# (whitened, reduced) virtual-sensor space. Let the 45-degree criterion pick it.
ranks, p_pwr, p_cor, k_opt = recipsiicos_rank_curve(
    forward, info, method="whitened", noise_cov=noise_cov, return_optimal=True
)

filters = make_recipsiicos_lcmv(
    info, forward, data_cov, rank=k_opt, method="whitened", noise_cov=noise_cov
)
stc = apply_lcmv(evoked, filters)
```

**Estimate leakage-free connectivity with PW-/APW-MCMV** (resting-state alpha):

```python
from advance_beamlab import (
    pairwise_mcmv_connectivity,
    augmented_pairwise_mcmv_connectivity,
    ar1_surrogate_significance,
)

# rois: grid indices of the regions of interest. Band-pass the data to the
# analysis band (e.g. 8-12 Hz) and estimate data_cov from the *same* band, so
# the weights are tuned to it (broadband weights invent spurious connectivity).
conn = pairwise_mcmv_connectivity(
    band_data, info, forward, data_cov, rois, method="envelope", noise_cov=noise_cov
)

# keep the significant edges, then re-estimate them with neighbour augmentation
sig = ar1_surrogate_significance(
    conn, reference_time_courses, method="envelope", sfreq=info["sfreq"]
)
conn_apw = augmented_pairwise_mcmv_connectivity(
    band_data,
    info,
    forward,
    data_cov,
    rois,
    conn,
    sig,
    method="envelope",
    noise_cov=noise_cov,
)
```

---

<!-- doc-split: background -->

# Mathematical background

Notation: $M$ sensors, $N$ source locations. Bold lowercase are vectors, bold
uppercase are matrices, $^{\mathsf T}$ is transpose, $\langle\cdot\rangle$ is the
time average. $\mathbf{I_n}$ is the $n\times n$ identity.

## 1. The measurement model

At each instant the sensors measure a linear mixture of the active sources plus
noise:

$$\mathbf{x}(t)=\sum_{i} \mathbf{g_i}\, s_i(t) + \mathbf{n}(t)=\mathbf{G}\,\mathbf{s}(t)+\mathbf{n}(t).$$

- $\mathbf{x}(t)\in\mathbb{R}^{M}$ is the sensor reading at time $t$.
- $s_i(t)$ is the (scalar) time course of source $i$.
- $\mathbf{g_i}\in\mathbb{R}^{M}$ is the **forward field** (a.k.a. leadfield,
  topography) of source $i$: the pattern that source produces across the sensors
  when it has unit amplitude. It is computed once from the head model and sensor
  geometry, and is therefore *known*.
- $\mathbf{n}(t)$ is additive noise with covariance
  $\mathbf{C_n}=\langle\mathbf{n}\mathbf{n}^{\mathsf T}\rangle$.

For a source at location $\mathbf{r}$ with **free orientation**, the forward
field is $\mathbf{g}=\mathbf{L}(\mathbf{r})\,\mathbf{u}$, where
$\mathbf{L}(\mathbf{r})\in\mathbb{R}^{M\times 3}$ holds the fields of unit
dipoles along $x,y,z$ and $\mathbf{u}$ is the (unit) orientation. Fixing the
orientation collapses this to a single column.

Everything downstream is driven by the **data covariance**

$$\mathbf{R}=\langle\mathbf{x}\,\mathbf{x}^{\mathsf T}\rangle .$$

If the sources have covariance $\mathbf{C_s}=\langle\mathbf{s}\mathbf{s}^{\mathsf T}\rangle$
and are uncorrelated with the noise, then
$\mathbf{R}=\mathbf{G}\,\mathbf{C_s}\,\mathbf{G}^{\mathsf T}+\mathbf{C_n}$. Keep
this identity in mind: it is the reason the localizers below peak at the true
sources, and the reason the ReciPSIICOS decomposition works.

## 2. The spatial filter, and the LCMV solution

A **spatial filter** is a vector $\mathbf{w}$ that estimates one source's time
course as a weighted sum of sensors, $\hat s(t)=\mathbf{w}^{\mathsf T}\mathbf{x}(t)$.
We want two things:

1. **Unit gain on the target**: $\mathbf{w}^{\mathsf T}\mathbf{g}=1$, so the
   target passes through untouched.
2. **Minimum output power**: minimise
   $\langle\hat s^2\rangle=\mathbf{w}^{\mathsf T}\mathbf{R}\,\mathbf{w}$. Since
   the target is pinned by the constraint, minimising *total* output power
   forces the filter to suppress everything else: other sources and noise.

This is the Linearly Constrained Minimum Variance (LCMV) problem. Minimising
$\mathbf{w}^{\mathsf T}\mathbf{R}\mathbf{w}$ subject to
$\mathbf{w}^{\mathsf T}\mathbf{g}=1$ with a Lagrange multiplier $\lambda$,

$$\mathcal{L}=\mathbf{w}^{\mathsf T}\mathbf{R}\mathbf{w}-\lambda(\mathbf{w}^{\mathsf T}\mathbf{g}-1),$$

$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}}=2\mathbf{R}\mathbf{w}-\lambda\mathbf{g}=\mathbf{0}\;\Rightarrow\;\mathbf{w}=\tfrac{\lambda}{2}\mathbf{R}^{-1}\mathbf{g}.$$

Enforcing the constraint $\mathbf{w}^{\mathsf T}\mathbf{g}=1$ fixes $\lambda$ and gives the LCMV filter, whose reconstructed power is $\langle\hat s^2\rangle=1/(\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g})$:

$$\boxed{\;\mathbf{w}=\dfrac{\mathbf{R}^{-1}\mathbf{g}}{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}}\;}$$

Scanning $1/(\mathbf{g}(\mathbf{r})^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}(\mathbf{r}))$
over $\mathbf{r}$ is the classic beamformer power map.

## 3. Signal cancellation with correlated sources

Suppose two sources with fields $\mathbf{g_1},\mathbf{g_2}$ have correlation
$\rho$. The LCMV filter for source 1 is free to place a null anywhere except
along $\mathbf{g_1}$. Because source 2 is correlated with source 1, the filter
can *lower its own output power* by passing a scaled, sign-flipped copy of
source 2 that partially cancels source 1 in the average. The constraint on
$\mathbf{g_1}$ is still satisfied instant by instant, but the variance is
reduced by exploiting the correlation. The result is **signal cancellation**:
the reconstructed power of source 1 is suppressed by roughly a factor
$(1-\rho^2)$, collapsing to zero as $\rho\to 1$. Amplitudes are underestimated,
locations are pulled towards each other, and spurious "connectivity" appears
because each filter leaks a copy of the other source. This single failure mode
is what MCMV and ReciPSIICOS each remove, in two different ways.

## 4. MCMV: the joint constraint

Instead of one filter that only protects its own source, MCMV solves for $n$
filters **at once** and forbids each from responding to the *others'* fields.
Collect the $n$ forward fields as the columns of
$\mathbf{H}=[\mathbf{g_1},\dots,\mathbf{g_n}]\in\mathbb{R}^{M\times n}$ and the
$n$ filters as the columns of $\mathbf{W}=[\mathbf{w_1},\dots,\mathbf{w_n}]$. The
constraint is

$$\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I_n},\qquad\text{i.e.}\qquad \mathbf{w_i}^{\mathsf T}\mathbf{g_j}=\delta_{ij}.$$

The diagonal ($i=j$) is the familiar **unit-gain** condition; the off-diagonal
($i\ne j$) **zero-gain** conditions are the new ingredient. Filter $i$ is forced
to be *blind* to every other constrained source. A blind filter cannot exploit a
correlation it cannot see, so the cancellation of Section 3 disappears.

Minimise the total output power $\mathrm{Tr}(\mathbf{W}^{\mathsf T}\mathbf{R}\mathbf{W})$
subject to $\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I_n}$. With a matrix of
Lagrange multipliers $\mathbf{\Lambda}$,

$$\mathcal{L}=\mathrm{Tr}(\mathbf{W}^{\mathsf T}\mathbf{R}\mathbf{W})-\mathrm{Tr}\big(\mathbf{\Lambda}(\mathbf{W}^{\mathsf T}\mathbf{H}-\mathbf{I_n})\big),$$

$$\frac{\partial\mathcal{L}}{\partial\mathbf{W}}=2\mathbf{R}\mathbf{W}-\mathbf{H}\mathbf{\Lambda}^{\mathsf T}=\mathbf{0}\;\Rightarrow\;\mathbf{W}=\tfrac12\mathbf{R}^{-1}\mathbf{H}\,\mathbf{\Lambda}^{\mathsf T}.$$

Substituting into the constraint $\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}$
fixes $\mathbf{\Lambda}$ and yields the **MCMV weights** (Moiseev et al. 2011,
Eq. 5):

$$\boxed{\;\mathbf{W}=\mathbf{R}^{-1}\mathbf{H}\,\big(\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}\big)^{-1}\;}$$

Reading it line by line: $\mathbf{R}^{-1}\mathbf{H}$ is the stack of ordinary
inverse-covariance filters; multiplying by
$(\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H})^{-1}$ mixes them just enough to
enforce all $n^2$ gain conditions simultaneously. For $n=1$ the bracket is a
scalar and this is *exactly* the LCMV filter of Section 2. MCMV is a strict
generalisation. The matrix $\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}$ is
invertible as long as the constrained fields are linearly independent; it
becomes singular if two sources coincide in both location and orientation (then
you have asked for two filters to do contradictory things), which the
implementation detects and reports.

In this package the joint solve is `make_mcmv(...)`; `apply_mcmv(...)` runs the
filters on `Raw`/`Epochs`/`Evoked`/arrays.

## 5. Whitening, and why it is mandatory for mixed sensors

The data covariance mixes sensor types with *different physical units*:
magnetometers in tesla ($\sim10^{-13}$), gradiometers in T/m, EEG in volts.
The scalar $\mathbf{w}^{\mathsf T}\mathbf{R}\mathbf{w}$ then sums
$\mathrm{T}^2$, $(\mathrm{T/m})^2$ and cross terms (literally adding
incommensurable quantities), and $\mathbf{R}^{-1}$ is numerically dominated by
whichever block is largest, making the filter effectively blind to the
smaller-scale sensor type. This is a *correctness* problem, not just
conditioning.

The fix is a change of coordinates. Build a **whitener** $\mathbf{W_n}$ from
the noise covariance so that $\mathbf{W_n}\mathbf{C_n}\mathbf{W_n}^{\mathsf T}=\mathbf{I}$,
from the eigendecomposition
$\mathbf{C_n}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^{\mathsf T}$,
$\mathbf{W_n}=\mathbf{\Lambda_r}^{-1/2}\mathbf{U_r}^{\mathsf T}$ (keeping only
the $r$ non-negligible eigenpairs, i.e. the numerical rank, which drops below $M$
after SSS/Maxwell filtering or ICA). Then whiten leadfield and covariance,

$$\tilde{\mathbf{H}}=\mathbf{W_n}\mathbf{H},\qquad \tilde{\mathbf{R}}=\mathbf{W_n}\mathbf{R}\,\mathbf{W_n}^{\mathsf T},$$

solve MCMV there, and fold the whitener back into the returned weights so they
still act on raw sensor data ($\hat{\mathbf{s}}=\tilde{\mathbf{W}}^{\mathsf T}\mathbf{W_n}\mathbf{x}$).
In whitened coordinates every channel is dimensionless with unit noise variance,
so all sensor types are commensurable.

Two consequences follow:

- **Single sensor type** (e.g. a CTF gradiometer array): with no `noise_cov` the
  ad-hoc model is a *scalar* multiple of the identity, so the whitener is a
  global scale that cancels out of the unit-gain filter and the result is
  unchanged. A **measured** single-type `noise_cov` is not a global scale, and
  because the `reg` diagonal loading is applied in the whitened space it is not
  equivariant under whitening: at `reg=0` the unit-gain filter is exactly
  invariant to the choice of noise covariance (verified to 1e-13), but at the
  default `reg=0.05` the weights change materially, by about 30% on the MNE
  `sample` gradiometer array. Whitening never mixes incommensurable units, but it
  is not a no-op for a single sensor type once `reg > 0`.
  Note also that MNE counts magnetometers and gradiometers as *two* types, so a
  combined MEG array requires a `noise_cov`; `make_mcmv` raises exactly as
  `make_lcmv` does if one is not supplied.
- **Per-type rank matters.** A single global eigenvalue threshold would discard
  the smaller-unit type as if it were the null space; the rank must be resolved
  *per sensor type*. The package uses `mne.cov.compute_whitener`, the same
  routine `make_lcmv` uses, so a single-source MCMV reproduces MNE's whitening
  exactly. With no `noise_cov`, MNE's ad-hoc per-type model is used.

## 6. Weight normalisation

- **`unit-gain`** (`weight_norm="unit-gain"` or `None`): the literal Eq. 5
  filter. The gain constraint $\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}$ holds
  on the raw leadfield, so reconstructed amplitudes are in physical source
  units. Use it when amplitudes/units matter.
- **`unit-noise-gain`**: rescale each filter so that
  $\mathbf{w_i}^{\mathsf T}\mathbf{C_n}\mathbf{w_i}=1$, i.e. unit output noise.
  Because the solve is in whitened space this is exactly unit Euclidean norm of
  the whitened filter (MNE's definition). It equalises the noise floor across
  locations, which is what you want for *maps* (so deep, low-SNR locations are
  not penalised) at the cost of no longer preserving physical amplitude.

## 7. The scanning localizers: MAI, MPZ, MER and rMER

MCMV needs to know *which* sources to constrain. The localizers are scalar maps
that peak at the true sources. They are built from four $n\times n$ matrices
(Moiseev et al. 2011, Table 2), each a leadfield sandwiched around a covariance:

$$\mathbf{S}=\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H},\qquad \mathbf{G}=\mathbf{H}^{\mathsf T}\mathbf{C_n}^{-1}\mathbf{H},$$

$$\mathbf{T}=\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{C_n}\mathbf{R}^{-1}\mathbf{H},\qquad \mathbf{E}=\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\bar{\mathbf{R}}\,\mathbf{R}^{-1}\mathbf{H},$$

where $\bar{\mathbf{R}}=\langle\bar{\mathbf{b}}\bar{\mathbf{b}}^{\mathsf T}\rangle$
is the covariance of the epoch-**averaged** (evoked) field. The four localizers
(Table 1) and what each measures:

| Localizer | Formula | Reduces at $n=1$ to | Use it for |
|---|---|---|---|
| **MAI** (activity index) | $\mathrm{Tr}(\mathbf{G}\mathbf{S}^{-1})-n$ | $\zeta-1=\dfrac{\mathbf{g}^{\mathsf T}\mathbf{C_n}^{-1}\mathbf{g}}{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}}-1$ | robust, broad peaks |
| **MPZ** (pseudo-Z) | $\mathrm{Tr}(\mathbf{S}\mathbf{T}^{-1})-n$ | $\bar Z-1=\dfrac{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}}{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{C_n}\mathbf{R}^{-1}\mathbf{g}}-1$ | sharper localisation |
| **MER** (evoked) | $\mathrm{Tr}(\mathbf{E}\mathbf{T}^{-1})$ | evoked pseudo-Z | phase-locked responses |
| **rMER** (reduced evoked) | $\mathrm{Tr}(\mathbf{E}\mathbf{S}^{-1})$ | reduced evoked pseudo-Z | evoked, no clean $\mathbf{C_n}$ |

Intuition: $\zeta$ is a *(signal+noise)/noise* ratio. Here
$(\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g})^{-1}$ is the total
reconstructed power (signal + leaked noise) and
$(\mathbf{g}^{\mathsf T}\mathbf{C_n}^{-1}\mathbf{g})^{-1}$ is the noise-only power,
so subtracting $1$ gives a *pure* signal-to-noise ratio that is zero where there
is no source. The multi-source versions generalise this to the whole
constrained set. Two facts proved in the paper and verified in the tests: both
power localizers are non-negative and ordered (the MAI value $\ge$ the MPZ value)
(MPZ is sharper but noisier), and each localizer's global maximum is the true
source configuration (they are *unbiased* for any $\mathbf{C_n}$). The power
localizers (MAI, MPZ) work for induced/oscillatory activity; the event-related
ones (MER, rMER) target phase-locked responses and need $\bar{\mathbf{R}}$.

Everything is computed in the whitened space where $\mathbf{C_n}=\mathbf{I}$; the
localizers are **invariant** under whitening, so the values are identical to the
raw-space formulas above.

## 8. Data-driven orientation in closed form

For a free-orientation forward each candidate location contributes an $M\times 3$
block $\mathbf{H_k}=[\mathbf{h}^x_k,\mathbf{h}^y_k,\mathbf{h}^z_k]$, and we must
pick the orientation $\mathbf{u_k}$ (so that $\mathbf{g_k}=\mathbf{H_k}\mathbf{u_k}$)
that maximises the localizer, *given* the sources already found (their fields
form the reference block $\mathbf{H_R}$). Moiseev et al. show the maximiser is
the top eigenvector of a $3\times 3$ generalised eigenproblem
$\mathbf{D}\,\mathbf{u_k}=\lambda\,\mathbf{F}\,\mathbf{u_k}$ (no scan over angles
is needed):

$$\mathbf{F}=\mathbf{A}_{kk}-\mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk},$$

$$\mathbf{D}=\mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{B}_{RR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk}-\mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{B}_{Rk}-\mathbf{B}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk}+\mathbf{B}_{kk}.$$

Here $(\mathbf{A},\mathbf{B})$ is the localizer's (denominator, numerator) pair:
$(\mathbf{S},\mathbf{G})$ for MAI, $(\mathbf{T},\mathbf{S})$ for MPZ,
$(\mathbf{T},\mathbf{E})$ for MER, $(\mathbf{S},\mathbf{E})$ for rMER. The
subscripted blocks are the Table-2 matrices evaluated between $\mathbf{H_R}$ and
$\mathbf{H_k}$. $\mathbf{F}$ is the Schur complement of the denominator (it
"conditions out" the already-found sources); $\mathbf{D}$ is the corresponding
numerator form. With no references (the first source), it collapses to

$$\mathbf{B}_{kk}\mathbf{u}=\lambda\mathbf{A}_{kk}\mathbf{u}.$$

Exposed as `optimal_orientation(...)`.

## 9. The sequential source search

The full parameter space of $n$ sources is far too large to scan jointly, so the
search is greedy and iterative (Moiseev et al. 2011):

1. **Find source 1** by scanning the single-source localizer over the grid
   (with the optimal orientation at each location).
2. **Fix** it as a reference and scan the **multi-source** localizer for
   source 2. Every candidate is now evaluated *jointly* with source 1.
3. Repeat, adding one source per iteration. Each new source is added to the
   joint constraint, so the cancellation that hid correlated activity is
   progressively removed and previously masked sources emerge.

**Knowing when to stop.** After adding source $k$, monitor its pseudo-Z
$\bar z_k=(\mathbf{w_k}^{\mathsf T}\mathbf{R}\mathbf{w_k})/(\mathbf{w_k}^{\mathsf T}\mathbf{C_n}\mathbf{w_k})$,
where $\mathbf{w_k}$ is that source's **row of the joint $k$-source MCMV
filter**, not a single-source (LCMV) filter at the same location, which would
still suffer the cancellation the joint constraint has just removed.
Genuine sources have large $\bar z_k$; once you start adding noise, $\bar z_k$
drops to a baseline and fluctuates. The baseline is generally **not** $1$ and
must be judged from the data, so `scan_mcmv` runs a requested `n_sources` and
returns the whole `pseudo_z` sequence for you to read the elbow. Note the greedy
search does not order the sources by strength: each step maximises the *joint*
localizer given the sources already fixed, so a later source can carry the larger
$\bar z_k$. Exposed as `scan_mcmv(...)`, which returns the discovered sources,
their orientations (in head coordinates), the pseudo-Z sequence, the
per-iteration localizer maps, and a ready-to-apply `MCMVBeamformer`.

## 10. ReciPSIICOS: cleaning the covariance instead of constraining sources

ReciPSIICOS attacks the same cancellation from the other end: rather than a new
filter, it *repairs the data covariance* so that an ordinary LCMV no longer
cancels correlated sources.

**The vec view.** Stack the columns of the $M\times M$ covariance into a vector
$\mathrm{vec}(\mathbf{R})\in\mathbb{R}^{M^2}$. Using
$\mathbf{R}=\mathbf{G}\mathbf{C_s}\mathbf{G}^{\mathsf T}+\mathbf{C_n}$ and
$\mathrm{vec}(\mathbf{g_i}\mathbf{g_j}^{\mathsf T})=\mathbf{g_j}\otimes\mathbf{g_i}$,

$$\mathrm{vec}(\mathbf{R})=\sum_i [\mathbf{C_s}]_{ii}\,\mathrm{vec}(\mathbf{g_i}\mathbf{g_i}^{\mathsf T})+\sum_{i<j}[\mathbf{C_s}]_{ij}\,\mathrm{vec}(\mathbf{g_i}\mathbf{g_j}^{\mathsf T}+\mathbf{g_j}\mathbf{g_i}^{\mathsf T})+\mathrm{vec}(\mathbf{C_n}).$$

The first sum collects the **auto-products** (the source powers); the second
collects the symmetric **cross-products** (the source couplings). The couplings
are *exactly* what an LCMV beamformer exploits to cancel correlated sources.
Remove the cross-product part of the covariance and the cancellation has nothing
to feed on.

**A working space that makes it device-agnostic and tractable.** The original
study used one MEG array of a single sensor type and built the projector in raw
sensor space. Two changes are needed for general use. The first is ours, the
second follows the paper:

- *Noise whitening.* Section 2.6 of the paper discusses whitening only to note
  that it changes the forward operator and would therefore require rebuilding
  the projector in the whitened space; the study did not apply it and worked in
  raw sensor space. Whitening here (via `mne.cov.compute_whitener`, per sensor
  type) is our own addition rather than a step the paper prescribes: it is what
  makes the method valid when magnetometers (T) and gradiometers (T/m) are
  mixed, since otherwise the covariance is dominated by the larger-unit type.
- *Virtual sensors.* The whitened leadfield is reduced by a truncated SVD to the
  $q$ directions carrying a chosen fraction of its variance (`pct_var`; the
  paper keeps 90%, or 95% for its real data, while the default here is a more
  conservative 0.99). On the 306-channel `sample` array that default gives
  $q=29$ magnetometer, $q=43$ gradiometer and $q=76$ combined-MEG virtual
  sensors. The $M^2\times M^2$ correlation Gram would be
  $93\text{k}\times 93\text{k}$ for a 306-channel array; in the $q$-dimensional
  working space it is a few thousand square.

Write $\mathbf{B}=\mathbf{U_q}^{\mathsf T}\mathbf{W}$ for the composite operator
(whiten by $\mathbf{W}$, then keep the $q$ principal directions $\mathbf{U_q}$ of
the whitened leadfield). Everything below lives in this working space: the
leadfield $\mathbf{B}\mathbf{G}$, the covariance
$\mathbf{B}\mathbf{R}\mathbf{B}^{\mathsf T}$, and the projector.

**Building the projector, from the forward model alone.** Enumerate the
(working-space) auto-product vectors over the source grid as the columns of $G_p$
and the symmetric cross-product vectors as $G_c$.

- **`recipsiicos`** (Eq. 10): take the SVD of $G_p$, keep the
  top $K$ left singular vectors $\mathbf{U_K}$, and project *onto* the power
  subspace, $\mathbf{P}=\mathbf{U_K}\mathbf{U_K}^{\mathsf T}$. This retains
  power, and whatever of the correlation subspace is (near-)orthogonal to it is
  removed.
- **`whitened`** (Eqs. 15–17): first *whiten by the power subspace*, forming
  $C_p=G_pG_p^{\mathsf T}$
  and its range-restricted inverse square root
  $W_p=\mathbf{E}\mathbf{\Lambda}^{-1/2}\mathbf{E}^{\mathsf T}$
  (the auto-products span at most the symmetric subspace of dimension
  $q(q{+}1)/2$, so $C_p$ is *never* full rank: its null space is dropped rather
  than ridge-filled, and only the retained eigenvalues are then stabilised by a
  ridge of `reg` times their mean). Then, in that whitened space,
  project *away from* the top $K$ correlation directions and unwhiten. Because
  the power directions are flattened to unit scale first, this spares them far
  better than the plain variant. $W_p$ only spans that symmetric subspace, so a
  rank $K\ge q(q{+}1)/2$ removes all of it and annihilates the covariance; the
  code warns when a requested rank reaches that bound.

**Applying it** (Eq. 11): reshape the projected vector back to a matrix and
symmetrise,

$$\tilde{\mathbf{R}}=\tfrac12\big(\mathbf{M}+\mathbf{M}^{\mathsf T}\big),\qquad \mathbf{M}=\mathrm{vec}^{-1}\big(\mathbf{P}\,\mathrm{vec}(\mathbf{B}\mathbf{R}\mathbf{B}^{\mathsf T})\big).$$

Projecting out a subspace can make $\tilde{\mathbf{R}}$ indefinite, but a
covariance used by a beamformer must be positive semi-definite, so we
**spectral-flip** (Eq. 12): eigendecompose and replace each eigenvalue by its
absolute value. If a large fraction of the covariance energy sat in negative
eigenvalues (default warn threshold 20%, Eq. 24), the rank $K$ was too aggressive
and the method warns.

**Solving the beamformer in the working space.** Because whitening and the
virtual-sensor reduction both change the space, the cleaned covariance cannot be
handed back to `make_lcmv` (which would whiten a second time and solve against
the wrong leadfield). Instead the LCMV is solved *in the working space* by
reusing MNE's own filter computation (its orientation selection, weight
normalisation and rank handling) with an identity whitener there (the
working-space noise is white by construction). The reduction operator
$\mathbf{B}$ is then folded into the returned `Beamformer` as its whitener, so
`apply_lcmv` applies the whole pipeline to sensor data unchanged.

**Free orientation** (Eqs. 22–23): at each location the local $M\times 3$
leadfield is reduced to its two dominant *tangential* topographies by a local
SVD; the power set then expands to three columns per location and the
correlation set to four columns per location pair.

**The one free parameter is $K$.** The projector depends only on the forward model, so it
is built once and reused across datasets sharing that forward.
`recipsiicos_rank_curve(...)` returns the retained power/correlation energy
versus $K$ (Eqs. 20–21), computed in closed form over all ranks from a single
decomposition rather than one per rank. With `return_optimal=True` it also
returns the rank $K^*$ at the 45° point where the correlation subspace stops
emptying faster than the power subspace (Section 2.4). The two methods traverse
the rank axis in *opposite* directions: the identity is $K=q^2$ for
`recipsiicos` and $K=0$ for `whitened` (which is why Fig. 19 of the paper puts
the ReciPSIICOS scale in descending order). Both curves therefore rise with $K$
for the former and fall with it for the latter, and $K^*$ is located
accordingly. Note $K$ lives in the $q^2$-dimensional working covariance space.
`make_recipsiicos_cov(...)` returns the cleaned `mne.Covariance` (for
inspection); `make_recipsiicos_lcmv(...)` builds the beamformer end to end.

**Practical notes for real data.** Three things matter when moving from
simulations to recordings:

- *Free-orientation MEG needs `reduce_rank=True`.* In a spherical conductor a
  radial source produces no external magnetic field, so a three-orientation MEG
  leadfield is effectively rank two per location and the working-space LCMV is
  singular unless the per-source forward rank is reduced, exactly as in
  `make_lcmv`. Pass `reduce_rank=True`, or use a fixed-orientation forward.
- *The correlation Gram is $O(N^2)$ in the source count.* The `whitened`
  projector and both rank curves enumerate every source *pair*, so a
  full-resolution grid (tens of thousands of vertices) is impractical: build the
  projector on a decimated source space (a few thousand vertices as in the
  paper, or a cortical label), then reuse it across datasets sharing that
  forward. The `recipsiicos` projector uses only the auto-products and stays
  linear in $N$. Runtime scales with $N^2$; *memory* does not. The cross-product
  columns are accumulated into the Gram in blocks whose size is chosen from a
  64 MiB budget, so peak memory is set by the $q^2\times q^2$ Gram itself
  (99 MiB at $q=60$, 1.5 GiB at $q=120$). Keep $q$ modest: it, not $N$, is what
  can make the projector unaffordable.
- *The rank curve needs a forward with rich leadfield structure.* On a
  single-shell sphere model the tangential leadfields are so low-rank that the
  power subspace collapses to a handful of directions and the curve degenerates.
  There is nothing to separate. A realistic BEM forward gives the smooth,
  separable curve the 45° criterion expects; on a degenerate curve $K^*$ falls
  back to a near-identity rank rather than one that empties the covariance.

## 11. Connectivity: pairwise and augmented-pairwise MCMV

Functional connectivity asks whether two regions' time courses are coupled,
as measured by coherence, phase locking, or amplitude-envelope correlation. Any inverse
operator that leaks one region into another manufactures coupling that is not
there, and the correlated-source cancellation of Section 3 can equally *hide*
real coupling. Connectivity is therefore exactly where a leakage-free filter
pays off (Nunes et al. 2020).

**Direct leakage, and PW-MCMV.** Reconstruct a single pair $(a,b)$ with a
2-source MCMV constraining $\{a,b\}$. The zero-gain condition
$\mathbf{w}_a^{\mathsf T}\mathbf{g}_b=0$ (Section 4) means $\hat s_a$ contains no
copy of source $b$, and $\hat s_b$ none of $a$: the pair carries **no direct
leakage**, so their connectivity is not biased by spatial spread or by the
mutual cancellation that collapses correlated LCMV estimates. Doing this for
every pair is *pairwise MCMV* (`pairwise_mcmv_connectivity`). Note that the
reconstruction of a given region differs from pair to pair; that is intrinsic to
the method, since each pair uses its own two-column constraint.

**Indirect leakage, and APW-MCMV.** PW-MCMV nulls the *partner* but not third
regions. If $a$ leaks into a neighbour $k$ that is genuinely coupled to $b$, that
leaked copy of $k$ correlates with $\hat s_b$ and a spurious $a$–$b$ edge
survives. The sharpest measure of the residual is the leakage coefficient
$\alpha_k=\mathbf{w}_a^{\mathsf T}\mathbf{g}_k$: for a pairwise filter it is
nonzero (the indirect path), and adding $k$ to the beamformer drives it to
*machine zero* by construction (the new zero-gain row). This is **augmented
pairwise MCMV**: for every statistically significant pair, add up to two
neighbouring regions of *each* source (those within a 4 cm radius that
themselves carry significant connections, ranked by their number of
connections), giving a beamformer of order 2 to 6, and re-estimate the pair
(`augmented_pairwise_mcmv_connectivity`). The order is capped for a concrete
reason: an $n$-source filter spends $n$ of its $M$ degrees of freedom on the
constraints (≈ "losing $n$ sensors"), and sources sharing a lobe are seen by far
fewer than $M$ sensors, so the effective $n/M_\text{actual}$ degrades SNR well
before $n=M$.

**Two things the paper insists on.** (i) Tune the weights to the analysis band.
For the resting-state envelope analyses (the setting APW-MCMV targets), the
sensor data are band-passed (8–12 Hz in the paper) before the covariance and the
filters are built, so pass band-limited `data`/`data_cov`. The Discussion shows
why: broadband weights stay near-optimal only where the power concentrates (the
low end), and for close source pairs they become suboptimal at higher
frequencies and report spurious connectivity there, which a narrow-band
covariance removes. (Task coherence/PLV analyses in the paper instead use
broadband weights with a multitaper spectral estimate.) (ii) Use *plain*
connectivity on the MCMV output. MCMV already removes leakage, so applying
leakage-orthogonalisation as well (the symmetric-orthogonalisation baseline)
would double-correct and discard the genuine zero-lag coupling MCMV is designed
to preserve; hence `orthogonalize=False` and a *signed* envelope correlation
(`absolute=False`, matching the paper's Pearson-of-envelopes) are the defaults.

**The envelope metric.** The paper's amplitude-envelope correlation is
specified in two steps: "the envelopes of the signals were computed by taking
the absolute values of the analytic Hilbert transform of the signals and then
low-pass filtering to 0.5 Hz", and the correlations are taken on the
*downsampled* envelopes. `mne_connectivity.envelope_correlation` takes the
Hilbert transform internally and correlates the envelopes directly, so the
0.5 Hz step cannot be applied from outside it; the envelope metric is therefore
computed in this module (`envelope_lowpass=0.5`, `envelope_resample=None`),
reducing exactly to `envelope_correlation` when `envelope_lowpass=None`.
Coherence and the phase measures are still delegated to
`mne_connectivity.spectral_connectivity_epochs`. The low-pass is not cosmetic:
it is what makes the surrogate null below correctly sized (see next paragraph).

**Significance.** Edges are tested against an AR(1) surrogate null
(`ar1_surrogate_significance`): fit a first-order autoregressive model to each
reconstructed course, generate independent Gaussian surrogates with the same
temporal smoothness, Fisher-transform their connectivity, and *standardise the
real edges by the null mean and standard deviation* (giving z-scores that are
zero-mean, unit-variance under the null, per Colclough et al. 2015) before
converting to p-values and thresholding with a False Discovery Rate of 0.05. A
first-order model captures the temporal smoothness of a reconstruction but not
the slow amplitude fluctuation of a strongly narrow-band signal, so the null is
anticonservative if the envelopes still carry their fast sub-band ripple: over
eight independent 9–11 Hz sources (a complete null) the test rejects 7.5 % of
edges at $\alpha=0.05$ without the 0.5 Hz envelope low-pass, and 0.4 % with it.
Use the same envelope settings for the null as for the matrix being tested. The
default is `method='envelope'`, which is what the paper prescribes for the
resting-state envelope correlations. The spectral metrics (`'coh'`, `'plv'`,
`'imcoh'`, `'wpli'`) are supported too, but they need an *epoched*
`reference_time_courses` of shape `(n_epochs, n_sources, n_times)`: a single
continuous surrogate segment carries no usable coherence or phase-locking
estimate. An edge whose null degenerates (zero or non-finite
spread) is reported as *not* significant, with a warning, never as an effect.

---

## 12. ABMC: a Bayesian beamformer for spike-like sources

LCMV localises by output *power*, which fails for the low-power, morphologically
distinctive transients of interest in epilepsy: interictal epileptiform
discharges (IEDs) and delayed responses (DRs) to single-pulse electrical
stimulation. ABMC (Shirani et al., 2024) addresses this in two stages.

**Stage 1: sparse Bayesian learning covariance** (`sbl_covariance`). Model the
data as $x(t) = G s(t) + \varepsilon(t)$ with $x\sim\mathcal N(0,R)$ and

$$ R = G\,\alpha\,G^\mathsf{T} + \Lambda, $$

where $\alpha=\mathrm{diag}(\alpha_1,\dots)$ are per-source prior variances (one
per leadfield column) and $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_M)$ is a
diagonal sensor-noise covariance. Fitting $(\alpha,\Lambda)$ by type-II maximum
likelihood, minimising $F=\mathrm{tr}(CR^{-1})+\log|R|$ over the data covariance
$C$, with the convex-bounding updates

$$ \alpha_n \leftarrow \alpha_n\sqrt{\tfrac{g_n^\mathsf{T}R^{-1}CR^{-1}g_n}{g_n^\mathsf{T}R^{-1}g_n}}, \qquad \lambda_m \leftarrow \lambda_m\sqrt{\tfrac{(R^{-1}CR^{-1})_{mm}}{(R^{-1})_{mm}}}, $$

yields a *model* covariance $R$ that, because the sources are modelled as mutually
uncorrelated ($\alpha$ diagonal), does not carry the cross-source correlation that
makes LCMV cancel correlated sources. The fit runs in a *noise-normalised* sensor
space: every channel is divided by its noise standard deviation, from `noise_cov`
when given and otherwise from MNE's ad-hoc per-type model, with the scaling undone
on the returned covariance. That is what makes the fit independent of the physical
units of the data (the paper's equations are written for single-sensor-type
intracranial EEG, where an $O(1)$ initialisation is harmless; against an SI-unit
MEG leadfield the same initialisation puts $G\alpha G^\mathsf{T}$ and $\Lambda$
eighteen orders of magnitude apart and $R$ is singular on the first iteration) and
what makes a *diagonal* $\Lambda$ meaningful across magnetometers (T),
gradiometers (T/m) and EEG (V). For a single sensor type it is a global scalar and
changes nothing. Note also that for a free-orientation forward the prior treats
the x, y and z columns of a grid point as three independent scalar sources, so it
is not covariant under a rotation of the source frame.

**Stage 2: template-constrained beamformer** (`make_abmc`). Per grid
point and orientation, solve

$$ \min_W \tfrac12 W^\mathsf{T} R W \quad\text{s.t.}\quad G^\mathsf{T}W=f \;\text{ and }\; \max_W (W^\mathsf{T}X\cdot u), $$

a distortionless minimum-variance beamformer with an added
maximum-cross-correlation-to-template constraint, where $u$ is the **caller-supplied
template of the target waveform**, passed via the `template` argument. In the
paper it is an expert-annotated IED or DR, but in general it can be any known
source morphology. The paper descends the Lagrangian,

$$ W(n{+}1) = W(n) - \mu\big(R W(n) - \beta_1 G - \beta_2 X u^\mathsf{T}\big), $$

with $\beta_1$ eliminated via the gain constraint (which the update provably
maintains at every step) and $\beta_2 = P\beta_1$; the template lag $j^\ast$ is
fixed once per segment, seeded from an initial LCMV output.

**This implementation solves that descent at its fixed point instead of stepping
towards it.** Setting the update to zero gives $RW=(G+P\,Xu^\mathsf{T})\beta_1$, and
the consistency of the $\beta_1$ expression then forces $G^\mathsf{T}W=f$, so

$$ \boxed{\;W^\ast = f\,\frac{R^{-1}\big(G + P\,Xu^\mathsf{T}\big)}{G^\mathsf{T}R^{-1}\big(G + P\,Xu^\mathsf{T}\big)}\;} $$

This is the *same* estimator the iteration converges to (the descent's own relative
step evaluated at $W^\ast$ is $\sim 2\times10^{-12}$), but it removes the tuning.
That matters in practice: on a real gradiometer covariance the descent needed
several thousand steps, and its stopping rule (on the size of the *step*) reported
convergence while the weights were still ~40% away from $W^\ast$, moving the
localised peak by about 9 cm. There is consequently no step size, iteration count or
tolerance to set; `reg` regularises $R$ for this solve exactly as it does in
`make_lcmv` and `make_mcmv`.

**Read-out.** Following the paper, the source is localised by the **maximum
cross-correlation between the beamformer output and the template**,
$|\mathrm{corr}(W^\mathsf{T}X,\,u_{j^\ast})|$, maximised over orientation: the grid
location whose output best matches the desired morphology at the best lag (LCMV, by
contrast, localises on output power). The same correlation also picks the
orientation at each grid point. The output variance $\tfrac12 W^\mathsf{T}RW$ is the
beamformer's minimisation objective, not the localizer; it is returned per grid point
as a diagnostic only and nothing in the scan reads it.

The ratio $P$ trades the two constraints: too small and the template term vanishes
(→ plain LCMV); too large and the weights blow up (the paper's
non-convergence regime). ABMC reports the blow-up fraction. For $P$ to mean
that, it has to be dimensionless: $g_n^\mathsf{T}g_n$ carries the units of the
squared forward model while $g_n^\mathsf{T}c_n$, with $c_n = Xu_{j^\ast}^\mathsf{T}$,
carries data × template units, so each $c_n$ is first rescaled to the norm of its
leadfield column. Without that rescaling $P\,g^\mathsf{T}c/g^\mathsf{T}g$ is
$\sim10^{-19}$ on SI-unit MEG at any sane $P$ and the paper's second constraint is
inert; with it, $P\sim0.01$–$0.1$ gives the template a perceptible but subordinate
weight regardless of the units. A warning fires if the realised coupling is so small
that the constraint does nothing.

**Multiple templates.** The paper matches several expert-annotated templates
per case; `make_abmc_dictionary` runs the ABMC scan for a whole dictionary of
desired waveforms in one call, estimating the SBL covariance once (it depends
only on the data) and reusing it for every template.

<!-- doc-split: tuning -->

# Parameter tuning

## `make_mcmv` / `scan_mcmv`

| Parameter | What it controls | Effect of changing it |
|---|---|---|
| `sources` (make_mcmv) | Which grid locations to constrain jointly | More sources → more nulls placed, better correlated-source separation, but the constraint uses more of the data rank; too many → ill-conditioning. |
| `orientations` | Source orientations (free-ori forward) | If omitted, use `scan_mcmv`/`optimal_orientation` to estimate them from data. Hand-set orientations that are off reduce SNR sharply. |
| `n_sources` (scan_mcmv) | Beamformer order reached by the search | Increase until `pseudo_z` drops to baseline; extra sources past the real ones add noise-level components. |
| `localizer` (scan_mcmv) | Which scanning statistic | `mai` = robust, broad; `mpz` = sharper but noisier; `mer`/`rmer` = phase-locked/evoked (need `evoked_cov`). |
| `noise_cov` | Whitening model | A measured `noise_cov` is essential for **mixed sensor types** and for a meaningful `unit-noise-gain`; `None` uses an ad-hoc per-type model (fine for a single sensor type). |
| `reg` | Diagonal loading of the (whitened) covariance, as a fraction of its mean eigenvalue | Larger `reg` → more stable inverse, smoother maps, lower resolution and slightly biased amplitudes; `reg=0` is the exact but fragile inverse. Default `0.05`, matching `make_lcmv`. |
| `weight_norm` | Output scaling | `unit-gain` preserves physical amplitude; `unit-noise-gain` equalises the noise floor (better maps, no amplitude). |
| `rank` | Numerical rank of the whitener **and** the covariance inverse | Use an integer after SSP/ICA/SSS (data are rank-deficient); `'full'` assumes full rank and will over-fit noise if the data are not. |

## `make_recipsiicos_cov` / `make_recipsiicos_lcmv`

| Parameter | What it controls | Effect of changing it |
|---|---|---|
| `rank` (the projection rank $K$) | Size of the retained power subspace (`recipsiicos`) or removed correlation subspace (`whitened`), in the $q^2$-dimensional virtual-sensor space | The single most important knob. Too small → the projector removes real power (over-smoothing, lost sources); too large → correlation leaks back and cancellation returns. Use `recipsiicos_rank_curve` (optionally `return_optimal=True` for the 45° $K^*$) and pick $K$ near that point. |
| `method` | Which projector | `recipsiicos` (project onto power) is simpler; `whitened` (project away from correlation in power-whitened space) spares source power better and is usually preferred for real data. |
| `pct_var` / `n_virtual` | Virtual-sensor count $q$ | Fraction of whitened-leadfield variance kept (default 0.99), or an explicit count. Fewer virtual sensors → smaller and faster $M^2$-space but coarser subspace separation; too few and the power subspace fills the space, leaving the projector nothing to remove. |
| `noise_cov` | Whitening model | Whitens per sensor type; **essential for mixed sensor types**. `None` uses an ad-hoc per-type model (a global scaling for a single type, which leaves the projector subspaces unchanged). |
| `whitener_rank` | Numerical rank of the whitener | Use an integer after SSP/ICA/SSS (data are rank-deficient); `'full'` assumes full rank. |
| `reg` | Tikhonov loading of the working-space LCMV inverse (and the whitening ridge for `whitened`) | Same trade-off as MCMV's `reg`: stability vs resolution. Default `0.05`. |
| `pick_ori`, `weight_norm`, `reduce_rank`, `inversion` | Orientation and normalisation of the working-space LCMV | Reuse MNE's own filter computation, so they behave exactly as in `make_lcmv`, including that **free-orientation MEG needs `reduce_rank=True`** (the radial-silent leadfield is rank-deficient). |

If a ReciPSIICOS run warns about negative-eigenvalue energy above the threshold,
lower `rank`: too much of the covariance was projected away and the
positive-definite repair had to flip a large amount of energy.

## `pairwise_mcmv_connectivity` / `augmented_pairwise_mcmv_connectivity`

The reconstruction parameters (`reg`, `weight_norm`, `noise_cov`, `rank`,
`orientations`) go straight to `make_mcmv` and behave exactly as in the MCMV
table above. The connectivity-specific knobs:

| Parameter | What it controls | Effect of changing it |
|---|---|---|
| `method` | Connectivity metric | `'envelope'` (signed amplitude-envelope correlation) for resting-state coupling; `'coh'`, `'plv'`, `'imcoh'`, … for task phase/spectral measures. The spectral metrics require `sfreq`, `fmin`, `fmax`. `'cohy'` is refused: coherency is complex and the returned matrix is real. Use `'coh'` (magnitude) or `'imcoh'` (imaginary part). |
| `radius` (APW) | Neighbour search radius for augmentation | Default 0.04 m (4 cm), from the paper's ~2 cm resolution rule. Larger admits more candidate conductors (higher order → better indirect-leakage suppression but lower SNR); smaller admits fewer. |
| `max_neighbours` (APW) | Neighbours added per source of the pair | Default 2, capping beamformer order at 2 + 2·`max_neighbours` = 6. Raising it suppresses more indirect leakage but erodes SNR (an $n$-source filter spends $n$ degrees of freedom; see §11), so keep the total order ≲ 8. |
| `orthogonalize` | Leakage-orthogonalisation of the envelopes | Default `False` (plain correlation). MCMV already removes leakage; enabling this is the competing symmetric-orthogonalisation baseline and discards genuine zero-lag coupling. |
| `absolute` | Sign of the envelope correlation | Default `False` (signed Pearson of envelopes, as in the paper); `True` returns the magnitude. Honoured for both `orthogonalize` settings, unlike `mne_connectivity.envelope_correlation`, which ignores it unless `orthogonalize='pairwise'`. |
| `envelope_lowpass` | Envelope low-pass before correlating (§11) | Default `0.5` Hz, per the paper. `None` correlates the unsmoothed envelopes (exactly `envelope_correlation`) and leaves the AR(1) null anticonservative. Needs `sfreq`, which defaults to `info['sfreq']`. |
| `envelope_resample` | Envelope downsampling before correlating | Default `None`. A target rate (Hz) reproduces the paper's "downsampled envelope correlations"; it changes the cost, not the expected value. |
| `fmin`, `fmax`, `mt_bandwidth` | Band and multitaper smoothing (spectral methods) | Set the band for `coh`/`plv`/…; the paper uses 2 Hz multitaper smoothing for its task analyses. For resting envelope work, band-pass the data and covariance to the analysis band instead (§11). |

For `ar1_surrogate_significance`, `n_surrogates` (default 200) trades null-estimate
precision against runtime, and `alpha` (default 0.05) is the FDR level; pass the
same `orthogonalize`, `absolute`, `envelope_lowpass` and `envelope_resample` used
for the matrix under test, plus its `sfreq`.

---

## `make_abmc` / `sbl_covariance`

| Parameter | What it controls | Effect of changing it |
|---|---|---|
| `cov` | The beamformer covariance $R$ | `None` (default) estimates the SBL covariance from the data, which is the intended ABMC pipeline. Pass a precomputed `sbl_covariance` result, or any `mne.Covariance`, to override. |
| `P` | Ratio $\beta_2/\beta_1$ weighting the template constraint | The one genuinely free parameter. The paper states it "is empirically adjusted" and reports no value, because the useful setting depends on the recording (its own data is 20–32 subdural contacts). Pass **`P="auto"`** to have `abmc_stability_curve` choose it on your data (see below), or set it yourself: the constraint column is rescaled to its leadfield column so `P` is dimensionless, and 0.01–0.1 is the working range. Below it the constraint is inert and ABMC reduces to a plain LCMV (a warning fires). Above it the localised peak starts to move well before anything blows up: measured on the example fixture, every peak is still on its $P\to 0$ location up to $P=0.18$, but by $P=1$ half of them have moved and the mean error has risen from 0.85 cm to 2.20 cm, while the first weights only blow up at $P=2.3$. So do not read `blowup_fraction` as an all-clear: it is zero again for large $P$, where the weights are finite but the answer is wrong. Use `result.critical_p`, the smallest $P$ at which some column's gain denominator vanishes, which is predicted before the solve and is never below 1. |
| `reg` | Diagonal loading of $R$ for the Stage-2 solve | **Default 0**, which is what the paper does. Its $R = G\alpha G^{\mathsf T} + \Lambda$ carries an *estimated* per-channel noise term and is positive definite by construction (condition number ~20 on the MNE `sample` data, against ~10¹⁵ for the empirical covariance of the same segment), so no loading is needed. Raise it only when you supply your own ill-conditioned `cov`. |
| `method` | How Stage 2 is solved | `"closed-form"` (default) solves at the fixed point of the paper's Eqs. 17–19 directly. `"iterative"` runs the paper's gradient descent verbatim, for exact reproduction; `mu`, `max_iter` and `tol` apply only to that path. The two agree to ~1e-8 when the descent is run to convergence, and a test pins them together. Note the descent's step count grows with the condition number of $R$: on an ill-conditioned covariance it can need 10⁵ steps, which is why it is not the default. |
| `mu` / `max_iter` / `tol` (`method="iterative"`) | Descent step size, budget and tolerance | `mu=None` uses $1/\lambda_{\max}(R)$. **`tol` is a distance to the fixed point, not a step size.** Those are not interchangeable: on an ill-conditioned $R$ the steps go small precisely because the descent is crawling along a shallow direction, so a step-size rule reports convergence while the weights are still far away. Because the fixed point is known in closed form, the honest test is available. |
| `max_lag` | Template-lag search window (samples) | `None` searches all lags; restrict it when the true delay range is known, to avoid spurious matches. |
| `max_iter` / `tol` (`sbl_covariance`) | SBL convergence | Iterate the Champagne updates until the Eq. 6 cost changes by less than `tol` (default 1e-5) or `max_iter` (default 100) is reached. |

<!-- doc-split: reference -->

# API

| Function | Purpose |
|---|---|
| `make_mcmv` / `apply_mcmv` / `apply_mcmv_cov` | Build and apply the joint MCMV filters for a known source set |
| `localizer_value` / `optimal_orientation` | The Table-1 localizers and the closed-form orientation |
| `scan_mcmv` → `MCMVScanResult` | Sequential source discovery |
| `make_recipsiicos_cov` | ReciPSIICOS-cleaned `mne.Covariance` |
| `make_recipsiicos_lcmv` | ReciPSIICOS + `make_lcmv` in one call |
| `recipsiicos_rank_curve` | Retained-energy-versus-$K$ curve (and optional 45° optimum $K^*$) for choosing the rank |
| `reconstruct_pairwise_mcmv` | Per-pair 2-source MCMV reconstructions (the PW-MCMV primitive) |
| `pairwise_mcmv_connectivity` | PW-MCMV connectivity matrix (direct-leakage-free) |
| `augmented_pairwise_mcmv_connectivity` | APW-MCMV: re-estimate significant pairs with neighbour augmentation |
| `ar1_surrogate_significance` | AR(1)-surrogate significance mask (Fisher-$z$ + FDR) |
| `sbl_covariance` | ABMC Stage 1: sparse Bayesian learning (Champagne) model covariance |
| `make_abmc` → `ABMCResult` | ABMC Stage 2: template-constrained beamformer for spike-like sources |
| `make_abmc_dictionary` | Run ABMC for a dictionary of desired templates, reusing one SBL covariance |
| `abmc_stability_curve` | Explore and refine the template-constraint trade-off $P$ on your own data |

# References

- Moiseev, A., Gaspar, J. M., Schneider, J. A., & Herdman, A. T. (2011).
  Application of multi-source minimum variance beamformers for reconstruction of
  correlated neural activity. *NeuroImage*, 58(2), 481–496.
  [doi:10.1016/j.neuroimage.2011.05.081](https://doi.org/10.1016/j.neuroimage.2011.05.081)
- Nunes, A. S., Moiseev, A., Kozhemiako, N., Cheung, T., Ribary, U., & Doesburg,
  S. M. (2020). Multiple constrained minimum variance beamformer (MCMV)
  performance in connectivity analyses. *NeuroImage*, 208, 116386.
  [doi:10.1016/j.neuroimage.2019.116386](https://doi.org/10.1016/j.neuroimage.2019.116386)
- Kuznetsova, A., Nurislamova, Y., & Ossadtchi, A. (2021). Modified covariance
  beamformer for solving MEG inverse problem in the environment with correlated
  sources. *NeuroImage*, 228, 117677.
  [doi:10.1016/j.neuroimage.2020.117677](https://doi.org/10.1016/j.neuroimage.2020.117677)
- Shirani, S., Abdi-Sargezeh, B., Valentin, A., Alarcon, G., Bird, J., & Sanei, S.
  (2024). Do interictal epileptiform discharges and brain responses to electrical
  stimulation come from the same location? An advanced source localization
  solution. *IEEE Trans. Biomed. Eng.*, 71(9), 2771–2780.
  [doi:10.1109/TBME.2024.3392603](https://doi.org/10.1109/TBME.2024.3392603)
- Van Veen, B. D., van Drongelen, W., Yuchtman, M., & Suzuki, A. (1997).
  Localization of brain electrical activity via linearly constrained minimum
  variance spatial filtering. *IEEE Trans. Biomed. Eng.*, 44(9), 867–880.
  [doi:10.1109/10.623056](https://doi.org/10.1109/10.623056)
- Frost, O. L. (1972). An algorithm for linearly constrained adaptive array
  processing. *Proceedings of the IEEE*, 60(8), 926–935.
  [doi:10.1109/PROC.1972.8817](https://doi.org/10.1109/PROC.1972.8817)
- Sekihara, K., & Nagarajan, S. S. (2008). *Adaptive Spatial Filters for
  Electromagnetic Brain Imaging*. Springer.
  [doi:10.1007/978-3-540-79370-0](https://doi.org/10.1007/978-3-540-79370-0)

# Maintainers and contributors

- **Sepehr Shirani**, maintainer and contributor (<sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>)
- **Muzhi Wang**, contributor

Contributions are welcome. Please open an issue or pull request.

# License

BSD-3-Clause. Copyright (c) 2026, Sepehr Shirani and Muzhi Wang.