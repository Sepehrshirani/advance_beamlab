# mne-beamlab

Advanced minimum-variance beamformers for MEG/EEG source reconstruction, built
to be fully compatible with [MNE-Python](https://mne.tools) and to its
contribution standards, so that each algorithm can be upstreamed into
`mne.beamformer`.

Every algorithm is implemented exactly from its peer-reviewed source, cites the
official paper, uses the algorithm's published name, and mirrors the
`mne.beamformer` API. This README is written to be **self-contained**: the
[Mathematical background](#mathematical-background) section derives every
equation from first principles, so you can understand and tune each method
without reading the papers.

## Implemented algorithms

- **Multiple Constrained Minimum Variance (MCMV)** — a multi-source beamformer
  that constrains several sources *jointly*, which removes the signal
  cancellation that biases single-source LCMV when sources are correlated. It
  comes with the four **scanning localizers** (MAI, MPZ, MER, rMER) and the
  **sequential source search** that turn it into a discovery tool. After
  Moiseev et al. (2011); connectivity/APW-MCMV after Nunes et al. (2020).
- **ReciPSIICOS** — makes an *ordinary* LCMV beamformer robust to correlated
  sources by cleaning the data covariance before the beamformer is built. After
  Kuznetsova, Nurislamova & Ossadtchi (2021).

## Installation

```bash
pip install -e ".[dev]"     # from a clone, for development (tests + docs + lint)
# or the runtime install only:
pip install -e .
```

Requires Python ≥ 3.10 and MNE-Python ≥ 1.6.

## Quick start

**Discover correlated sources with MCMV** (no need to know where they are):

```python
import mne
from mne_beamlab import scan_mcmv, apply_mcmv

# info, forward (free orientation), data_cov, noise_cov as usual in MNE
result = scan_mcmv(info, forward, data_cov, noise_cov=noise_cov,
                   localizer="mpz", n_sources=2)

print(result["sources"])       # discovered grid indices
print(result["pseudo_z"])      # per-source pseudo-Z (judge how many are real)
stc_time_courses = apply_mcmv(epochs, result["filters"])  # jointly-optimal filters
```

**Make LCMV robust to correlation with ReciPSIICOS**:

```python
from mne_beamlab import make_recipsiicos_lcmv
from mne.beamformer import apply_lcmv

filters = make_recipsiicos_lcmv(info, forward, data_cov, rank=70, method="whitened")
stc = apply_lcmv(evoked, filters)
```

---

# Mathematical background

Notation: $M$ sensors, $N$ source locations. Bold lowercase are vectors, bold
uppercase are matrices, $^{\mathsf T}$ is transpose, $\langle\cdot\rangle$ is the
time average. $\mathbf{I}_n$ is the $n\times n$ identity.

## 1. The measurement model

At each instant the sensors measure a linear mixture of the active sources plus
noise:

$$\mathbf{x}(t)=\sum_{i} \mathbf{g}_i\, s_i(t) + \mathbf{n}(t)=\mathbf{G}\,\mathbf{s}(t)+\mathbf{n}(t).$$

- $\mathbf{x}(t)\in\mathbb{R}^{M}$ is the sensor reading at time $t$.
- $s_i(t)$ is the (scalar) time course of source $i$.
- $\mathbf{g}_i\in\mathbb{R}^{M}$ is the **forward field** (a.k.a. leadfield,
  topography) of source $i$: the pattern that source produces across the sensors
  when it has unit amplitude. It is computed once from the head model and sensor
  geometry — it is *known*.
- $\mathbf{n}(t)$ is additive noise with covariance
  $\mathbf{C}_n=\langle\mathbf{n}\mathbf{n}^{\mathsf T}\rangle$.

For a source at location $\mathbf{r}$ with **free orientation**, the forward
field is $\mathbf{g}=\mathbf{L}(\mathbf{r})\,\mathbf{u}$, where
$\mathbf{L}(\mathbf{r})\in\mathbb{R}^{M\times 3}$ holds the fields of unit
dipoles along $x,y,z$ and $\mathbf{u}$ is the (unit) orientation. Fixing the
orientation collapses this to a single column.

Everything downstream is driven by the **data covariance**

$$\mathbf{R}=\langle\mathbf{x}\,\mathbf{x}^{\mathsf T}\rangle .$$

If the sources have covariance $\mathbf{C}_s=\langle\mathbf{s}\mathbf{s}^{\mathsf T}\rangle$
and are uncorrelated with the noise, then
$\mathbf{R}=\mathbf{G}\,\mathbf{C}_s\,\mathbf{G}^{\mathsf T}+\mathbf{C}_n$. Keep
this identity in mind — it is the reason the localizers below peak at the true
sources, and the reason the ReciPSIICOS decomposition works.

## 2. The beamformer, and LCMV

A **spatial filter** is a vector $\mathbf{w}$ that estimates one source's time
course as a weighted sum of sensors, $\hat s(t)=\mathbf{w}^{\mathsf T}\mathbf{x}(t)$.
We want two things:

1. **Unit gain on the target**: $\mathbf{w}^{\mathsf T}\mathbf{g}=1$, so the
   target passes through untouched.
2. **Minimum output power**: minimise
   $\langle\hat s^2\rangle=\mathbf{w}^{\mathsf T}\mathbf{R}\,\mathbf{w}$. Since
   the target is pinned by the constraint, minimising *total* output power
   forces the filter to suppress everything else — other sources and noise.

This is the Linearly Constrained Minimum Variance (LCMV) problem. Minimising
$\mathbf{w}^{\mathsf T}\mathbf{R}\mathbf{w}$ subject to
$\mathbf{w}^{\mathsf T}\mathbf{g}=1$ with a Lagrange multiplier $\lambda$,

$$\mathcal{L}=\mathbf{w}^{\mathsf T}\mathbf{R}\mathbf{w}-\lambda(\mathbf{w}^{\mathsf T}\mathbf{g}-1),\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{w}}=2\mathbf{R}\mathbf{w}-\lambda\mathbf{g}=\mathbf{0}\;\Rightarrow\;\mathbf{w}=\tfrac{\lambda}{2}\mathbf{R}^{-1}\mathbf{g}.$$

Enforcing the constraint $\mathbf{w}^{\mathsf T}\mathbf{g}=1$ fixes $\lambda$ and gives the LCMV filter

$$\boxed{\;\mathbf{w}=\dfrac{\mathbf{R}^{-1}\mathbf{g}}{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}}\;}\qquad\text{with reconstructed power}\quad \langle\hat s^2\rangle=\dfrac{1}{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}} .$$

Scanning $1/(\mathbf{g}(\mathbf{r})^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}(\mathbf{r}))$
over $\mathbf{r}$ is the classic beamformer power map.

## 3. Why correlated sources break LCMV

Suppose two sources with fields $\mathbf{g}_1,\mathbf{g}_2$ have correlation
$\rho$. The LCMV filter for source 1 is free to place a null anywhere except
along $\mathbf{g}_1$. Because source 2 is correlated with source 1, the filter
can *lower its own output power* by passing a scaled, sign-flipped copy of
source 2 that partially cancels source 1 in the average — the constraint on
$\mathbf{g}_1$ is still satisfied instant by instant, but the variance is
reduced by exploiting the correlation. The result is **signal cancellation**:
the reconstructed power of source 1 is suppressed by roughly a factor
$(1-\rho^2)$, collapsing to zero as $\rho\to 1$. Amplitudes are underestimated,
locations are pulled toward each other, and spurious "connectivity" appears
because each filter leaks a copy of the other source. This single failure mode
is what MCMV and ReciPSIICOS each remove, in two different ways.

## 4. MCMV: the joint constraint

Instead of one filter that only protects its own source, MCMV solves for $n$
filters **at once** and forbids each from responding to the *others'* fields.
Collect the $n$ forward fields as the columns of
$\mathbf{H}=[\mathbf{g}_1,\dots,\mathbf{g}_n]\in\mathbb{R}^{M\times n}$ and the
$n$ filters as the columns of $\mathbf{W}=[\mathbf{w}_1,\dots,\mathbf{w}_n]$. The
constraint is

$$\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}_n,\qquad\text{i.e.}\qquad \mathbf{w}_i^{\mathsf T}\mathbf{g}_j=\delta_{ij}.$$

The diagonal ($i=j$) is the familiar **unit-gain** condition; the off-diagonal
($i\ne j$) **zero-gain** conditions are the new ingredient — filter $i$ is forced
to be *blind* to every other constrained source. A blind filter cannot exploit a
correlation it cannot see, so the cancellation of Section 3 disappears.

Minimise the total output power $\operatorname{Tr}(\mathbf{W}^{\mathsf T}\mathbf{R}\mathbf{W})$
subject to $\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}_n$. With a matrix of
Lagrange multipliers $\mathbf{\Lambda}$,

$$\mathcal{L}=\operatorname{Tr}(\mathbf{W}^{\mathsf T}\mathbf{R}\mathbf{W})-\operatorname{Tr}\!\big(\mathbf{\Lambda}(\mathbf{W}^{\mathsf T}\mathbf{H}-\mathbf{I}_n)\big),\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{W}}=2\mathbf{R}\mathbf{W}-\mathbf{H}\mathbf{\Lambda}^{\mathsf T}=\mathbf{0}\;\Rightarrow\;\mathbf{W}=\tfrac12\mathbf{R}^{-1}\mathbf{H}\,\mathbf{\Lambda}^{\mathsf T}.$$

Substituting into the constraint $\mathbf{W}^{\mathsf T}\mathbf{H}=\mathbf{I}$
fixes $\mathbf{\Lambda}$ and yields the **MCMV weights** (Moiseev et al. 2011,
Eq. 5):

$$\boxed{\;\mathbf{W}=\mathbf{R}^{-1}\mathbf{H}\,\big(\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}\big)^{-1}\;}$$

Reading it line by line: $\mathbf{R}^{-1}\mathbf{H}$ is the stack of ordinary
inverse-covariance filters; multiplying by
$(\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H})^{-1}$ mixes them just enough to
enforce all $n^2$ gain conditions simultaneously. For $n=1$ the bracket is a
scalar and this is *exactly* the LCMV filter of Section 2 — MCMV is a strict
generalisation. The matrix $\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H}$ is
invertible as long as the constrained fields are linearly independent; it
becomes singular if two sources coincide in both location and orientation (then
you have asked for two filters to do contradictory things), which the
implementation detects and reports.

In this package the joint solve is `make_mcmv(...)`; `apply_mcmv(...)` runs the
filters on `Raw`/`Epochs`/`Evoked`/arrays.

## 5. Whitening, and why it is mandatory for mixed sensors

The data covariance mixes sensor types with *different physical units* —
magnetometers in tesla ($\sim\!10^{-13}$), gradiometers in T/m, EEG in volts.
The scalar $\mathbf{w}^{\mathsf T}\mathbf{R}\mathbf{w}$ then sums
$\mathrm{T}^2$, $(\mathrm{T/m})^2$ and cross terms — literally adding
incommensurable quantities — and $\mathbf{R}^{-1}$ is numerically dominated by
whichever block is largest, making the filter effectively blind to the
smaller-scale sensor type. This is a *correctness* problem, not just
conditioning.

The fix is a change of coordinates. Build a **whitener** $\mathbf{W}_{\!n}$ from
the noise covariance so that $\mathbf{W}_{\!n}\mathbf{C}_n\mathbf{W}_{\!n}^{\mathsf T}=\mathbf{I}$,
from the eigendecomposition
$\mathbf{C}_n=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^{\mathsf T}$,
$\mathbf{W}_{\!n}=\mathbf{\Lambda}_r^{-1/2}\mathbf{U}_r^{\mathsf T}$ (keeping only
the $r$ non-negligible eigenpairs — the numerical rank, which drops below $M$
after SSS/Maxwell filtering or ICA). Then whiten leadfield and covariance,

$$\tilde{\mathbf{H}}=\mathbf{W}_{\!n}\mathbf{H},\qquad \tilde{\mathbf{R}}=\mathbf{W}_{\!n}\mathbf{R}\,\mathbf{W}_{\!n}^{\mathsf T},$$

solve MCMV there, and fold the whitener back into the returned weights so they
still act on raw sensor data ($\hat{\mathbf{s}}=\tilde{\mathbf{W}}^{\mathsf T}\mathbf{W}_{\!n}\mathbf{x}$).
In whitened coordinates every channel is dimensionless with unit noise variance,
so all sensor types are commensurable.

Two consequences worth internalising:

- **Single sensor type** (e.g. a CTF gradiometer array): the whitener is
  effectively a global scale that cancels out of the unit-gain filter, so
  single-type results are unchanged. Whitening is harmless there and essential
  everywhere else.
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
  $\mathbf{w}_i^{\mathsf T}\mathbf{C}_n\mathbf{w}_i=1$, i.e. unit output noise.
  Because the solve is in whitened space this is exactly unit Euclidean norm of
  the whitened filter — MNE's definition. It equalises the noise floor across
  locations, which is what you want for *maps* (so deep, low-SNR locations are
  not penalised) at the cost of no longer preserving physical amplitude.

## 7. Finding the sources: the localizers

MCMV needs to know *which* sources to constrain. The localizers are scalar maps
that peak at the true sources. They are built from four $n\times n$ matrices
(Moiseev et al. 2011, Table 2), each a leadfield sandwiched around a covariance:

$$\mathbf{S}=\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{H},\quad
\mathbf{G}=\mathbf{H}^{\mathsf T}\mathbf{C}_n^{-1}\mathbf{H},\quad
\mathbf{T}=\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\mathbf{C}_n\mathbf{R}^{-1}\mathbf{H},\quad
\mathbf{E}=\mathbf{H}^{\mathsf T}\mathbf{R}^{-1}\bar{\mathbf{R}}\,\mathbf{R}^{-1}\mathbf{H},$$

where $\bar{\mathbf{R}}=\langle\bar{\mathbf{b}}\bar{\mathbf{b}}^{\mathsf T}\rangle$
is the covariance of the epoch-**averaged** (evoked) field. The four localizers
(Table 1) and what each measures:

| Localizer | Formula | Reduces at $n=1$ to | Use it for |
|---|---|---|---|
| **MAI** (activity index) | $\operatorname{Tr}(\mathbf{G}\mathbf{S}^{-1})-n$ | $\zeta-1=\dfrac{\mathbf{g}^{\mathsf T}\mathbf{C}_n^{-1}\mathbf{g}}{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}}-1$ | robust, broad peaks |
| **MPZ** (pseudo-Z) | $\operatorname{Tr}(\mathbf{S}\mathbf{T}^{-1})-n$ | $\bar Z-1=\dfrac{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g}}{\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{C}_n\mathbf{R}^{-1}\mathbf{g}}-1$ | sharper localisation |
| **MER** (evoked) | $\operatorname{Tr}(\mathbf{E}\mathbf{T}^{-1})$ | evoked pseudo-Z | phase-locked responses |
| **rMER** (reduced evoked) | $\operatorname{Tr}(\mathbf{E}\mathbf{S}^{-1})$ | reduced evoked pseudo-Z | evoked, no clean $\mathbf{C}_n$ |

Intuition: $\zeta$ is a *(signal+noise)/noise* ratio — $(\mathbf{g}^{\mathsf T}\mathbf{R}^{-1}\mathbf{g})^{-1}$
is the total reconstructed power (signal + leaked noise) and
$(\mathbf{g}^{\mathsf T}\mathbf{C}_n^{-1}\mathbf{g})^{-1}$ is the noise-only power,
so subtracting $1$ gives a *pure* signal-to-noise ratio that is zero where there
is no source. The multi-source versions generalise this to the whole
constrained set. Two facts proved in the paper and verified in the tests: both
power localizers are non-negative and ordered, $P_{\text{MAI}}\ge P_{\text{MPZ}}$
(MPZ is sharper but noisier), and each localizer's global maximum is the true
source configuration (they are *unbiased* for any $\mathbf{C}_n$). The power
localizers (MAI, MPZ) work for induced/oscillatory activity; the event-related
ones (MER, rMER) target phase-locked responses and need $\bar{\mathbf{R}}$.

Everything is computed in the whitened space where $\mathbf{C}_n=\mathbf{I}$; the
localizers are **invariant** under whitening, so the values are identical to the
raw-space formulas above.

## 8. Data-driven orientation (no orientation search)

For a free-orientation forward each candidate location contributes an $M\times 3$
block $\mathbf{H}_k=[\mathbf{h}^x_k,\mathbf{h}^y_k,\mathbf{h}^z_k]$, and we must
pick the orientation $\mathbf{u}_k$ (so that $\mathbf{g}_k=\mathbf{H}_k\mathbf{u}_k$)
that maximises the localizer, *given* the sources already found (their fields
form the reference block $\mathbf{H}_R$). Moiseev et al. show the maximiser is
the top eigenvector of a $3\times 3$ generalized eigenproblem
$\mathbf{D}\,\mathbf{u}_k=\lambda\,\mathbf{F}\,\mathbf{u}_k$ — no scan over angles
is needed:

$$\mathbf{F}=\mathbf{A}_{kk}-\mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk},\qquad
\mathbf{D}=\mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{B}_{RR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk}
-\mathbf{A}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{B}_{Rk}
-\mathbf{B}_{kR}\mathbf{A}_{RR}^{-1}\mathbf{A}_{Rk}+\mathbf{B}_{kk}.$$

Here $(\mathbf{A},\mathbf{B})$ is the localizer's (denominator, numerator) pair —
$(\mathbf{S},\mathbf{G})$ for MAI, $(\mathbf{T},\mathbf{S})$ for MPZ,
$(\mathbf{T},\mathbf{E})$ for MER, $(\mathbf{S},\mathbf{E})$ for rMER — and the
subscripted blocks are the Table-2 matrices evaluated between $\mathbf{H}_R$ and
$\mathbf{H}_k$. $\mathbf{F}$ is the Schur complement of the denominator (it
"conditions out" the already-found sources); $\mathbf{D}$ is the corresponding
numerator form. With no references (the first source), it collapses to
$\mathbf{B}_{kk}\mathbf{u}=\lambda\mathbf{A}_{kk}\mathbf{u}$. Exposed as
`optimal_orientation(...)`.

## 9. The sequential source search

The full parameter space of $n$ sources is far too large to scan jointly, so the
search is greedy and iterative (Moiseev et al. 2011):

1. **Find source 1** by scanning the single-source localizer over the grid
   (with the optimal orientation at each location).
2. **Fix** it as a reference and scan the **multi-source** localizer for source
   2 — now every candidate is evaluated *jointly* with source 1.
3. Repeat, adding one source per iteration. Each new source is added to the
   joint constraint, so the cancellation that hid correlated activity is
   progressively removed and previously masked sources emerge.

**Knowing when to stop.** After adding source $k$, monitor its single-source
pseudo-Z $\bar z_k=(\mathbf{w}_k^{\mathsf T}\mathbf{R}\mathbf{w}_k)/(\mathbf{w}_k^{\mathsf T}\mathbf{C}_n\mathbf{w}_k)$.
Genuine sources have large $\bar z_k$; once you start adding noise, $\bar z_k$
drops to a baseline and fluctuates. The baseline is generally **not** $1$ and
must be judged from the data — so `scan_mcmv` runs a requested `n_sources` and
returns the whole `pseudo_z` sequence for you to read the elbow. Exposed as
`scan_mcmv(...)`, which returns the discovered sources, their orientations, the
pseudo-Z sequence, the per-iteration localizer maps, and a ready-to-apply
`MCMVBeamformer`.

## 10. ReciPSIICOS: cleaning the covariance instead of constraining sources

ReciPSIICOS attacks the same cancellation from the other end: rather than a new
filter, it *repairs the data covariance* so that an ordinary `make_lcmv` no
longer cancels correlated sources.

**The vec view.** Stack the columns of the $M\times M$ covariance into a vector
$\operatorname{vec}(\mathbf{R})\in\mathbb{R}^{M^2}$. Using
$\mathbf{R}=\mathbf{G}\mathbf{C}_s\mathbf{G}^{\mathsf T}+\mathbf{C}_n$ and
$\operatorname{vec}(\mathbf{g}_i\mathbf{g}_j^{\mathsf T})=\mathbf{g}_j\otimes\mathbf{g}_i$,

$$\operatorname{vec}(\mathbf{R})=\underbrace{\sum_i [\mathbf{C}_s]_{ii}\,\operatorname{vec}(\mathbf{g}_i\mathbf{g}_i^{\mathsf T})}_{\text{auto-products: source \textit{power}}}
+\underbrace{\sum_{i\ne j}[\mathbf{C}_s]_{ij}\,\operatorname{vec}(\mathbf{g}_i\mathbf{g}_j^{\mathsf T})}_{\text{cross-products: source \textit{coupling}}}
+\operatorname{vec}(\mathbf{C}_n).$$

The **auto-product** directions $\operatorname{vec}(\mathbf{g}_i\mathbf{g}_i^{\mathsf T})$
carry the source powers; the **cross-product** directions
$\operatorname{vec}(\mathbf{g}_i\mathbf{g}_j^{\mathsf T})$ carry the couplings —
and the couplings are *exactly* what an LCMV beamformer exploits to cancel
correlated sources. Kill the cross-product part of the covariance and the
cancellation has nothing to feed on.

**Building the projector — from the forward model alone.** Enumerate the
auto-product vectors over the source grid as the columns of $\mathbf{G}_{\text{pwr}}$
and (for the whitened variant) the cross-product vectors as $\mathbf{G}_{\text{cor}}$.

- **`recipsiicos`** (Eq. 10): take the SVD of $\mathbf{G}_{\text{pwr}}$, keep the
  top $K$ left singular vectors $\mathbf{U}_K$, and project *onto* the power
  subspace, $\mathbf{P}=\mathbf{U}_K\mathbf{U}_K^{\mathsf T}$. This retains
  power, and whatever of the correlation subspace is (near-)orthogonal to it is
  removed.
- **`whitened`** (Eqs. 15–17): first *whiten by the power subspace* — form
  $\mathbf{C}_{\text{pwr}}=\mathbf{G}_{\text{pwr}}\mathbf{G}_{\text{pwr}}^{\mathsf T}$,
  its range-restricted inverse square root
  $\mathbf{W}_{\text{pwr}}=\mathbf{E}\mathbf{\Lambda}^{-1/2}\mathbf{E}^{\mathsf T}$
  (drop the null space — the auto-products live in the symmetric subspace of
  dimension $M(M{+}1)/2$, so $\mathbf{C}_{\text{pwr}}$ is *never* full rank and
  must be range-restricted, not ridge-filled) — then, in that whitened space,
  project *away from* the top $K$ correlation directions and unwhiten. Because
  the power directions have been flattened to unit scale first, this spares them
  far better than the plain variant.

**Applying it** (Eq. 11): reshape the projected vector back to a matrix and
symmetrise,

$$\tilde{\mathbf{R}}=\tfrac12\big(\mathbf{M}+\mathbf{M}^{\mathsf T}\big),\qquad \mathbf{M}=\operatorname{vec}^{-1}\!\big(\mathbf{P}\,\operatorname{vec}(\mathbf{R})\big).$$

Projecting out a subspace can make $\tilde{\mathbf{R}}$ indefinite, but a
covariance fed to `make_lcmv` must be positive definite, so we **spectral-flip**
(Eq. 12): eigendecompose and replace each eigenvalue by its absolute value. If a
large fraction of the covariance energy sat in negative eigenvalues (default
warn threshold 20%, Eq. 24), the rank $K$ was too aggressive and the method
warns.

**Free orientation** (Eqs. 22–23): at each location the local $M\times 3$
leadfield is reduced to its two dominant *tangential* topographies by a local
SVD; the power set then expands to three columns per location and the
correlation set to four columns per location pair.

**The one knob is $K$.** The projector depends only on the forward model, so it
is built once and reused across datasets sharing that forward. In this package:
`make_recipsiicos_cov(...)` returns the cleaned `mne.Covariance`;
`make_recipsiicos_lcmv(...)` wraps it straight into a beamformer;
`recipsiicos_rank_curve(...)` gives the retained power/correlation energy versus
$K$ to choose it.

---

# Parameter tuning

## `make_mcmv` / `scan_mcmv`

| Parameter | What it controls | Effect of changing it |
|---|---|---|
| `sources` (make_mcmv) | Which grid locations to constrain jointly | More sources → more nulls placed, better correlated-source separation, but the constraint uses more of the data rank; too many → ill-conditioning. |
| `orientations` | Source orientations (free-ori forward) | If omitted, use `scan_mcmv`/`optimal_orientation` to estimate them from data — hand-set orientations that are off reduce SNR sharply. |
| `n_sources` (scan_mcmv) | Beamformer order reached by the search | Increase until `pseudo_z` drops to baseline; extra sources past the real ones add noise-level components. |
| `localizer` (scan_mcmv) | Which scanning statistic | `mai` = robust, broad; `mpz` = sharper but noisier; `mer`/`rmer` = phase-locked/evoked (need `evoked_cov`). |
| `noise_cov` | Whitening model | A measured `noise_cov` is essential for **mixed sensor types** and for a meaningful `unit-noise-gain`; `None` uses an ad-hoc per-type model (fine for a single sensor type). |
| `reg` | Diagonal loading of the (whitened) covariance, as a fraction of its mean eigenvalue | Larger `reg` → more stable inverse, smoother maps, lower resolution and slightly biased amplitudes; `reg=0` is the exact but fragile inverse. Default `0.05`, matching `make_lcmv`. |
| `weight_norm` | Output scaling | `unit-gain` preserves physical amplitude; `unit-noise-gain` equalises the noise floor (better maps, no amplitude). |
| `rank` | Numerical rank of the whitener **and** the covariance inverse | Use an integer after SSP/ICA/SSS (data are rank-deficient); `'full'` assumes full rank and will over-fit noise if the data are not. |

## `make_recipsiicos_cov` / `make_recipsiicos_lcmv`

| Parameter | What it controls | Effect of changing it |
|---|---|---|
| `rank` (the projection rank $K$) | Size of the retained power subspace (`recipsiicos`) or removed correlation subspace (`whitened`) | The single most important knob. Too small → the projector removes real power (over-smoothing, lost sources); too large → correlation leaks back and cancellation returns. Use `recipsiicos_rank_curve` and pick $K$ near the elbow. |
| `method` | Which projector | `recipsiicos` (project onto power) is simpler; `whitened` (project away from correlation in power-whitened space) spares source power better and is usually preferred for real data. |
| `reg` | Regularisation of the LCMV inverse (and the whitening ridge for `whitened`) | Same trade-off as MCMV's `reg`: stability vs resolution. |
| `noise_cov` (lcmv wrapper) | Whitening for the downstream LCMV | Needed for mixed sensor types; see the note above. |
| `pick_ori`, `weight_norm`, … | Passed straight through to `make_lcmv` | Standard MNE beamformer options. |

If a ReciPSIICOS run warns about negative-eigenvalue energy above the threshold,
lower `rank`: too much of the covariance was projected away and the
positive-definite repair had to flip a large amount of energy.

---

# API

| Function | Purpose |
|---|---|
| `make_mcmv` / `apply_mcmv` / `apply_mcmv_cov` | Build and apply the joint MCMV filters for a known source set |
| `localizer_value` / `optimal_orientation` | The Table-1 localizers and the closed-form orientation |
| `scan_mcmv` → `MCMVScanResult` | Sequential source discovery |
| `make_recipsiicos_cov` | ReciPSIICOS-cleaned `mne.Covariance` |
| `make_recipsiicos_lcmv` | ReciPSIICOS + `make_lcmv` in one call |
| `recipsiicos_rank_curve` | Retained-energy-versus-$K$ curve for choosing the rank |

# References

- Moiseev, A., Gaspar, J. M., Schneider, J. A., & Herdman, A. T. (2011).
  Application of multi-source minimum variance beamformers for reconstruction of
  correlated neural activity. *NeuroImage*, 58(2), 481–496.
  [doi:10.1016/j.neuroimage.2011.05.081](https://doi.org/10.1016/j.neuroimage.2011.05.081)
- Nunes, A. S., Moiseev, A., Kozhemiako, N., Cheung, T., Ribary, U., & Doesburg,
  S. M. (2020). Multiple constrained minimum variance beamformer (MCMV) and its
  application to MEG. *NeuroImage*, 208, 116386.
- Kuznetsova, A., Nurislamova, Y., & Ossadtchi, A. (2021). Modified covariance
  beamformer for solving MEG inverse problem in the environment with correlated
  sources. *NeuroImage*, 228, 117677.
  [doi:10.1016/j.neuroimage.2020.117677](https://doi.org/10.1016/j.neuroimage.2020.117677)
- Van Veen, B. D., van Drongelen, W., Yuchtman, M., & Suzuki, A. (1997).
  Localization of brain electrical activity via linearly constrained minimum
  variance spatial filtering. *IEEE Trans. Biomed. Eng.*, 44(9), 867–880.
- Sekihara, K., & Nagarajan, S. S. (2008). *Adaptive Spatial Filters for
  Electromagnetic Brain Imaging*. Springer.

# License

BSD-3-Clause. Copyright (c) 2026, Sepehr Shirani.