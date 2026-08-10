"""
.. _ex-fem-head-model:

=========================================================
Beamforming EEG on a finite-element head model
=========================================================

Every forward solution MNE-Python can compute uses the boundary element method:
nested, closed, homogeneous surfaces for scalp, outer skull and inner skull.
That model has no way to represent the cerebrospinal fluid, the split of the
skull into compact and spongy bone, or the openings at the orbits and the
auditory meatus. For EEG those omissions bias the forward field, because
current has to flow *through* all of it to reach the electrodes. The finite
element method (FEM) discretises the whole head volume instead and gives every
tetrahedron its own conductivity. MNE-Python has no FEM solver.

This example uses a precomputed one. The New York Head :footcite:`HuangEtAl2016`
is a six-tissue FEM model of the ICBM152 template, solved once at high
resolution and distributed as a lead field for 231 electrodes of the 10-05
system at 74382 cortical nodes.
:func:`~advance_beamlab.read_ny_head_forward` wraps it as an ordinary
:class:`mne.Forward`, which is the only thing any beamformer needs.
:func:`mne.beamformer.make_lcmv` and the multi-source methods in this package
therefore run on it unchanged.

Three things are shown: what the FEM forward differs from a BEM forward *by*,
that the beamformers localise correctly on it, and where the approach stops
(it is EEG-only, for a reason that is geometric rather than technical).

.. note::
   The model is a 678 MB download, licensed by its authors under the GPL v3.
   It is fetched on demand and is not redistributed with this BSD-licensed
   package. Publications using it must cite :footcite:`HuangEtAl2016`.
"""
# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
import pyvista as pv
from mne.beamformer import apply_lcmv_cov, make_lcmv

from advance_beamlab import (
    apply_mcmv,
    make_mcmv,
    make_ny_head_info,
    ny_head_montage,
    ny_head_scalp,
    read_ny_head_forward,
    scan_mcmv,
)

# %%
# The FEM forward
# ---------------
# ``resolution`` selects one of the model's nested cortical meshes; they are
# strict subsets of one another, so a coarse mesh is a genuine subsampling of
# the same geometry. 5K is ample for EEG, whose spatial resolution is far
# coarser than the mesh.

fwd = read_ny_head_forward(resolution="5K", orientation="normal")
info = make_ny_head_info(sfreq=250.0)
print(fwd)

# %%
# Note the rank. The lead field is supplied in common average reference, so it
# is rank deficient by exactly one and any data generated from it will be too.
# That is not a defect to be regularised away: it is what average referencing
# *means*. Every beamformer here resolves it through :func:`mne.compute_rank`
# rather than inverting a singular matrix.

gain = fwd["sol"]["data"]
print(
    f"electrodes: {fwd['nchan']}, rank of the lead field: {np.linalg.matrix_rank(gain)}"
)

# %%
# What the FEM changes
# --------------------
# The comparison that matters is against the model MNE would otherwise have
# used. We build a three-layer BEM forward on ``fsaverage`` for the *same* 231
# electrodes and cortical-normal dipoles, and compare gain magnitudes as a
# function of how deep the source sits below the nearest electrode.

subjects_dir = mne.datasets.fetch_fsaverage().parent
montage = ny_head_montage()
info_bem = mne.create_info(list(montage.ch_names), 250.0, "eeg")
info_bem.set_montage(montage)

bem = mne.make_bem_solution(
    mne.make_bem_model(
        "fsaverage", ico=3, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir
    )
)
src_bem = mne.setup_source_space(
    "fsaverage", spacing="ico4", subjects_dir=subjects_dir, add_dist=False
)
fwd_bem = mne.make_forward_solution(
    info_bem, trans="fsaverage", src=src_bem, bem=bem, eeg=True, meg=False
)
fwd_bem = mne.convert_forward_solution(fwd_bem, force_fixed=True, use_cps=True)
# Match the FEM convention, which is average referenced.
gain_bem = fwd_bem["sol"]["data"]
gain_bem = gain_bem - gain_bem.mean(0, keepdims=True)

# %%
# Both models are in MNE's units of V/(A m), so the curves are directly
# comparable. The FEM gains are consistently *lower*: the cerebrospinal fluid is
# the most conductive tissue in the head, and modelling it shunts current that a
# three-layer BEM instead pushes out to the scalp. This is the systematic bias a
# BEM introduces for EEG, and it is roughly a factor of 1.7 here.

elec = np.array([c["loc"][:3] for c in info["chs"]])
depth_fem = np.linalg.norm(elec[:, None] - fwd["source_rr"][None], axis=2).min(0)
depth_bem = np.linalg.norm(elec[:, None] - fwd_bem["source_rr"][None], axis=2).min(0)

edges = np.arange(0.012, 0.056, 0.004)
centres, med_fem, med_bem = [], [], []
for lo, hi in zip(edges[:-1], edges[1:], strict=True):
    mf, mb = (depth_fem >= lo) & (depth_fem < hi), (depth_bem >= lo) & (depth_bem < hi)
    if mf.sum() > 20 and mb.sum() > 20:
        centres.append((lo + hi) / 2 * 1000)
        med_fem.append(np.median(np.abs(gain[:, mf])))
        med_bem.append(np.median(np.abs(gain_bem[:, mb])))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
ax.plot(centres, med_bem, "o-", label="BEM (fsaverage, 3 layers)")
ax.plot(centres, med_fem, "s-", label="FEM (New York Head, 6 tissues)")
ax.set(
    xlabel="depth below nearest electrode (mm)",
    ylabel="median |gain|  (V / A m)",
    title="Forward gain: FEM vs BEM",
)
ax.legend()
ax2.plot(centres, np.array(med_bem) / np.array(med_fem), "k^-")
ax2.axhline(1.0, color="0.6", lw=1)
ax2.set(
    xlabel="depth below nearest electrode (mm)",
    ylabel="BEM / FEM",
    title="BEM overestimates scalp potential",
    ylim=(0, None),
)

# %%
# The head model itself
# ---------------------
# Both surfaces travel with the model, so they can be rendered directly. The
# scalp is drawn here as well as the cortex, and it is worth doing: against the
# cortex alone the montage looks like a cloud of points floating around and
# through the brain. It is not. Every electrode lies on this scalp surface
# (median 4.6 mm from the nearest vertex, on a mesh whose own spacing is about
# 10 mm), and the closest any electrode comes to the cortex is 11.2 mm, which
# is scalp plus skull plus CSF, as it should be.
#
# The points reaching well below the brain are real too: the set includes four
# neck electrodes and a band of face and cheek positions, because the model was
# built for transcranial stimulation targeting as well as for EEG. Use ``picks``
# to restrict the forward to the electrodes you actually recorded.
#
# This is also how a source estimate on this model is visualised: the source
# space is the model's own mesh rather than a FreeSurfer subject, so
# ``stc.plot()``, which looks a subject up in a ``subjects_dir``, does not
# apply here.

scalp_rr, scalp_tris = ny_head_scalp()


def _mesh(rr, tris):
    """Assemble a triangulation as a PyVista mesh, in millimetres."""
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    return pv.PolyData(rr * 1000.0, faces)


# ``off_screen`` per plotter, not globally: it lets ``screenshot`` render
# without opening a window, while the global setting stays on-screen for the
# ``stc.plot()`` figures elsewhere in the gallery, which need a display.
plotter = pv.Plotter(window_size=(900, 600), off_screen=True)
plotter.add_mesh(
    _mesh(scalp_rr, scalp_tris), color="#e8d5c4", opacity=0.25, smooth_shading=True
)
for hemi in fwd["src"]:
    plotter.add_mesh(
        _mesh(hemi["rr"], hemi["tris"]), color="#c8b7a6", smooth_shading=True
    )
plotter.add_points(
    elec * 1000.0, color="#0072B2", point_size=9, render_points_as_spheres=True
)
plotter.camera_position = "yz"
plotter.camera.azimuth = 210
plotter.camera.elevation = 15
img = plotter.screenshot()
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax.imshow(img)
ax.set_axis_off()

# %%
# Beamforming on it
# -----------------
# Nothing below is FEM-specific: this is the ordinary pipeline, with ``fwd``
# supplied by the FEM reader. Two nearby, near-perfectly correlated sources are
# simulated. This is the regime in which a single-source LCMV is expected to
# cancel.
#
# Read the result below carefully, because it does not go the textbook way: on
# this array LCMV finds both sources exactly, and it is the greedy MCMV *search*
# that misses. That is worth showing rather than hiding. The cancellation costs
# amplitude, not position, and these two topographies are distinguishable enough
# (their correlation is negative) for the power map to separate them.

rng = np.random.default_rng(0)
rr = fwd["source_rr"]
i1 = int(np.argmax(rr[:, 1]))
i2 = int(np.argmin(np.linalg.norm(rr - (rr[i1] + np.array([0.045, 0.0, 0.0])), axis=1)))

n_ep, n_t = 60, 250
t = np.arange(n_t) / 250.0
sources = np.zeros((n_ep, fwd["nsource"], n_t))
for e in range(n_ep):
    a = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 0.2)) * 20e-9  # 20 nA m
    sources[e, i1] = a
    sources[e, i2] = 0.95 * a + 0.05 * np.sin(2 * np.pi * 10 * t + 1.0) * 20e-9

sensor = np.einsum("cs,est->ect", gain, sources)
noise = rng.standard_normal((n_ep, fwd["nchan"], n_t)) * 0.15e-6
noise -= noise.mean(1, keepdims=True)  # keep the noise average referenced too

epochs = mne.EpochsArray(sensor + noise, info, tmin=0, baseline=None)
epochs_noise = mne.EpochsArray(noise, info, tmin=0, baseline=None)
data_cov = mne.compute_covariance(epochs, method="empirical")
noise_cov = mne.compute_covariance(epochs_noise, method="empirical")
print(
    f"separation {np.linalg.norm(rr[i1] - rr[i2]) * 1000:.0f} mm, "
    f"source correlation r = {np.corrcoef(sources[0, i1], sources[0, i2])[0, 1]:.3f}, "
    f"data rank {mne.compute_rank(data_cov, info=info)}"
)

# %%
# LCMV first, then the MCMV search that looks for a *pair*.

filters = make_lcmv(
    info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, weight_norm="unit-noise-gain"
)
power = apply_lcmv_cov(data_cov, filters).data.ravel()

scan = scan_mcmv(info, fwd, data_cov, n_sources=2, noise_cov=noise_cov, reg=0.05)
print(f"MCMV found sources {scan['sources']}, pseudo-Z {np.round(scan['pseudo_z'], 1)}")

# %%
# With the pair fixed, MCMV returns each source's time course with the other
# constrained to zero gain, which is what recovers the true dipole moments
# rather than the mutually cancelled ones a single-source filter would give.

beamformer = make_mcmv(info, fwd, data_cov, [i1, i2], noise_cov=noise_cov, reg=0.05)
recovered = apply_mcmv(epochs.average(), beamformer)
print(
    "recovered amplitudes: "
    f"{np.round(np.abs(recovered.data).max(1) * 1e9, 1)} nA m (simulated 20 and 19)"
)

# %%
# The LCMV power map on the FEM cortical surface, with the two simulated
# locations marked. Both are recovered.

peaks = np.argsort(power)[::-1][:2]
errors = [np.linalg.norm(rr[peaks] - rr[k], axis=1).min() * 1000 for k in (i1, i2)]
print(f"LCMV peak localisation errors: {errors[0]:.1f} and {errors[1]:.1f} mm")

scalars = [np.zeros(s["np"]) for s in fwd["src"]]
n_lh = fwd["src"][0]["nuse"]
scalars[0][fwd["src"][0]["vertno"]] = power[:n_lh]
scalars[1][fwd["src"][1]["vertno"]] = power[n_lh:]

plotter = pv.Plotter(window_size=(800, 500), off_screen=True)
for hemi, values in zip(fwd["src"], scalars, strict=True):
    mesh = _mesh(hemi["rr"], hemi["tris"])
    mesh["power"] = values
    plotter.add_mesh(
        mesh, scalars="power", cmap="hot", smooth_shading=True, show_scalar_bar=False
    )
plotter.add_points(
    rr[[i1, i2]] * 1000.0,
    color="#0072B2",
    point_size=18,
    render_points_as_spheres=True,
)
plotter.camera_position = "xy"
plotter.camera.elevation = 65
img = plotter.screenshot()
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax.imshow(img)
ax.set_axis_off()

# %%
# Why this is EEG-only
# --------------------
# A precomputed template lead field can exist for EEG and cannot exist for MEG,
# and the reason is geometric. EEG electrodes are placed by a standardised
# layout, so "Cz on the average head" is the same position for every recording
# made with that montage. Solve once, reuse forever. MEG sensors sit in a
# fixed helmet while the head sits wherever the participant put it, so the
# sensor positions *relative to the brain* differ for every session and are
# recorded per session in ``info['dev_head_t']``. There is no template MEG array
# to solve for, and an MEG FEM forward therefore needs a solver run on the
# individual anatomy and that session's head position.
#
# This costs less than it appears. The tissues a BEM misrepresents are precisely
# the ones MEG is least sensitive to: the magnetic field is unaffected by the
# radially symmetric part of the volume currents, which is why MEG source
# estimates are famously insensitive to skull conductivity while EEG estimates
# are not :footcite:`HuangEtAl2016`. BEM is a reasonable model for MEG. For EEG
# it is the weak link, which is the case this addresses.
#
# References
# ----------
# .. footbibliography::
