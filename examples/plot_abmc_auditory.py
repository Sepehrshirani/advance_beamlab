"""
.. _ex-abmc-auditory:

============================================
ABMC on a real recording, and where it stops
============================================

Every other ABMC demonstration in this package is a simulation, which is a fair
criticism of it: simulations are where a method is guaranteed to meet its own
assumptions. This example runs ABMC :footcite:`Shirani2024` on real MEG, the
auditory response of the MNE ``sample`` dataset, and reports the result whether
or not it flatters the method.

The paper's own cases are epileptic spikes and delayed responses to single-pulse
stimulation. The ``sample`` dataset has neither, but a single-trial auditory
evoked response is the same shape of problem: a short transient with a
reproducible morphology, known from the average, carrying a small share of the
trial's variance. That is the regime ABMC exists for.

There is no ground truth on real data, so "correct" has to be defined in advance
rather than judged by eye. Here it is anatomical: the primary auditory cortex is
the transverse temporal (Heschl's) gyrus, and both methods are scored by the
distance from their peak to the nearest grid point inside that label. The
comparison is against :func:`mne.beamformer.make_lcmv` on the same data.

The summary, which the numbers below reproduce: ABMC lands 5 to 11 mm from
Heschl's gyrus at every averaging level, against 12 mm for LCMV, and its best
result is on a single trial. Two caveats keep that from being a stronger claim
than it is. Grid spacing here is several millimetres, so differences of a few
millimetres are not resolved, and a distance to a label is a weak criterion
compared with a simulated ground truth. What the example does establish is that
ABMC runs correctly on a real recording and is not worse than LCMV on one.
"""
# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.beamformer import apply_lcmv_cov, make_lcmv

from advance_beamlab import make_abmc

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
meg_dir = data_path / "MEG" / "sample"

# %%
# Gradiometers only, so the comparison is single-unit and needs no cross-type
# whitening argument to be made here; that is covered in the MCMV auditory
# example. The 2 Hz high-pass removes the drift that would otherwise dominate a
# single trial's variance.

raw = mne.io.read_raw_fif(meg_dir / "sample_audvis_filt-0-40_raw.fif", preload=True)
raw.pick("grad").filter(2, 40)
events = mne.read_events(meg_dir / "sample_audvis_filt-0-40_raw-eve.fif")
epochs = mne.Epochs(
    raw, events, {"aud_l": 1}, tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True
)
evoked = epochs.average()

fwd = mne.read_forward_solution(meg_dir / "sample_audvis-meg-oct-6-fwd.fif")
fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True)
fwd = mne.pick_channels_forward(fwd, epochs.ch_names, ordered=True)
source_rr = fwd["source_rr"]

# %%
# The scoring target, fixed before either method is run: the grid points inside
# the transverse temporal label of either hemisphere.

labels = [
    label
    for label in mne.read_labels_from_annot(
        "sample", "aparc", subjects_dir=subjects_dir
    )
    if "transversetemporal" in label.name
]
auditory, offset = [], 0
for hemi, src_hemi in zip(("lh", "rh"), fwd["src"], strict=True):
    for label in labels:
        if label.hemi != hemi:
            continue
        shared = np.intersect1d(src_hemi["vertno"], label.vertices)
        auditory += [
            offset + int(np.where(src_hemi["vertno"] == v)[0][0]) for v in shared
        ]
    offset += src_hemi["nuse"]
auditory = np.array(sorted(set(auditory)))


def distance_to_auditory(index):
    """Millimetres from a grid point to the nearest Heschl's gyrus point."""
    return np.linalg.norm(source_rr[auditory] - source_rr[index], axis=1).min() * 1000


print(f"grid points inside Heschl's gyrus: {len(auditory)}")

# %%
# The template is the dominant temporal component of the average, which is what
# an experimenter actually has: a prototype of the waveform, not a copy of the
# trial being localised.

noise_cov = mne.compute_covariance(epochs, tmax=0.0, method="shrunk")
data_cov = mne.compute_covariance(epochs, tmin=0.0, method="shrunk")

_, _, vt = np.linalg.svd(evoked.data, full_matrices=False)
template = vt[0] / np.abs(vt[0]).max()

filters = make_lcmv(
    epochs.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    weight_norm="unit-noise-gain",
)
lcmv_peak = int(np.argmax(apply_lcmv_cov(data_cov, filters).data.ravel()))
lcmv_error = distance_to_auditory(lcmv_peak)
print(f"LCMV peak: {lcmv_error:.0f} mm from Heschl's gyrus")

# %%
# Now ABMC, over a sweep of how many trials are averaged. Averaging raises the
# signal-to-noise ratio and simultaneously raises the share of the variance the
# response carries, so the sweep moves the recording out of the regime ABMC is
# designed for while making the problem easier. Both effects are visible.

trials = epochs.get_data()
rows = []
for n_avg in (1, 2, 4, 8, 16):
    segment = trials[:n_avg].mean(0)
    share = np.var(evoked.data) / np.var(segment)
    result = make_abmc(
        epochs.info, fwd, segment, template, noise_cov=noise_cov, P="auto"
    )
    peak = int(np.argmax(np.abs(result.template_match)))
    rows.append((n_avg, share * 100, distance_to_auditory(peak), result.critical_p))
    print(
        f"  n_avg={n_avg:2d}  variance share {share * 100:5.1f}%  "
        f"ABMC {rows[-1][2]:5.0f} mm from Heschl  critical_p {result.critical_p:.2f}"
    )

# %%
# ABMC is at or inside LCMV's error across the whole sweep, and the single-trial
# point is the best of the five, which is the end of the sweep the method is
# designed for. Do not over-read the ordering between 5 and 11 mm: the source
# grid is spaced at several millimetres and the criterion is a distance to an
# anatomical label, so the honest reading is that ABMC is not worse here, not
# that it is reliably better by 7 mm.
#
# One result is worth singling out because it was different a day earlier. Before
# the automatic selection of ``P`` was made to reject values above
# ``critical_p``, this same recording gave 53 mm at a single trial: the plateau
# rule had settled on the degenerate large-``P`` limit, which is a fixed point
# and so looks perfectly stable while localising badly. ``critical_p`` is about
# 1.25 here, so that constraint is doing real work on this dataset rather than
# guarding a case that never arises.

n_avg, shares, errors, _ = map(np.array, zip(*rows, strict=True))

fig, ax = plt.subplots(figsize=(7.2, 4), constrained_layout=True)
ax.plot(n_avg, errors, "o-", label="ABMC")
ax.axhline(lcmv_error, color="#D55E00", ls="--", label=f"LCMV ({lcmv_error:.0f} mm)")
ax.set(
    xlabel="trials averaged",
    ylabel="distance from Heschl's gyrus (mm)",
    xscale="log",
    title="ABMC on real MEG, scored against auditory cortex",
)
ax.set_xticks(n_avg)
ax.set_xticklabels(n_avg)
for x, y, s in zip(n_avg, errors, shares, strict=True):
    ax.annotate(
        f"{s:.0f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color="0.35",
    )
ax.legend()

# %%
# The percentages annotated on the points are the share of the segment's
# variance carried by the evoked response. Even the single-trial point, at about
# 4%, sits above the 0.14% to 1.05% the simulation example works at, so this
# recording does not reach the low-variance regime the method is really aimed
# at. No dataset shipped with MNE-Python would: none of them contain
# epileptiform spikes. The simulation remains the place where ABMC is tested
# against a known answer, and this example is the place where it is shown to
# survive contact with a real recording.
#
# References
# ----------
# .. footbibliography::
