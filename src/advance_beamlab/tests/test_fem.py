# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from advance_beamlab import (  # noqa: E402
    apply_mcmv,
    make_mcmv,
    make_recipsiicos_lcmv,
    read_ny_head_forward,
)
from advance_beamlab._fem import (  # noqa: E402
    _N_VERT_HEMI,
    _N_VERT_TOTAL,
    _ny_head_dir,
    make_ny_head_info,
    ny_head_montage,
)

# The model is a 678 MB download under a licence that forbids redistribution, so
# it is never present on a bare runner. Every test here is skipped unless a user
# (or the dataset CI job) has already fetched it; none of them will download it.
_HAVE = (_ny_head_dir() / "sa_nyhead.mat").is_file()
requires_ny_head = pytest.mark.skipif(
    not _HAVE, reason="New York Head model not downloaded (see fetch_ny_head)"
)
pytestmark = requires_ny_head


@pytest.fixture(scope="module")
def fwd():
    """A coarse fixed-orientation forward, shared across tests (reading is slow)."""
    return read_ny_head_forward(resolution="2K", orientation="normal")


@pytest.fixture(scope="module")
def info():
    return make_ny_head_info(sfreq=250.0)


def test_forward_structure(fwd):
    """The forward must be a well-formed MNE Forward, not merely a dict."""
    assert fwd["nchan"] == 231
    assert fwd["nsource"] == 2004
    assert fwd["sol"]["data"].shape == (231, 2004)
    assert fwd["coord_frame"] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
    assert mne.forward.is_fixed_orient(fwd)
    # Two hemispheres, left first, in ascending vertex order -- MNE relies on it.
    assert len(fwd["src"]) == 2
    assert [s["np"] for s in fwd["src"]] == [_N_VERT_HEMI, _N_VERT_HEMI]
    assert sum(s["nuse"] for s in fwd["src"]) == fwd["nsource"]
    for s in fwd["src"]:
        assert s["coord_frame"] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
        assert np.all(np.diff(s["vertno"]) > 0)
        assert s["tris"].min() >= 0 and s["tris"].max() < s["np"]
    # Left hemisphere on the left, right on the right.
    assert fwd["src"][0]["rr"][:, 0].mean() < 0 < fwd["src"][1]["rr"][:, 0].mean()


def test_free_orientation(fwd):
    """Free orientation gives three columns per source in head coordinates."""
    free = read_ny_head_forward(resolution="2K", orientation="free")
    assert free["sol"]["data"].shape == (231, 3 * 2004)
    assert not mne.forward.is_fixed_orient(free)
    assert_allclose = np.testing.assert_allclose
    assert_allclose(free["source_rr"], fwd["source_rr"])
    # The normal-constrained lead field must be reproducible from the free one by
    # projecting onto the surface normals; this is the check that the rotation
    # into head coordinates was applied consistently to gains and normals alike.
    g3 = free["sol"]["data"].reshape(231, 2004, 3)
    projected = np.einsum("csk,sk->cs", g3, fwd["source_nn"])
    assert_allclose(projected, fwd["sol"]["data"], rtol=1e-6, atol=1e-9)


def test_fif_roundtrip(fwd, tmp_path):
    """MNE must be able to write and re-read the forward it was handed."""
    fname = tmp_path / "ny-fwd.fif"
    mne.write_forward_solution(fname, fwd, overwrite=True)
    got = mne.read_forward_solution(fname)
    got = mne.convert_forward_solution(got, force_fixed=True, use_cps=False)
    assert got["nsource"] == fwd["nsource"]
    # FIF stores the gain in single precision, hence the loose tolerance.
    np.testing.assert_allclose(got["sol"]["data"], fwd["sol"]["data"], rtol=1e-6)
    np.testing.assert_allclose(got["source_rr"], fwd["source_rr"], atol=1e-7)


def test_average_reference(fwd, info):
    """The lead field is average referenced; the info must carry the projector."""
    assert np.abs(fwd["sol"]["data"].sum(0)).max() < 1e-9
    assert [p["desc"] for p in info["projs"]] == ["Average EEG reference"]
    # Which means the forward -- and so any data it generates -- is rank
    # deficient by exactly one. Every beamformer here has to resolve that rank
    # correctly rather than invert a singular covariance.
    assert np.linalg.matrix_rank(fwd["sol"]["data"]) == 230


def test_units_are_volts_per_ampere_metre(fwd):
    """Gains must be in MNE's units, i.e. tens of V/(A m) for cortical dipoles.

    The distributed lead field is in microvolts per nanoampere-metre; getting the
    conversion wrong is a silent factor of 1000, which would leave localisation
    untouched (beamformer weights are scale invariant) and every reconstructed
    amplitude wrong. The band below is wide enough to be model-independent and
    narrow enough to catch any power-of-ten slip: an MNE 3-layer BEM forward on
    fsaverage with these same 231 electrodes gives a median of 32 V/(A m), and a
    10 nA m cortical dipole must produce a peak scalp potential of a few
    microvolts, as it does physiologically.
    """
    g = np.abs(fwd["sol"]["data"])
    assert 5.0 < np.median(g) < 100.0
    peak_uv = g.max() * 10e-9 * 1e6
    assert 0.5 < peak_uv < 50.0


def test_montage_matches_standard_1005():
    """Electrode geometry must agree with MNE's own 10-05 layout.

    Both are template heads with independently placed fiducials, so they are not
    expected to coincide exactly; they are expected to agree to within the
    difference between two template heads, which is about a centimetre. A frame
    or unit error would show up here as centimetres of systematic offset or a
    wrong overall scale.
    """
    ny = ny_head_montage().get_positions()["ch_pos"]
    p = mne.channels.make_standard_montage("standard_1005").get_positions()
    # MNE stores its standard montages in MRI coordinates, so put both in head
    # coordinates via their own fiducials before comparing.
    t = mne.transforms.get_ras_to_neuromag_trans(p["nasion"], p["lpa"], p["rpa"])
    shared = sorted(set(ny) & set(p["ch_pos"]))
    assert len(shared) > 150
    a = np.array([ny[c] for c in shared])
    b = mne.transforms.apply_trans(t, np.array([p["ch_pos"][c] for c in shared]))
    assert np.median(np.linalg.norm(a - b, axis=1)) < 0.015
    for i in range(3):
        assert np.corrcoef(a[:, i], b[:, i])[0, 1] > 0.99


def test_geometry_is_anatomical(fwd, info):
    """Sanity bounds a wrong transform would violate."""
    e = np.array([c["loc"][:3] for c in info["chs"]])
    assert 0.09 < np.linalg.norm(e, axis=1).mean() < 0.14  # head-sized
    assert e[info["ch_names"].index("Cz"), 2] > 0.10  # Cz on top
    assert abs(e[info["ch_names"].index("Cz"), 0]) < 0.01  # Cz on the midline
    assert e[info["ch_names"].index("T7"), 0] < -0.05  # T7 on the left
    assert e[info["ch_names"].index("T8"), 0] > 0.05
    # Cortex sits inside the electrodes, separated by scalp, skull and CSF.
    d = np.linalg.norm(e[:, None] - fwd["source_rr"][None], axis=2).min()
    assert 0.005 < d < 0.030


@pytest.mark.parametrize("resolution", ["1K", "2K", "5K"])
def test_resolutions_are_nested(resolution):
    """The meshes are strict subsets, so a coarse one is a true subsampling."""
    f = read_ny_head_forward(resolution=resolution, orientation="normal")
    assert f["nsource"] <= _N_VERT_TOTAL
    v = np.concatenate([f["src"][0]["vertno"], f["src"][1]["vertno"] + _N_VERT_HEMI])
    assert len(np.unique(v)) == f["nsource"]


def test_picks_subset(fwd):
    names = fwd["sol"]["row_names"]
    sub = ["Cz", "Fz", "Pz", "Oz", "C3", "C4"]
    f = read_ny_head_forward(resolution="1K", picks=sub)
    assert f["sol"]["row_names"] == sub
    assert f["nchan"] == len(sub)
    with pytest.raises(ValueError, match="not New York Head electrodes"):
        read_ny_head_forward(resolution="1K", picks=["Cz", "NOT_AN_ELECTRODE"])
    assert "Cz" in names


def test_bad_arguments():
    with pytest.raises(ValueError, match="resolution"):
        read_ny_head_forward(resolution="3K")
    with pytest.raises(ValueError, match="orientation"):
        read_ny_head_forward(orientation="loose")


def _simulate(fwd, info, i1, i2, seed=0):
    """Two correlated dipoles of 20 nA m in average-referenced noise."""
    rng = np.random.default_rng(seed)
    g = fwd["sol"]["data"]
    n_ch, n_src = g.shape
    n_t, n_ep = 200, 40
    t = np.arange(n_t) / info["sfreq"]
    s = np.zeros((n_ep, n_src, n_t))
    for e in range(n_ep):
        a = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 0.2)) * 20e-9
        s[e, i1] = a
        s[e, i2] = 0.95 * a + 0.05 * np.sin(2 * np.pi * 10 * t + 1.0) * 20e-9
    x = np.einsum("cs,est->ect", g, s)
    nz = rng.standard_normal((n_ep, n_ch, n_t)) * 0.15e-6
    nz -= nz.mean(1, keepdims=True)  # average reference the noise too
    ep = mne.EpochsArray(x + nz, info, tmin=0, verbose=False)
    epn = mne.EpochsArray(nz, info, tmin=0, verbose=False)
    return (
        ep,
        mne.compute_covariance(ep, method="empirical", verbose=False),
        mne.compute_covariance(epn, method="empirical", verbose=False),
    )


def test_lcmv_localises(fwd, info):
    """MNE's own LCMV must localise on the FEM forward without any adaptation."""
    from mne.beamformer import apply_lcmv_cov, make_lcmv

    rr = fwd["source_rr"]
    i1 = int(np.argmax(rr[:, 1]))
    i2 = int(np.argmin(np.linalg.norm(rr - (rr[i1] + np.array([0.045, 0, 0])), axis=1)))
    ep, dc, nc = _simulate(fwd, info, i1, i2)
    flt = make_lcmv(
        info, fwd, dc, reg=0.05, noise_cov=nc, weight_norm="unit-noise-gain"
    )
    power = apply_lcmv_cov(dc, flt).data.ravel()
    top = np.argsort(power)[::-1][:2]
    for i in (i1, i2):
        assert np.linalg.norm(rr[top] - rr[i], axis=1).min() < 0.020


def test_mcmv_recovers_amplitudes(fwd, info):
    """MCMV must return the true dipole moment in A m, which pins the units."""
    rr = fwd["source_rr"]
    i1 = int(np.argmax(rr[:, 1]))
    i2 = int(np.argmin(np.linalg.norm(rr - (rr[i1] + np.array([0.045, 0, 0])), axis=1)))
    ep, dc, nc = _simulate(fwd, info, i1, i2)
    bf = make_mcmv(info, fwd, dc, [i1, i2], noise_cov=nc, reg=0.05)
    amp = np.abs(apply_mcmv(ep.average(), bf).data).max(1)
    np.testing.assert_allclose(amp, [20e-9, 19e-9], rtol=0.25)


def test_recipsiicos_runs(fwd, info):
    """ReciPSIICOS must accept the FEM forward and return a usable filter."""
    from mne.beamformer import apply_lcmv_cov

    rr = fwd["source_rr"]
    i1 = int(np.argmax(rr[:, 1]))
    i2 = int(np.argmin(np.linalg.norm(rr - (rr[i1] + np.array([0.045, 0, 0])), axis=1)))
    _, dc, nc = _simulate(fwd, info, i1, i2)
    flt = make_recipsiicos_lcmv(info, fwd, dc, rank=40, noise_cov=nc, reg=0.05)
    power = apply_lcmv_cov(dc, flt).data.ravel()
    assert power.shape == (fwd["nsource"],)
    assert np.all(np.isfinite(power)) and power.max() > 0
