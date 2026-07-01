import mne, numpy as np
mne.set_log_level("ERROR")
from mne_beamlab import scan_mcmv, apply_mcmv

montage = mne.channels.make_standard_montage("standard_1020")
info = mne.create_info(montage.ch_names[:32], 200.0, "eeg"); info.set_montage("standard_1020")
sphere = mne.make_sphere_model("auto", "auto", info)
src = mne.setup_volume_source_space(sphere=sphere, pos=30.0)
fwd = mne.make_forward_solution(info, None, src, sphere, eeg=True, meg=False)
fwd = mne.convert_forward_solution(fwd, force_fixed=False, surf_ori=False)

G = fwd["sol"]["data"]; locs = [10, 45]
oris = [np.array([1.,0,.3]), np.array([0,1.,-.2])]
H = np.column_stack([G[:,3*l:3*l+3] @ (u/np.linalg.norm(u)) for l,u in zip(locs,oris)])
C = np.array([[1.,0.8],[0.8,1.]]); N = np.eye(G.shape[0])
R = N + 16.0*np.trace(N)/np.trace(H@C@H.T)*(H@C@H.T)
cov = mne.Covariance(R, info["ch_names"], [], list(info["projs"]), nfree=1)

res = scan_mcmv(info, fwd, cov, localizer="mpz", n_sources=2)
print("implanted :", sorted(locs))
print("recovered :", sorted(res["sources"]))
X = np.random.default_rng(0).standard_normal((len(res["filters"]["ch_names"]), 100))
print("apply out :", apply_mcmv(X, res["filters"]).shape, "(expect (2, 100))")
print("RESULT    :", "MATCH" if sorted(res["sources"])==sorted(locs) else "MISMATCH")
