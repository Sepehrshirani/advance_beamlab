import numpy as np
def test_probe(scenario):
    from advance_beamlab import pairwise_mcmv_connectivity, augmented_pairwise_mcmv_connectivity
    from advance_beamlab._connectivity import _select_neighbours
    import sys
    tc = sys.modules['advance_beamlab.tests.test_connectivity']
    _env_corr = tc._env_corr
    d = scenario
    conn = pairwise_mcmv_connectivity(d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
        method="envelope", noise_cov=d["ncov"], absolute=False)
    sig = np.zeros((4,4), bool); sig[0,1]=sig[1,0]=True; sig[1,2]=sig[2,1]=True
    apw = augmented_pairwise_mcmv_connectivity(d["evoked"], d["info"], d["fwd"], d["dcov"], d["sources"],
        conn, sig, positions=d["positions"], method="envelope", noise_cov=d["ncov"], absolute=False)
    truth = _env_corr(d["s_a"], d["s_b"])
    print("\ntruth        =", truth)
    print("PW  conn[0,1]=", conn[0,1], " |err| =", abs(conn[0,1]-truth))
    print("APW apw [0,1]=", apw[0,1], " |err| =", abs(apw[0,1]-truth))
    print("ratio apw_err/pw_err =", abs(apw[0,1]-truth)/abs(conn[0,1]-truth))
    print("apw == conn on augmented edge?", apw[0,1]==conn[0,1])
    print("no-op APW would pass line-328 assertion?", abs(conn[0,1]-truth) <= abs(conn[0,1]-truth)+1e-9)
    pos = d["positions"]; s = np.ones((4,4),bool); np.fill_diagonal(s,False)
    ch = _select_neighbours(0,1,pos,s,s.sum(1),radius=1.0,max_neighbours=2)
    print("caps_order chosen =", ch, "len =", len(ch), "(assertion is len<=4)")

from advance_beamlab.tests.test_connectivity import scenario  # noqa: F401,E402
