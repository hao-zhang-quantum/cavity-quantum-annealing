"""Check cqa.py against the paper's data (N = 24, w_c = 3.4448, lam = 0.5, self-energy included).

Reference errors 1 - F_spin come from an independent code (dense adaptive Runge-Kutta, DOP853,
rtol 1e-12, n_max = 12), the data behind Fig. 2(c) of the paper.  Run: python test_reference.py
"""
from cqa import CavityAnneal, evolve

WC = 3.4448
REFERENCE = {  # T: (bare, cavity)
    11.787686347935873: (9.584452e-01, 8.961818e-01),
    44.668359215096324: (7.644521e-01, 3.394532e-01),
    169.26666150378762: (3.275843e-01, 7.062785e-03),
    411.4: (6.459493e-02, 4.080259e-05),
}
bare, cavity = CavityAnneal(24, wc=WC, lam=0.0, n_max=0), CavityAnneal(24, wc=WC, lam=0.5)
worst = 0.0
for T, (rb, rc) in REFERENCE.items():
    eb = 1 - bare.fidelity(evolve(bare, T))
    ec = 1 - cavity.fidelity(evolve(cavity, T))
    rel = max(abs(eb - rb) / rb, abs(ec - rc) / rc); worst = max(worst, rel)
    print(f"T={T:7.2f}  bare {eb:.6e} (ref {rb:.6e})  cavity {ec:.6e} (ref {rc:.6e})  rel.dev {rel:.1e}")
assert worst < 5e-4, f"deviation from the reference {worst:.1e} exceeds 5e-4"
print(f"OK: largest relative deviation {worst:.1e}")
