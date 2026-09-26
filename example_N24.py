"""Reference point of the paper: N = 24, w_c = 3.4448, lam = 0.5.

Runs the bare anneal and cavity quantum annealing on a grid of annealing times T, prints the
error 1 - F_spin and the final photon number, fits each error curve with exp(-Gamma T), and saves
the figure error_vs_T.png.  Runtime: about two minutes on a laptop.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cqa import CavityAnneal, evolve

N, WC, LAM = 24, 3.4448, 0.5
# the annealing times of Fig. 2(c) in the paper
T_GRID = np.array([11.787686347935873, 28.651202696637817, 69.63974029624322, 108.57111194022039,
                   169.26666150378762, 263.9, 411.4, 641.4])

bare = CavityAnneal(N, wc=WC, lam=0.0, n_max=0)
cavity = CavityAnneal(N, wc=WC, lam=LAM)

err_bare, err_cav = [], []
print(f"{'T':>8} {'bare 1-F':>12} {'cavity 1-F':>12} {'photons':>8}")
for T in T_GRID:
    eb = 1 - bare.fidelity(evolve(bare, T))
    psi = evolve(cavity, T)
    ec = 1 - cavity.fidelity(psi)
    err_bare.append(eb); err_cav.append(ec)
    print(f"{T:8.1f} {eb:12.4e} {ec:12.4e} {cavity.photons(psi):8.3f}")


def fit_rate(T, err):
    """One-parameter fit err = exp(-Gamma T) (least squares on log err), as in the paper."""
    T, err = np.asarray(T), np.asarray(err)
    return float(-np.sum(T * np.log(err)) / np.sum(T ** 2))


fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
tt = np.linspace(0, T_GRID[-1] * 1.05, 200)
handles, labels = [], []
for err, mk, col, lab in ((err_bare, "o", "#555555", "bare anneal"),
                          (err_cav, "s", "#E63946", rf"cavity, $\lambda={LAM}$")):
    G = fit_rate(T_GRID, err)
    ax.plot(tt, np.exp(-G * tt), "-", color=col, lw=1.2)                        # fit line
    ax.plot(T_GRID, err, mk, ms=6, mfc="white", mec=col, mew=1.2, ls="none")    # data
    handles.append(Line2D([], [], marker=mk, ms=6, mfc="white", mec=col, mew=1.2, color=col, lw=1.2))
    labels.append(rf"{lab}: $\Gamma={G:.4f}$")
    print(f"{lab}: fitted Gamma = {G:.4f}")
ax.set_yscale("log"); ax.set_ylim(min(err_cav) / 3, 1.5)
ax.set_xlabel("annealing time $T$"); ax.set_ylabel(r"error $1-F_{\rm spin}$")
ax.set_title(rf"$N={N}$, $\omega_c={WC}$")
ax.legend(handles, labels, frameon=False, fontsize=8)
fig.savefig("error_vs_T.png", dpi=200)
print("saved error_vs_T.png")
