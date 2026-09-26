"""Cavity quantum annealing: the p-spin register coupled to one cavity mode.

    H(s) = (1-s) H_D + s H_P + w_c a^dag a + g(s) (a + a^dag) O + (g(s)^2 / w_c) O^2,   s = t/T,

    H_D = -2 S_x,  H_P = -N (2 S_z / N)^p,  O = 2 S_x / sqrt(N),
    g(s) = g0 sin^2(pi s),  g0 = lam * w_c / sqrt(N).

The register lives in the symmetric subspace S = N/2 (dimension N+1); the cavity is truncated at
n_max photons.  The last term is the dipole self-energy.  g0 = 0 gives the bare anneal.

Time evolution uses the fourth-order commutator-free Magnus integrator (CF4): every step is a
product of two exact matrix exponentials, so the evolution is unitary to machine precision and
the error falls as h^4 with the step h.
"""
import numpy as np
from scipy.linalg import eigh


def spin_ops(N):
    """Collective S_x, S_z in the S = N/2 multiplet, basis m = N/2, ..., -N/2."""
    S = N / 2
    m = S - np.arange(N + 1)
    Sz = np.diag(m)
    sp = np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1))          # <m+1|S+|m>
    Splus = np.diag(sp, k=1)
    Sx = 0.5 * (Splus + Splus.T)
    return Sx, Sz


class CavityAnneal:
    """Hamiltonian terms and observables of the coupled register + cavity."""

    def __init__(self, N, wc=3.4448, lam=0.5, n_max=6, p=3):
        self.N, self.wc, self.n_max = N, wc, n_max
        self.g0 = lam * wc / np.sqrt(N)
        Sx, Sz = spin_ops(N)
        HD, HP, O = -2 * Sx, -N * np.linalg.matrix_power(2 * Sz / N, p), 2 * Sx / np.sqrt(N)
        d = n_max + 1
        a = np.diag(np.sqrt(np.arange(1, d)), k=1)
        Ip, Is = np.eye(d), np.eye(N + 1)
        self.HD, self.HP = np.kron(Ip, HD), np.kron(Ip, HP)
        self.Hc = np.kron(wc * a.T @ a, Is)
        self.Hg = np.kron(a + a.T, O)
        self.Hdse = np.kron(Ip, O @ O)
        self.target = eigh(HP)[1][:, 0]                       # ground state of H_P (all spins up)
        self.dims = (d, N + 1)

    def H(self, s):
        g = self.g0 * np.sin(np.pi * s) ** 2
        H = (1 - s) * self.HD + s * self.HP + self.Hc
        if self.g0:
            H = H + g * self.Hg + (g ** 2 / self.wc) * self.Hdse
        return H

    def initial_state(self):
        return eigh(self.H(0.0))[1][:, 0].astype(complex)     # register ground state, cavity vacuum

    def fidelity(self, psi):
        """F_spin: probability that the register is in the ground state of H_P (photons traced out)."""
        amp = psi.reshape(self.dims) @ self.target
        return float(np.sum(np.abs(amp) ** 2))

    def photons(self, psi):
        pn = np.sum(np.abs(psi.reshape(self.dims)) ** 2, axis=1)
        return float(np.arange(len(pn)) @ pn)


def _expm_herm(H, tau):
    w, v = eigh(H)
    return (v * np.exp(-1j * tau * w)) @ v.conj().T


def evolve(model, T, h=0.2, psi0=None):
    """Propagate from s=0 to s=1 in time T with the CF4 Magnus integrator (step h in t)."""
    psi = model.initial_state() if psi0 is None else psi0.astype(complex)
    n = max(1, int(np.ceil(T / h)))
    dt = T / n
    c1, c2 = 0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6        # Gauss nodes
    a1, a2 = (3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12
    for k in range(n):
        H1, H2 = model.H((k + c1) / n), model.H((k + c2) / n)
        psi = _expm_herm(a2 * H1 + a1 * H2, dt) @ psi          # first factor acts first
        psi = _expm_herm(a1 * H1 + a2 * H2, dt) @ psi
    return psi
