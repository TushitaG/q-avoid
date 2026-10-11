"""
Quantum amplitude estimation of collision probability.

Encoding
--------
The encounter plane is cut into a 2^n x 2^n grid (in covariance principal axes).

    A |0>|0>|0> = sum_i sqrt(p_i) |i> ( sqrt(1 - f_i)|0> + sqrt(f_i)|1> )

p_i : Gaussian probability mass of cell i (the uncertainty distribution)
f_i : conditional probability of a hit given the object is in cell i

so P(objective qubit = 1) = sum_i p_i f_i = Pc (up to the box truncation Z,
which we correct for). Because f_i is the exact conditional hit probability the
grid introduces no discretisation bias -- the grid only shapes the circuit.

Because the principal-axis Gaussian factorises, the x and y registers are
prepared independently, which keeps the state-preparation circuit shallow.

Estimation
----------
Maximum-likelihood amplitude estimation (Suzuki et al., 2020): run A Q^k for
k = 0, 1, 2, 4, ... and fit theta to the observed hit counts, where
P(hit | k) = sin^2((2k+1) theta) and a = sin^2(theta).
No phase estimation, no ancilla register -- friendlier to near-term devices.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate, grover_operator

from .classical import disk_quadrature, _gauss_pdf


@dataclass
class GridEncoding:
    px: np.ndarray       # marginal cell masses along x (normalised)
    py: np.ndarray       # marginal cell masses along y (normalised)
    f: np.ndarray        # conditional hit probability, shape (2^n, 2^n), indexed [ix, iy]
    box_mass: float      # Gaussian mass inside the box (Z)
    edges: tuple         # (x_edges, y_edges) for plotting

    @property
    def amplitude(self):
        """The value the circuit encodes, a = sum p_i f_i."""
        return float(self.px @ self.f @ self.py)


def encode_grid(enc, n_bits=3, n_sigma=4.0):
    """Discretise an Encounter into the (p_i, f_i) needed for the circuit."""
    edges, marginals, z = [], [], 1.0
    for mu, s in zip(enc.mean, enc.sigma):
        # Box covers +/- n_sigma of the uncertainty and always contains the disk.
        lo = min(mu - n_sigma * s, -1.01 * enc.hbr)
        hi = max(mu + n_sigma * s, 1.01 * enc.hbr)
        e = np.linspace(lo, hi, 2 ** n_bits + 1)
        cdf = ndtr((e - mu) / s)
        m = np.diff(cdf)
        z *= m.sum()
        edges.append(e)
        marginals.append(m)

    # Hit mass per cell: integrate the pdf over the disk, binned by cell.
    x, y, w = disk_quadrature(enc.hbr)
    mass = (w * _gauss_pdf(x, y, enc)).ravel()
    ix = np.clip(np.searchsorted(edges[0], x.ravel()) - 1, 0, 2 ** n_bits - 1)
    iy = np.clip(np.searchsorted(edges[1], y.ravel()) - 1, 0, 2 ** n_bits - 1)
    hit = np.zeros((2 ** n_bits, 2 ** n_bits))
    np.add.at(hit, (ix, iy), mass)

    cell = np.outer(marginals[0], marginals[1])
    f = np.divide(hit, cell, out=np.zeros_like(hit), where=cell > 0)
    return GridEncoding(
        px=marginals[0] / marginals[0].sum(),
        py=marginals[1] / marginals[1].sum(),
        f=np.clip(f, 0.0, 1.0),
        box_mass=z,
        edges=tuple(edges),
    )


def build_state_preparation(grid):
    """The operator A. Qubit layout: [x register | y register | objective]."""
    n = int(np.log2(len(grid.px)))
    x, y = QuantumRegister(n, "x"), QuantumRegister(n, "y")
    obj = QuantumRegister(1, "hit")
    qc = QuantumCircuit(x, y, obj, name="A")

    qc.append(StatePreparation(np.sqrt(grid.px)), x)
    qc.append(StatePreparation(np.sqrt(grid.py)), y)

    # UCRY selects its angle by the control basis state, first control = LSB,
    # so the flat index is ix + 2^n * iy -> transpose before flattening.
    angles = 2 * np.arcsin(np.sqrt(grid.f.T.ravel()))
    qc.append(UCRYGate(list(angles)), [obj[0], *x, *y])
    return qc


def build_grover(a_circuit):
    """Q = A S_0 A^dagger S_chi, where S_chi flips the phase of |hit=1>."""
    oracle = QuantumCircuit(a_circuit.num_qubits)
    oracle.z(a_circuit.num_qubits - 1)
    return grover_operator(oracle, state_preparation=a_circuit)


def amplified_circuit(a_circuit, grover, k):
    """A followed by k Grover iterations, measuring only the objective qubit."""
    qc = QuantumCircuit(a_circuit.num_qubits)
    qc.compose(a_circuit, inplace=True)
    for _ in range(k):
        qc.compose(grover, inplace=True)
    out = ClassicalRegister(1, "c")
    qc.add_register(out)
    qc.measure(a_circuit.num_qubits - 1, out[0])
    return qc


# ---- maximum-likelihood amplitude estimation --------------------------------

def mlae_schedule(n_powers):
    """k = 0, 1, 2, 4, ..., 2^(n_powers - 2). Exponentially spaced powers are
    what give the Heisenberg-like ~1/N error scaling."""
    return [0] + [2 ** j for j in range(n_powers - 1)]


def oracle_queries(schedule, shots):
    """Each shot of A Q^k calls A (2k + 1) times. This is the cost we compare
    against one Monte Carlo sample = one call."""
    return shots * sum(2 * k + 1 for k in schedule)


def mlae_fit(schedule, hits, shots, grid_points=20_000):
    """Return the maximum-likelihood amplitude a = sin^2(theta)."""
    ks = np.asarray(schedule)[:, None]
    h = np.asarray(hits)[:, None]
    eps = 1e-15

    def nll(theta):
        p = np.sin((2 * ks + 1) * theta) ** 2
        return -np.sum(h * np.log(p + eps) + (shots - h) * np.log(1 - p + eps), axis=0)

    # The likelihood is multimodal, so do a dense scan first. Spacing must be
    # well below the period pi / (2 k_max + 1) of the highest power.
    n = max(grid_points, 40 * (2 * ks.max() + 1))
    thetas = np.linspace(0, np.pi / 2, int(n))
    best = thetas[np.argmin(nll(thetas))]
    step = thetas[1] - thetas[0]
    res = minimize_scalar(lambda t: nll(np.array([t]))[0],
                          bounds=(max(best - step, 0), min(best + step, np.pi / 2)),
                          method="bounded")
    return float(np.sin(res.x) ** 2)


def mlae_circuits(a_circuit, schedule, seed=None):
    """Build and transpile A Q^k for every k in the schedule (done once, reused)."""
    grover = build_grover(a_circuit)
    circuits = [amplified_circuit(a_circuit, grover, k) for k in schedule]
    return transpile(circuits, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=seed)


def run_mlae_qiskit(circuits, schedule, shots, sampler):
    """Full run on a Qiskit sampler (V2 primitive). Returns (estimate, hits)."""
    result = sampler.run(circuits, shots=shots).result()
    hits = [r.data.c.get_counts().get("1", 0) for r in result]
    return mlae_fit(schedule, hits, shots), hits


def run_mlae_ideal(amplitude, schedule, shots, rng=None):
    """Shot-sampled ideal MLAE without building circuits.

    On a noiseless device the objective qubit after A Q^k is exactly
    Bernoulli(sin^2((2k+1) theta)), so drawing Binomial counts reproduces a
    shot-based ideal simulator. Used for large sweeps; the notebook checks it
    against run_mlae_qiskit.
    """
    rng = np.random.default_rng(rng)
    theta = np.arcsin(np.sqrt(amplitude))
    p = np.sin((2 * np.asarray(schedule) + 1) * theta) ** 2
    hits = rng.binomial(shots, p)
    return mlae_fit(schedule, hits, shots), hits
