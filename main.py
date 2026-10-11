"""
Q-Avoid: quantum amplitude estimation for satellite collision risk.

Self-contained version of the notebook (q_avoid.ipynb). Runs the core
experiments and prints the results:

  1. build the 7-qubit circuit whose hit-qubit probability equals Pc
  2. run maximum-likelihood amplitude estimation (MLAE) on a Qiskit sampler
  3. error vs. cost: MLAE vs Monte Carlo at the same number of oracle calls
  4. cost of a 10%-accurate estimate as events get rarer
  5. (if the ESA dataset is present) maneuver-decision accuracy over real events

Without the dataset it uses a real ESA conjunction (event 12555) built in below.
Dataset: https://zenodo.org/records/4463683 -> data/train_data.csv

    python main.py              # full run, ~1-2 min without the dataset
    python main.py --quick      # fewer repetitions
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate, grover_operator
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

HBR = 10.0           # combined hard-body radius (m); not given in the dataset
THRESHOLD = 1e-4     # maneuver if Pc >= this
SHOTS = 100          # shots per MLAE circuit
DATA_PATH = "data/train_data.csv"


# ---------------------------------------------------------------------------
# Encounter geometry
# ---------------------------------------------------------------------------

@dataclass
class Encounter:
    """Conjunction in the encounter plane, in covariance principal axes (metres)."""
    event_id: int
    mean: np.ndarray
    sigma: np.ndarray
    hbr: float = HBR
    esa_log_risk: float = np.nan


# Last CDM of ESA Kelvins event 12555 (ESA risk 3.4e-4), already projected.
SHOWCASE = Encounter(
    event_id=12555,
    mean=np.array([-37.66286738, -545.44658324]),
    sigma=np.array([81.25517471, 1747.82296127]),
    esa_log_risk=-3.4710,
)

KELVINS_COLUMNS = [
    "event_id", "time_to_tca", "risk",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "t_ct_r", "t_cn_r", "t_cn_t",
    "c_sigma_r", "c_sigma_t", "c_sigma_n", "c_ct_r", "c_cn_r", "c_cn_t",
]


def _rtn_covariance(row, prefix):
    s = np.array([row[f"{prefix}_sigma_r"], row[f"{prefix}_sigma_t"], row[f"{prefix}_sigma_n"]])
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = row[f"{prefix}_ct_r"]
    corr[0, 2] = corr[2, 0] = row[f"{prefix}_cn_r"]
    corr[1, 2] = corr[2, 1] = row[f"{prefix}_cn_t"]
    return corr * np.outer(s, s)


def encounter_from_row(row):
    """Project a CDM onto the plane normal to the relative velocity.

    Both covariances are treated as if in the target's RTN frame -- the
    anonymised dataset has no chaser state to rotate with.
    """
    r = np.array([row["relative_position_r"], row["relative_position_t"], row["relative_position_n"]])
    v = np.array([row["relative_velocity_r"], row["relative_velocity_t"], row["relative_velocity_n"]])
    cov = _rtn_covariance(row, "t") + _rtn_covariance(row, "c")

    z = v / np.linalg.norm(v)
    x = r - (r @ z) * z
    x /= np.linalg.norm(x)
    P = np.vstack([x, np.cross(z, x)])

    eigval, eigvec = np.linalg.eigh(P @ cov @ P.T)
    return Encounter(int(row["event_id"]), eigvec.T @ (P @ r),
                     np.sqrt(np.maximum(eigval, 1e-12)), HBR, float(row["risk"]))


def load_kelvins(path, min_log_risk=-6.0):
    """Last CDM per event, keeping events ESA rated above 1e-6."""
    import pandas as pd
    df = pd.read_csv(path, usecols=KELVINS_COLUMNS)
    last = df.sort_values("time_to_tca").groupby("event_id").head(1)
    last = last[last["risk"] > min_log_risk].dropna()
    return [encounter_from_row(row) for _, row in last.iterrows()]


# ---------------------------------------------------------------------------
# Classical baseline
# ---------------------------------------------------------------------------

def gauss_pdf(x, y, enc):
    (mx, my), (sx, sy) = enc.mean, enc.sigma
    return np.exp(-0.5 * (((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2)) / (2 * np.pi * sx * sy)


def disk_quadrature(hbr, n_r=48, n_theta=96):
    """Gauss-Legendre in radius, uniform in angle, over the hard-body disk."""
    nodes, w = np.polynomial.legendre.leggauss(n_r)
    r, w = 0.5 * hbr * (nodes + 1), 0.5 * hbr * w
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(r, theta, indexing="ij")
    W = np.outer(w * r, np.full(n_theta, 2 * np.pi / n_theta))
    return R * np.cos(T), R * np.sin(T), W


def pc_reference(enc):
    """Ground-truth Pc by integrating the Gaussian over the disk."""
    x, y, w = disk_quadrature(enc.hbr)
    return float(np.sum(w * gauss_pdf(x, y, enc)))


def pc_monte_carlo(enc, n, rng):
    """Plain Monte Carlo: sample relative positions, count hits."""
    pts = enc.mean + enc.sigma * rng.standard_normal((n, 2))
    return np.count_nonzero(np.einsum("ij,ij->i", pts, pts) < enc.hbr ** 2) / n


def pc_monte_carlo_fast(pc, n, rng, size):
    """Same distribution as pc_monte_carlo: the hit count is Binomial(n, Pc)."""
    return rng.binomial(n, pc, size=size) / n


# ---------------------------------------------------------------------------
# Quantum encoding
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    px: np.ndarray       # normalised cell masses along x
    py: np.ndarray       # normalised cell masses along y
    f: np.ndarray        # P(hit | cell), indexed [ix, iy]
    box_mass: float      # Gaussian mass inside the grid box

    @property
    def amplitude(self):
        return float(self.px @ self.f @ self.py)


def encode_grid(enc, n_bits=3, n_sigma=4.0):
    """Cut the plane into a 2^n x 2^n grid and compute p_i and f_i.

    f_i is the exact conditional hit probability of cell i, so
    sum_i p_i f_i equals Pc with no discretisation bias.
    """
    edges, marginals, z = [], [], 1.0
    for mu, s in zip(enc.mean, enc.sigma):
        lo = min(mu - n_sigma * s, -1.01 * enc.hbr)
        hi = max(mu + n_sigma * s, 1.01 * enc.hbr)
        e = np.linspace(lo, hi, 2 ** n_bits + 1)
        m = np.diff(ndtr((e - mu) / s))
        z *= m.sum()
        edges.append(e)
        marginals.append(m)

    x, y, w = disk_quadrature(enc.hbr)
    mass = (w * gauss_pdf(x, y, enc)).ravel()
    ix = np.clip(np.searchsorted(edges[0], x.ravel()) - 1, 0, 2 ** n_bits - 1)
    iy = np.clip(np.searchsorted(edges[1], y.ravel()) - 1, 0, 2 ** n_bits - 1)
    hit = np.zeros((2 ** n_bits, 2 ** n_bits))
    np.add.at(hit, (ix, iy), mass)

    cell = np.outer(*marginals)
    f = np.divide(hit, cell, out=np.zeros_like(hit), where=cell > 0)
    return Grid(marginals[0] / marginals[0].sum(), marginals[1] / marginals[1].sum(),
                np.clip(f, 0, 1), z)


def build_state_preparation(grid):
    """A|0> = sum_i sqrt(p_i)|i>(sqrt(1-f_i)|0> + sqrt(f_i)|1>).

    Qubits: [x register | y register | hit]. The Gaussian factorises in
    principal axes, so x and y are prepared independently.
    """
    n = int(np.log2(len(grid.px)))
    x, y, hit = QuantumRegister(n, "x"), QuantumRegister(n, "y"), QuantumRegister(1, "hit")
    qc = QuantumCircuit(x, y, hit, name="A")
    qc.append(StatePreparation(np.sqrt(grid.px)), x)
    qc.append(StatePreparation(np.sqrt(grid.py)), y)
    # UCRY picks its angle by control state with the first control as LSB,
    # i.e. flat index ix + 2^n * iy, hence the transpose.
    angles = 2 * np.arcsin(np.sqrt(grid.f.T.ravel()))
    qc.append(UCRYGate(list(angles)), [hit[0], *x, *y])
    return qc


def build_grover(a_circuit):
    """Q = A S_0 A^dagger S_chi; each application rotates by 2*theta."""
    oracle = QuantumCircuit(a_circuit.num_qubits)
    oracle.z(a_circuit.num_qubits - 1)
    return grover_operator(oracle, state_preparation=a_circuit)


def amplified_circuit(a_circuit, grover, k, measure=True):
    qc = QuantumCircuit(a_circuit.num_qubits)
    qc.compose(a_circuit, inplace=True)
    for _ in range(k):
        qc.compose(grover, inplace=True)
    if measure:
        qc.add_register(ClassicalRegister(1, "c"))
        qc.measure(a_circuit.num_qubits - 1, 0)
    return qc


# ---------------------------------------------------------------------------
# Maximum-likelihood amplitude estimation (Suzuki et al., 2020)
# ---------------------------------------------------------------------------

def mlae_schedule(n_powers):
    """k = 0, 1, 2, 4, ... -- exponential spacing gives the ~1/N scaling."""
    return [0] + [2 ** j for j in range(n_powers - 1)]


def oracle_queries(schedule, shots):
    """One shot of A Q^k calls A (2k+1) times; one MC sample calls it once."""
    return shots * sum(2 * k + 1 for k in schedule)


def mlae_fit(schedule, hits, shots):
    """Maximise the likelihood of the hit counts over theta, a = sin^2(theta)."""
    ks = np.asarray(schedule)[:, None]
    h = np.asarray(hits)[:, None]

    def nll(theta):
        p = np.sin((2 * ks + 1) * theta) ** 2
        return -np.sum(h * np.log(p + 1e-15) + (shots - h) * np.log(1 - p + 1e-15), axis=0)

    # Multimodal likelihood: dense scan first, then refine around the best point.
    thetas = np.linspace(0, np.pi / 2, int(max(20_000, 40 * (2 * ks.max() + 1))))
    best = thetas[np.argmin(nll(thetas))]
    step = thetas[1] - thetas[0]
    res = minimize_scalar(lambda t: nll(np.array([t]))[0], method="bounded",
                          bounds=(max(best - step, 0), min(best + step, np.pi / 2)))
    return float(np.sin(res.x) ** 2)


def run_mlae_qiskit(circuits, schedule, sampler):
    result = sampler.run(circuits, shots=SHOTS).result()
    hits = [r.data.c.get_counts().get("1", 0) for r in result]
    return mlae_fit(schedule, hits, SHOTS)


def run_mlae_ideal(amplitude, schedule, rng):
    """On a noiseless device the hit qubit after A Q^k is exactly
    Bernoulli(sin^2((2k+1) theta)); drawing binomial counts reproduces a
    shot-based ideal simulator. Used for the large sweeps."""
    theta = np.arcsin(np.sqrt(amplitude))
    hits = rng.binomial(SHOTS, np.sin((2 * np.asarray(schedule) + 1) * theta) ** 2)
    return mlae_fit(schedule, hits, SHOTS)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def rel_rmse(estimates, truth):
    return np.sqrt(np.mean((np.asarray(estimates) - truth) ** 2)) / truth


def check_circuit(enc, grid, rng):
    pc = pc_reference(enc)
    A = build_state_preparation(grid)
    p_hit = Statevector(A).probabilities([A.num_qubits - 1])[1] * grid.box_mass
    print(f"Reference Pc (integral):        {pc:.6e}")
    print(f"Circuit P(hit = 1), statevector: {p_hit:.6e}")

    # Grover amplification should follow sin^2((2k+1) theta)
    G = build_grover(A)
    theta = np.arcsin(np.sqrt(grid.amplitude))
    for k in (1, 8, 32):
        sv = Statevector(amplified_circuit(A, G, k, measure=False))
        print(f"  k = {k:2d}: P(hit) = {sv.probabilities([A.num_qubits - 1])[1]:.5f}"
              f"   theory {np.sin((2 * k + 1) * theta) ** 2:.5f}")

    mc = pc_monte_carlo(enc, 100_000, rng)
    print(f"Monte Carlo, 100,000 samples:    {mc:.3e}")
    return A, pc


def qiskit_mlae(A, grid, pc, rng, runs):
    schedule = mlae_schedule(7)
    circuits = transpile([amplified_circuit(A, build_grover(A), k) for k in schedule],
                         basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=7)
    sampler = StatevectorSampler(seed=np.random.default_rng(11))
    est = np.array([run_mlae_qiskit(circuits, schedule, sampler) for _ in range(runs)]) * grid.box_mass
    ideal = np.array([run_mlae_ideal(grid.amplitude, schedule, rng) for _ in range(200)]) * grid.box_mass

    print(f"schedule k = {schedule}, {oracle_queries(schedule, SHOTS):,} oracle calls per estimate,"
          f" deepest circuit {circuits[-1].count_ops().get('cx', 0)} CNOTs")
    print(f"Qiskit sampler ({runs} runs):  mean {est.mean():.3e}, rel. RMSE {rel_rmse(est, pc):.3f}")
    print(f"Ideal fast path (200 runs): mean {ideal.mean():.3e}, rel. RMSE {rel_rmse(ideal, pc):.3f}")


def error_vs_queries(grid, pc, rng, reps):
    rows = []
    print(f"{'oracle calls':>13} {'Monte Carlo':>12} {'MLAE':>8}")
    for m in range(1, 12):
        sch = mlae_schedule(m)
        n = oracle_queries(sch, SHOTS)
        q = np.array([run_mlae_ideal(grid.amplitude, sch, rng) for _ in range(reps)]) * grid.box_mass
        c = pc_monte_carlo_fast(pc, n, rng, reps)
        rows.append((n, rel_rmse(c, pc), rel_rmse(q, pc)))
        print(f"{n:>13,} {rows[-1][1]:>12.3f} {rows[-1][2]:>8.3f}")

    rows = np.array(rows)
    slope = lambda col: np.polyfit(np.log10(rows[3:, 0]), np.log10(rows[3:, col]), 1)[0]
    print(f"fitted slope: Monte Carlo {slope(1):.2f} (theory -0.5), MLAE {slope(2):.2f} (theory -1)")
    return rows


def cost_vs_rarity(rng, reps, target=0.1):
    pcs = np.logspace(-2, -6, 5)
    rows = []
    print(f"{'Pc':>7} {'MC calls':>13} {'MLAE calls':>11} {'saving':>7}")
    for a in pcs:
        mc = (1 - a) / (a * target ** 2)
        qae = np.nan
        for m in range(1, 16):
            sch = mlae_schedule(m)
            if rel_rmse([run_mlae_ideal(a, sch, rng) for _ in range(reps)], a) < target:
                qae = oracle_queries(sch, SHOTS)
                break
        rows.append((a, mc, qae))
        print(f"{a:>7.0e} {mc:>13,.0f} {qae:>11,.0f} {mc / qae:>6.0f}x")
    return np.array(rows)


def decision_accuracy(encounters, rng, reps, n_powers=6):
    sch = mlae_schedule(n_powers)
    budget = oracle_queries(sch, SHOTS)
    pcs = np.array([pc_reference(e) for e in encounters])
    keep = (pcs > 1e-6) & (pcs < 1e-2)

    res = []
    for e, pc in zip([e for e, k in zip(encounters, keep) if k], pcs[keep]):
        g = encode_grid(e)
        q = np.array([run_mlae_ideal(g.amplitude, sch, rng) for _ in range(reps)]) * g.box_mass
        c = pc_monte_carlo_fast(pc, budget, rng, reps)
        truth = pc >= THRESHOLD
        res.append((pc, np.mean((c >= THRESHOLD) == truth), np.mean((q >= THRESHOLD) == truth)))
    res = np.array(res)
    near = (res[:, 0] > THRESHOLD / 5) & (res[:, 0] < THRESHOLD * 5)

    print(f"{len(res)} events, {budget:,} oracle calls per estimate")
    print(f"correct maneuver calls:            Monte Carlo {res[:, 1].mean():.1%}   MLAE {res[:, 2].mean():.1%}")
    print(f"within 5x of threshold (n={near.sum()}):    Monte Carlo {res[near, 1].mean():.1%}"
          f"   MLAE {res[near, 2].mean():.1%}")
    return res


def save_figures(err, cost, pc, folder="figures"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(err[:, 0], err[:, 1], "o-", color="#d1495b", label="Monte Carlo")
    ax.loglog(err[:, 0], err[:, 2], "s-", color="#2e86ab", label="Quantum MLAE")
    ax.set(xlabel="oracle calls", ylabel="relative RMSE of Pc", title=f"Error vs cost, Pc = {pc:.1e}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{folder}/main_error_vs_queries.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(cost[:, 0], cost[:, 1], "o-", color="#d1495b", label="Monte Carlo")
    ax.loglog(cost[:, 0], cost[:, 2], "s-", color="#2e86ab", label="Quantum MLAE")
    ax.invert_xaxis()
    ax.set(xlabel="collision probability", ylabel="oracle calls for 10% error", title="Cost vs rarity")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{folder}/main_cost_vs_rarity.png", dpi=120)
    print(f"figures saved to {folder}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="fewer repetitions")
    parser.add_argument("--data", default=DATA_PATH, help="path to Kelvins train_data.csv")
    args = parser.parse_args()
    reps = 60 if args.quick else 200
    rng = np.random.default_rng(2026)

    encounters = None
    if os.path.exists(args.data):
        encounters = load_kelvins(args.data)
        print(f"Loaded {len(encounters)} ESA events with risk > 1e-6")
    else:
        print(f"No dataset at {args.data}; using built-in ESA event {SHOWCASE.event_id}")

    enc = SHOWCASE
    grid = encode_grid(enc)

    print("\n== 1. Circuit check ==")
    A, pc = check_circuit(enc, grid, rng)

    print("\n== 2. MLAE on the Qiskit sampler ==")
    qiskit_mlae(A, grid, pc, rng, runs=10 if args.quick else 30)

    print("\n== 3. Error vs cost (equal oracle calls) ==")
    err = error_vs_queries(grid, pc, rng, reps)

    print("\n== 4. Calls needed for a 10% accurate Pc ==")
    cost = cost_vs_rarity(rng, reps)

    if encounters is not None:
        print("\n== 5. Maneuver decisions over real events ==")
        decision_accuracy(encounters, rng, reps=min(reps, 100))

    save_figures(err, cost, pc)


if __name__ == "__main__":
    main()
