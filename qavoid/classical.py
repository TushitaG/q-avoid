"""Classical collision-probability estimators: reference integral and Monte Carlo."""

import numpy as np


def _gauss_pdf(x, y, enc):
    (mx, my), (sx, sy) = enc.mean, enc.sigma
    return np.exp(-0.5 * (((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2)) / (2 * np.pi * sx * sy)


def disk_quadrature(hbr, n_r=48, n_theta=96):
    """Gauss-Legendre in radius, uniform in angle. Returns points and weights."""
    r_nodes, r_w = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * hbr * (r_nodes + 1)
    r_w = 0.5 * hbr * r_w
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(r, theta, indexing="ij")
    W = np.outer(r_w * r, np.full(n_theta, 2 * np.pi / n_theta))
    return R * np.cos(T), R * np.sin(T), W


def pc_reference(enc):
    """Collision probability by numerical integration over the hard-body disk.

    This is our ground truth. In 2D it's cheap; the point of the project is the
    estimation stage, which is what scales badly in higher-fidelity models
    (non-linear motion, non-Gaussian uncertainty) where only sampling works.
    """
    x, y, w = disk_quadrature(enc.hbr)
    return float(np.sum(w * _gauss_pdf(x, y, enc)))


def pc_monte_carlo(enc, n_samples, rng=None, chunk=1_000_000):
    """Plain Monte Carlo: draw relative positions, count hits inside the disk."""
    rng = np.random.default_rng(rng)
    hits, left = 0, n_samples
    while left > 0:
        m = min(chunk, left)
        pts = enc.mean + enc.sigma * rng.standard_normal((m, 2))
        hits += int(np.count_nonzero(np.einsum("ij,ij->i", pts, pts) < enc.hbr ** 2))
        left -= m
    return hits / n_samples


def pc_monte_carlo_fast(pc_true, n_samples, rng=None, size=None):
    """Statistically identical shortcut for sweeps.

    The hit count of n independent samples is exactly Binomial(n, Pc), so for
    large parameter sweeps we draw that count directly instead of generating
    every sample. pc_monte_carlo() is used in the notebook to show they agree.
    """
    rng = np.random.default_rng(rng)
    return rng.binomial(n_samples, pc_true, size=size) / n_samples
