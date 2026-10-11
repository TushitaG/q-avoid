"""
Loading ESA Kelvins conjunction data and reducing each event to a 2D
encounter-plane problem.

Short-term encounter assumption: near TCA the relative motion is a straight
line, so collision probability is the mass of a 2D Gaussian (relative position
uncertainty projected onto the plane normal to the relative velocity) that
falls inside a disk of radius R (the combined hard-body radius).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

KELVINS_COLUMNS = [
    "event_id", "time_to_tca", "mission_id", "risk", "miss_distance",
    "relative_speed", "c_object_type",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "t_ct_r", "t_cn_r", "t_cn_t",
    "c_sigma_r", "c_sigma_t", "c_sigma_n", "c_ct_r", "c_cn_r", "c_cn_t",
]


@dataclass
class Encounter:
    """A conjunction reduced to the encounter plane, in principal axes.

    mean  -- miss vector (m), expressed in the covariance principal frame
    sigma -- 1-sigma along each principal axis (m)
    hbr   -- combined hard-body radius (m)
    """
    event_id: int
    mean: np.ndarray
    sigma: np.ndarray
    hbr: float
    esa_log_risk: float = np.nan
    miss_distance: float = np.nan


def load_kelvins(path, min_log_risk=-6.0):
    """Return the latest CDM of every event whose ESA risk is above a floor.

    The final CDM before TCA is what an operator actually decides on, so we
    keep one row per event. Most events sit at the -30 floor (no risk) and are
    not interesting for a maneuver decision.
    """
    df = pd.read_csv(path, usecols=KELVINS_COLUMNS)
    last = df.sort_values("time_to_tca").groupby("event_id").head(1)
    last = last[last["risk"] > min_log_risk]
    return last.dropna(subset=KELVINS_COLUMNS[7:]).reset_index(drop=True)


def _rtn_covariance(row, prefix):
    s = np.array([row[f"{prefix}_sigma_r"], row[f"{prefix}_sigma_t"], row[f"{prefix}_sigma_n"]])
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = row[f"{prefix}_ct_r"]
    corr[0, 2] = corr[2, 0] = row[f"{prefix}_cn_r"]
    corr[1, 2] = corr[2, 1] = row[f"{prefix}_cn_t"]
    return corr * np.outer(s, s)


def encounter_from_row(row, hbr=10.0):
    """Project the RTN miss vector and combined covariance onto the encounter plane.

    Simplification: both covariances are treated as if expressed in the target's
    RTN frame. The dataset is anonymised and has no chaser state vector, so we
    cannot rotate the chaser covariance properly.
    """
    r = np.array([row["relative_position_r"], row["relative_position_t"], row["relative_position_n"]])
    v = np.array([row["relative_velocity_r"], row["relative_velocity_t"], row["relative_velocity_n"]])
    cov = _rtn_covariance(row, "t") + _rtn_covariance(row, "c")

    # Encounter-plane basis: z along relative velocity, x along the projected miss.
    z = v / np.linalg.norm(v)
    r_perp = r - (r @ z) * z
    x = r_perp / np.linalg.norm(r_perp)
    y = np.cross(z, x)
    P = np.vstack([x, y])

    mean_2d = P @ r
    cov_2d = P @ cov @ P.T

    # Rotate into principal axes so the Gaussian factorises per axis. The disk is
    # centred at the origin, so it is unaffected by the rotation.
    eigval, eigvec = np.linalg.eigh(cov_2d)
    return Encounter(
        event_id=int(row["event_id"]),
        mean=eigvec.T @ mean_2d,
        sigma=np.sqrt(np.maximum(eigval, 1e-12)),
        hbr=hbr,
        esa_log_risk=float(row["risk"]),
        miss_distance=float(row["miss_distance"]),
    )
