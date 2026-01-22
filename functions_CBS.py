import os
import math
import numpy as np


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ISAC_data")


# =========================
# System params
# =========================
def load_system_params(param_file: str):
    """
    Load system parameters saved by channel_generation.py.
    """
    data = np.load(param_file)

    # allow both "BW" and "B"
    if "BW" in data:
        BW = float(data["BW"])
    else:
        BW = float(data["B"])

    params = {
        "D_rayleigh": float(np.asarray(data["D_rayleigh"]).reshape(())),
        "fc": float(np.asarray(data["fc"]).reshape(())),
        "BW": BW,
        "N_el": int(np.asarray(data["N_el"]).reshape(())),
        "N_az": int(np.asarray(data["N_az"]).reshape(())),
        "f_scs": float(np.asarray(data["f_scs"]).reshape(())),
        "Delta_T": float(np.asarray(data["Delta_T"]).reshape(())),
        "M": int(np.asarray(data["M"]).reshape(())),
        "K": int(np.asarray(data["K"]).reshape(())),
        "user_height": float(np.asarray(data["user_height"]).reshape(())),
        "BS_height": float(np.asarray(data["BS_height"]).reshape(())),
        "d": float(np.asarray(data["d"]).reshape(())),
    }

    print("[Info] Loaded system params from: {}".format(param_file))
    print("[Info] N_az={}, N_el={}, M={}, fc={}, BW={}, K={}".format(
        params["N_az"], params["N_el"], params["M"], params["fc"], params["BW"], params["K"]
    ))
    return params


def build_subcarrier_frequencies(fc: float, f_scs: float, M: int) -> np.ndarray:
    return fc + f_scs * (np.arange(M) - (M - 1) / 2.0)


# =========================
# Rx signal + argmax
# =========================
def received_signal_argmax_idx(BW: float, H: np.ndarray, PS: np.ndarray, TTD: np.ndarray, fm_list: np.ndarray):
    """
    Simulate received signal and return peak index (hard argmax).
    """
    # Thermal noise (per-subcarrier)
    k_B = 1.380649e-23
    T_sys = 290.0
    num, M, N = H.shape

    noise_power = k_B * T_sys * BW * 1e3 / M
    noise_std = math.sqrt(noise_power / 2.0)

    f0 = fm_list[0]
    fm_rel = (fm_list - f0).reshape(1, M, 1, 1)  # (1,M,1,1)

    H_H = np.conj(H).reshape(num, M, 1, N)  # (num,M,1,N)

    # BF: exp(j*(PS - 2*pi*fm_rel*TTD))
    BF = np.exp(1j * (PS - 2.0 * np.pi * fm_rel * TTD))  # broadcast to (num,M,N,1)
    Y = math.sqrt(1e4 / M) * (H_H @ BF) / math.sqrt(N)    # (num,M,1,1)
    Y = Y.squeeze(-1).squeeze(-1)                         # (num,M)

    noise = (np.random.randn(*Y.shape) + 1j * np.random.randn(*Y.shape)) * noise_std
    Y = Y + noise

    mag_sq = np.abs(Y) ** 2
    mag_db = 10.0 * np.log10(mag_sq + 1e-30)

    max_val_db = np.max(mag_db, axis=-1)
    max_idx = np.argmax(mag_db, axis=-1)
    return max_val_db, max_idx, Y


# =========================
# CBS rainbow beam (baseline)
# =========================
def beam_squint_trajectory(BW: float, M: int, f: np.ndarray, theta0: float, r0: float, thetac: float, rc: float):
    """
    Compute beam-squint trajectory (theta_m, r_m) across subcarriers.
    """
    theta_m = np.zeros(M, dtype=float)
    r_m = np.zeros(M, dtype=float)

    f0 = f[0]
    for m in range(M):
        fm = f[m]
        theta_m[m] = np.arcsin(
            (BW - (fm - f0)) * f0 / BW / fm * np.sin(theta0)
            + (BW + f0) * (fm - f0) / BW / fm * np.sin(thetac)
        )
        r_m[m] = 1.0 / (
            (1.0 / r0) * (BW - (fm - f0)) * f0 / BW / fm * (np.cos(theta0) ** 2) / (np.cos(theta_m[m]) ** 2)
            + (1.0 / rc) * (BW + f0) * (fm - f0) / BW / fm * (np.cos(thetac) ** 2) / (np.cos(theta_m[m]) ** 2)
        )
    return theta_m, r_m


def generate_beamfocusing_vector_CBS(Nt: int, M: int, BW: float, d: float, f: np.ndarray,
                                     r0: float, theta0: float, rc: float, thetac: float):
    """
    Generate CBS beam focusing vector (PS, TTD) for a ULA of Nt antennas.
    """
    c = 3e8
    nn = np.arange(-(Nt - 1) / 2.0, (Nt - 1) / 2.0 + 1.0)  # antenna index centered

    # distances to control points
    rr = np.sqrt(r0 ** 2 + (nn * d) ** 2 - 2.0 * r0 * nn * d * np.sin(theta0))
    rrc = np.sqrt(rc ** 2 + (nn * d) ** 2 - 2.0 * rc * nn * d * np.sin(thetac))

    phi = np.zeros(Nt, dtype=float)
    t = np.zeros(Nt, dtype=float)

    f0 = f[0]
    fM = f[M - 1]

    # Per-antenna PS/TTD
    for n in range(Nt):
        phi[n] = f0 / c * rr[n]
        t[n] = fM / BW / c * rrc[n] - phi[n] / BW

    theta_traj, r_traj = beam_squint_trajectory(BW, M, f, theta0, r0, thetac, rc)

    TTD = t.reshape(1, 1, Nt, 1)
    PS = (-2.0 * np.pi * phi).reshape(1, 1, Nt, 1)
    return TTD, PS, theta_traj, r_traj


def rainbow_beam_batch_CBS(N_az: int, d: float, fm_list: np.ndarray,
                           phi_est_deg: np.ndarray, r_init: float,
                           delta_phi_deg: float, delta_r: float):
    """
    Per-sample CBS rainbow beam for distance search.
    """
    c = 3e8
    M = len(fm_list)

    f0 = fm_list[0]
    fM = fm_list[-1]
    BW = fM - f0

    num = phi_est_deg.shape[0]

    # endpoints (in radians)
    phi0 = np.deg2rad((phi_est_deg.reshape(-1) - delta_phi_deg))
    phi1 = np.deg2rad((phi_est_deg.reshape(-1) + delta_phi_deg))
    r0 = (r_init - delta_r) * np.ones(num, dtype=float)
    r1 = (r_init + delta_r) * np.ones(num, dtype=float)

    nn = np.arange(-(N_az - 1) / 2.0, (N_az - 1) / 2.0 + 1.0)

    phi = np.zeros((num, N_az), dtype=float)
    t = np.zeros((num, N_az), dtype=float)

    phi_list_rad = np.zeros((num, M), dtype=float)
    r_list = np.zeros((num, M), dtype=float)

    # Build per-sample PS/TTD from endpoint constraints + trajectory
    for i in range(num):
        rr = np.sqrt(r0[i] ** 2 + (nn * d) ** 2 - 2.0 * r0[i] * nn * d * np.sin(phi0[i]))
        rrc = np.sqrt(r1[i] ** 2 + (nn * d) ** 2 - 2.0 * r1[i] * nn * d * np.sin(phi1[i]))

        for n in range(N_az):
            phi[i, n] = f0 / c * rr[n]
            t[i, n] = fM / BW / c * rrc[n] - phi[i, n] / BW

        # per-subcarrier trajectory (phi_m, r_m)
        for m in range(M):
            fm = fm_list[m]
            phi_list_rad[i, m] = np.arcsin(
                (BW - (fm - f0)) * f0 / BW / fm * np.sin(phi0[i])
                + (BW + f0) * (fm - f0) / BW / fm * np.sin(phi1[i])
            )
            r_list[i, m] = 1.0 / (
                (1.0 / r0[i]) * (BW - (fm - f0)) * f0 / BW / fm * (np.cos(phi0[i]) ** 2) / (np.cos(phi_list_rad[i, m]) ** 2)
                + (1.0 / r1[i]) * (BW + f0) * (fm - f0) / BW / fm * (np.cos(phi1[i]) ** 2) / (np.cos(phi_list_rad[i, m]) ** 2)
            )

    PS = (-2.0 * np.pi * phi).reshape(num, 1, N_az, 1)
    TTD = t.reshape(num, 1, N_az, 1)
    return TTD, PS, phi_list_rad, r_list


def rainbow_estimation_distance_CBS(BW: float, H: np.ndarray, N_az: int, N_el: int, d: float, fm_list: np.ndarray,
                                    phi_est_deg: np.ndarray, r_init: float, delta_phi_deg: float, delta_r: float):
    """
    Distance estimation using batch-designed Gao-style rainbow beams.
    """
    TTD, PS, phi_list_rad, r_list = rainbow_beam_batch_CBS(
        N_az=N_az, d=d, fm_list=fm_list,
        phi_est_deg=phi_est_deg, r_init=r_init,
        delta_phi_deg=delta_phi_deg, delta_r=delta_r
    )

    _, max_idx, Y = received_signal_argmax_idx(BW=BW, H=H, PS=PS, TTD=TTD, fm_list=fm_list)
    r_est = r_list[np.arange(max_idx.shape[0]), max_idx]
    return phi_est_deg, r_est, Y
