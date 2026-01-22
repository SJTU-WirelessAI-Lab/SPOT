import os
import numpy as np

import functions_CBS as fcbs


# =========================
# Paths (relative)
# =========================
CHANNEL_DIR = os.path.join(fcbs.BASE_DIR, "channels")
USER_DIR = os.path.join(fcbs.BASE_DIR, "user_data")
PARAM_DIR = os.path.join(fcbs.BASE_DIR, "params")


def mode_tag(mode: str, arch: str, dis_min: int, dis_max: int) -> str:
    """Build the filename tag consistent with channel_generation.py."""
    return "{}{}_{:d}to{:d}".format(mode, arch, dis_min, dis_max)


def percentile_err(arr, p=95):
    return float(np.percentile(np.asarray(arr).reshape(-1), p))


def main():
    # -------------------------
    # Experiment config (test only)
    # -------------------------
    arch = "_ULA"
    dis_min, dis_max = 5, 300
    K = 1  # CBS baseline here assumes single-user channel file (K=1)

    tag = mode_tag("test", arch, dis_min, dis_max)

    # -------------------------
    # System params
    # -------------------------
    param_path = os.path.join(PARAM_DIR, "params{}.npz".format(arch))
    params = fcbs.load_system_params(param_path)

    N_az = params["N_az"]
    N_el = params["N_el"]
    M = params["M"]
    fc = params["fc"]
    BW = params["BW"]
    f_scs = params["f_scs"]
    d = params["d"]
    user_height = params["user_height"]
    BS_height = params["BS_height"]

    print("[Info] BS_height={}, user_height={}".format(BS_height, user_height))

    # subcarrier frequencies (absolute, Hz)
    fm_list = fcbs.build_subcarrier_frequencies(fc, f_scs, M)

    # -------------------------
    # Load test data
    # -------------------------
    channel_path = os.path.join(CHANNEL_DIR, "channel_K{}_{}.npz".format(K, tag))
    user_path = os.path.join(USER_DIR, "user_data_{}.npz".format(tag))

    if not os.path.exists(channel_path):
        raise FileNotFoundError("Channel file not found: {}".format(channel_path))
    if not os.path.exists(user_path):
        raise FileNotFoundError("User data not found: {}".format(user_path))

    ch = np.load(channel_path)
    H = ch["H"]
    # expected: (num, K, M, N) when saved; for K=1 we squeeze to (num, M, N)
    if H.ndim == 4:
        H = H.squeeze(axis=1)
    H = np.asarray(H)  # (num, M, N)

    ud = np.load(user_path)
    phi_gt = ud["phi"]  # radians, shape (num, K) or (num, 1)
    r_gt = ud["r"]      # meters, shape (num, K) or (num, 1)
    x_gt = ud["x"]
    y_gt = ud["y"]

    # squeeze to (num, 1)
    phi_gt = phi_gt.reshape(-1, 1)
    r_gt = r_gt.reshape(-1, 1)
    x_gt = x_gt.reshape(-1, 1)
    y_gt = y_gt.reshape(-1, 1)

    # -------------------------
    # Step-1: CBS rainbow beam focusing vector (fixed)
    # -------------------------
    # NOTE: these are CBS control points; you can adjust as needed.
    TTD, PS, phi_traj_rad, r_traj = fcbs.generate_beamfocusing_vector_CBS(
        Nt=N_az, M=M, BW=BW, d=d, f=fm_list,
        r0=200.0, theta0=np.pi / 3,  # start control point
        rc=200.0, thetac=-np.pi / 3  # end control point
    )

    # -------------------------
    # Step-2: Angle estimation from peak subcarrier index
    # -------------------------
    max_val_db, max_idx, Y = fcbs.received_signal_argmax_idx(
        BW=BW, H=H, PS=PS, TTD=TTD, fm_list=fm_list
    )

    # phi_traj_rad: length M, radians
    phi_est_deg = np.rad2deg(phi_traj_rad[max_idx]).reshape(-1, 1)
    phi_gt_deg = np.rad2deg(phi_gt)

    ang_rmse = np.sqrt(np.mean((phi_est_deg - phi_gt_deg) ** 2))
    print("[Test] Angle RMSE (deg): {:.6f}".format(float(ang_rmse)))

    # -------------------------
    # Step-3: Distance estimation (batch rainbow beam search)
    # -------------------------
    _, r_est, _ = fcbs.rainbow_estimation_distance_CBS(
        BW=BW, H=H, N_az=N_az, N_el=N_el, d=d, fm_list=fm_list,
        phi_est_deg=phi_est_deg,
        r_init=(dis_max + dis_min) / 2.0,
        delta_phi_deg=0.0,
        delta_r=(dis_max - dis_min) / 2.0
    )

    r_est = r_est.reshape(-1, 1)
    dist_rmse = np.sqrt(np.mean((r_est - r_gt) ** 2))
    print("[Test] Distance RMSE (m): {:.6f}".format(float(dist_rmse)))

    # -------------------------
    # Step-4: 2D position RMSE
    # -------------------------
    x_est = r_est * np.cos(np.deg2rad(phi_est_deg))
    y_est = r_est * np.sin(np.deg2rad(phi_est_deg))

    rmse_2d = np.sqrt(np.mean((x_est - x_gt) ** 2 + (y_est - y_gt) ** 2))
    print("[Test] 2D RMSE (m): {:.6f}".format(float(rmse_2d)))

    # Optional diagnostics
    e2d = np.sqrt((x_est - x_gt) ** 2 + (y_est - y_gt) ** 2).reshape(-1)
    print("[Test] 2D error p95 (m): {:.6f}".format(percentile_err(e2d, 95)))


if __name__ == "__main__":
    main()
