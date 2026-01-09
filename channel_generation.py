import os
import numpy as np


# =========================
# Global config
# =========================
ARCH = "_ULA"          # "_ULA" or "_UPA" (this script currently uses ULA-style steering)
DIS_MIN = 5
DIS_MAX = 300

# System parameters
BS_HEIGHT = 1.5
USER_HEIGHT = 1.5
N_AZ = 256
N_EL = 1
K = 1

C = 3e8
FC = 28e9
F_SCS = 240e3
M = 1584
BANDWIDTH = F_SCS * M

# Dataset sizes (samples, not users)
SPLIT_SAMPLES = {
    "train": 5000,
    "val": 100,
    "test": 100,
}

# Train chunking (in samples). Each chunk writes one file for channels.
TRAIN_CHUNK_SAMPLES = 10000


# =========================
# Paths (relative)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "ISAC_data")
CHANNEL_DIR = os.path.join(BASE_DIR, "channels")
USER_DIR = os.path.join(BASE_DIR, "user_data")
PARAM_DIR = os.path.join(BASE_DIR, "params")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def tag(split: str) -> str:
    """Build a consistent tag for file naming."""
    return "{}{}_{:d}to{:d}".format(split, ARCH, DIS_MIN, DIS_MAX)


def compute_rayleigh_distance(fc, n_az, d):
    """Compute Rayleigh distance for a ULA aperture."""
    lam = C / fc
    D = (n_az - 1) * d
    return 2.0 * D * D / lam


def build_fm_list(fc, f_scs, M):
    """Build subcarrier center-frequency list."""
    return fc + f_scs * (np.arange(M) - (M - 1) / 2.0)


def generate_user_positions(num_users: int, dis_min: float, dis_max: float,
                            bs_height: float, user_height: float,
                            phi_min_deg: float = -60.0, phi_max_deg: float = 60.0):
    """Generate random user positions in 2D (x,y) with fixed height difference."""
    u = np.random.rand(num_users)
    v = np.random.rand(num_users)

    # Uniform in r^2 to ensure uniform area distribution
    r2d = np.sqrt(u * (dis_max ** 2 - dis_min ** 2) + dis_min ** 2)
    angle_deg = phi_min_deg + (phi_max_deg - phi_min_deg) * v
    phi = np.deg2rad(angle_deg)

    x = r2d * np.cos(phi)
    y = r2d * np.sin(phi)
    z = np.full(num_users, user_height - bs_height)

    # Note: theta definition preserved from your original code style
    theta = np.arctan2(r2d, (user_height - bs_height))
    r_3d = np.sqrt(r2d ** 2 + (bs_height - user_height) ** 2)

    return x, y, z, phi, theta, r_3d


def compute_channel_chunk(phi, theta, r, fm_list, c_az, n_az, d):
    """Compute single-user LoS-like steering-based channel for a chunk."""
    # Shapes:
    #   phi/theta/r: (U,)
    # Return:
    #   H: (U, M, N_az) complex64/complex128 (later cast)
    U = phi.shape[0]
    M_ = fm_list.shape[0]

    # Broadcast helpers
    c_az = c_az.reshape(1, 1, n_az)                  # (1,1,N)
    k_list = (2.0 * np.pi * fm_list / C).reshape(1, M_, 1)  # (1,M,1)

    sin_phi = np.sin(phi).reshape(U, 1, 1)
    r_k = r.reshape(U, 1, 1)

    # Your current version uses a simplified exponent (ULA-only, near-field-ish)
    exponent = -1j * k_list * (np.sqrt(r_k ** 2 + (c_az ** 2) * (d ** 2) - 2.0 * r_k * c_az * d * sin_phi) - r_k)
    a = np.exp(exponent)  # (U, M, N_az)

    fm_ext = fm_list.reshape(1, M_, 1)
    beta = (C / (4.0 * np.pi)) / (fm_ext * r_k)                 # (U, M, 1)
    beta_exp = np.exp(1j * 2.0 * np.pi * fm_ext * r_k / C)      # (U, M, 1)

    H = beta * beta_exp * a  # (U, M, N_az)
    return H


def write_split(split: str, fm_list: np.ndarray, d: float):
    """Generate and save (channels, user_data) for a given split."""
    num_samples = int(SPLIT_SAMPLES[split])
    num_users = num_samples * K

    # User positions
    x, y, z, phi, theta, r = generate_user_positions(
        num_users=num_users,
        dis_min=DIS_MIN,
        dis_max=DIS_MAX,
        bs_height=BS_HEIGHT,
        user_height=USER_HEIGHT,
    )

    # Save user data (reshape to [num_samples, K])
    ensure_dir(USER_DIR)
    user_path = os.path.join(USER_DIR, "user_data_{}.npz".format(tag(split)))
    np.savez(
        user_path,
        x=x.reshape(num_samples, K),
        y=y.reshape(num_samples, K),
        z=z.reshape(num_samples, K),
        phi=phi.reshape(num_samples, K),
        theta=theta.reshape(num_samples, K),
        r=r.reshape(num_samples, K),
    )
    print("[Info] Saved user_data:", user_path)

    # Channel generation
    ensure_dir(CHANNEL_DIR)

    c_az = np.arange(N_AZ) - (N_AZ + 1) / 2.0  # (N_az,)

    if split == "train":
        # Chunked saving for train
        chunk_samples = int(TRAIN_CHUNK_SAMPLES)
        if chunk_samples <= 0:
            raise ValueError("TRAIN_CHUNK_SAMPLES must be positive.")

        num_chunks = int(np.ceil(num_samples / float(chunk_samples)))
        for ci in range(num_chunks):
            s0 = ci * chunk_samples
            s1 = min((ci + 1) * chunk_samples, num_samples)

            # user index range
            u0 = s0 * K
            u1 = s1 * K

            H_u = compute_channel_chunk(
                phi=phi[u0:u1],
                theta=theta[u0:u1],
                r=r[u0:u1],
                fm_list=fm_list,
                c_az=c_az,
                n_az=N_AZ,
                d=d
            )  # (U, M, N)

            # reshape to (samples_in_chunk, K, M, N)
            H_chunk = H_u.reshape((s1 - s0), K, M, N_AZ).astype(np.complex64)

            ch_path = os.path.join(
                CHANNEL_DIR,
                "channel_K{}_{}_{}.npz".format(K, tag(split), ci)
            )
            np.savez(ch_path, H=H_chunk)
            print("[Info] Saved train channel chunk {}/{}: {}".format(ci + 1, num_chunks, ch_path))

    else:
        # Single-file saving for val/test
        H_u = compute_channel_chunk(
            phi=phi,
            theta=theta,
            r=r,
            fm_list=fm_list,
            c_az=c_az,
            n_az=N_AZ,
            d=d
        )  # (num_users, M, N)

        H_all = H_u.reshape(num_samples, K, M, N_AZ).astype(np.complex64)

        ch_path = os.path.join(
            CHANNEL_DIR,
            "channel_K{}_{}.npz".format(K, tag(split))
        )
        np.savez(ch_path, H=H_all)
        print("[Info] Saved {} channel: {}".format(split, ch_path))


def save_params(fm_list, d):
    """Save system parameters once under ./ISAC_data/params/."""
    ensure_dir(PARAM_DIR)
    D_rayleigh = compute_rayleigh_distance(FC, N_AZ, d)

    param_path = os.path.join(PARAM_DIR, "params{}.npz".format(ARCH))
    np.savez(
        param_path,
        D_rayleigh=D_rayleigh,
        fc=FC,
        B=BANDWIDTH,
        N_el=N_EL,
        N_az=N_AZ,
        f_scs=F_SCS,
        Delta_T=1e-3,  # keep your original placeholder
        M=M,
        K=K,
        d=d,
        user_height=USER_HEIGHT,
        BS_height=BS_HEIGHT,
    )
    print("[Info] Saved params:", param_path)
    print("[Info] Rayleigh distance:", D_rayleigh)


def main():
    ensure_dir(BASE_DIR)
    ensure_dir(CHANNEL_DIR)
    ensure_dir(USER_DIR)
    ensure_dir(PARAM_DIR)

    fm_list = build_fm_list(FC, F_SCS, M)

    lam = C / FC
    d = lam / 2.0

    print("[Info] tmax(ns):", 1e9 / F_SCS)
    print("[Info] BW(Hz):", BANDWIDTH)
    print("[Info] N_az={}, N_el={}, K={}, M={}".format(N_AZ, N_EL, K, M))

    save_params(fm_list, d)

    # Generate splits in one run (train/val/test)
    for split in ["train", "val", "test"]:
        print("\n========== Generating split:", split, "==========")
        write_split(split, fm_list, d)

    print("\n[Done] All splits generated.")


if __name__ == "__main__":
    main()
