import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# =========================
# Path
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "ISAC_data")
PARAM_DIR = os.path.join(BASE_DIR, "params")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_system_params(arch: str = "_ULA", param_dir: str = None):
    """Load system parameters saved by channel_generation.py."""
    if param_dir is None:
        param_dir = PARAM_DIR
    param_path = os.path.join(param_dir, "params{}.npz".format(arch))
    data = np.load(param_path)

    N_el = int(data["N_el"])
    N_az = int(data["N_az"])
    M = int(data["M"])
    K = int(data["K"])

    fc = float(data["fc"])
    B = float(data["B"])
    f_scs = float(data["f_scs"])
    Delta_T = float(data["Delta_T"])
    D_rayleigh = float(data["D_rayleigh"])

    user_height = float(data["user_height"])
    BS_height = float(data["BS_height"])
    d = float(data["d"])

    print("[Info] Loaded params from:", param_path)
    print("[Info] N_az={}, N_el={}, M={}, K={}, fc={}, B={}".format(N_az, N_el, M, K, fc, B))
    return N_az, N_el, M, fc, B, f_scs, Delta_T, D_rayleigh, K, user_height, BS_height, d


# =========================
# Dataset
# =========================
class ISACDataset(Dataset):
    """Dataset wrapper for (H, position labels) samples."""
    def __init__(self, phi, theta, r, x, y, z, H):
        super().__init__()
        self.H = H
        self.phi = phi
        self.theta = theta
        self.r = r
        self.x = x
        self.y = y
        self.z = z
        self.num = H.shape[0]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # Keep complex for downstream physics-based operations
        return {
            "H": torch.tensor(self.H[idx], dtype=torch.complex128),
            "phi_gt": torch.tensor(np.rad2deg(self.phi[idx]), dtype=torch.float32),
            "theta_gt": torch.tensor(np.rad2deg(self.theta[idx]), dtype=torch.float32),
            "r_gt": torch.tensor(self.r[idx], dtype=torch.float32),
            "x_gt": torch.tensor(self.x[idx], dtype=torch.float32),
            "y_gt": torch.tensor(self.y[idx], dtype=torch.float32),
            "z_gt": torch.tensor(self.z[idx], dtype=torch.float32),
        }


# =========================
# Signal model
# =========================
def received_signal(BW, H, PS, TTD, fm_list):
    """Generate received signal with thermal noise under given PS/TTD beamformer."""
    k_B = 1.380649e-23
    T_sys = 290.0

    num, M, N = H.shape
    noise_power = k_B * T_sys * BW * 1e3 / M
    noise_std = float((noise_power / 2.0) ** 0.5)

    device = H.device
    fm = fm_list.reshape(1, M, 1, 1).to(device=device, dtype=torch.float64)

    PS = PS.reshape(num, 1, N, 1).to(device=device, dtype=torch.float64)
    TTD = TTD.reshape(num, 1, N, 1).to(device=device, dtype=torch.float64)

    H_H = torch.conj(H).unsqueeze(-2)  # (num, M, 1, N)
    BF = torch.exp(1j * (PS - 2 * torch.pi * fm * TTD)).to(torch.complex128)  # (num, M, N, 1)

    Y = math.sqrt(1e4 / M) * (H_H @ BF) / math.sqrt(N)  # (num, M, 1, 1)
    Y = Y.squeeze()  # (num, M)

    noise_real = torch.randn(Y.shape, device=device) * noise_std
    noise_imag = torch.randn(Y.shape, device=device) * noise_std
    noise = (noise_real + 1j * noise_imag).to(torch.complex128)

    return (Y + noise).reshape(num, M)


# =========================
# Initialization (fallback)
# =========================
def initial_rainbow_beam(N_az, N_el, d, fm_list, user_height, BS_height, phi_1, phi_M):
    """Compute a deterministic rainbow-beam initialization (used as fallback)."""
    c = 3e8
    phi_1 = np.deg2rad(phi_1)
    phi_M = np.deg2rad(phi_M)

    f_M = fm_list[-1]
    f_1 = fm_list[0]

    dis = 150.0
    theta = np.arctan2(dis, user_height - BS_height)
    r = np.sqrt((user_height - BS_height) ** 2 + dis ** 2)

    c_az = (np.arange(N_az) - (N_az + 1) / 2.0).reshape(N_az, 1)
    c_el = (np.arange(N_el) - (N_el + 1) / 2.0).reshape(1, N_el)

    Psi_1 = (-c_az * d * np.sin(phi_1) * np.sin(theta)
             + (c_az ** 2) * (d ** 2) * (1 - (np.sin(phi_1) * np.sin(theta)) ** 2) / (2 * r)
             - c_el * d * np.cos(theta)
             + (c_el ** 2) * (d ** 2) * (np.sin(theta) ** 2) / (2 * r))
    Psi_M = (-c_az * d * np.sin(phi_M) * np.sin(theta)
             + (c_az ** 2) * (d ** 2) * (1 - (np.sin(phi_M) * np.sin(theta)) ** 2) / (2 * r)
             - c_el * d * np.cos(theta)
             + (c_el ** 2) * (d ** 2) * (np.sin(theta) ** 2) / (2 * r))

    Psi_1 = 2.0 * f_1 / c * Psi_1
    Psi_M = 2.0 * f_M / c * Psi_M

    PS = np.pi * (f_1 * Psi_M - f_M * Psi_1) / (f_M - f_1)
    TTD = (Psi_M - Psi_1) / (2.0 * (f_M - f_1))

    PS = PS.reshape(1, -1).astype(np.float32)
    TTD = (1e9 * TTD).reshape(1, -1).astype(np.float32)  # ns
    return PS, TTD


# =========================
# Training utilities
# =========================
def softmax_peak(mag_db):
    """Soft-argmax peak extraction for differentiable peak index/value."""
    B, M = mag_db.shape
    W = F.softmax(mag_db, dim=-1)            # (B, M)
    val_soft = torch.sum(W * mag_db, dim=-1) # (B,)
    m_range = torch.arange(M, device=mag_db.device, dtype=mag_db.dtype).view(1, M)
    idx_soft = torch.sum(W * m_range, dim=-1)  # (B,)
    return val_soft, idx_soft


def quantize(data, bit_width, scale, bias):
    """Uniform quantization with external (scale, bias) calibration."""
    q_min = -2 ** (bit_width - 1)
    q_max = 2 ** (bit_width - 1) - 1
    q = torch.round(data / scale) + bias
    q = torch.clamp(q, q_min, q_max)
    return q


def loss_rainet(pos_est, phi_gt, r_gt, x_gt, y_gt, delta_height, K):
    B = pos_est.shape[0]
    phi_gt = phi_gt.view(B, K)
    r_gt = r_gt.view(B, K)
    x_est = pos_est[:, :K]
    y_est = pos_est[:, K:2 * K]

    x_rmse = torch.sqrt(torch.mean((x_est - x_gt) ** 2))
    y_rmse = torch.sqrt(torch.mean((y_est - y_gt) ** 2))
    r_est = torch.sqrt(x_est ** 2 + y_est ** 2)  
    phi_est = torch.rad2deg(torch.atan2(y_est, x_est)) 
    phi_rmse = torch.sqrt(torch.mean((phi_est - phi_gt) ** 2))
    r_rmse = torch.sqrt(torch.mean((r_est - r_gt) ** 2))
    r_error = torch.sqrt(torch.mean((x_est - x_gt) ** 2 + (y_est - y_gt) ** 2))
    # total_loss = x_rmse + y_rmse
    total_loss = r_error
    return total_loss, x_rmse, y_rmse, r_error, phi_rmse, r_rmse


def loss_fn(pos_est, phi_gt, r_gt, x_gt, y_gt, delta_height, K=1):
    """2D localization loss (distance error in xy-plane) with auxiliary metrics."""
    B = pos_est.shape[0]
    phi_gt = phi_gt.view(B, K)
    r_gt = r_gt.view(B, K)

    # Output convention preserved from your original code
    phi_est = pos_est[:, :K] / 10.0
    r3d_est = pos_est[:, K:2 * K] / 10.0
    r2d_est = r3d_est

    angle = torch.deg2rad(phi_est)
    x_est = r2d_est * torch.cos(angle)
    y_est = r2d_est * torch.sin(angle)

    x_rmse = torch.sqrt(torch.mean((x_est - x_gt) ** 2))
    y_rmse = torch.sqrt(torch.mean((y_est - y_gt) ** 2))
    phi_rmse = torch.sqrt(torch.mean((phi_est - phi_gt) ** 2))
    r_rmse = torch.sqrt(torch.mean((r3d_est - r_gt) ** 2))
    r_error = torch.sqrt(torch.mean((x_est - x_gt) ** 2 + (y_est - y_gt) ** 2))

    total_loss = r_error
    return total_loss, x_rmse, y_rmse, r_error, phi_rmse, r_rmse
