import os
import glob
import gc
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsRegressor

from functions import (
    BASE_DIR, ensure_dir,
    load_system_params, ISACDataset,
    initial_rainbow_beam, received_signal
)


# =========================
# Paths
# =========================
CHANNEL_DIR = os.path.join(BASE_DIR, "channels")
USER_DIR = os.path.join(BASE_DIR, "user_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def cleanup(*objs, empty_cuda=True):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if empty_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def mode_tag(mode: str, arch: str, dis_min: int, dis_max: int) -> str:
    return "{}{}_{:d}to{:d}".format(mode, arch, dis_min, dis_max)


def find_channel_files(mode: str, K: int, arch: str, dis_min: int, dis_max: int):
    """
    Locate channel npz files for a given split.
    - train: multiple chunks -> channel_K{K}_train{arch}_{dis_min}to{dis_max}_*.npz
    - test: single file      -> channel_K{K}_test{arch}_{dis_min}to{dis_max}.npz
    """
    tag = mode_tag(mode, arch, dis_min, dis_max)
    if mode == "train":
        pattern = os.path.join(CHANNEL_DIR, "channel_K{}_{}_*.npz".format(K, tag))
        files = sorted(glob.glob(pattern))
    else:
        path = os.path.join(CHANNEL_DIR, "channel_K{}_{}.npz".format(K, tag))
        files = [path] if os.path.exists(path) else []

    if not files:
        raise FileNotFoundError("No channel files found for mode='{}' with tag='{}' in {}".format(
            mode, tag, CHANNEL_DIR
        ))
    return files


# ================================
# Load channels
# ================================
def load_channels(files, squeeze_k=True, max_samples=None):
    """
    Load channel npz files and concatenate along sample dimension.
    """
    H_all = []
    total = 0
    need = None if max_samples is None else int(max_samples)

    for fp in files:
        data = np.load(fp)
        H = data["H"]

        # stored as (num, K, M, N) when K=1 -> optionally squeeze to (num, M, N)
        if squeeze_k and H.ndim == 4:
            H = H.squeeze(axis=1)

        if need is None:
            H_all.append(H)
            total += H.shape[0]
        else:
            remaining = need - total
            if remaining <= 0:
                break

            # Read only what we need from this file (slice before concatenation)
            take = min(remaining, H.shape[0])
            H_all.append(H[:take])
            total += take

            if total >= need:
                break

    if not H_all:
        raise RuntimeError("No channel data loaded. Check file paths or max_samples setting.")

    return np.concatenate(H_all, axis=0)


def load_user_npz(mode: str, arch: str, dis_min: int, dis_max: int):
    """Load user position labels saved by channel_generation.py."""
    tag = mode_tag(mode, arch, dis_min, dis_max)
    path = os.path.join(USER_DIR, "user_data_{}.npz".format(tag))
    if not os.path.exists(path):
        raise FileNotFoundError("User data not found: {}".format(path))
    return np.load(path)


def build_loader(H, user_npz, batch_size, shuffle):
    """Construct DataLoader for a given split."""
    num = H.shape[0]
    phi = user_npz["phi"][:num]
    theta = user_npz["theta"][:num]
    r = user_npz["r"][:num]
    x = user_npz["x"][:num]
    y = user_npz["y"][:num]
    z = user_npz["z"][:num]

    dataset = ISACDataset(phi=phi, theta=theta, r=r, x=x, y=y, z=z, H=H)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =========================
# Beam model (fixed PS/TTD)
# =========================
class RainbowBeamModel(nn.Module):
    def __init__(self, BW, f_scs, N_az, N_el, PS_init, TTD_init):
        super().__init__()
        self.N = N_az * N_el
        self.f_scs = f_scs
        self.BW = BW

        self.PS = nn.Parameter(torch.tensor(PS_init, dtype=torch.float32), requires_grad=False)
        self.TTD = nn.Parameter(torch.tensor(TTD_init, dtype=torch.float32), requires_grad=False)  # ns

    def forward(self, H, fm_list):
        B, M, N = H.shape

        PS_limited = torch.remainder(self.PS, 2 * torch.pi)               # [0, 2pi)
        TTD_limited = torch.remainder(1e-9 * self.TTD, 1.0 / self.f_scs)  # seconds, wrapped

        PS_exp = PS_limited.expand(B, -1)
        TTD_exp = TTD_limited.expand(B, -1)

        Y = received_signal(self.BW, H, PS_exp, TTD_exp, fm_list)  # (B,M) complex
        mag = torch.abs(Y) ** 2
        mag_dbm = 10 * torch.log10(mag + 1e-12) + 30.0
        mag_dbm = torch.maximum(mag_dbm, -80.0 * torch.ones_like(mag_dbm))
        return mag_dbm


# =========================
# kNN estimator
# =========================
class KNNEstimator:
    """scikit-learn kNN regressor mapping feature -> (phi_deg, r_m)."""
    def __init__(self, k=11, weighting="distance"):
        self.model = KNeighborsRegressor(
            n_neighbors=int(k),
            weights=weighting,
            metric="euclidean"
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    # experiment config
    arch = "_ULA"
    dis_min, dis_max = 5, 300
    K = 1

    k_neighbors = 11
    weighting = "distance"
    batch_size = 300
    device = torch.device("cpu")
    print("[Info] device:", device)

    # ================================
    # Codebook size
    # ================================
    train_max_samples = 30000
    test_max_samples = None

    # -------------------------
    # System params
    # -------------------------
    N_az, N_el, M, fc, BW, f_scs, Delta_T, D_rayleigh, Kp, user_height, BS_height, d = load_system_params(arch=arch)
    if int(Kp) != int(K):
        print("[Warn] params K={} but script uses K={}. Proceeding with K={}.".format(int(Kp), int(K), int(K)))

    fm_list_np = fc + f_scs * (np.arange(M) - (M - 1) / 2.0)
    fm_list = torch.from_numpy(fm_list_np.astype(np.float32)).to(device)

    # -------------------------
    # Load train (read only as needed)
    # -------------------------
    train_files = find_channel_files("train", K, arch, dis_min, dis_max)
    H_train = load_channels(train_files, squeeze_k=True, max_samples=train_max_samples)
    user_train = load_user_npz("train", arch, dis_min, dis_max)
    train_loader = build_loader(H_train, user_train, batch_size=batch_size, shuffle=False)
    print("[Info] Loaded train samples:", H_train.shape[0])

    # -------------------------
    # Init fixed rainbow beam
    # -------------------------
    PS_init, TTD_init = initial_rainbow_beam(
        N_az, N_el, d, fm_list_np, user_height, BS_height, -60, 60
    )
    beam_model = RainbowBeamModel(BW, f_scs, N_az, N_el, PS_init, TTD_init).to(device)
    beam_model.eval()

    # -------------------------
    # Build train features
    # -------------------------
    X_list, y_list = [], []

    with torch.no_grad():
        for batch in train_loader:
            H_b = batch["H"].to(device)  # (B,M,N) complex
            phi_deg = batch["phi_gt"].cpu().numpy().reshape(-1)  # degrees
            r_m = batch["r_gt"].cpu().numpy().reshape(-1)        # meters

            feat = beam_model(H_b, fm_list).detach().cpu().numpy()  # (B,M)
            X_list.append(feat.reshape(feat.shape[0], -1))
            y_list.append(np.stack([phi_deg, r_m], axis=1))         # (B,2)

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    print("[Info] Train shapes: X={}, y={}".format(X_train.shape, y_train.shape))

    # -------------------------
    # Train kNN
    # -------------------------
    knn = KNNEstimator(k=k_neighbors, weighting=weighting)
    knn.fit(X_train, y_train)
    print("[Info] kNN trained: k={}, weighting={}".format(k_neighbors, weighting))

    # -------------------------
    # Load test
    # -------------------------
    test_files = find_channel_files("test", K, arch, dis_min, dis_max)
    H_test = load_channels(test_files, squeeze_k=True, max_samples=test_max_samples)
    user_test = load_user_npz("test", arch, dis_min, dis_max)
    test_loader = build_loader(H_test, user_test, batch_size=batch_size, shuffle=False)
    print("[Info] Loaded test samples:", H_test.shape[0])

    # -------------------------
    # Evaluate
    # -------------------------
    preds_all, gts_all = [], []

    with torch.no_grad():
        for batch in test_loader:
            H_b = batch["H"].to(device)
            phi_deg = batch["phi_gt"].cpu().numpy().reshape(-1)
            r_m = batch["r_gt"].cpu().numpy().reshape(-1)

            feat = beam_model(H_b, fm_list).detach().cpu().numpy()
            X_test = feat.reshape(feat.shape[0], -1)

            pred = knn.predict(X_test)  # (B,2): [phi_deg, r_m]
            gt = np.stack([phi_deg, r_m], axis=1)

            preds_all.append(pred)
            gts_all.append(gt)

    preds = np.concatenate(preds_all, axis=0)
    gts = np.concatenate(gts_all, axis=0)

    # -------------------------
    # Metrics
    # -------------------------
    err = np.linalg.norm(preds - gts, axis=1)
    mean_err = float(np.mean(err))
    print("[Test] Mean L2 error in (phi_deg, r_m) space: {:.6f}".format(mean_err))

    out_path = os.path.join(OUTPUT_DIR, "knn_result{}_{:d}to{:d}.npz".format(arch, dis_min, dis_max))
    np.savez(out_path, preds=preds, gts=gts)
    print("[Info] Saved:", out_path)

    cleanup(H_train, user_train, H_test, user_test, X_train, y_train, preds, gts)


if __name__ == "__main__":
    main()
