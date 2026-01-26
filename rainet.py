import os
import glob
import gc
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from functions import (
    BASE_DIR, ensure_dir,
    load_system_params, ISACDataset,
    initial_rainbow_beam, received_signal, loss_rainet
)

# =========================
# Paths
# =========================
CHANNEL_DIR = os.path.join(BASE_DIR, "channels")
USER_DIR = os.path.join(BASE_DIR, "user_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(BASE_DIR, "figure", "range")


def cleanup(*objs, close_figs=True, empty_cuda=True):
    """Release references and clear caches to reduce peak memory."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    if close_figs:
        try:
            plt.close("all")
        except Exception:
            pass
    gc.collect()
    if empty_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def mode_tag(mode: str, arch: str, dis_min: int, dis_max: int) -> str:
    """Build the filename tag consistent with channel_generation.py."""
    return "{}{}_{:d}to{:d}".format(mode, arch, dis_min, dis_max)


def find_channel_files(mode: str, K: int, arch: str, dis_min: int, dis_max: int):
    """Locate channel npz files for a given split (train supports multiple chunks)."""
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
# [channel loading
# ================================
def load_channels(files, squeeze_k=True, max_samples=None):
    """
    Load and concatenate channel files along sample dimension
    """
    H_all = []
    total = 0
    need = None if max_samples is None else int(max_samples)

    for fp in files:
        data = np.load(fp)
        H = data["H"]
        if squeeze_k and H.ndim == 4:
            H = H.squeeze(axis=1)  # (num, M, N)

        if need is None:
            H_all.append(H)
            total += H.shape[0]
        else:
            remain = need - total
            if remain <= 0:
                break
            take = min(remain, H.shape[0])
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
# RaiNet
# =========================
class RaiNet(nn.Module):
    """
      Conv1d(1->64, k=7, s=2, tanh) ×3
      FC: 128 -> 84 -> 2, tanh activations.
    """
    def __init__(self, input_len: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.1),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            feat = self.features(dummy)
            flat_dim = feat.numel()

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Linear(128, 84),
            nn.BatchNorm1d(84),
            nn.Tanh(),

            nn.Linear(84, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.to(torch.float32)          # (B, M)
        x = x.unsqueeze(1)               # (B, 1, M)
        z = self.features(x)             # (B, C, L)
        z = z.view(z.size(0), -1)        # (B, flat)
        out = self.classifier(z)         # (B, 2)
        return out


# =========================
# Beam model
# =========================
class RainbowBeamModel(nn.Module):
    def __init__(self, BW, f_scs, N_az, N_el, PS_init, TTD_init):
        super().__init__()
        self.N = N_az * N_el
        self.f_scs = f_scs
        self.BW = BW
        # self.PS = nn.Parameter(torch.tensor(PS_init, dtype=torch.float32), requires_grad=True)
        # self.TTD = nn.Parameter(torch.tensor(TTD_init, dtype=torch.float32), requires_grad=True)
        self.PS = torch.tensor(PS_init, dtype=torch.float32)
        self.TTD = torch.tensor(TTD_init, dtype=torch.float32)

    def forward(self, H, fm_list):
        B, M, N = H.shape
        PS_limited = torch.remainder(self.PS, 2 * torch.pi)
        TTD_limited = torch.remainder(1e-9 * self.TTD, 1.0 / self.f_scs)

        PS_exp = PS_limited.expand(B, -1)
        TTD_exp = TTD_limited.expand(B, -1)

        Y = received_signal(self.BW, H, PS_exp, TTD_exp, fm_list)  # (B,M) complex
        mag = torch.abs(Y) ** 2
        mag_dbm = 10 * torch.log10(mag + 1e-30) + 30.0
        mag_dbm = torch.maximum(mag_dbm, -80.0 * torch.ones_like(mag_dbm))
        return mag_dbm, PS_exp, TTD_exp


# =========================
# Train / Eval
# =========================
def train_one_epoch(model_bf, model_net, loader, fm_list, opt_net, delta_height, dis_max, device):
    model_bf.eval()
    model_net.train()

    total = 0
    agg = dict(loss=0.0, phi=0.0, r=0.0, x=0.0, y=0.0, dist=0.0)

    for batch in loader:
        H = batch["H"].to(device)
        phi_gt = batch["phi_gt"].to(device)
        r_gt = batch["r_gt"].to(device)
        x_gt = batch["x_gt"].to(device)
        y_gt = batch["y_gt"].to(device)

        opt_net.zero_grad()

        mag_dbm, PS, TTD = model_bf(H, fm_list)
        pos = model_net(mag_dbm) * float(dis_max)  # (B,2) scaled


        l, x_rmse, y_rmse, dist_err, phi_rmse, r_rmse = loss_rainet(
            pos, phi_gt, r_gt, x_gt, y_gt, delta_height, K=1
        )

        l.backward()
        opt_net.step()

        bs = H.shape[0]
        total += bs
        agg["loss"] += float(l.item()) * bs
        agg["phi"] += float(phi_rmse.item()) * bs
        agg["r"] += float(r_rmse.item()) * bs
        agg["x"] += float(x_rmse.item()) * bs
        agg["y"] += float(y_rmse.item()) * bs
        agg["dist"] += float(dist_err.item()) * bs

    for k in agg:
        agg[k] /= max(total, 1)
    return agg


@torch.no_grad()
def eval_full(model_bf, model_net, loader, fm_list, delta_height, dis_max, device):
    model_bf.eval()
    model_net.eval()

    total = 0
    agg = dict(loss=0.0, phi=0.0, r=0.0, x=0.0, y=0.0, dist=0.0)
    last_pos = None
    last_phi_gt = None
    last_r_gt = None

    for batch in loader:
        H = batch["H"].to(device)
        phi_gt = batch["phi_gt"].to(device)
        r_gt = batch["r_gt"].to(device)
        x_gt = batch["x_gt"].to(device)
        y_gt = batch["y_gt"].to(device)

        mag_dbm, _, _ = model_bf(H, fm_list)
        pos = model_net(mag_dbm) * float(dis_max)

        l, x_rmse, y_rmse, dist_err, phi_rmse, r_rmse = loss_rainet(
            pos, phi_gt, r_gt, x_gt, y_gt, delta_height, K=1
        )

        bs = H.shape[0]
        total += bs
        agg["loss"] += float(l.item()) * bs
        agg["phi"] += float(phi_rmse.item()) * bs
        agg["r"] += float(r_rmse.item()) * bs
        agg["x"] += float(x_rmse.item()) * bs
        agg["y"] += float(y_rmse.item()) * bs
        agg["dist"] += float(dist_err.item()) * bs

        last_pos = pos
        last_phi_gt = phi_gt
        last_r_gt = r_gt

    for k in agg:
        agg[k] /= max(total, 1)
    return agg, last_pos, last_phi_gt, last_r_gt


def save_scatter_and_cdf(dis_tag, dis_max, r_true, r_est, phi_true, phi_est):
    ensure_dir(FIG_DIR)

    # scatter (distance)
    plt.figure()
    plt.scatter(r_true, r_est, s=20, alpha=0.7)
    mx = max(float(np.max(r_true)), float(np.max(r_est)), 1.0)
    plt.plot([0, mx], [0, mx], "r--")
    plt.xlabel("True Distance (m)")
    plt.ylabel("Estimated Distance (m)")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "distance_scatter{}.png".format(dis_tag)), dpi=120)

    # scatter (angle)
    plt.figure()
    plt.scatter(phi_true, phi_est, s=20, alpha=0.7)
    plt.plot([-60, 60], [-60, 60], "r--")
    plt.xlabel("True Angle (deg)")
    plt.ylabel("Estimated Angle (deg)")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "angle_scatter{}.png".format(dis_tag)), dpi=120)

    # CDF distance error
    plt.figure()
    e = np.abs(r_true.reshape(-1) - r_est.reshape(-1))
    e_sorted = np.sort(e)
    cdf = np.arange(1, len(e_sorted) + 1) / float(len(e_sorted))
    plt.plot(e_sorted, cdf)
    plt.axvline(x=np.percentile(e, 95), linestyle="--")
    plt.xlabel("Absolute Distance Error (m)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "distance_cdf{}.png".format(dis_tag)), dpi=120)

    # CDF angle error
    plt.figure()
    e = np.abs(phi_true.reshape(-1) - phi_est.reshape(-1))
    e_sorted = np.sort(e)
    cdf = np.arange(1, len(e_sorted) + 1) / float(len(e_sorted))
    plt.plot(e_sorted, cdf)
    plt.axvline(x=np.percentile(e, 95), linestyle="--")
    plt.xlabel("Absolute Angle Error (deg)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "angle_cdf{}.png".format(dis_tag)), dpi=120)


# =========================
# Main (train.py style)
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIG_DIR)

    # experiment config
    arch = "_ULA"
    dis_min = 5
    dis_max_list = [300]  # keep your loop behavior
    K = 1
    epochs = 100

    # runtime config
    device = torch.device("cpu")
    print("[Info] device:", device)

    # ================================
    # Dataset size
    # ================================
    train_max_samples = None 
    val_max_samples = None
    test_max_samples = None

    # system params
    N_az, N_el, M, fc, BW, f_scs, Delta_T, D_rayleigh, Kp, user_height, BS_height, d = load_system_params(arch=arch)
    delta_height = BS_height - user_height

    fm_list_np = fc + f_scs * (np.arange(M) - (M - 1) / 2.0)
    fm_list = torch.from_numpy(fm_list_np.astype(np.float32)).to(device)

    for dis_max in dis_max_list:
        dis_tag = "_{}to{}".format(dis_min, dis_max)

        # -------------------------
        # Load splits (read-as-needed)
        # -------------------------
        train_files = find_channel_files("train", K, arch, dis_min, dis_max)
        H_train = load_channels(train_files, squeeze_k=True, max_samples=train_max_samples)
        user_train = load_user_npz("train", arch, dis_min, dis_max)
        train_loader = build_loader(H_train, user_train, batch_size=300, shuffle=True)
        train_size = H_train.shape[0]
        print("[Info] Loaded train samples:", train_size)

        H_val = load_channels(find_channel_files("val", K, arch, dis_min, dis_max),
                              squeeze_k=True, max_samples=val_max_samples)
        user_val = load_user_npz("val", arch, dis_min, dis_max)
        val_loader = build_loader(H_val, user_val, batch_size=H_val.shape[0], shuffle=False)
        val_size = H_val.shape[0]
        print("[Info] Loaded val samples:", val_size)

        H_test = load_channels(find_channel_files("test", K, arch, dis_min, dis_max),
                               squeeze_k=True, max_samples=test_max_samples)
        user_test = load_user_npz("test", arch, dis_min, dis_max)
        test_loader = build_loader(H_test, user_test, batch_size=H_test.shape[0], shuffle=False)
        test_size = H_test.shape[0]
        print("[Info] Loaded test samples:", test_size)

        # -------------------------
        # Init models
        # -------------------------
        PS_init, TTD_init = initial_rainbow_beam(N_az, N_el, d, fm_list_np, user_height, BS_height, -60, 60)

        model_bf = RainbowBeamModel(BW, f_scs, N_az, N_el, PS_init, TTD_init).to(device)
        model_net = RaiNet(input_len=M).to(device)

        opt_net = optim.Adam(model_net.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt_net, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-4
        )

        best = dict(test_loss=float("inf"))

        # -------------------------
        # Train loop
        # -------------------------
        for ep in range(epochs):
            tr = train_one_epoch(model_bf, model_net, train_loader, fm_list, opt_net,
                                 delta_height, dis_max, device)
            print("[Epoch {}/{}][Train] loss={:.6f}, phi={:.4f}, r={:.4f}, x={:.4f}, y={:.4f}, dist={:.4f}".format(
                ep + 1, epochs, tr["loss"], tr["phi"], tr["r"], tr["x"], tr["y"], tr["dist"]
            ))

            va, _, _, _ = eval_full(model_bf, model_net, val_loader, fm_list, delta_height, dis_max, device)
            print("[Val]  loss={:.6f}, phi={:.4f}, r={:.4f}, dist={:.4f}".format(
                va["loss"], va["phi"], va["r"], va["dist"]
            ))
            scheduler.step(va["loss"])

            te, pos, phi_gt, r_gt = eval_full(model_bf, model_net, test_loader, fm_list, delta_height, dis_max, device)
            print("[Test] loss={:.6f}, phi={:.4f}, r={:.4f}, dist={:.4f}".format(
                te["loss"], te["phi"], te["r"], te["dist"]
            ))

            # -------------------------
            # Save best
            # -------------------------
            if va["loss"] < best["test_loss"]:
                best["test_loss"] = va["loss"]

                bf_path = os.path.join(OUTPUT_DIR, "best_rainbow_beam_model1{}_ANN.pt".format(dis_tag))
                net_path = os.path.join(OUTPUT_DIR, "best_rainbow_beam_model2{}_ANN.pt".format(dis_tag))
                torch.save(model_bf.state_dict(), bf_path)
                torch.save(model_net.state_dict(), net_path)

                # save PS/TTD
                ps_ttd_path = os.path.join(OUTPUT_DIR, "ps_ttd_result{}{}.npz".format(arch, dis_tag))
                # PS/TTD are expanded in forward; here save parameter base (single vector)
                np.savez(ps_ttd_path,
                         PS=model_bf.PS.detach().cpu().numpy(),
                         TTD=model_bf.TTD.detach().cpu().numpy())
                print("[Info] Saved best checkpoint:", bf_path, net_path)
                print("[Info] Saved PS/TTD:", ps_ttd_path)

                # figures
                pos_np = pos.detach().cpu().numpy()
                phi_est = pos_np[:, 0].reshape(-1)
                r_est = pos_np[:, 1].reshape(-1)

                phi_true = phi_gt.detach().cpu().numpy().reshape(-1)
                r_true = r_gt.detach().cpu().numpy().reshape(-1)

                # save_scatter_and_cdf(dis_tag, dis_max, r_true, r_est, phi_true, phi_est)

        print("[Done] Best val loss {}: {:.6f}".format(dis_tag, best["test_loss"]))

        cleanup(H_train, user_train, H_val, user_val, H_test, user_test)


if __name__ == "__main__":
    main()
