import os
import gc
import glob
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from functions import (
    BASE_DIR, ensure_dir,
    load_system_params, ISACDataset,
    initial_rainbow_beam, received_signal,
    softmax_peak, quantize, loss_fn
)


# =========================
# Paths (relative)
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
        # val/test usually a single file
        path = os.path.join(CHANNEL_DIR, "channel_K{}_{}.npz".format(K, tag))
        files = [path] if os.path.exists(path) else []
    if not files:
        raise FileNotFoundError("No channel files found for mode='{}' with tag='{}' in {}".format(
            mode, tag, CHANNEL_DIR
        ))
    return files


def load_channels(files, squeeze_k=True):
    """Load and concatenate channel files along sample dimension."""
    H_all = []
    for fp in files:
        data = np.load(fp)
        H = data["H"]
        # stored as (num, K, M, N) when K=1 -> optionally squeeze
        if squeeze_k and H.ndim == 4:
            H = H.squeeze(axis=1)  # (num, M, N)
        H_all.append(H)
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
    # user_npz fields are shaped (num, K). Train assumes K=1; keep original shape.
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
# Models
# =========================
class RainbowBeamModel(nn.Module):
    """Trainable PS/TTD beamformer that produces per-subcarrier received power."""
    def __init__(self, BW, f_scs, N_az, N_el, PS_init, TTD_init):
        super().__init__()
        self.N = N_az * N_el
        self.f_scs = f_scs
        self.BW = BW

        self.PS = nn.Parameter(torch.tensor(PS_init, dtype=torch.float32), requires_grad=True)
        self.TTD = nn.Parameter(torch.tensor(TTD_init, dtype=torch.float32), requires_grad=True)  # ns

    def forward(self, H, fm_list):
        B, M, N = H.shape
        PS_exp = self.PS.expand(B, -1)
        TTD_exp = 1e-9 * self.TTD.expand(B, -1)
        Y = received_signal(self.BW, H, PS_exp, TTD_exp, fm_list)  # (B, M) complex
        mag = torch.abs(Y) ** 2
        mag_dbm = 10 * torch.log10(mag + 1e-30) + 30.0
        # mag_dbm = torch.maximum(mag_dbm, -80.0 * torch.ones_like(mag_dbm))
        return mag_dbm, PS_exp, TTD_exp


class Estimation(nn.Module):
    """A small MLP that maps peak (idx, power) -> coarse (phi, r)."""
    def __init__(self):
        super().__init__()
        in_dim = 2
        h1, h2, h3 = 64, 128, 64

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.LeakyReLU(0.1),
        )
        self.fc_phi = nn.Linear(h3, 1)
        self.fc_r = nn.Linear(h3, 1)

    def forward(self, max_idx, max_val):
        x = torch.cat([max_idx, max_val], dim=1).to(torch.float32)
        h = self.net(x)
        phi = self.fc_phi(h)
        r = self.fc_r(h)
        return torch.cat([phi, r], dim=1)


# =========================
# Train / Eval loops
# =========================
def train_one_epoch(model_bf, model_est, loader, fm_list, opt_bf, opt_est, delta_height, device):
    """One epoch of joint training."""
    model_bf.train()
    model_est.train()

    total = 0
    agg = dict(loss=0.0, phi=0.0, r=0.0, x=0.0, y=0.0, dist=0.0)
    grad_norms = {"PS": [], "TTD": []}

    for batch in loader:
        H = batch["H"].to(device)
        phi_gt = batch["phi_gt"].to(device)
        r_gt = batch["r_gt"].to(device)
        x_gt = batch["x_gt"].to(device)
        y_gt = batch["y_gt"].to(device)

        opt_bf.zero_grad()
        opt_est.zero_grad()

        mag_dbm, PS, TTD = model_bf(H, fm_list)

        # differentiable peak (train)
        max_val, max_idx = softmax_peak(mag_dbm)
        max_idx = max_idx.reshape(-1, 1)
        max_val = max_val.reshape(-1, 1)

        pos = model_est(max_idx, max_val)
        l, x_rmse, y_rmse, dist_err, phi_rmse, r_rmse = loss_fn(
            pos, phi_gt, r_gt, x_gt, y_gt, delta_height, K=1
        )

        l.backward()

        grad_norms["PS"].append(float(model_bf.PS.grad.norm().item()))
        grad_norms["TTD"].append(float(model_bf.TTD.grad.norm().item()))

        opt_bf.step()
        opt_est.step()

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
    agg["grad_PS"] = float(np.mean(grad_norms["PS"]))
    agg["grad_TTD"] = float(np.mean(grad_norms["TTD"]))
    return agg


@torch.no_grad()
def eval_full(model_bf, model_est, loader, fm_list, delta_height, BW, f_scs, device):
    """Full-batch evaluation using hard argmax (as in your original val/test)."""
    model_bf.eval()
    model_est.eval()

    total = 0
    agg = dict(loss=0.0, phi=0.0, r=0.0, x=0.0, y=0.0, dist=0.0)

    for batch in loader:
        H = batch["H"].to(device)
        phi_gt = batch["phi_gt"].to(device)
        r_gt = batch["r_gt"].to(device)
        x_gt = batch["x_gt"].to(device)
        y_gt = batch["y_gt"].to(device)

        mag_dbm, PS, TTD = model_bf(H, fm_list)
        PS_limited = torch.remainder(PS, 2 * torch.pi)
        TTD_limited = torch.remainder(TTD, 1.0 / f_scs)
        Y = received_signal(BW, H, PS_limited, TTD_limited, fm_list)
        mag = torch.abs(Y) ** 2
        mag_dbm = 10 * torch.log10(mag + 1e-30) + 30.0
        # mag_dbm = torch.maximum(mag_dbm, -80.0 * torch.ones_like(mag_dbm))
        max_val, max_idx = torch.max(mag_dbm, dim=-1)
        max_idx = max_idx.reshape(-1, 1)
        max_val = max_val.reshape(-1, 1)

        pos = model_est(max_idx, max_val)
        l, x_rmse, y_rmse, dist_err, phi_rmse, r_rmse = loss_fn(
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

    for k in agg:
        agg[k] /= max(total, 1)
    return agg, pos, phi_gt, r_gt


def save_scatter_and_cdf(dis_tag, r_true, r_est, phi_true, phi_est):
    """Save scatter and CDF figures under ./ISAC_data/figure/range/."""
    ensure_dir(FIG_DIR)

    # scatter (distance)
    plt.figure()
    plt.scatter(r_true, r_est, s=20, alpha=0.7)
    mx = max(np.max(r_true), np.max(r_est), 1.0)
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
    plt.xlabel("Absolute Angle Error (deg)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "angle_cdf{}.png".format(dis_tag)), dpi=120)


def visualize_beam(PS, TTD, fm_list_np, N_az, d, user_x, user_y,
                   center_dis, center_angle_deg, cluster_radius,
                   epoch, save_dir):
    """Compute and plot beamforming power pattern on a 2D spatial grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PS_mod = PS % (2 * np.pi)
    TTD_mod = TTD % (1.0 / 240e3)

    x_min = max(center_dis * np.cos(np.deg2rad(center_angle_deg)) - cluster_radius * 2, 0)
    x_max = center_dis * np.cos(np.deg2rad(center_angle_deg)) + cluster_radius * 2
    y_min = center_dis * np.sin(np.deg2rad(center_angle_deg)) - cluster_radius * 2
    y_max = center_dis * np.sin(np.deg2rad(center_angle_deg)) + cluster_radius * 2

    Nx, Ny = 50, 50
    x_grid = np.linspace(x_min, x_max, Nx)
    y_grid = np.linspace(y_min, y_max, Ny)
    XX, YY = np.meshgrid(x_grid, y_grid)
    power_grid = np.zeros((Ny, Nx))

    c_az = np.arange(N_az) - (N_az + 1) / 2.0
    M = len(fm_list_np)

    PS_np = PS_mod.flatten().astype(np.float64)
    TTD_np = TTD_mod.flatten().astype(np.float64)
    fm = fm_list_np.astype(np.float64)

    chunk = 5
    for yi in range(0, Ny, chunk):
        y_end = min(yi + chunk, Ny)
        xx = XX[yi:y_end].reshape(-1)
        yy = YY[yi:y_end].reshape(-1)
        n_pts = len(xx)

        r_pts = np.sqrt(xx ** 2 + yy ** 2).reshape(n_pts, 1, 1)
        sin_phi = (yy / np.maximum(r_pts.squeeze(), 1e-6)).reshape(n_pts, 1, 1)

        c_az_3d = c_az.reshape(1, 1, N_az)
        k_3d = (2.0 * np.pi * fm / 3e8).reshape(1, M, 1)

        delta_r = np.sqrt(r_pts ** 2 + (c_az_3d * d) ** 2 -
                          2.0 * r_pts * c_az_3d * d * sin_phi)
        exponent = -1j * k_3d * (delta_r - r_pts)
        a = np.exp(exponent)

        bf_phase = PS_np.reshape(1, 1, N_az) - \
            2.0 * np.pi * fm.reshape(1, M, 1) * TTD_np.reshape(1, 1, N_az)
        w = np.exp(1j * bf_phase)  # (1, M, N_az)

        Y = a @ w.transpose(0, 2, 1)  # (n_pts, M, N_az) @ (1, N_az, M) -> need different approach
        # Actually: sum over antennas for each subcarrier
        Y = np.sum(a * w, axis=2)  # (n_pts, M)
        P = np.sum(np.abs(Y) ** 2, axis=1)  # (n_pts,)
        power_grid[yi:y_end, :] = P.reshape(y_end - yi, Nx)

    power_db = 10.0 * np.log10(power_grid + 1e-30)

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(XX, YY, power_db, shading="auto", cmap="jet")
    plt.colorbar(label="Beam Power (a.u. dB)")
    plt.scatter(user_x, user_y, c="white", s=8, alpha=0.6, label="Users")
    cx = center_dis * np.cos(np.deg2rad(center_angle_deg))
    cy = center_dis * np.sin(np.deg2rad(center_angle_deg))
    plt.plot(cx, cy, "r+", markersize=15, markeredgewidth=2, label="Cluster center")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Beam Pattern (Epoch {})".format(epoch))
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.tight_layout()
    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, "beam_pattern_ep{:03d}.png".format(epoch)), dpi=120)
    plt.close()


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIG_DIR)

    # experiment config
    arch = "_ULA"
    dis_min, dis_max = 5, 300
    epochs = 100

    # cluster mode (must match channel_generation.py settings)
    CLUSTER_MODE = True
    CLUSTER_CENTER_DIS = 100.0
    CLUSTER_CENTER_ANGLE_DEG = 30.0
    CLUSTER_RADIUS = 10.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] device:", device)

    if CLUSTER_MODE:
        print("[Info] CLUSTER MODE: center=({:.0f}m, {:.0f}deg), radius={:.0f}m".format(
            CLUSTER_CENTER_DIS, CLUSTER_CENTER_ANGLE_DEG, CLUSTER_RADIUS))

    # system params
    N_az, N_el, M, fc, BW, f_scs, Delta_T, D_rayleigh, K, user_height, BS_height, d = load_system_params(arch=arch)
    delta_height = BS_height - user_height

    fm_list_np = fc + f_scs * (np.arange(M) - (M - 1) / 2.0)
    fm_list = torch.from_numpy(fm_list_np.astype(np.float32)).to(device)

    # file tag (must match channel_generation.py tag function)
    cluster_suffix = "_cluster" if CLUSTER_MODE else ""

    # load splits
    train_tag = "train{}_{:d}to{:d}{}".format(arch, dis_min, dis_max, cluster_suffix)
    val_tag = "val{}_{:d}to{:d}{}".format(arch, dis_min, dis_max, cluster_suffix)
    test_tag = "test{}_{:d}to{:d}{}".format(arch, dis_min, dis_max, cluster_suffix)

    # train (may have multiple chunks)
    train_pattern = os.path.join(CHANNEL_DIR, "channel_K{}_{}*.npz".format(K, train_tag))
    train_files = sorted(glob.glob(train_pattern))
    if not train_files:
        raise FileNotFoundError("No train channel files: {}".format(train_pattern))
    H_train = load_channels(train_files, squeeze_k=True)
    user_train = np.load(os.path.join(USER_DIR, "user_data_{}.npz".format(train_tag)))

    H_val = load_channels([os.path.join(CHANNEL_DIR, "channel_K{}_{}.npz".format(K, val_tag))], squeeze_k=True)
    user_val = np.load(os.path.join(USER_DIR, "user_data_{}.npz".format(val_tag)))
    val_loader = build_loader(H_val, user_val, batch_size=H_val.shape[0], shuffle=False)

    H_test = load_channels([os.path.join(CHANNEL_DIR, "channel_K{}_{}.npz".format(K, test_tag))], squeeze_k=True)
    user_test = np.load(os.path.join(USER_DIR, "user_data_{}.npz".format(test_tag)))
    test_loader = build_loader(H_test, user_test, batch_size=H_test.shape[0], shuffle=False)

    train_loader = build_loader(H_train, user_train, batch_size=200, shuffle=True)

    # user positions for beam visualization
    user_x_train = user_train["x"][:H_train.shape[0]].flatten()
    user_y_train = user_train["y"][:H_train.shape[0]].flatten()

    # init PS/TTD
    PS_init, TTD_init = initial_rainbow_beam(N_az, N_el, d, fm_list_np, user_height, BS_height, -60, 60)

    model_bf = RainbowBeamModel(BW, f_scs, N_az, N_el, PS_init, TTD_init).to(device)
    model_est = Estimation().to(device)

    opt_bf = optim.Adam(model_bf.parameters(), lr=1e-3)
    opt_est = optim.Adam(model_est.parameters(), lr=5e-3)

    best = dict(val_loss=float("inf"))
    dis_tag = "_{}to{}{}".format(dis_min, dis_max, cluster_suffix)

    # history tracking
    ps_history = []
    ttd_history = []
    grad_ps_history = []
    grad_ttd_history = []

    # initial beam visualization
    visualize_beam(
        model_bf.PS.detach().cpu().numpy(),
        model_bf.TTD.detach().cpu().numpy() * 1e-9,
        fm_list_np, N_az, d,
        user_x_train, user_y_train,
        CLUSTER_CENTER_DIS, CLUSTER_CENTER_ANGLE_DEG, CLUSTER_RADIUS,
        epoch=0, save_dir=FIG_DIR,
    )

    for ep in range(epochs):
        tr = train_one_epoch(
            model_bf, model_est, train_loader, fm_list,
            opt_bf, opt_est, delta_height, device
        )

        msg = "[Epoch {}/{}][Train] loss={:.6f}, phi={:.4f}, r={:.4f}, dist={:.4f}, |grad_PS|={:.6f}, |grad_TTD|={:.6f}".format(
            ep + 1, epochs, tr["loss"], tr["phi"], tr["r"], tr["dist"],
            tr["grad_PS"], tr["grad_TTD"]
        )
        print(msg)

        grad_ps_history.append(tr["grad_PS"])
        grad_ttd_history.append(tr["grad_TTD"])

        va, _, _, _ = eval_full(model_bf, model_est, val_loader, fm_list, delta_height, BW, f_scs, device)
        print("[Val]  loss={:.6f}, phi={:.4f}, r={:.4f}, dist={:.4f}".format(va["loss"], va["phi"], va["r"], va["dist"]))

        te, pos, phi_gt, r_gt = eval_full(model_bf, model_est, test_loader, fm_list, delta_height, BW, f_scs, device)
        print("[Test] loss={:.6f}, phi={:.4f}, r={:.4f}, dist={:.4f}".format(te["loss"], te["phi"], te["r"], te["dist"]))

        # record PS/TTD
        ps_history.append(model_bf.PS.detach().cpu().numpy().copy())
        ttd_history.append(model_bf.TTD.detach().cpu().numpy().copy())

        # beam visualization every 10 epochs
        if (ep + 1) % 10 == 0 or ep == 0:
            visualize_beam(
                model_bf.PS.detach().cpu().numpy(),
                model_bf.TTD.detach().cpu().numpy() * 1e-9,
                fm_list_np, N_az, d,
                user_x_train, user_y_train,
                CLUSTER_CENTER_DIS, CLUSTER_CENTER_ANGLE_DEG, CLUSTER_RADIUS,
                epoch=ep + 1, save_dir=FIG_DIR,
            )

        if va["loss"] < best["val_loss"]:
            best["val_loss"] = va["loss"]

            bf_path = os.path.join(OUTPUT_DIR, "best_model_bf{}.pt".format(dis_tag))
            est_path = os.path.join(OUTPUT_DIR, "best_model_est{}.pt".format(dis_tag))
            torch.save(model_bf.state_dict(), bf_path)
            torch.save(model_est.state_dict(), est_path)

            ps_ttd_path = os.path.join(OUTPUT_DIR, "ps_ttd_result{}{}.npz".format(arch, dis_tag))
            np.savez(ps_ttd_path,
                     PS=model_bf.PS.detach().cpu().numpy(),
                     TTD=model_bf.TTD.detach().cpu().numpy())
            print("[Info] Saved best checkpoint:", bf_path, est_path)
            print("[Info] Saved PS/TTD:", ps_ttd_path)

    print("[Done] Best val loss:", best["val_loss"])

    # plot PS/TTD evolution
    ps_arr = np.array(ps_history)    # (epochs, N)
    ttd_arr = np.array(ttd_history)  # (epochs, N)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    im0 = axes[0].imshow(ps_arr.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_ylabel("Antenna index")
    axes[0].set_title("PS evolution (rad)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(ttd_arr.T, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Antenna index")
    axes[1].set_title("TTD evolution (ns)")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ps_ttd_evolution{}.png".format(dis_tag)), dpi=120)
    plt.close()

    # plot gradient norms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.semilogy(grad_ps_history, label="|grad PS|")
    ax1.set_ylabel("Gradient norm")
    ax1.set_title("PS gradient norm")
    ax1.legend()
    ax1.grid(True)
    ax2.semilogy(grad_ttd_history, label="|grad TTD|", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient norm")
    ax2.set_title("TTD gradient norm")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "grad_norms{}.png".format(dis_tag)), dpi=120)
    plt.close()

    # plot PS/TTD before vs after
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(ps_history[0].flatten(), label="Initial", alpha=0.8)
    ax1.plot(ps_history[-1].flatten(), label="Final", alpha=0.8)
    ax1.set_ylabel("PS (rad)")
    ax1.set_title("PS: initial vs final")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(ttd_history[0].flatten(), label="Initial", alpha=0.8)
    ax2.plot(ttd_history[-1].flatten(), label="Final", alpha=0.8)
    ax2.set_ylabel("TTD (ns)")
    ax2.set_xlabel("Antenna index")
    ax2.set_title("TTD: initial vs final")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ps_ttd_before_after{}.png".format(dis_tag)), dpi=120)
    plt.close()

    # proactive cleanup
    cleanup(H_train, user_train, H_val, H_test)


if __name__ == "__main__":
    main()


