# Trains a small PyTorch MLP on the 3-spiral classification task (3 classes),
# then replace each torch.nn.ReLU with a *fault-tolerant ReLU* implemented using
# your default_architecture_class.py (phase code + decoder + encoder_map).

# Optional:
#    python train_and_swap_fault_tolerant_relu.py --epochs 2000 --hidden 64
#    python train_and_swap_fault_tolerant_relu.py --sigma_phase 0.01 --p_syn 0.05


import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


from default_architecture_class import arhitecture_specs, default_arhitecture



# 3-spiral dataset generator
def make_3_spirals(n_per_class: int = 500, noise: float = 0.2, seed: int = 0):
    """
    Classic 3-spiral synthetic dataset (2D -> 3 classes).
    Returns: X (N,2), y (N,)
    """
    rng = np.random.default_rng(seed)
    K = 3
    N = n_per_class
    X = []
    y = []
    for k in range(K):
        # angle runs outward
        t = np.linspace(0, 4 * math.pi, N)
        # class offset
        offset = 2 * math.pi * k / K
        r = t
        x1 = r * np.cos(t + offset)
        x2 = r * np.sin(t + offset)
        pts = np.stack([x1, x2], axis=1)
        pts += rng.normal(0.0, noise, size=pts.shape)
        X.append(pts)
        y.append(np.full(N, k, dtype=np.int64))
    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0)
    # normalize for easier training
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    return X, y



# Baseline PyTorch MLP

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    

import torch
import torch.nn as nn


# Shallow MLP variant with 1 hidden layer
class ShallowMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)



# Fault-tolerant ReLU implementation using default_architecture_class.py
@dataclass
class FTReluConfig:
    M: int = 7
    # pick lambdas "large enough" so y in [0, y_max] doesn't wrap in mod 1
    lambdas: tuple = (19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0)
    x_min: float = -6.0
    x_max: float = 6.0
    S: int = 241  # number of decoded values (grid size)
    sigma_phase: float = 0.0
    sigma_trig: float = 0.0
    sigma_score: float = 0.0
    p_syn: float = 0.0


class FaultTolerantReLU(nn.Module):
    """
    Drop-in replacement for nn.ReLU during *inference*.

    It uses:
      - input encoding: theta = mod1(x / lambdas)
      - decoder chooses nearest x_k in x_values
      - encoder_map: x_k -> relu(x_k)
      - output phases represent y / lambdas (no wrap if lambdas > y_max)
      - reconstruct y as average_j(out_phase[j] * lambda[j])
    """
    def __init__(self, cfg: FTReluConfig):
        super().__init__()
        self.cfg = cfg
        self.lambdas = np.asarray(cfg.lambdas, dtype=float)
        self.x_values = np.linspace(cfg.x_min, cfg.x_max, cfg.S, dtype=float)

        # Precompute candidate ReLU outputs and their phase codes for robust circular decoding.
        # The FT architecture maps decoded x_k -> ReLU(x_k), so valid outputs are y_k = max(0, x_k).
        self.y_values = np.maximum(0.0, self.x_values)
        # (S, M) table: phase_table[k, j] = (y_k / lambda_j) mod 1
        self.phase_table = (self.y_values[:, None] / self.lambdas[None, :])
        self.phase_table = self.phase_table - np.floor(self.phase_table)

        spec = arhitecture_specs(
            M=cfg.M,
            m0=1,  # one input
            S=cfg.S,
            lambdas=self.lambdas,
            x_values=self.x_values,
        )
        self.ft = default_arhitecture(
            spec,
            sigma_phase=cfg.sigma_phase,
            sigma_trig=cfg.sigma_trig,
            sigma_score=cfg.sigma_score,
            p=cfg.p_syn,
        )

        # logical weight = 1 for pass-through to decoder
        self.ft.set_logical_weights([1.0])

        # encoder map implements ReLU on the decoded grid
        enc = {float(x): float(max(0.0, x)) for x in self.x_values}
        self.ft.set_encoder_map(enc)
        self.ft.compile_encoder_weights()

    @staticmethod
    def _mod1(x):
        return x - np.floor(x)

    def _encode_scalar(self, x: float) -> np.ndarray:
        # encode x into phases per modulus
        return self._mod1(x / self.lambdas)

    def _decode_output_scalar(self, out_phases: np.ndarray) -> float:
        """Decode y from noisy phases by solving a discrete least-squares problem on the circle.

        We choose y among the valid outputs y_k = ReLU(x_k) such that its expected phase code
        (y_k / lambdas) mod 1 is closest to out_phases in circular distance.

        This is robust when phase noise causes wrap-around near 0/1 boundaries.
        """
        p = np.asarray(out_phases, dtype=float).reshape(1, -1)  # (1, M)
        # circular distance on [0,1): d(a,b) = min(|a-b|, 1-|a-b|)
        diff = np.abs(self.phase_table - p)                     # (S, M)
        diff = np.minimum(diff, 1.0 - diff)
        scores = np.sum(diff * diff, axis=1)                    # (S,)
        k_hat = int(np.argmin(scores))
        return float(self.y_values[k_hat])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU-only numpy pipeline (evaluation replacement; not for training)
        x_cpu = x.detach().to("cpu")
        x_np = x_cpu.numpy()

        flat = x_np.reshape(-1)
        out = np.empty_like(flat, dtype=np.float32)

        for i in range(flat.shape[0]):
            theta = self._encode_scalar(float(flat[i]))[None, :]  # (1, M)
            res = self.ft.forward(theta)

            if "out_phases" not in res:
                raise KeyError(f"default_arhitecture.forward() did not return 'out_phases'. Keys: {list(res.keys())}")

            out_ph = np.asarray(res["out_phases"], dtype=float).reshape(-1)
            out_ph = out_ph - np.floor(out_ph)  # keep phases in [0,1) for circular decode
            
            y = self._decode_output_scalar(out_ph)
            out[i] = y

        out_np = out.reshape(x_np.shape)
        out_t = torch.from_numpy(out_np).to(x.device)
        return out_t


def replace_relu_with_ftrelu(module: nn.Module, ft_relu: nn.Module):
    """
    In-place: replace all nn.ReLU in a model with the provided ft_relu *instance*.
    If you want independent noise RNG per layer, pass a factory instead; here we keep it simple.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, ft_relu)
        else:
            replace_relu_with_ftrelu(child, ft_relu)


# Training / Eval helpers
@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: str, epochs: int, lr: float):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            acc_tr = accuracy(model, train_loader, device)
            acc_te = accuracy(model, test_loader, device)
            print(f"epoch {ep:4d} | train acc {acc_tr:.3f} | test acc {acc_te:.3f}")

    return model


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_class", type=int, default=600)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", type=str, default="cpu")

    # fault-tolerant ReLU params
    ap.add_argument("--M", type=int, default=7)
    ap.add_argument("--S", type=int, default=241)
    ap.add_argument("--x_min", type=float, default=-6.0)
    ap.add_argument("--x_max", type=float, default=6.0)
    ap.add_argument("--sigma_phase", type=float, default=0.00)
    ap.add_argument("--sigma_trig", type=float, default=0.00)
    ap.add_argument("--sigma_score", type=float, default=0.00)
    ap.add_argument("--p_syn", type=float, default=0.00)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data
    X, y = make_3_spirals(args.n_per_class, args.noise, seed=args.seed)
    # train/test split
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    test_ds  = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    # 1) Train normal ReLU MLP
    base = MLP(2, args.hidden, 3)
    print("\nTraining baseline MLP with nn.ReLU ...")
    train(base, train_loader, test_loader, args.device, args.epochs, args.lr)
    base_acc = accuracy(base, test_loader, args.device)
    print(f"\nBaseline test accuracy: {base_acc:.4f}")

    # 2) Copy weights into a fresh model, then replace ReLU with fault-tolerant ReLU
    swapped = MLP(2, args.hidden, 3)
    swapped.load_state_dict(base.state_dict())

    cfg = FTReluConfig(
        M=args.M,
        lambdas=FTReluConfig.lambdas,  # keep default (large) lambdas
        x_min=args.x_min,
        x_max=args.x_max,
        S=args.S,
        sigma_phase=args.sigma_phase,
        sigma_trig=args.sigma_trig,
        sigma_score=args.sigma_score,
        p_syn=args.p_syn,
    )
    ft_relu = FaultTolerantReLU(cfg)

    replace_relu_with_ftrelu(swapped, ft_relu)
    swapped.to(args.device)

    # 3) Evaluate swapped model (fault-tolerant ReLU)
    print("\nEvaluating swapped model (fault-tolerant ReLU) ...")
    swapped_acc = accuracy(swapped, test_loader, args.device)
    print(f"Swapped test accuracy:  {swapped_acc:.4f}")

    # quick sanity print
    print("\nDone.")
    print("Tip: try noise/failure sweeps, e.g.:")
    print("  python train_and_swap_fault_tolerant_relu.py --sigma_phase 0.01")
    print("  python train_and_swap_fault_tolerant_relu.py --p_syn 0.05")


if __name__ == "__main__":
    main()