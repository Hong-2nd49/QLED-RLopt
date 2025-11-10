import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


FEATURE_COLS = [
    "ZnO_ratio",
    "QD_layers",
    "HTL_thickness_nm",
    "ZnO_thickness_nm",
    "bias_V",
]

TARGET_COLS = [
    "EQE_sim",
    "recomb_overlap",
    "penalty",
]


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[["EQE_sim", "recomb_overlap", "penalty"]].to_numpy(dtype=float)
    return X, y


def train_model(X, y, epochs: int = 500, lr: float = 1e-3):
    device = torch.device("cpu")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    model = SurrogateMLP(in_dim=X.shape[1], out_dim=y.shape[1])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}] loss = {loss.item():.6f}")

    return model


def save_model(model: nn.Module, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)


def save_meta(out_path: Path):
    meta = {
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train surrogate model from simulation CSV data.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/generated_designs.csv",
        help="Path to CSV containing designs and metrics.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="surrogate_model/artifacts",
        help="Directory to store trained model and metadata.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    X, y = load_data(csv_path)
    model = train_model(X, y, epochs=args.epochs, lr=args.lr)

    save_model(model, out_dir / "surrogate_mlp.pt")
    save_meta(out_dir / "meta.json")

    print(f"Saved surrogate model and metadata to {out_dir}")


if __name__ == "__main__":
    main()
