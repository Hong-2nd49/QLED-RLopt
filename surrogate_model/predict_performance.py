import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


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


class SurrogatePredictor:
    def __init__(self, artifacts_dir: str = "surrogate_model/artifacts"):
        artifacts = Path(artifacts_dir)
        with open(artifacts / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_cols"]
        self.target_cols = meta["target_cols"]

        self.model = SurrogateMLP(
            in_dim=len(self.feature_cols),
            out_dim=len(self.target_cols),
        )
        state_dict = torch.load(artifacts / "surrogate_mlp.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, design: Dict) -> Dict:
        x = np.array([design[col] for col in self.feature_cols], dtype=float)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred = self.model(x_t).numpy().squeeze(0)

        out = dict(zip(self.target_cols, y_pred.tolist()))
        # Map back to metrics naming used elsewhere
        return {
            "EQE": float(out.get("EQE_sim", 0.0)),
            "recomb_overlap": float(out.get("recomb_overlap", 0.0)),
            "penalty": float(out.get("penalty", 0.0)),
        }
