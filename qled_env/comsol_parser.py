import pandas as pd
import numpy as np
from pathlib import Path

def parse_comsol_csv(path) -> dict:
    """
    Parse COMSOL-exported CSV for QLED device simulation.

    Expected columns (adapt this to your COMSOL export):
      - x, z (and optionally y)
      - n_electron
      - n_hole
      - R_rad        # radiative recombination rate density
      - R_nrad       # non-radiative recombination rate density
    """
    path = Path(path)
    df = pd.read_csv(path)

    required = {"x", "n_electron", "n_hole", "R_rad", "R_nrad"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in COMSOL CSV: {missing}")

    ne = df["n_electron"].to_numpy()
    nh = df["n_hole"].to_numpy()
    R_rad = df["R_rad"].to_numpy()
    R_nrad = df["R_nrad"].to_numpy()

    # Spatial overlap between electrons and holes
    if np.all(ne == 0) or np.all(nh == 0):
        overlap = 0.0
    else:
        overlap = float(
            np.sum(ne * nh)
            / (np.sqrt(np.sum(ne**2)) * np.sqrt(np.sum(nh**2)) + 1e-12)
        )

    total_rad = float(np.sum(R_rad))
    total_nrad = float(np.sum(R_nrad))
    total = total_rad + total_nrad + 1e-18

    # Internal EQE proxy
    eqe_proxy = total_rad / total

    # Simple penalty: large non-rad fraction
    penalty = float((total_nrad / total) * 0.2)

    has_y = "y" in df.columns

    carrier_cols = ["x", "z", "n_electron", "n_hole"]
    recomb_cols = ["x", "z", "R_rad", "R_nrad"]
    if has_y:
        carrier_cols.insert(1, "y")
        recomb_cols.insert(1, "y")

    return {
        "carrier_map": df[carrier_cols],
        "recomb_profile": df[recomb_cols],
        "EQE": float(eqe_proxy),
        "recomb_overlap": overlap,
        "penalty": penalty,
    }
