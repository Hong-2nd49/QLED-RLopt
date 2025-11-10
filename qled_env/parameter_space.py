import numpy as np

def sample_design() -> dict:
    """
    Sample a QLED design configuration.
    This is the 'state' the RL agent sees and the simulator evaluates.
    """
    return {
        "ZnO_ratio": float(np.random.uniform(0.3, 0.7)),      # lateral ZnO coverage
        "QD_layers": int(np.random.randint(1, 4)),            # number of QD sub-layers
        "HTL_thickness_nm": float(np.random.uniform(15, 30)),
        "ZnO_thickness_nm": float(np.random.uniform(15, 35)),
        "bias_V": float(np.random.uniform(2.5, 4.0)),
        # optional: "comsol_csv": "path/to/file.csv"
    }
