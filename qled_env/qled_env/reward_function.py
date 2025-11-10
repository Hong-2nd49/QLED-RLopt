def compute_reward(metrics: dict) -> float:
    """
    Physics-inspired reward:
    - Encourage high EQE
    - Encourage strong e–h recombination overlap
    - Penalize penalty term (e.g. non-rad losses, too high bias, hotspots)
    """
    eqe = metrics.get("EQE", 0.0)
    overlap = metrics.get("recomb_overlap", 0.0)
    penalty = metrics.get("penalty", 0.0)

    # Tunable weights
    w_eqe = 1.0
    w_overlap = 0.2
    w_penalty = 1.0

    return w_eqe * eqe + w_overlap * overlap - w_penalty * penalty
