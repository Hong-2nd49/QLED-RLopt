from __future__ import annotations
from typing import Dict, Any, Tuple


def compute_reward(
    metrics: Dict[str, Any],
    violation: Any = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Physics-inspired reward for QLED device optimization.

    Reward encourages:
    - High EQE
    - Strong e-h recombination overlap
    - Low physical / fabrication penalties

    Args:
        metrics:
            Dictionary returned by simulator, e.g.
            {
              "EQE": float,
              "recomb_overlap": float,
              "penalty": float (optional)
            }

        violation:
            Constraint violation from ParameterSpace.
            Can be:
            - scalar
            - dict of constraint_name -> value

    Returns:
        reward:
            Scalar reward for RL
        reward_info:
            Extra logging info (for analysis / TensorBoard)
    """

    # --- core metrics ---
    eqe = float(metrics.get("EQE", metrics.get("eqe", 0.0)))
    overlap = float(metrics.get("recomb_overlap", 0.0))

    # --- penalties ---
    metric_penalty = float(metrics.get("penalty", 0.0))

    if isinstance(violation, dict):
        constraint_penalty = sum(max(0.0, float(v)) for v in violation.values())
    else:
        constraint_penalty = max(0.0, float(violation))

    total_penalty = metric_penalty + constraint_penalty

    # --- tunable weights (freeze interface, tune values later) ---
    w_eqe = 1.0
    w_overlap = 0.2
    w_penalty = 1.0

    reward = (
        w_eqe * eqe
        + w_overlap * overlap
        - w_penalty * total_penalty
    )

    reward_info = {
        "eqe": eqe,
        "overlap": overlap,
        "metric_penalty": metric_penalty,
        "constraint_penalty": constraint_penalty,
        "total_penalty": total_penalty,
        "reward": reward,
    }

    return reward, reward_info
