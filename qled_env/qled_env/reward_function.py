from __future__ import annotations
from typing import Dict, Any, Tuple
import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe(x: float, default: float = 0.0) -> float:
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _log_eps(x: float, eps: float = 1e-6) -> float:
    # log(eps + max(0,x))
    return math.log(eps + max(0.0, x))


def _log1p(x: float) -> float:
    return math.log1p(max(0.0, x))


def compute_reward(
    metrics: Dict[str, Any],
    violation: Any = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Reward v4: physics-consistent + RL-stable.

    Optional keys in metrics:
      EQE, recomb_overlap, inj_balance, droop, penalty
      brightness, lifetime, auger_rate, leakage
      U_prev (previous utility), delta_params_norm (||Δparams||), step (int)
    """

    # ---- read metrics (robust) ----
    eqe = _safe(metrics.get("EQE", metrics.get("eqe", 0.0)), 0.0)
    overlap = _clamp(_safe(metrics.get("recomb_overlap", metrics.get("overlap", 0.0)), 0.0), 0.0, 1.0)
    balance = _clamp(_safe(metrics.get("inj_balance", metrics.get("balance", 0.0)), 0.0), 0.0, 1.0)
    droop = _clamp(_safe(metrics.get("droop", 1.0), 1.0), 0.0, 1.0)

    brightness = _safe(metrics.get("brightness", 0.0), 0.0)
    lifetime = _safe(metrics.get("lifetime", 0.0), 0.0)
    auger = _safe(metrics.get("auger_rate", metrics.get("auger", 0.0)), 0.0)
    leakage = _safe(metrics.get("leakage", 0.0), 0.0)

    metric_penalty = _safe(metrics.get("penalty", 0.0), 0.0)
    delta_params_norm = _safe(metrics.get("delta_params_norm", 0.0), 0.0)
    U_prev = metrics.get("U_prev", None)
    U_prev = None if U_prev is None else _safe(U_prev, None)

    # ---- constraint penalties ----
    if isinstance(violation, dict):
        constraint_penalty = sum(max(0.0, _safe(v, 0.0)) for v in violation.values())
    else:
        constraint_penalty = max(0.0, _safe(violation, 0.0))

    total_penalty = metric_penalty + constraint_penalty

    # ---- utility in log-space (multiplicative in disguise, but stable) ----
    # EQE main trunk, others are gates in log domain.
    # Add eps so low overlap/balance still has gradient (doesn't go -inf).
    eps_gate = 1e-3

    U_eqe = _log_eps(eqe, eps=1e-3)                # main objective
    U_overlap = math.log(eps_gate + overlap)       # [-6.9, 0] roughly
    U_balance = math.log(eps_gate + balance)
    U_droop = math.log(eps_gate + droop)

    # optional “application” objectives (small weights)
    U_bri = _log1p(brightness)                     # >=0
    U_life = _log1p(lifetime)                      # >=0

    # non-radiative / leakage penalties (>=0)
    P_auger = _log1p(auger)
    P_leak = _log1p(leakage)

    # ---- weights (start conservative) ----
    w_eqe = 1.0
    w_overlap = 0.6
    w_balance = 0.6
    w_droop = 0.3

    w_bri = 0.05
    w_life = 0.08

    w_auger = 0.12
    w_leak = 0.12

    w_pen = 0.8                 # penalty weight
    w_jump = 0.15               # discourage huge parameter jumps
    w_delta = 0.25              # progress shaping (keep modest)

    # ---- combine utility ----
    U = (
        w_eqe * U_eqe
        + w_overlap * U_overlap
        + w_balance * U_balance
        + w_droop * U_droop
        + w_bri * U_bri
        + w_life * U_life
        - w_auger * P_auger
        - w_leak * P_leak
        - w_pen * _log1p(total_penalty)
        - w_jump * (delta_params_norm ** 2)
    )

    # ---- progress shaping (optional, needs env to pass U_prev) ----
    dU = 0.0
    if U_prev is not None:
        dU = U - U_prev

    U_shaped = U + w_delta * dU

    # ---- soft threshold bonus (tiny) ----
    # Smoothly adds a small bonus when EQE surpasses practical thresholds.
    bonus = 0.0
    if eqe > 0.0:
        bonus += 0.10 * (1.0 / (1.0 + math.exp(-0.6 * (eqe - 10.0))))
        bonus += 0.15 * (1.0 / (1.0 + math.exp(-0.6 * (eqe - 20.0))))

    U_shaped += bonus

    # ---- scale + bound for RL stability ----
    # tanh keeps reward nicely bounded; multiply to get around [-10, 10]
    reward = 10.0 * math.tanh(U_shaped)

    info = {
        "reward": reward,
        "U": U,
        "U_shaped": U_shaped,
        "dU": dU,
        "bonus": bonus,
        "eqe": eqe,
        "overlap": overlap,
        "inj_balance": balance,
        "droop": droop,
        "brightness": brightness,
        "lifetime": lifetime,
        "auger_rate": auger,
        "leakage": leakage,
        "metric_penalty": metric_penalty,
        "constraint_penalty": constraint_penalty,
        "total_penalty": total_penalty,
        "delta_params_norm": delta_params_norm,
        "weights": {
            "w_eqe": w_eqe,
            "w_overlap": w_overlap,
            "w_balance": w_balance,
            "w_droop": w_droop,
            "w_bri": w_bri,
            "w_life": w_life,
            "w_auger": w_auger,
            "w_leak": w_leak,
            "w_pen": w_pen,
            "w_jump": w_jump,
            "w_delta": w_delta,
        }
    }

    return float(reward), info
