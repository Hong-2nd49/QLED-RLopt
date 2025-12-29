from __future__ import annotations
from typing import Dict, Any, Tuple
import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _log1p_norm(x: float, ref: float) -> float:
    """
    Normalize non-negative x into ~[0,1] with log1p scaling.
    x=ref -> ~1.
    """
    x = max(0.0, float(x))
    ref = max(1e-12, float(ref))
    return _clamp(math.log1p(x) / math.log1p(ref), 0.0, 1.5)


def compute_reward(
    metrics: Dict[str, Any],
    violation: Any = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Reward v5 (RL-stable + physics-consistent, avoids log-gate negativity):
      - main: shaped EQE
      - helpers: overlap/balance/droop bonuses (quadratic shaping)
      - penalties: simulator penalty + constraint violation (log1p), plus jump penalty
      - progress shaping: small + dU using U_prev (optional)

    Returns (reward, reward_info). reward is clipped to [-10, 10].
    """

    # ---- core metrics ----
    eqe = _safe_float(metrics.get("EQE", metrics.get("eqe", 0.0)), 0.0)
    overlap = _clamp(_safe_float(metrics.get("recomb_overlap", 0.0), 0.0), 0.0, 1.0)
    balance = _clamp(_safe_float(metrics.get("inj_balance", 0.0), 0.0), 0.0, 1.0)
    droop = _clamp(_safe_float(metrics.get("droop", 1.0), 1.0), 0.0, 1.0)

    # optional metrics (may be missing)
    brightness = _safe_float(metrics.get("brightness", 0.0), 0.0)
    lifetime = _safe_float(metrics.get("lifetime", 0.0), 0.0)
    auger = _safe_float(metrics.get("auger_rate", metrics.get("auger", 0.0)), 0.0)
    leakage = _safe_float(metrics.get("leakage", 0.0), 0.0)

    metric_penalty = _safe_float(metrics.get("penalty", 0.0), 0.0)

    # shaping signals injected by env
    U_prev = metrics.get("U_prev", None)
    U_prev = None if U_prev is None else _safe_float(U_prev, None)
    delta_params_norm = _safe_float(metrics.get("delta_params_norm", 0.0), 0.0)

    # ---- constraint violation aggregation ----
    if isinstance(violation, dict):
        constraint_penalty = sum(max(0.0, _safe_float(v, 0.0)) for v in violation.values())
    else:
        constraint_penalty = max(0.0, _safe_float(violation, 0.0))

    total_penalty = metric_penalty + constraint_penalty

    # ---- normalize / shape ----
    # EQE: assume percent-like 0..20 (from your SurrogateSim). adjust ref if needed.
    eqe_s = _log1p_norm(eqe, ref=20.0)          # ~0..1.5
    bri_s = _log1p_norm(brightness, ref=1000.0)
    life_s = _log1p_norm(lifetime, ref=1000.0)
    auger_s = _log1p_norm(auger, ref=1.0)
    leak_s = _log1p_norm(leakage, ref=1.0)

    # quadratic shaping to encourage near-1 without log negativity
    overlap_q = overlap * overlap
    balance_q = balance * balance
    droop_q = droop * droop

    # ---- IMPORTANT: avoid double-counting droop ----
    # In your SurrogateSim, EQE already includes droop multiplier.
    # So we keep droop bonus small, as a gentle preference (mainly for stability).
    w_eqe = 3.0
    w_overlap = 1.0
    w_balance = 1.0
    w_droop = 0.25

    w_bri = 0.10
    w_life = 0.15
    w_auger = 0.20
    w_leak = 0.20

    # penalties
    w_pen = 1.2
    w_jump = 0.6       # encourages smooth parameter changes
    w_delta = 0.6      # progress shaping weight (keep moderate)

    # ---- base utility (always roughly >= 0 if things are decent) ----
    U = (
        w_eqe * eqe_s
        + w_overlap * overlap_q
        + w_balance * balance_q
        + w_droop * droop_q
        + w_bri * bri_s
        + w_life * life_s
        - w_auger * auger_s
        - w_leak * leak_s
        - w_pen * math.log1p(max(0.0, total_penalty))
        - w_jump * (delta_params_norm ** 2)
    )

    dU = 0.0
    if U_prev is not None:
        dU = U - U_prev

    # ---- sparse-ish bonus (tiny, smooth-ish) ----
    bonus = 0.0
    if eqe > 10.0:
        bonus += 0.10
    if eqe > 20.0:
        bonus += 0.15

    raw_reward = U + w_delta * dU + bonus

    # scale + clip for PPO stability
    reward = _clamp(5.0 * raw_reward, -10.0, 10.0)

    info = {
        "U": U,
        "dU": dU,
        "bonus": bonus,
        "eqe": eqe,
        "eqe_s": eqe_s,
        "overlap": overlap,
        "overlap_q": overlap_q,
        "inj_balance": balance,
        "balance_q": balance_q,
        "droop": droop,
        "droop_q": droop_q,
        "brightness": brightness,
        "lifetime": lifetime,
        "auger_rate": auger,
        "leakage": leakage,
        "metric_penalty": metric_penalty,
        "constraint_penalty": constraint_penalty,
        "total_penalty": total_penalty,
        "delta_params_norm": delta_params_norm,
        "raw_reward": raw_reward,
        "reward": reward,
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
        },
    }

    return float(reward), info
