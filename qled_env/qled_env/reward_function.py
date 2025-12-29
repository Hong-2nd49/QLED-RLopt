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
    """log1p scaling -> about [0,1] when x in [0, ref]."""
    x = max(0.0, float(x))
    ref = max(1e-12, float(ref))
    return _clamp(math.log1p(x) / math.log1p(ref), 0.0, 2.0)


def compute_reward(
    metrics: Dict[str, Any],
    violation: Any = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    # ---- core metrics ----
    eqe = _safe_float(metrics.get("EQE", metrics.get("eqe", 0.0)), 0.0)
    overlap = _clamp(_safe_float(metrics.get("recomb_overlap", 0.0), 0.0), 0.0, 1.0)
    balance = _clamp(_safe_float(metrics.get("inj_balance", 0.0), 0.0), 0.0, 1.0)
    droop = _clamp(_safe_float(metrics.get("droop", 1.0), 1.0), 0.0, 1.0)

    metric_penalty = _safe_float(metrics.get("penalty", 0.0), 0.0)

    # shaping signals from env
    U_prev = metrics.get("U_prev", None)
    U_prev = None if U_prev is None else _safe_float(U_prev, None)
    delta_params_norm = _safe_float(metrics.get("delta_params_norm", 0.0), 0.0)

    # ---- constraint penalties ----
    if isinstance(violation, dict):
        constraint_penalty = sum(max(0.0, _safe_float(v, 0.0)) for v in violation.values())
    else:
        constraint_penalty = max(0.0, _safe_float(violation, 0.0))

    total_penalty = metric_penalty + constraint_penalty

    # ---- shaping: make it easy to climb from near-zero ----
    # IMPORTANT: sqrt has larger gradient near 0 than square
    overlap_s = math.sqrt(overlap + 1e-8)   # 0..1
    balance_s = math.sqrt(balance + 1e-8)   # 0..1
    droop_s = droop  # droop already inside EQE in your SurrogateSim; keep mild

    # EQE can exceed 20 in your surrogate; use ref 40 to avoid early saturation
    eqe_s = _log1p_norm(eqe, ref=40.0)      # ~0..1+

    # ---- multiplicative gating with a floor (no gradient death) ----
    gate_floor = 0.25
    gate_overlap = gate_floor + (1.0 - gate_floor) * overlap_s
    gate_balance = gate_floor + (1.0 - gate_floor) * balance_s

    # core objective (physics-ish): EQE needs overlap & balance to "unlock"
    core = eqe_s * gate_overlap * gate_balance

    # helper dense rewards: explicitly push overlap/balance upward
    helper = 0.55 * overlap_s + 0.55 * balance_s + 0.05 * droop_s

    # ---- penalties: keep soft, but present ----
    pen_soft = math.log1p(max(0.0, total_penalty))

    # jump penalty (reduced): first learn to move, later we can tighten it
    jump_pen = delta_params_norm ** 2

    # ---- weights ----
    w_core = 3.0
    w_helper = 1.0
    w_pen = 1.6
    w_jump = 0.15
    w_delta = 0.25

    U = w_core * core + w_helper * helper - w_pen * pen_soft - w_jump * jump_pen

    # progress shaping
    dU = 0.0
    if U_prev is not None:
        dU = U - U_prev

    # small milestone bonus (won't dominate)
    bonus = 0.0
    if overlap > 0.6:
        bonus += 0.08
    if balance > 0.6:
        bonus += 0.08
    if eqe > 20.0:
        bonus += 0.08

    raw = U + w_delta * dU + bonus

    # bounded reward: avoids hard clipping saturation
    reward = 10.0 * math.tanh(raw)

    info = {
        "U": U,
        "dU": dU,
        "bonus": bonus,
        "raw": raw,
        "reward": reward,
        "eqe": eqe,
        "eqe_s": eqe_s,
        "overlap": overlap,
        "overlap_s": overlap_s,
        "inj_balance": balance,
        "balance_s": balance_s,
        "droop": droop,
        "gate_overlap": gate_overlap,
        "gate_balance": gate_balance,
        "core": core,
        "helper": helper,
        "total_penalty": total_penalty,
        "pen_soft": pen_soft,
        "delta_params_norm": delta_params_norm,
        "weights": {
            "w_core": w_core,
            "w_helper": w_helper,
            "w_pen": w_pen,
            "w_jump": w_jump,
            "w_delta": w_delta,
            "gate_floor": gate_floor,
        },
    }
    return float(reward), info
