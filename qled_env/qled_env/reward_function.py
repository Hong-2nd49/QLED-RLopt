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
    """log1p scaling; x=ref -> ~1.0."""
    x = max(0.0, float(x))
    ref = max(1e-12, float(ref))
    return _clamp(math.log1p(x) / math.log1p(ref), 0.0, 2.0)


def compute_reward(metrics: Dict[str, Any], violation: Any = 0.0) -> Tuple[float, Dict[str, Any]]:
    """
    Reward v9:
      - core = EQE unlocked by (overlap, balance) gates (physics-like)
      - hinge penalties to prevent "balance-only" or "overlap-only" cheating
      - brightness window: avoid trivial low-drive solutions
      - boundary_penalty: discourage hugging param-space edges
      - tanh(raw / scale): avoid reward saturating at +10 too easily
    """

    # ---- metrics (robust) ----
    eqe = _safe_float(metrics.get("EQE", 0.0), 0.0)
    overlap = _clamp(_safe_float(metrics.get("recomb_overlap", 0.0), 0.0), 0.0, 1.0)
    balance = _clamp(_safe_float(metrics.get("inj_balance", 0.0), 0.0), 0.0, 1.0)
    droop = _clamp(_safe_float(metrics.get("droop", 1.0), 1.0), 0.0, 1.0)

    brightness = _safe_float(metrics.get("brightness", 0.0), 0.0)
    lifetime = _safe_float(metrics.get("lifetime", 0.0), 0.0)
    leakage = _safe_float(metrics.get("leakage", 0.0), 0.0)
    auger = _safe_float(metrics.get("auger_rate", metrics.get("auger", 0.0)), 0.0)

    metric_penalty = _safe_float(metrics.get("penalty", 0.0), 0.0)

    # shaping signals injected by env
    U_prev = metrics.get("U_prev", None)
    U_prev = None if U_prev is None else _safe_float(U_prev, None)
    delta_params_norm = _safe_float(metrics.get("delta_params_norm", 0.0), 0.0)

    # optional env-injected signals
    boundary_penalty = _safe_float(metrics.get("boundary_penalty", 0.0), 0.0)
    V_drive = _safe_float(metrics.get("V_drive", 0.0), 0.0)

    # ---- constraint violation aggregation ----
    if isinstance(violation, dict):
        constraint_penalty = sum(max(0.0, _safe_float(v, 0.0)) for v in violation.values())
    else:
        constraint_penalty = max(0.0, _safe_float(violation, 0.0))

    total_penalty = metric_penalty + constraint_penalty

    # ---- shaping: easy gradients near 0 ----
    overlap_s = math.sqrt(overlap + 1e-8)     # 0..1
    balance_s = math.sqrt(balance + 1e-8)     # 0..1

    # Eqe can be >20 in surrogate; let it breathe a bit
    eqe_s = _log1p_norm(eqe, ref=45.0)

    # ---- multiplicative gates with floor ----
    gate_floor = 0.25
    gate_overlap = gate_floor + (1.0 - gate_floor) * overlap_s
    gate_balance = gate_floor + (1.0 - gate_floor) * balance_s
    core = eqe_s * gate_overlap * gate_balance

    # ---- explicit helper terms (dense guidance) ----
    bri_s = _log1p_norm(brightness, ref=1500.0)
    life_s = _log1p_norm(lifetime, ref=2000.0)
    leak_s = _log1p_norm(leakage, ref=1.0)
    auger_s = _log1p_norm(auger, ref=1.0)

    helper = (
        0.55 * overlap_s
        + 0.35 * balance_s
        + 0.08 * bri_s
        + 0.08 * life_s
        - 0.10 * leak_s
        - 0.10 * auger_s
        + 0.02 * droop
    )

    # ---- hinge penalties: force fixing bottlenecks ----
    target_overlap = 0.40
    target_balance = 0.30
    low_overlap_pen = max(0.0, target_overlap - overlap) ** 2
    low_balance_pen = max(0.0, target_balance - balance) ** 2

    # ---- brightness window: avoid trivial low-drive optimum ----
    # encourage brightness to be within [B_min, B_max]
    B_min, B_max = 600.0, 1800.0
    bri_low_pen = max(0.0, B_min - brightness) ** 2 / (B_min ** 2)
    bri_high_pen = max(0.0, brightness - B_max) ** 2 / (B_max ** 2)

    # ---- soft penalties ----
    pen_soft = math.log1p(max(0.0, total_penalty))
    jump_pen = delta_params_norm ** 2

    # ---- weights (start here; later你可以用optuna扫) ----
    w_core = 3.0
    w_helper = 1.0
    w_pen = 1.4
    w_jump = 0.10
    w_bound = 0.8
    w_low_overlap = 3.0
    w_low_balance = 1.2
    w_bri_window = 1.6

    # mild preference: not too low drive (avoid “全部在2.7V躺赢”)
    # (如果你后续把 surrogate 里亮度/损耗做得更真实，这条可以更弱或去掉)
    v_floor = 2.6
    v_low_pen = max(0.0, v_floor - V_drive) ** 2

    U = (
        w_core * core
        + w_helper * helper
        - w_low_overlap * low_overlap_pen
        - w_low_balance * low_balance_pen
        - w_bri_window * (bri_low_pen + bri_high_pen)
        - w_pen * pen_soft
        - w_jump * jump_pen
        - w_bound * boundary_penalty
        - 0.15 * v_low_pen
    )

    # progress shaping
    dU = 0.0
    if U_prev is not None:
        dU = U - U_prev

    bonus = 0.0
    if overlap > 0.65:
        bonus += 0.08
    if balance > 0.75:
        bonus += 0.05
    if eqe > 20.0:
        bonus += 0.05

    raw = U + 0.20 * dU + bonus

    # IMPORTANT: de-saturate reward (avoid always ~10)
    # raw/2.8 makes tanh less trigger-happy.
    reward = 10.0 * math.tanh(raw / 2.8)

    info = {
        "U": U,
        "dU": dU,
        "bonus": bonus,
        "raw": raw,
        "reward": reward,
        "eqe": eqe,
        "eqe_s": eqe_s,
        "overlap": overlap,
        "inj_balance": balance,
        "droop": droop,
        "brightness": brightness,
        "lifetime": lifetime,
        "leakage": leakage,
        "auger_rate": auger,
        "V_drive": V_drive,
        "delta_params_norm": delta_params_norm,
        "boundary_penalty": boundary_penalty,
        "pen_soft": pen_soft,
        "hinge": {
            "target_overlap": target_overlap,
            "target_balance": target_balance,
            "low_overlap_pen": low_overlap_pen,
            "low_balance_pen": low_balance_pen,
            "bri_low_pen": bri_low_pen,
            "bri_high_pen": bri_high_pen,
        },
        "weights": {
            "w_core": w_core,
            "w_helper": w_helper,
            "w_low_overlap": w_low_overlap,
            "w_low_balance": w_low_balance,
            "w_bri_window": w_bri_window,
            "w_pen": w_pen,
            "w_jump": w_jump,
            "w_bound": w_bound,
            "gate_floor": gate_floor,
        },
    }
    return float(reward), info
