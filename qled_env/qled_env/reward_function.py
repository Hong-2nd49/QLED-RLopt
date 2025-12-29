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
    """log1p scaling; x=ref -> ~1.0, clamped."""
    x = max(0.0, float(x))
    ref = max(1e-12, float(ref))
    return _clamp(math.log1p(x) / math.log1p(ref), 0.0, 2.0)


def compute_reward(metrics: Dict[str, Any], violation: Any = 0.0) -> Tuple[float, Dict[str, Any]]:
    """
    Reward v10: Efficiency + Reliability + Anti-cheat (boundaries & low-V tricks)

    - core: EQE unlocked by overlap & injection balance gates (multiplicative)
    - reliability: explicitly penalize droop/leakage/auger and reward lifetime
    - windows: brightness window avoids trivial low-V or extreme-V wins
    - anti-cheat: boundary_penalty + eml_thin_penalty from env
    - stable output: tanh(raw/scale) to avoid saturating at 10 too easily
    """

    # --- base metrics ---
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

    boundary_penalty = _safe_float(metrics.get("boundary_penalty", 0.0), 0.0)
    eml_thin_penalty = _safe_float(metrics.get("eml_thin_penalty", 0.0), 0.0)
    V_drive = _safe_float(metrics.get("V_drive", 0.0), 0.0)

    # --- constraint penalty aggregation ---
    if isinstance(violation, dict):
        constraint_penalty = sum(max(0.0, _safe_float(v, 0.0)) for v in violation.values())
    else:
        constraint_penalty = max(0.0, _safe_float(violation, 0.0))

    total_penalty = metric_penalty + constraint_penalty
    pen_soft = math.log1p(max(0.0, total_penalty))
    jump_pen = delta_params_norm ** 2

    # --- shaping (good gradients near 0) ---
    overlap_s = math.sqrt(overlap + 1e-8)
    balance_s = math.sqrt(balance + 1e-8)
    eqe_s = _log1p_norm(eqe, ref=30.0)

    # multiplicative gates with floor
    gate_floor = 0.25
    gate_overlap = gate_floor + (1.0 - gate_floor) * overlap_s
    gate_balance = gate_floor + (1.0 - gate_floor) * balance_s
    core = eqe_s * gate_overlap * gate_balance

    # normalized reliability terms
    bri_s = _log1p_norm(brightness, ref=1500.0)
    life_s = _log1p_norm(lifetime, ref=2000.0)
    leak_s = _log1p_norm(leakage, ref=1.0)
    auger_s = _log1p_norm(auger, ref=1.0)

    # --- hinge targets (force good regions) ---
    # overlap/balance not too low
    target_overlap = 0.45
    target_balance = 0.40
    low_overlap_pen = max(0.0, target_overlap - overlap) ** 2
    low_balance_pen = max(0.0, target_balance - balance) ** 2

    # droop floor: avoid winning by pushing V too high (droop collapse)
    droop_floor = 0.60
    low_droop_pen = max(0.0, droop_floor - droop) ** 2

    # lifetime floor: avoid "high EQE but dies fast"
    life_floor = 800.0
    low_life_pen = max(0.0, (life_floor - lifetime) / life_floor) ** 2

    # brightness window: keep within a practical range
    B_min, B_max = 700.0, 1800.0
    bri_low_pen = max(0.0, (B_min - brightness) / B_min) ** 2
    bri_high_pen = max(0.0, (brightness - B_max) / B_max) ** 2

    # mild high-V regularizer (optional safety rail)
    v_high = 4.2
    v_high_pen = max(0.0, (V_drive - v_high) / v_high) ** 2

    # --- helper dense reward (still useful) ---
    helper = (
        0.55 * overlap_s
        + 0.35 * balance_s
        + 0.10 * bri_s
        + 0.18 * life_s
        - 0.25 * leak_s
        - 0.25 * auger_s
        + 0.03 * droop
    )

    # --- weights (tuned for your current failure mode: high V -> droop/auger/leak -> low life) ---
    w_core = 3.0
    w_helper = 1.0
    w_pen = 1.2
    w_jump = 0.10
    w_bound = 1.2
    w_eml_thin = 1.2

    w_low_overlap = 3.0
    w_low_balance = 1.5
    w_low_droop = 4.0
    w_low_life = 3.0
    w_bri_window = 1.4
    w_v_high = 0.4

    U = (
        w_core * core
        + w_helper * helper
        - w_low_overlap * low_overlap_pen
        - w_low_balance * low_balance_pen
        - w_low_droop * low_droop_pen
        - w_low_life * low_life_pen
        - w_bri_window * (bri_low_pen + bri_high_pen)
        - w_pen * pen_soft
        - w_jump * jump_pen
        - w_bound * boundary_penalty
        - w_eml_thin * eml_thin_penalty
        - w_v_high * v_high_pen
    )

    # progress shaping (small)
    dU = 0.0
    if U_prev is not None:
        dU = U - U_prev

    # tiny milestones
    bonus = 0.0
    if overlap > 0.70:
        bonus += 0.05
    if balance > 0.80:
        bonus += 0.03
    if lifetime > 900.0:
        bonus += 0.05

    raw = U + 0.18 * dU + bonus

    # de-saturate: reduce "always near +10"
    reward = 10.0 * math.tanh(raw / 3.2)

    info = {
        "U": U,
        "dU": dU,
        "bonus": bonus,
        "raw": raw,
        "reward": reward,
        "eqe": eqe,
        "overlap": overlap,
        "inj_balance": balance,
        "droop": droop,
        "brightness": brightness,
        "leakage": leakage,
        "auger_rate": auger,
        "lifetime": lifetime,
        "V_drive": V_drive,
        "delta_params_norm": delta_params_norm,
        "boundary_penalty": boundary_penalty,
        "eml_thin_penalty": eml_thin_penalty,
        "pen_soft": pen_soft,
        "hinge": {
            "target_overlap": target_overlap,
            "target_balance": target_balance,
            "droop_floor": droop_floor,
            "life_floor": life_floor,
            "low_overlap_pen": low_overlap_pen,
            "low_balance_pen": low_balance_pen,
            "low_droop_pen": low_droop_pen,
            "low_life_pen": low_life_pen,
            "bri_low_pen": bri_low_pen,
            "bri_high_pen": bri_high_pen,
        },
    }
    return float(reward), info
