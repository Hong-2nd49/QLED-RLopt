from qled_env.reward_function import compute_reward

def test_reward_increases_with_eqe_and_overlap():
    low = compute_reward({"EQE": 0.1, "recomb_overlap": 0.6, "penalty": 0.0})
    high = compute_reward({"EQE": 0.2, "recomb_overlap": 0.8, "penalty": 0.0})
    assert high > low

def test_reward_penalizes_penalty():
    base = compute_reward({"EQE": 0.15, "recomb_overlap": 0.7, "penalty": 0.0})
    penalized = compute_reward({"EQE": 0.15, "recomb_overlap": 0.7, "penalty": 0.1})
    assert penalized < base
