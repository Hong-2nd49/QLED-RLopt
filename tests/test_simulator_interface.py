from qled_env.simulator_interface import QLEDSimulator
from qled_env.parameter_space import sample_design

def test_mock_simulator_runs():
    sim = QLEDSimulator(use_surrogate=False, use_comsol=False)
    design = sample_design()
    metrics = sim.evaluate(design)

    assert "EQE" in metrics
    assert "recomb_overlap" in metrics
    assert "penalty" in metrics
    assert 0.0 <= metrics["recomb_overlap"] <= 1.0
