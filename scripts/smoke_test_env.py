import numpy as np

from qled_env.qled_env.parameter_space import ParameterSpace
from qled_env.qled_env.rl_env import QLEDRLEnv
from qled_env.qled_env.simulator_interface import SimulatorInterface


class DummySim(SimulatorInterface):
    def evaluate(self, params):
        # simple smooth "hill" for EQE, peak near mid-range
        x = np.array(list(params.values()), dtype=float)
        eqe = float(np.exp(-np.sum((x - np.mean(x)) ** 2) / (np.var(x) + 1e-6)))
        overlap = float(np.clip(1.0 - abs(params["ps_fill_frac"] - 0.25), 0.0, 1.0))
        penalty = 0.0
        return {"EQE": eqe, "recomb_overlap": overlap, "penalty": penalty}


if __name__ == "__main__":
    ps = ParameterSpace()
    env = QLEDRLEnv(simulator=DummySim(), param_space=ps, max_steps=5)

    obs, info = env.reset()
    print("RESET OK")
    print("  obs shape:", obs.shape)
    print("  metrics:", info["metrics"])
    print("  params keys:", list(info["params"].keys())[:5], "...")

    for i in range(5):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        print(f"STEP {i}: r={r:.4f}, term={terminated}, trunc={truncated}, EQE={info['metrics'].get('EQE')}")
    print("SMOKE TEST PASSED ✅")
