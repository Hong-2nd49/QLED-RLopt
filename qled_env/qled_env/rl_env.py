from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .parameter_space import ParameterSpace
from .reward_function import compute_reward
from .simulator_interface import SimulatorInterface


class QLEDRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: SimulatorInterface,
        param_space: ParameterSpace,
        max_steps: int = 30,
        action_scale: float = 0.05,
        include_metrics_in_obs: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self.simulator = simulator
        self.ps = param_space
        self.max_steps = max_steps
        self.action_scale = float(action_scale)
        self.include_metrics_in_obs = include_metrics_in_obs

        self.rng = np.random.default_rng(seed)

        obs_dim = self.ps.dim
        if self.include_metrics_in_obs:
            obs_dim += self.ps.metrics_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.ps.dim,), dtype=np.float32
        )

        self.step_count = 0
        self.x = None
        self.last_metrics = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        if options and "init_params" in options:
            params = options["init_params"]
            self.x = self.ps.to_normalized(params)
        else:
            self.x = self.ps.sample_normalized(self.rng)

        params_real = self.ps.to_real(self.x)
        self.last_metrics = self.simulator.evaluate(params_real)

        obs = self._build_obs(self.x, self.last_metrics)
        info = {"params": params_real, "metrics": self.last_metrics}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        self.x = self.x + self.action_scale * action
        self.x = np.clip(self.x, -1.0, 1.0)

        params_real = self.ps.to_real(self.x)

        violation = self.ps.constraint_violation(params_real)
        metrics = self.simulator.evaluate(params_real)

        reward, reward_info = compute_reward(metrics, violation)

        terminated = False
        truncated = self.step_count >= self.max_steps

        if self.ps.is_hard_invalid(violation):
            terminated = True

        self.last_metrics = metrics

        obs = self._build_obs(self.x, metrics)
        info = {
            "params": params_real,
            "metrics": metrics,
            "violation": violation,
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def _build_obs(self, x_norm, metrics):
        x = x_norm.astype(np.float32)
        if not self.include_metrics_in_obs:
            return x
        m = self.ps.metrics_to_vec(metrics).astype(np.float32)
        return np.concatenate([x, m], axis=0)
