from stable_baselines3 import PPO

from qled_env.qled_env.rl_env import QLEDRLEnv
from qled_env.qled_env.parameter_space import ParameterSpace
from qled_env.qled_env.surrogate_sim import SurrogateSim


def make_env():
    ps = ParameterSpace()
    env = QLEDRLEnv(
        simulator=SurrogateSim(),
        param_space=ps,
        max_steps=30,
        action_scale=0.05,
    )
    return env


if __name__ == "__main__":
    env = make_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        gamma=0.98,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=10_000)

    obs, info = env.reset()
    total_r = 0.0

    for _ in range(30):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r)
        if terminated or truncated:
            break

    print("Eval total reward:", total_r)
    print("Last metrics:", info["metrics"])
