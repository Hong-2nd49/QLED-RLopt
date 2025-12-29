from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from qled_env.qled_env.rl_env import QLEDRLEnv
from qled_env.qled_env.parameter_space import ParameterSpace
from qled_env.qled_env.surrogate_sim import SurrogateSim


def make_env(seed: int = 0):
    ps = ParameterSpace()
    env = QLEDRLEnv(
        simulator=SurrogateSim(),
        param_space=ps,
        max_steps=30,
        action_scale=0.08,
        include_metrics_in_obs=False,
        seed=seed,
    )
    return Monitor(env)


if __name__ == "__main__":
    env = make_env(seed=0)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        gamma=0.98,
        learning_rate=3e-4,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=50_000)

    obs, info = env.reset()
    total_r = 0.0
    last_info = info

    for _ in range(30):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r)
        last_info = info
        if terminated or truncated:
            break

    print("Eval total reward:", total_r)
    print("Last metrics:", last_info["metrics"])

    # print key params for diagnosing overlap bottleneck
    params = last_info.get("params", {})
    keys = ["t_EML_nm", "t_HTL_nm", "t_ETL_nm", "V_drive", "ps_radius_nm", "ps_fill_frac"]
    print("Last params (key):", {k: params[k] for k in keys if k in params})
