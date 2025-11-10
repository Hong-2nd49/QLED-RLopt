import argparse
from agent.dqn_agent import DQNAgent
from qled_env.simulator_interface import QLEDSimulator
from qled_env.parameter_space import sample_design
from qled_env.reward_function import compute_reward

def parse_args():
    parser = argparse.ArgumentParser(description="Run RL-based QLED architecture optimization.")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of designs to evaluate (episodes).")
    parser.add_argument("--use_surrogate", action="store_true",
                        help="Use surrogate model instead of mock physics.")
    parser.add_argument("--use_comsol", action="store_true",
                        help="Use COMSOL CSV results (developer mode).")
    return parser.parse_args()

def main():
    args = parse_args()

    sim = QLEDSimulator(
        use_surrogate=args.use_surrogate,
        use_comsol=args.use_comsol,
    )

    agent = DQNAgent(
        state_dim=sim.state_dim,
        action_dim=sim.action_dim,
    )

    for ep in range(args.episodes):
        # Stateless design proposal for now (contextual bandit style).
        design = sample_design()
        metrics = sim.evaluate(design)
        reward = compute_reward(metrics)

        agent.learn(
            state=design,
            action=None,
            reward=reward,
            next_state=None,
            done=True,
        )

        print(
            f"[Episode {ep+1:03d}] "
            f"EQE={metrics.get('EQE', 0):.3f}  "
            f"overlap={metrics.get('recomb_overlap', 0):.3f}  "
            f"reward={reward:.3f}"
        )

    print("Optimization run completed.")

if __name__ == "__main__":
    main()
