class DQNAgent:
    """
    Minimal placeholder agent.

    This stub keeps the pipeline runnable.
    Replace with a real DQN / PPO implementation later.
    """

    def __init__(self, state_dim=None, action_dim=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def learn(self, state, action, reward, next_state, done: bool):
        # TODO: implement actual learning.
        # For now, this is a no-op so scripts run without error.
        pass
