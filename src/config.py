"""Single source of truth for the shared experiment limits (identical for B and C)."""
from src.environment.coordination_env import EnvConfig
from src.environment.reward import RewardConfig

MAX_ATTEMPTS = 4          # 1 greedy + up to 3 sampled solver attempts
ENV_CONFIG = EnvConfig(max_attempts=MAX_ATTEMPTS, allow_verify_more=True)
REWARD_CONFIG = RewardConfig()
