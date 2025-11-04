"""Default configuration generators for MyoSuite environments."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class MyoSuiteEnvCfg:
  """Minimal configuration for MyoSuite environments.

  This is a placeholder config that allows MyoSuite environments to work
  with mjlab's training infrastructure. The actual environment configuration
  comes from MyoSuite itself.
  """
  num_envs: int = 1
  """Number of parallel environments."""
  device: str = "cpu"
  """Device to use (MyoSuite uses CPU MuJoCo, so GPU acceleration is limited)."""


def get_default_myosuite_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  """Get default RL configuration for MyoSuite environments.

  Returns:
    Default RslRlOnPolicyRunnerCfg with reasonable defaults for MyoSuite tasks
  """
  return RslRlOnPolicyRunnerCfg(
    experiment_name="myosuite",
    run_name="",
    max_iterations=1000,
    num_steps_per_env=24,
    policy=RslRlPpoActorCriticCfg(
      actor_hidden_dims=(256, 256),
      critic_hidden_dims=(256, 256),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      learning_rate=3e-4,
      num_learning_epochs=5,
      num_mini_batches=4,
      gamma=0.99,
      lam=0.95,
    ),
    clip_actions=None,  # Let MyoSuite handle action bounds
  )

