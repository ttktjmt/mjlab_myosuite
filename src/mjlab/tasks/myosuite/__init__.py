"""Auto-registration of MyoSuite environments for mjlab."""

# Import registration utilities
from mjlab.tasks.myosuite.registration import register_myosuite_envs

# Import wrapper
from mjlab.tasks.myosuite.wrapper import MyoSuiteVecEnvWrapper


def _register_all_myosuite_envs():
  """Automatically register all MyoSuite environments."""
  try:
    register_myosuite_envs()
  except ImportError:
    # MyoSuite not installed, skip registration
    pass


# Auto-register on import
_register_all_myosuite_envs()

__all__ = ["MyoSuiteVecEnvWrapper", "register_myosuite_envs"]


