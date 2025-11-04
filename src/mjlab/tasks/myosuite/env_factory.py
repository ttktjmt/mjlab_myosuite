"""Factory for creating MyoSuite environments compatible with mjlab."""

from mjlab.tasks.myosuite.config import MyoSuiteEnvCfg
from mjlab.tasks.myosuite.wrapper import MyoSuiteVecEnvWrapper


def make_myosuite_env(
  myosuite_env_id: str,
  cfg: MyoSuiteEnvCfg | None = None,
  device: str = "cpu",
  render_mode: str | None = None,
  **kwargs,
) -> MyoSuiteVecEnvWrapper:
  """Create a MyoSuite environment wrapped for mjlab.

  Args:
    myosuite_env_id: The original MyoSuite environment ID
    cfg: Environment configuration (MyoSuiteEnvCfg)
    device: Device to use for tensors (default: "cpu")
    render_mode: Render mode (ignored for now, MyoSuite handles rendering differently)
    **kwargs: Additional arguments passed to MyoSuite environment

  Returns:
    Wrapped MyoSuite environment compatible with mjlab
  """
  try:
    from myosuite.utils import gym as myosuite_gym
  except ImportError:
    raise ImportError(
      "MyoSuite is not installed. Install it with: pip install -U myosuite"
    ) from None

  # Don't set environment variables here - let the system use defaults
  # Setting DISPLAY=:99 or MUJOCO_GL=egl breaks the viewer when a real display is available
  # These should only be set when running in a truly headless environment (e.g., during training)
  # For play/viewer mode, the system should use the default display

  # Use cfg if provided, otherwise use defaults
  if cfg is None:
    cfg = MyoSuiteEnvCfg()

  num_envs = cfg.num_envs if hasattr(cfg, "num_envs") else 1
  device = cfg.device if hasattr(cfg, "device") and cfg.device else device

  # Try to create the environment with compatibility workarounds
  try:
    # Create the base MyoSuite environment
    myosuite_env = myosuite_gym.make(myosuite_env_id, **kwargs)
  except AttributeError as e:
    print(e)
    raise RuntimeError(
      f"MyoSuite environment creation failed: {e}\n"
      "This may be due to MuJoCo compatibility issues. "
      "See docs/myosuite_troubleshooting.md for solutions."
    ) from e

  # Wrap it for mjlab compatibility
  wrapped_env = MyoSuiteVecEnvWrapper(
    env=myosuite_env,
    num_envs=num_envs,
    device=device,
  )

  # Get the spec from the original MyoSuite environment and set it on the wrapper
  # This is required by gymnasium's environment checker
  if hasattr(myosuite_env, "spec") and myosuite_env.spec is not None:
    wrapped_env.spec = myosuite_env.spec

  return wrapped_env
