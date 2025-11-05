"""Wrapper for MyoSuite environments to work with mjlab's training infrastructure."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import vector
from gymnasium.vector import SyncVectorEnv
from mjlab.envs import ManagerBasedRlEnvCfg
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class MyoSuiteVecEnvWrapper(VecEnv, gym.Env):
  """Wraps a vectorized MyoSuite environment to work with mjlab's RslRlVecEnvWrapper.

  This wrapper:
  1. Vectorizes single MyoSuite environments
  2. Converts numpy arrays to torch tensors
  3. Handles observation/action space conversions
  4. Provides a mock cfg attribute for compatibility
  """

  # Mark as vectorized environment for gymnasium
  is_vector_env = True

  # Add metadata for gymnasium
  metadata = {"render_modes": [None, "rgb_array"]}

  def __init__(
    self,
    env: gym.Env | vector.VectorEnv,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    clip_actions: float | None = None,
  ):
    """Initialize the wrapper.

    Args:
      env: Either a single MyoSuite environment or already vectorized environment
      num_envs: Number of environments (if env is single, will vectorize)
      device: Device to use for tensors
      clip_actions: Optional action clipping value
    """
    # Initialize gym.Env parent (no-op but required for proper inheritance)
    gym.Env.__init__(self)

    self.clip_actions = clip_actions

    # Normalize device specification (accepts "CUDA:0", "cuda:0", torch.device, etc.)
    def _normalize_device(d: str | torch.device) -> torch.device:
      if isinstance(d, torch.device):
        return d
      d_str = str(d).strip()
      d_lc = d_str.lower()
      if d_lc.startswith("cuda"):
        # Accept forms like "cuda" or "cuda:0" or uppercase variants
        if ":" in d_lc:
          idx = d_lc.split(":", 1)[1]
          dev = f"cuda:{idx}"
        else:
          dev = "cuda"
        try:
          return torch.device(dev)
        except Exception:
          return torch.device("cpu")
      # Default to cpu for anything else
      return torch.device("cpu")

    norm_device = _normalize_device(device)
    self.device_str = str(norm_device)
    self.device = norm_device

    # Vectorize if needed
    if isinstance(env, vector.VectorEnv):
      self.env = env
      self.num_envs = env.num_envs
    else:
      # Single environment - need to vectorize
      if num_envs is None:
        num_envs = 1

      # Create environment factories - each lambda captures env in its own closure
      # This ensures each environment is independent (though for MyoSuite they share the same instance)
      def make_env_factory(env_instance):
        """Create a factory that returns the env instance."""
        return lambda: env_instance

      env_factories = [make_env_factory(env) for _ in range(num_envs)]
      self.env = SyncVectorEnv(env_factories)
      self.num_envs = num_envs

    # Get action/observation spaces
    self.single_action_space = self.env.single_action_space
    original_obs_space = self.env.single_observation_space
    self.action_space = self.env.action_space
    original_obs_space_vec = self.env.observation_space

    # Modify observation space to include both 'policy' and 'critic' groups for RSL-RL
    # RSL-RL expects observations as a dict with 'policy' and 'critic' keys
    if not isinstance(original_obs_space, gym.spaces.Dict):
      # Convert single observation space to Dict with both policy and critic
      # Both use the same space (since we don't have privileged info for critic)
      self.single_observation_space = gym.spaces.Dict(
        {
          "policy": original_obs_space,
          "critic": original_obs_space,
        }
      )
      # Update vectorized observation space
      self.observation_space = gym.vector.utils.batch_space(
        self.single_observation_space, self.num_envs
      )
    elif "policy" not in original_obs_space or "critic" not in original_obs_space:
      # If it's already a Dict but missing policy/critic, add them
      spaces_dict = dict(original_obs_space.spaces)
      if "policy" not in spaces_dict:
        # Use the first available space or create a default
        first_space = (
          next(iter(spaces_dict.values()))
          if spaces_dict
          else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        )
        spaces_dict["policy"] = first_space
      if "critic" not in spaces_dict:
        # Use policy space for critic
        spaces_dict["critic"] = spaces_dict["policy"]
      self.single_observation_space = gym.spaces.Dict(spaces_dict)
      self.observation_space = gym.vector.utils.batch_space(
        self.single_observation_space, self.num_envs
      )
    else:
      # Already has both policy and critic, use as-is
      self.single_observation_space = original_obs_space
      self.observation_space = original_obs_space_vec

    # Extract action dimension
    if isinstance(self.single_action_space, gym.spaces.Box):
      self.num_actions = int(np.prod(self.single_action_space.shape))
    elif isinstance(self.single_action_space, gym.spaces.Discrete):
      self.num_actions = 1
    else:
      # Try to get shape attribute
      self.num_actions = int(np.prod(getattr(self.single_action_space, "shape", (1,))))

    # Estimate max episode length (MyoSuite environments typically have timeout)
    # Default to 1000 steps if not available
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    max_episode_length = None

    # Try to get max_episode_steps from spec
    if hasattr(unwrapped_env, "spec") and unwrapped_env.spec is not None:
      max_episode_length = getattr(unwrapped_env.spec, "max_episode_steps", None)

    # If not found in spec, try direct attribute
    if max_episode_length is None:
      max_episode_length = getattr(unwrapped_env, "max_episode_steps", None)

    # If still None, try to get from the vectorized environment
    if max_episode_length is None and isinstance(self.env, vector.VectorEnv):
      # Try to get from the first environment in the vector
      if hasattr(self.env, "envs") and len(self.env.envs) > 0:
        first_env = self.env.envs[0]
        if hasattr(first_env, "spec") and first_env.spec is not None:
          max_episode_length = getattr(first_env.spec, "max_episode_steps", None)
        if max_episode_length is None:
          max_episode_length = getattr(first_env, "max_episode_steps", None)

    # Set with default fallback
    self.max_episode_length = (
      int(max_episode_length) if max_episode_length is not None else 1000
    )

    # Create mock cfg for compatibility
    self._mock_cfg = self._create_mock_cfg()

    # Track episode lengths
    self.episode_length_buf = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )

    # Track last observation for get_observations()
    self._last_obs_dict: dict[str, Any] = {}

    # Create mock sim object for viewer compatibility
    # MyoSuite environments have mj_model and mj_data directly on the unwrapped env
    self._mock_sim = self._create_mock_sim()

    # Modify action space if clipping is enabled
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset
    obs, _ = self.env.reset()
    self._last_obs_dict = self._convert_obs_to_dict(obs)
    # Verify that observations are on the correct device after initial conversion
    for key, value in self._last_obs_dict.items():
      if isinstance(value, torch.Tensor):
        if value.device != self.device:
          # Force move to correct device immediately
          self._last_obs_dict[key] = value.to(device=self.device)

  def _create_mock_sim(self) -> Any:
    """Create a mock sim object that exposes mj_model, mj_data, and wp_data for viewer compatibility.

    Returns:
      Mock object compatible with both native and Viser viewers
    """
    # Get the underlying MyoSuite environment
    # For SyncVectorEnv, get the first environment
    if isinstance(self.env, vector.VectorEnv):
      if hasattr(self.env, "envs") and len(self.env.envs) > 0:
        myosuite_env = self.env.envs[0]
      elif hasattr(self.env, "single_env"):
        myosuite_env = self.env.single_env
      else:
        # Fallback: try to unwrap
        myosuite_env = getattr(self.env, "unwrapped", None)
    else:
      myosuite_env = self.env

    # Unwrap to get the actual MyoSuite environment
    while (
      hasattr(myosuite_env, "unwrapped") and myosuite_env.unwrapped is not myosuite_env
    ):
      myosuite_env = myosuite_env.unwrapped

    # Create a mock wp_data object that provides numpy arrays from mj_data
    class MockWpData:
      """Mock wp_data that converts mj_data arrays to the format Viser expects."""

      def __init__(self, env, num_envs: int = 1):
        self._env = env
        self._num_envs = num_envs

      @property
      def _mj_model(self):
        """Access mj_model dynamically from environment."""
        return self._env.mj_model

      @property
      def _mj_data(self):
        """Access mj_data dynamically from environment to ensure it's always current."""
        return self._env.mj_data

      def _to_batched(self, arr: np.ndarray) -> np.ndarray:
        """Convert a 1D or 2D array to batched format (batch_size, ...)"""
        arr = np.asarray(arr)
        if arr.ndim == 0:
          return arr.reshape(1)
        elif arr.ndim == 1:
          return arr[np.newaxis, :]  # (1, n)
        elif arr.ndim == 2:
          return arr[np.newaxis, :, :]  # (1, n, m)
        else:
          return arr[np.newaxis, ...]  # (1, ...)

      def _make_array_proxy(self, data: np.ndarray) -> Any:
        """Create an object with .numpy() method that returns the data."""
        batched = self._to_batched(data)

        class ArrayProxy:
          def __init__(self, arr: np.ndarray):
            self._arr = arr

          def numpy(self) -> np.ndarray:
            return self._arr

        return ArrayProxy(batched)

      @property
      def qpos(self):
        """Joint positions. Shape: (batch_size, nq)"""
        return self._make_array_proxy(self._mj_data.qpos)

      @property
      def qvel(self):
        """Joint velocities. Shape: (batch_size, nv)"""
        return self._make_array_proxy(self._mj_data.qvel)

      @property
      def xpos(self):
        """Body positions. Shape: (batch_size, nbody, 3)"""
        # Ensure mj_forward has been called to update xpos
        # MyoSuite environments update mj_data during step, but we need to ensure
        # forward kinematics are computed for visualization
        import mujoco

        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # xpos is shape (nbody, 3), we need to add batch dimension
        xpos_array = self._mj_data.xpos
        return self._make_array_proxy(xpos_array)

      @property
      def xmat(self):
        """Body orientation matrices. Shape: (batch_size, nbody, 3, 3)"""
        # Ensure mj_forward has been called to update xmat
        import mujoco

        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # xmat is shape (nbody, 9), reshape to (nbody, 3, 3), then add batch dimension
        xmat_array = self._mj_data.xmat.reshape(-1, 3, 3)
        return self._make_array_proxy(xmat_array)

      @property
      def geom_xpos(self):
        """Geometry positions. Shape: (batch_size, ngeom, 3)"""
        # Ensure mj_forward has been called to update geom_xpos
        import mujoco

        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # geom_xpos is shape (ngeom, 3), we need to add batch dimension
        return self._make_array_proxy(self._mj_data.geom_xpos)

      @property
      def geom_xmat(self):
        """Geometry orientation matrices. Shape: (batch_size, ngeom, 3, 3)"""
        # Ensure mj_forward has been called to update geom_xmat
        import mujoco

        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # geom_xmat is shape (ngeom, 9), reshape to (ngeom, 3, 3), then add batch dimension
        return self._make_array_proxy(self._mj_data.geom_xmat.reshape(-1, 3, 3))

      @property
      def mocap_pos(self):
        """Mocap positions. Shape: (batch_size, nmocap, 3)"""
        if self._mj_model.nmocap > 0:
          return self._make_array_proxy(self._mj_data.mocap_pos)
        else:
          # Return empty array with correct shape
          empty = np.zeros((0, 3), dtype=np.float32)
          return self._make_array_proxy(empty)

      @property
      def mocap_quat(self):
        """Mocap quaternions. Shape: (batch_size, nmocap, 4)"""
        if self._mj_model.nmocap > 0:
          return self._make_array_proxy(self._mj_data.mocap_quat)
        else:
          # Return empty array with correct shape
          empty = np.zeros((0, 4), dtype=np.float32)
          return self._make_array_proxy(empty)

    # Create a mock sim object that inherits from Simulation to pass isinstance checks
    # We need to import Simulation here to avoid circular imports
    try:
      from mjlab.sim.sim import Simulation

      class MockSim(Simulation):
        """Mock Simulation that works with MyoSuite environments.

        This class inherits from Simulation to pass isinstance checks,
        but bypasses the normal __init__ to avoid MuJoCo Warp requirements.
        """

        def __init__(self, env):
          # Don't call super().__init__() - we're bypassing MuJoCo Warp setup
          # Instead, set up minimal attributes needed for compatibility
          self._env = env
          self._mj_model = env.mj_model
          self._mj_data = env.mj_data
          self._wp_data = MockWpData(env, num_envs=1)

          # Set minimal attributes that Simulation expects
          self.num_envs = 1
          self.device = "cpu"  # MyoSuite runs on CPU
          # Set cfg to None or a minimal object if needed
          self.cfg = None

        @property
        def mj_model(self):
          return self._env.mj_model

        @property
        def mj_data(self):
          """Return mj_data, ensuring forward kinematics are up to date."""
          import mujoco

          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)  # type: ignore[attr-defined]
          return self._env.mj_data

        @property
        def wp_data(self):
          """Return mock wp_data for Viser viewer compatibility."""
          # Ensure forward kinematics are computed before accessing wp_data
          import mujoco

          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)  # type: ignore[attr-defined]
          return self._wp_data

        @property
        def data(self):
          """Return mj_data for compatibility with sim.data access."""
          import mujoco

          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)  # type: ignore[attr-defined]
          return self._env.mj_data

        # Override methods that might be called but aren't needed
        def create_graph(self) -> None:
          """No-op for MyoSuite (no CUDA graphs needed)."""
          pass

        def forward(self) -> None:
          """Update forward kinematics for visualization."""
          import mujoco

          # Ensure forward kinematics are computed for visualization
          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)  # type: ignore[attr-defined]

        def step(self) -> None:
          """No-op for MyoSuite (step handled by MyoSuite)."""
          pass

      # Create and return the mock sim
      mock_sim = MockSim(myosuite_env)

    except (ImportError, TypeError):
      # Fallback: if Simulation import fails or inheritance doesn't work,
      # create a regular class and use __class__ manipulation
      class MockSim:
        def __init__(self, env):
          self._env = env
          self._mj_model = env.mj_model
          self._mj_data = env.mj_data
          self._wp_data = MockWpData(env, num_envs=1)
          self.num_envs = 1
          self.device = "cpu"
          self.cfg = None

        @property
        def mj_model(self):
          return self._env.mj_model

        @property
        def mj_data(self):
          """Return mj_data, ensuring forward kinematics are up to date."""
          import mujoco

          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)
          return self._env.mj_data

        @property
        def wp_data(self):
          """Return mock wp_data for Viser viewer compatibility."""
          import mujoco

          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)
          return self._wp_data

        @property
        def data(self):
          """Return mj_data for compatibility with sim.data access."""
          import mujoco

          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)
          return self._env.mj_data

        def create_graph(self) -> None:
          pass

        def forward(self) -> None:
          """Update forward kinematics for visualization."""
          import mujoco

          mujoco.mj_forward(self._env.mj_model, self._env.mj_data)

        def step(self) -> None:
          pass

      mock_sim = MockSim(myosuite_env)

      # Try to make it pass isinstance check by manipulating __class__
      try:
        from mjlab.sim.sim import Simulation

        # Create a dynamic subclass that inherits from Simulation
        class MockSimulationSubclass(Simulation):
          pass

        # Change the instance's class to the subclass
        object.__setattr__(mock_sim, "__class__", MockSimulationSubclass)
      except (ImportError, TypeError, AttributeError):
        # If this fails, the isinstance check in Viser will use the interface check instead
        pass

    return mock_sim

  def _create_mock_cfg(self) -> ManagerBasedRlEnvCfg:
    """Create a mock cfg for compatibility with RslRlVecEnvWrapper."""
    # Create a dataclass-based mock cfg so it can be converted to dict for logging
    from dataclasses import dataclass, field

    from mjlab.viewer import ViewerConfig

    @dataclass
    class MockCfg:
      is_finite_horizon: bool = False
      viewer: ViewerConfig = field(default_factory=ViewerConfig)

      def to_dict(self) -> dict:
        """Convert to dictionary for wandb logging."""
        from dataclasses import asdict

        return asdict(self)

    return MockCfg()  # type: ignore[return-value]

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    """Return mock cfg for compatibility."""
    return self._mock_cfg  # type: ignore[return-value]

  @property
  def sim(self) -> Any:
    """Return mock sim object for viewer compatibility."""
    return self._mock_sim

  @property
  def unwrapped(self) -> "MyoSuiteVecEnvWrapper":
    """Return self as unwrapped (for viewer compatibility).

    This allows the viewer to access env.unwrapped.sim which will
    return our mock sim object with mj_model and mj_data.
    """
    return self

  @property
  def render_mode(self) -> str | None:
    """Get render mode."""
    return getattr(self.env, "render_mode", None)

  @classmethod
  def class_name(cls) -> str:
    """Return class name."""
    return cls.__name__

  # Gymnasium environments expose an attribute `spec: EnvSpec | None`.
  # We keep a plain attribute to satisfy static type checkers.
  spec: Any = None

  def seed(self, seed: int = -1) -> int:
    """Set seed for environment."""
    if hasattr(self.env, "seed"):
      return self.env.seed(seed)  # type: ignore[attr-defined]
    return seed

  def get_observations(self) -> TensorDict:
    """Get observations as TensorDict.

    CRITICAL: RSL-RL expects observations on the same device as the policy.
    This method ensures all observation tensors are on self.device.
    """
    # Rebuild observation dict, ensuring all tensors are on the correct device
    # This is necessary because tensors might have been created on CPU initially
    obs_dict_on_device = {}
    for key, value in self._last_obs_dict.items():
      if isinstance(value, torch.Tensor):
        # CRITICAL: Always create a NEW tensor on the target device
        # This ensures the tensor is definitely on the correct device
        # Use .contiguous() to ensure it's a proper tensor, not a view
        if value.device != self.device:
          # Move to device - this creates a new tensor if needed
          obs_dict_on_device[key] = value.to(device=self.device).contiguous()
        else:
          # Even if on correct device, ensure it's contiguous
          obs_dict_on_device[key] = value.contiguous()
      else:
        # For non-tensors, convert to tensor on target device
        if isinstance(value, np.ndarray):
          obs_dict_on_device[key] = (
            torch.from_numpy(value)
            .to(device=self.device, dtype=torch.float32)
            .contiguous()
          )
        else:
          obs_dict_on_device[key] = torch.tensor(
            value, device=self.device, dtype=torch.float32
          ).contiguous()

    # Create TensorDict
    td = TensorDict(obs_dict_on_device, batch_size=[self.num_envs])

    # CRITICAL: RSL-RL accesses obs['policy'] directly, so we MUST ensure it's on correct device
    # Force move 'policy' observation to correct device with explicit verification
    if "policy" in td:
      policy_val = td["policy"]
      if isinstance(policy_val, torch.Tensor):
        # Create a completely new tensor on the target device
        # Use .clone() to ensure it's a new tensor, then .to() to move it
        policy_on_device = policy_val.clone().to(device=self.device, non_blocking=False)
        # Verify it's actually on the device
        if policy_on_device.device != self.device:
          # If still not on device, force it using CUDA or CPU explicit placement
          if str(self.device).startswith("cuda"):
            policy_on_device = policy_on_device.cuda(device=self.device)
          else:
            policy_on_device = policy_on_device.cpu()
        td["policy"] = policy_on_device.contiguous()

      # Also ensure 'critic' is on correct device (same as policy for MyoSuite)
      if "critic" in td:
        critic_val = td["critic"]
        if isinstance(critic_val, torch.Tensor):
          critic_on_device = critic_val.clone().to(
            device=self.device, non_blocking=False
          )
          if critic_on_device.device != self.device:
            if str(self.device).startswith("cuda"):
              critic_on_device = critic_on_device.cuda(device=self.device)
            else:
              critic_on_device = critic_on_device.cpu()
          td["critic"] = critic_on_device.contiguous()

    return td

  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
  ) -> tuple[TensorDict, dict]:
    """Reset the environment (gym-compatible signature)."""
    try:
      obs, info = self.env.reset(seed=seed, options=options)
    except TypeError:
      # Fallback for older envs without seed/options
      obs, info = self.env.reset()

    # Update forward kinematics for visualization after reset
    # This is critical for the viewer to show the initial state
    if hasattr(self, "_mock_sim") and hasattr(self._mock_sim, "_env"):
      import mujoco

      mujoco.mj_forward(self._mock_sim._env.mj_model, self._mock_sim._env.mj_data)

    # Convert to torch tensors and store for get_observations()
    obs_dict = self._convert_obs_to_dict(obs)
    # CRITICAL: Ensure all tensors in obs_dict are on the correct device
    for key, value in obs_dict.items():
      if isinstance(value, torch.Tensor):
        if value.device != self.device:
          obs_dict[key] = value.to(device=self.device)
    self._last_obs_dict = obs_dict
    self.episode_length_buf.zero_()

    return TensorDict(obs_dict, batch_size=[self.num_envs]), info

  def step(  # type: ignore[override]
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    """Step the environment."""
    # Convert actions to numpy
    if isinstance(actions, torch.Tensor):
      actions_np = actions.cpu().numpy()
    else:
      actions_np = actions

    # Clip actions if needed
    if self.clip_actions is not None:
      actions_np = np.clip(actions_np, -self.clip_actions, self.clip_actions)

    # Step environment
    obs, rew, terminated, truncated, info = self.env.step(actions_np)

    # Update forward kinematics for visualization
    # This ensures the viewer has current position data
    if hasattr(self, "_mock_sim") and hasattr(self._mock_sim, "_env"):
      import mujoco

      mujoco.mj_forward(self._mock_sim._env.mj_model, self._mock_sim._env.mj_data)

    # Convert to torch tensors and store for get_observations()
    obs_dict = self._convert_obs_to_dict(obs)
    # CRITICAL: Ensure all tensors in obs_dict are on the correct device
    for key, value in obs_dict.items():
      if isinstance(value, torch.Tensor):
        if value.device != self.device:
          obs_dict[key] = value.to(device=self.device)
    self._last_obs_dict = obs_dict
    rew_tensor = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
    terminated_tensor = torch.as_tensor(
      terminated, device=self.device, dtype=torch.bool
    )
    truncated_tensor = torch.as_tensor(truncated, device=self.device, dtype=torch.bool)

    # Update episode lengths
    self.episode_length_buf += 1

    # Reset episode lengths for terminated/truncated envs
    done_mask = terminated_tensor | truncated_tensor
    self.episode_length_buf[done_mask] = 0

    # Combine terminated and truncated
    term_or_trunc = terminated_tensor | truncated_tensor
    dones = term_or_trunc.to(dtype=torch.long)

    # Add time_outs to extras
    extras = info.copy() if isinstance(info, dict) else {}
    extras["time_outs"] = truncated_tensor

    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      rew_tensor,
      dones,
      extras,
    )

  def _convert_obs_to_dict(self, obs: Any) -> dict[str, Any]:
    """Convert observations to dictionary of torch tensors.

    RSL-RL expects observation groups like 'policy' and 'critic'.
    For MyoSuite, we map all observations to both 'policy' and 'critic' groups
    (since we don't have privileged information for critic).
    """
    if isinstance(obs, dict):
      obs_dict = {}
      policy_tensors = []
      for key, value in obs.items():
        # Create tensor and ensure it's on the correct device
        # Always create on target device explicitly to avoid any device issues
        if isinstance(value, torch.Tensor):
          # If already a tensor, ensure it's on the correct device
          if value.device != self.device:
            tensor = value.to(device=self.device).to(dtype=torch.float32)
          else:
            tensor = value.to(dtype=torch.float32)
        elif isinstance(value, np.ndarray):
          # For numpy arrays, create directly on target device
          # Use from_numpy + to() to ensure proper device placement
          tensor = torch.from_numpy(value).to(device=self.device, dtype=torch.float32)
        else:
          # For other types, convert to tensor on target device
          tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
        # Map to 'policy' group for RSL-RL compatibility
        # If key is already 'policy' or 'critic', keep it; otherwise collect for policy
        if key in ["policy", "critic"]:
          obs_dict[key] = tensor
        else:
          # For MyoSuite observations, collect them to concatenate into 'policy' group
          policy_tensors.append(tensor)
      # Concatenate all non-policy/critic observations into a single 'policy' tensor
      if policy_tensors:
        policy_obs = (
          torch.cat(policy_tensors, dim=-1)
          if len(policy_tensors) > 1
          else policy_tensors[0]
        )
        obs_dict["policy"] = policy_obs
        # Also provide 'critic' observation (same as policy for MyoSuite)
        if "critic" not in obs_dict:
          obs_dict["critic"] = policy_obs
      return obs_dict
    elif isinstance(obs, np.ndarray):
      # Single observation array - map to both 'policy' and 'critic' groups for RSL-RL
      # Create tensor directly on the correct device using from_numpy + to()
      policy_obs = torch.from_numpy(obs).to(device=self.device, dtype=torch.float32)
      return {"policy": policy_obs, "critic": policy_obs}
    elif isinstance(obs, (list, tuple)):
      # Handle list/tuple of observations
      if isinstance(obs[0], dict):
        # List of dicts - convert to dict of arrays
        obs_dict = {}
        for key in obs[0].keys():
          tensor = torch.as_tensor(
            np.array([o[key] for o in obs]), device=self.device, dtype=torch.float32
          )
          if key in ["policy", "critic"]:
            obs_dict[key] = tensor
          else:
            obs_dict["policy"] = tensor
        # Ensure 'critic' exists (use 'policy' if not provided)
        if "critic" not in obs_dict and "policy" in obs_dict:
          obs_dict["critic"] = obs_dict["policy"]
        return obs_dict
      else:
        # List of arrays - map to both 'policy' and 'critic' groups
        policy_obs = torch.from_numpy(np.array(obs)).to(
          device=self.device, dtype=torch.float32
        )
        return {"policy": policy_obs, "critic": policy_obs}
    else:
      # Fallback - try to convert directly, map to both 'policy' and 'critic' groups
      # Convert to numpy first if not already, then use from_numpy for proper device placement
      if isinstance(obs, np.ndarray):
        policy_obs = torch.from_numpy(obs).to(device=self.device, dtype=torch.float32)
      else:
        policy_obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
      return {"policy": policy_obs, "critic": policy_obs}

  def close(self) -> None:
    """Close the environment."""
    return self.env.close()

  def _modify_action_space(self) -> None:
    """Modify action space if clipping is enabled."""
    if self.clip_actions is None:
      return

    if isinstance(self.single_action_space, gym.spaces.Box):
      self.single_action_space = gym.spaces.Box(
        low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
      )
      self.action_space = gym.vector.utils.batch_space(
        self.single_action_space, self.num_envs
      )
