from typing import Optional

import pytest


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


pytestmark = pytest.mark.skipif(not _has_myosuite(), reason="myosuite not installed")


def test_registration_and_make_env():
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401

  # Pick one canonical env id that should exist in MyoSuite
  raw_id = "myoElbowPose1D6MRandom-v0"
  wrapped_id = f"Mjlab-MyoSuite-{raw_id}"

  # Verify registration
  all_envs = list(gym.registry.keys())
  assert raw_id in all_envs or wrapped_id in all_envs

  # Make mjlab-wrapped env (prefer the wrapped id if present)
  env_id = wrapped_id if wrapped_id in all_envs else raw_id
  env = gym.make(env_id)

  try:
    # Basic reset/step
    obs, info = env.reset()
    assert isinstance(info, dict)

    if hasattr(env, "unwrapped"):
      unwrapped = env.unwrapped
    else:
      unwrapped = env

    # Verify sim interface expected by viewers
    assert hasattr(unwrapped, "sim")
    sim = unwrapped.sim
    assert hasattr(sim, "mj_model")
    assert hasattr(sim, "mj_data") or hasattr(sim, "data")
    assert hasattr(sim, "wp_data")

    # One step with zero action if possible
    action_space = getattr(
      unwrapped, "single_action_space", getattr(unwrapped, "action_space", None)
    )
    if action_space is not None:
      import numpy as np

      # Determine per-env action shape
      per_env_shape: Optional[tuple] = None
      if hasattr(action_space, "shape") and action_space.shape is not None:
        per_env_shape = action_space.shape
      # Build batched zero action for vector envs (num_envs x action_dim)
      num_envs = getattr(unwrapped, "num_envs", 1)
      if per_env_shape is not None:
        batched_shape = (num_envs,) + per_env_shape
        zero_act = np.zeros(batched_shape, dtype=float)
      else:
        # Fallback to sampling then zeroing shape
        sample = action_space.sample()
        zero_act = np.zeros_like(sample)
        if getattr(zero_act, "ndim", 0) == 1 and num_envs > 1:
          zero_act = np.tile(zero_act, (num_envs, 1))
      env.step(zero_act)

  finally:
    env.close()


def test_wrapper_observations_device():
  import gymnasium as gym
  import torch

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401

  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0")
  try:
    # If wrapper exposes get_observations, ensure tensors are on the env device
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(unwrapped, "get_observations") and hasattr(unwrapped, "device"):
      td = unwrapped.get_observations()
      assert "policy" in td
      pol = td["policy"]
      if isinstance(pol, torch.Tensor):
        assert pol.device == unwrapped.device
  finally:
    env.close()


def test_viewer_forward_kinematics_available():
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401

  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0")
  try:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    sim = getattr(unwrapped, "sim", None)
    assert sim is not None

    # Access properties used by the viewer; they should be callable without errors
    _ = sim.mj_model
    _ = sim.mj_data  # property should internally mj_forward
    _ = sim.wp_data  # property should internally mj_forward
  finally:
    env.close()
