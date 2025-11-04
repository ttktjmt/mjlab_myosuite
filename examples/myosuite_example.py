"""Example script showing how to use MyoSuite environments with mjlab.

This script demonstrates different approaches to using MyoSuite tasks with mjlab's
infrastructure. Note that MyoSuite uses standard MuJoCo (CPU) while mjlab uses
MuJoCo Warp (GPU), so full integration requires porting the task to mjlab's format.
"""

import gymnasium as gym


# Approach 1: Direct MyoSuite usage
def example_direct_myosuite():
  """Example of using MyoSuite directly."""
  try:
    from myosuite.utils import gym as myosuite_gym

    print("Creating MyoSuite environment directly...")
    env = myosuite_gym.make("myoElbowPose1D6MRandom-v0")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(
      f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}"
    )

    # Run a few steps
    for i in range(10):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      print(
        f"Step {i}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}"
      )
      if terminated or truncated:
        obs, info = env.reset()

    env.close()
    print("✓ Direct MyoSuite usage successful")

  except ImportError:
    print("✗ MyoSuite not installed. Install with: pip install -U myosuite")
  except Exception as e:
    print(f"✗ Error: {e}")


# Approach 2: Check if MyoSuite environments are registered
def example_check_registry():
  """Check what MyoSuite environments are available."""
  try:
    print("\nChecking MyoSuite environment registry...")
    myosuite_envs = [
      env_id for env_id in gym.registry.keys() if "myo" in env_id.lower()
    ]

    if myosuite_envs:
      print(f"Found {len(myosuite_envs)} MyoSuite environments:")
      for env_id in sorted(myosuite_envs)[:10]:  # Show first 10
        print(f"  - {env_id}")
      if len(myosuite_envs) > 10:
        print(f"  ... and {len(myosuite_envs) - 10} more")
    else:
      print("No MyoSuite environments found in registry")

  except ImportError:
    print("✗ MyoSuite not installed")
  except Exception as e:
    print(f"✗ Error: {e}")


# Approach 3: Attempt to use with mjlab's training infrastructure
def example_mjlab_compatibility_note():
  """Show note about compatibility limitations."""
  print("\n" + "=" * 60)
  print("NOTE: MyoSuite and mjlab Compatibility")
  print("=" * 60)
  print("""
MyoSuite environments are standard Gymnasium environments using CPU MuJoCo.
mjlab uses ManagerBasedRlEnv with GPU-accelerated MuJoCo Warp.

Key differences:
1. MyoSuite: CPU MuJoCo, single environments, standard Gym API
2. mjlab: GPU MuJoCo Warp, vectorized environments, manager-based API

To use MyoSuite tasks with mjlab's training infrastructure:
- Option A: Port the task to mjlab's ManagerBasedRlEnv format (see docs/myosuite_integration.md)
- Option B: Use MyoSuite directly for evaluation (shown above)
- Option C: Create a custom wrapper (experimental, limited GPU acceleration)

For full integration guide, see: docs/myosuite_integration.md
""")


if __name__ == "__main__":
  print("MyoSuite Integration Examples")
  print("=" * 60)

  example_direct_myosuite()
  example_check_registry()
  example_mjlab_compatibility_note()

  print("\n" + "=" * 60)
  print("For more information, see: docs/myosuite_integration.md")
  print("=" * 60)
