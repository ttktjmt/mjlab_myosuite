#!/usr/bin/env python3
"""Test script to verify the die reorientation environment loads correctly."""

import torch


def test_environment_import():
    """Test that the environment can be imported and registered."""
    print("Testing environment import...")
    try:
        from mjlab_myosuite.tasks import die_reorient_env_cfg

        print("✓ Environment configuration imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import environment: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_environment_creation():
    """Test that the environment can be created."""
    print("\nTesting environment creation...")
    try:
        from mjlab_myosuite.tasks.die_reorient_env_cfg import die_reorient_env_cfg

        # Get configuration
        cfg = die_reorient_env_cfg(play=True)
        print(f"✓ Configuration created successfully")
        print(f"  - Observations: {list(cfg.observations.keys())}")
        print(f"  - Actions: {list(cfg.actions.keys())}")
        print(f"  - Rewards: {list(cfg.rewards.keys())}")
        print(f"  - Terminations: {list(cfg.terminations.keys())}")
        print(f"  - Episode length: {cfg.episode_length_s}s")
        print(f"  - Decimation: {cfg.decimation}")

        return True
    except Exception as e:
        print(f"✗ Failed to create environment configuration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_task_registration():
    """Test that the task is registered in mjlab."""
    print("\nTesting task registration...")
    try:
        # Just import to trigger registration
        import mjlab_myosuite.tasks

        # Try to access the registry
        from mjlab.tasks.registry import _TASK_REGISTRY

        task_id = "Myosuite-Manipulation-DieReorient-Myohand"

        if task_id in _TASK_REGISTRY:
            info = _TASK_REGISTRY[task_id]
            print(f"✓ Task '{task_id}' registered successfully")
            print(f"  - Has env_cfg: {info.env_cfg is not None}")
            print(f"  - Has play_env_cfg: {info.play_env_cfg is not None}")
            print(f"  - Has rl_cfg: {info.rl_cfg is not None}")
            print(
                f"  - Runner class: {info.runner_cls.__name__ if info.runner_cls else 'None'}"
            )
            return True
        else:
            print(f"✗ Task '{task_id}' not found in registry")
            print(f"  Available tasks: {list(_TASK_REGISTRY.keys())}")
            return False

    except Exception as e:
        print(f"✗ Failed to check task registration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_rotation_utilities():
    """Test rotation utility functions."""
    print("\nTesting rotation utilities...")
    try:
        from mjlab_myosuite.tasks.die_reorient_env_cfg import (
            euler_to_quat,
            quat_to_euler,
            quat_distance,
        )

        # Test euler to quat conversion
        euler = torch.tensor([[0.0, 0.0, 0.0], [1.57, 0.0, 0.0]])  # 0 and 90 degrees
        quat = euler_to_quat(euler)
        print(f"✓ Euler to quaternion: {euler.shape} -> {quat.shape}")

        # Test quat to euler conversion
        euler_back = quat_to_euler(quat)
        print(f"✓ Quaternion to euler: {quat.shape} -> {euler_back.shape}")

        # Test quaternion distance
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
        q2 = torch.tensor([[0.707, 0.707, 0.0, 0.0]])  # 90 degrees around X
        dist = quat_distance(q1, q2)
        print(
            f"✓ Quaternion distance: {dist.item():.3f} radians (~{dist.item() * 180 / 3.14159:.1f} degrees)"
        )

        return True
    except Exception as e:
        print(f"✗ Failed rotation utilities test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MyoHand Die Reorientation Environment Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Import", test_environment_import()))
    results.append(("Creation", test_environment_creation()))
    results.append(("Registration", test_task_registration()))
    results.append(("Rotation Utils", test_rotation_utilities()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:20s}: {status}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Environment is ready.")
        print("\nNext steps:")
        print(
            "  1. Run with zero agent: uv run play Myosuite-Manipulation-DieReorient-Myohand --agent zero"
        )
        print(
            "  2. Run with random agent: uv run play Myosuite-Manipulation-DieReorient-Myohand --agent random"
        )
        print(
            "  3. Start training: uv run train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 512"
        )
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
