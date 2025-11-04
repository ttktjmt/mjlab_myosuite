"""Script to train RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import tyro

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.myosuite.config import MyoSuiteEnvCfg
from mjlab.tasks.myosuite.rl import MyoSuiteOnPolicyRunner
from mjlab.tasks.myosuite.wrapper import MyoSuiteVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainConfig:
  env: Any
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False


def run_train(task: str, cfg: TrainConfig) -> None:
  configure_torch_backends()

  registry_name: str | None = None

  if isinstance(cfg.env, TrackingEnvCfg):
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    cfg.env.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")

  # Enable NaN guard if requested
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  # Specify directory for logging experiments.
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir

  # Check if this is a MyoSuite environment
  is_myosuite = task.startswith("Mjlab-MyoSuite") or isinstance(cfg.env, MyoSuiteEnvCfg)

  env = gym.make(
    task, cfg=cfg.env, device=cfg.device, render_mode="rgb_array" if cfg.video else None
  )

  resume_path = (
    get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
    if cfg.agent.resume
    else None
  )

  # Helper function to find MyoSuiteVecEnvWrapper by unwrapping (needed for MyoSuite)
  def find_myosuite_wrapper(env_obj):
    """Unwrap environment to find MyoSuiteVecEnvWrapper."""
    current = env_obj
    max_depth = 10
    depth = 0
    visited = set()  # Prevent infinite loops

    while depth < max_depth:
      if isinstance(current, MyoSuiteVecEnvWrapper):
        return current

      # Prevent infinite loops
      obj_id = id(current)
      if obj_id in visited:
        break
      visited.add(obj_id)

      # Try to unwrap - check multiple possible attribute names
      next_env = None

      # Try .env attribute (most common in gymnasium wrappers like OrderEnforcing)
      if hasattr(current, "env"):
        try:
          candidate = current.env
          if candidate is not current and candidate is not None:
            next_env = candidate
        except (AttributeError, TypeError):
          pass

      # Try .unwrapped property
      if next_env is None and hasattr(current, "unwrapped"):
        try:
          candidate = current.unwrapped
          if candidate is not current and candidate is not None:
            next_env = candidate
        except (AttributeError, TypeError):
          pass

      # Try ._env (private attribute some wrappers use)
      if next_env is None and hasattr(current, "_env"):
        try:
          candidate = current._env
          if candidate is not current and candidate is not None:
            next_env = candidate
        except (AttributeError, TypeError):
          pass

      if next_env is None or next_env is current:
        # Can't unwrap further
        break

      current = next_env
      depth += 1

    return None

  # Find the MyoSuiteVecEnvWrapper before wrapping with RecordVideo (if MyoSuite)
  myosuite_wrapper = find_myosuite_wrapper(env) if is_myosuite else None

  if cfg.video:
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=os.path.join(log_dir, "videos", "train"),
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")
    # Re-find the wrapper after RecordVideo wraps it
    if is_myosuite and myosuite_wrapper is None:
      myosuite_wrapper = find_myosuite_wrapper(env)

  if is_myosuite:
    # Unwrap gymnasium's OrderEnforcing wrapper to get to our MyoSuiteVecEnvWrapper
    # The wrapper chain might be: OrderEnforcing -> RecordVideo -> MyoSuiteVecEnvWrapper
    # We need the unwrapped env because OrderEnforcing doesn't forward get_observations()

    if myosuite_wrapper is not None:
      # Use the unwrapped wrapper directly
      env = myosuite_wrapper
      env.clip_actions = cfg.agent.clip_actions
    else:
      # If we couldn't find it, try one more time with current env
      myosuite_wrapper = find_myosuite_wrapper(env)
      if myosuite_wrapper is not None:
        env = myosuite_wrapper
        env.clip_actions = cfg.agent.clip_actions
      else:
        # If still not found, raise an error with helpful info
        raise RuntimeError(
          "Could not find MyoSuiteVecEnvWrapper in environment wrapper chain. "
          f"Environment type: {type(env)}, "
          f"has .env: {hasattr(env, 'env')}, "
          f"has .unwrapped: {hasattr(env, 'unwrapped')}, "
          f"unwrapped type: {type(getattr(env, 'unwrapped', None))}"
        )
  else:
    # Standard mjlab environment - wrap with RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  if isinstance(cfg.env, TrackingEnvCfg):
    runner = MotionTrackingOnPolicyRunner(
      env, agent_cfg, str(log_dir), cfg.device, registry_name
    )
  elif is_myosuite:
    # MyoSuite environments use a custom runner that skips ONNX export
    runner = MyoSuiteOnPolicyRunner(env, agent_cfg, str(log_dir), cfg.device)
  else:
    runner = VelocityOnPolicyRunner(env, agent_cfg, str(log_dir), cfg.device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def main():
  # Parse first argument to choose the task.
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys()]  # if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig(env=env_cfg, agent=agent_cfg),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_train(chosen_task, args)


if __name__ == "__main__":
  main()
