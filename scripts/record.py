"""
Script to record RL agent visualization to .viser file.

Usage:
  uv run scripts/record.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --wandb-run-path team/project/run-id \
    --output-file recordings/<filename>.viser
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, cast

import gymnasium as gym
import torch
import tyro
import viser

# Import mjlab_myosuite to trigger auto-registration of MyoSuite environments
try:
  import mjlab_myosuite  # noqa: F401
except ImportError:
  pass  # MyoSuite not available, skip registration

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.sim.sim import Simulation
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viser_scene import ViserMujocoScene
from rsl_rl.runners import OnPolicyRunner

try:
  from mjlab_myosuite.config import MyoSuiteEnvCfg
  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper
except ImportError:
  MyoSuiteEnvCfg = None  # type: ignore
  MyoSuiteVecEnvWrapper = None  # type: ignore


@dataclass(frozen=True)
class RecordConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  output_file: str = "recordings/output.viser"
  num_steps: int = 1000
  env_idx: int = 0
  frame_skip: int = 2
  sleep_duration: float = 0.016
  random_goals: bool = True
  distance: float = 5.0
  azimuth: float = 45.0
  elevation: float = -20.0


def run_record(task: str, cfg: RecordConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Check if this is a MyoSuite environment
  is_myosuite = task.startswith("Mjlab-MyoSuite")

  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  # For MyoSuite environments, set device in config to match policy device
  if (
    is_myosuite and (MyoSuiteEnvCfg is not None) and isinstance(env_cfg, MyoSuiteEnvCfg)
  ):
    env_cfg.device = device

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if isinstance(env_cfg, TrackingEnvCfg):
    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      env_cfg.commands.motion.motion_file = str(
        Path(artifact.download()) / "motion.npz"
      )
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        env_cfg.commands.motion.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          env_cfg.commands.motion.motion_file = str(Path(art.download()) / "motion.npz")

    # Set random goal sampling if requested
    if cfg.random_goals:
      env_cfg.commands.motion.sampling_mode = "uniform"
      print("[INFO]: Using random goal sampling (uniform mode)")

  log_dir: Optional[Path] = None
  resume_path: Optional[Path] = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    if hasattr(env_cfg, "scene"):
      env_cfg.scene.num_envs = cfg.num_envs
    elif hasattr(env_cfg, "num_envs"):
      env_cfg.num_envs = cfg.num_envs

  env = gym.make(task, cfg=env_cfg, device=device)

  # Handle MyoSuite environments differently
  if is_myosuite:

    def find_myosuite_wrapper(env_obj):
      """Unwrap environment to find MyoSuiteVecEnvWrapper."""
      current = env_obj
      max_depth = 10
      depth = 0
      visited = set()

      while depth < max_depth:
        if (MyoSuiteVecEnvWrapper is not None) and isinstance(
          current, MyoSuiteVecEnvWrapper
        ):
          return current

        obj_id = id(current)
        if obj_id in visited:
          break
        visited.add(obj_id)

        next_env = None
        if hasattr(current, "env"):
          try:
            candidate = current.env
            if candidate is not current and candidate is not None:
              next_env = candidate
          except (AttributeError, TypeError):
            pass

        if next_env is None and hasattr(current, "unwrapped"):
          try:
            candidate = current.unwrapped
            if candidate is not current and candidate is not None:
              next_env = candidate
          except (AttributeError, TypeError):
            pass

        if next_env is None or next_env is current:
          break

        current = next_env
        depth += 1

      return None

    myosuite_wrapper = find_myosuite_wrapper(env)
    if myosuite_wrapper is not None:
      env = myosuite_wrapper
      env.clip_actions = agent_cfg.clip_actions
      env.device = torch.device(device)
      env.device_str = device
    else:
      raise RuntimeError(
        "Could not find MyoSuiteVecEnvWrapper in environment wrapper chain."
      )
  else:
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if DUMMY_MODE:
    action_shape = None
    if hasattr(env.unwrapped, "single_action_space"):
      action_shape = getattr(env.unwrapped.single_action_space, "shape", None)  # type: ignore
    else:
      action_shape = getattr(env.unwrapped.action_space, "shape", None)  # type: ignore
      if (
        isinstance(action_shape, tuple)
        and len(action_shape) > 1
        and getattr(env.unwrapped, "num_envs", 1) == action_shape[0]
      ):
        action_shape = action_shape[1:]

    env_device = (
      env.unwrapped.device if hasattr(env.unwrapped, "device") else torch.device("cpu")
    )

    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          per_env_shape = action_shape if isinstance(action_shape, tuple) else ()
          return torch.zeros(
            (getattr(env.unwrapped, "num_envs", 1),) + per_env_shape, device=env_device
          )

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          per_env_shape = action_shape if isinstance(action_shape, tuple) else ()
          return (
            2
            * torch.rand(
              (getattr(env.unwrapped, "num_envs", 1),) + per_env_shape,
              device=env_device,
            )
            - 1
          )

      policy = PolicyRandom()
  else:
    if isinstance(env_cfg, TrackingEnvCfg):
      runner = MotionTrackingOnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    else:
      runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  # For MyoSuite environments, ensure forward kinematics are computed
  if is_myosuite and hasattr(env.unwrapped, "sim"):
    import mujoco

    sim = env.unwrapped.sim
    if hasattr(sim, "_env"):
      mujoco.mj_forward(sim._env.mj_model, sim._env.mj_data)

  # Start recording
  print(f"[INFO]: Starting recording to {cfg.output_file}")
  print(f"[INFO]: Recording {cfg.num_steps} steps (frame skip: {cfg.frame_skip})")

  # Create viser server and scene
  server = viser.ViserServer(label="mjlab-recording", verbose=False)
  sim = env.unwrapped.sim
  assert isinstance(sim, Simulation)

  scene = ViserMujocoScene.create(
    server=server,
    mj_model=sim.mj_model,
    num_envs=env.num_envs,
  )
  scene.env_idx = cfg.env_idx

  # Get serializer
  serializer = server.get_scene_serializer()

  # Initialize environment
  env.reset()

  # Recording loop
  frame_count = 0
  for step in range(cfg.num_steps):
    # Get observation and action from policy
    obs = env.get_observations()
    action = policy(obs)

    # Detach action to avoid gradient tracking issues
    if isinstance(action, torch.Tensor):
      action = action.detach()

    # Step environment (use unwrapped to avoid wrapper issues)
    env.unwrapped.step(action)

    # Update visualization (only every Nth frame)
    if step % cfg.frame_skip == 0:
      with server.atomic():
        scene.update(sim.wp_data)
        server.flush()

      # Insert sleep for animation timing
      serializer.insert_sleep(cfg.sleep_duration)
      frame_count += 1

      # Progress indicator
      if (step + 1) % 100 == 0:
        print(
          f"[INFO]: Recorded {step + 1}/{cfg.num_steps} steps ({frame_count} frames)"
        )

  # Save recording
  output_path = Path(cfg.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  recording_data = serializer.serialize()
  output_path.write_bytes(recording_data)

  print(f"[INFO]: Recording saved to {output_path}")
  print(f"[INFO]: Total frames: {frame_count}")
  print(f"[INFO]: Duration: ~{frame_count * cfg.sleep_duration:.1f} seconds")
  print()
  print("[INFO]: To view the recording:")
  print("   1. Run: viser-build-client --out-dir viser-client")
  print(f"   2. Open: viser-client/?playbackPath=../{output_path}")

  # Cleanup
  server.stop()
  env.close()


def main():
  # Parse first argument to choose the task
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    RecordConfig,
    args=remaining_args,
    default=RecordConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_record(chosen_task, args)


if __name__ == "__main__":
  main()
