from mjlab.tasks.registry import register_mjlab_task

from .die_reorient_env_cfg import (
    die_reorient_env_cfg,
    DieReorientRlCfg
)

from rsl_rl.runners import OnPolicyRunner

# MyoChallenge 2022 Die Reorientation Task (Phase 1)
register_mjlab_task(
    task_id="Myosuite-Manipulation-DieReorient-Myohand",
    env_cfg=die_reorient_env_cfg(play=False),
    play_env_cfg=die_reorient_env_cfg(play=True),
    rl_cfg=DieReorientRlCfg(max_iterations=50_000),
    runner_cls=OnPolicyRunner,
)
