from mjlab.tasks.registry import register_mjlab_task

from .myohand_manipulation_env_cfg import (
    myohand_manipulation_env_cfg,
    MyohandRlCfg
)

from rsl_rl.runners import OnPolicyRunner
# from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

register_mjlab_task(
    task_id="Myosuite-Manipulation-Myohand-Reorientation",
    env_cfg=myohand_manipulation_env_cfg(reverse_knee=False),
    play_env_cfg=myohand_manipulation_env_cfg(reverse_knee=False, play=True),
    rl_cfg=MyohandRlCfg(max_iterations=50_000),
    runner_cls=OnPolicyRunner,
)
