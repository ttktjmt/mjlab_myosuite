# Quick Start Guide

## Installation

```bash
# Clone repository
git clone <your-repo>
cd mjlab_myosuite_ttktjmt

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Training

```bash
# Start training (2048 parallel environments)
uv run train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 2048

# Resume from checkpoint
uv run train Myosuite-Manipulation-DieReorient-Myohand --resume
```

### Testing/Playing

```bash
# Play with trained agent
uv run play Myosuite-Manipulation-DieReorient-Myohand --checkpoint-file logs/rsl_rl/die_reorient_p1/*/model_*.pt

# Debug with zero actions
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent zero

# Debug with random actions
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent random
```

## Task Description

**Objective**: Rotate a die held in MyoHand to match target orientation without dropping.

**Environment ID**: `Myosuite-Manipulation-DieReorient-Myohand`

**Phase**: MyoChallenge 2022 Phase 1 (limited rotation range ±90°)

## File Structure

```
src/mjlab_myosuite/
├── robot/
│   ├── myohand_constants.py    # Model configs, viewer, sim settings
│   └── __init__.py
└── tasks/
    ├── die_reorient_env_cfg.py  # Environment: obs, actions, rewards
    └── __init__.py               # Task registration
```

## Key Configuration Parameters

**In die_reorient_env_cfg.py:**
- `num_envs`: 512 (parallel environments)
- `episode_length_s`: 6.0 (training) / 20.0 (play)
- `decimation`: 5 (control frequency = 100 Hz)

**In myohand_constants.py:**
- `timestep`: 0.002 (simulation at 500 Hz)

## Important Notes

⚠️ **This is a skeleton implementation** - Several TODOs remain:
1. Body/Geom ID extraction from MuJoCo model
2. Rotation utilities (quaternion operations)
3. Contact sensor for drop detection
4. Goal marker visualization
5. Observation space refinement

See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for details.

## Troubleshooting

**Error: XML not found**
```bash
# Install MyoSuite with mjx branch
uv add "myosuite @ git+https://github.com/MyoHub/myosuite.git@mjx"
```

**Error: CUDA out of memory**
```bash
# Reduce parallel environments
uv run train ... --env.scene.num-envs 512
```

## Next Steps

1. Test environment loads: `uv run play ... --agent zero`
2. Implement TODOs (see IMPLEMENTATION_NOTES.md)
3. Run short training test: `--rl.max_iterations 1000`
4. Tune hyperparameters based on results

## References

- MyoChallenge: https://sites.google.com/view/myochallenge
- MyoSuite: https://github.com/MyoHub/myosuite
- mjlab: https://github.com/mujocolab/mjlab
