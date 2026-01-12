# mjlab_myosuite

MyoSuite MyoHand Die Reorientation task implementation using [mjlab](https://github.com/mujocolab/mjlab).

## Overview

This package provides an mjlab-compatible implementation of the MyoChallenge 2022 Die Reorientation task. The task involves manipulating a die held in a dexterous MyoHand to match a target orientation without dropping it.

**Task ID**: `Myosuite-Manipulation-DieReorient-Myohand`

# mjlab_myosuite — Quick Commands

Install (editable):

```bash
pip install -e .
```

Train:

```bash
uv run train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 2048
# or
python -m mjlab.train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 2048
```

Play / Evaluate:

```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --checkpoint-file [path-to-checkpoint]
# or
python -m mjlab.play Myosuite-Manipulation-DieReorient-Myohand --checkpoint-file [path-to-checkpoint]
```

Debug (quick agents):

```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent zero
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent random
```

## References

- [MyoChallenge 2022](https://sites.google.com/view/myochallenge)
- [MyoSuite](https://github.com/MyoHub/myosuite)
- [mjlab](https://github.com/mujocolab/mjlab)
- [mjlab_upkie](https://github.com/MarcDcls/mjlab_upkie)
- [mjlab_cartpole](https://github.com/Gregwar/mjlab_cartpole)
