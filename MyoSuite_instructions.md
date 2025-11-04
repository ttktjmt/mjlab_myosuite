MyoSuite installation instructions:

1. Install MyoSuite:

```bash
uv pip install "myosuite @ git+https://github.com/MyoHub/myosuite.git@remove_dmcontrol"
```

2. Train a MyoSuite environment:

```bash
uv run train Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --env.num_envs 4
```

3. Play a MyoSuite environment:

```bash
uv run play Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0-Play --wandb-run-path your-org/mjlab/run-id
```