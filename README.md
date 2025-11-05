MyoSuite installation instructions:

1. Install MyoSuite:

```bash
uv pip install -e .
uv pip install "myosuite @ git+https://github.com/MyoHub/myosuite.git@remove_dmcontrol"
```

2. Train a MyoSuite environment:

```bash
uv run sripts/train.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --env.num_envs 4
```

3. Play a MyoSuite environment:

```bash
uv run scripts/play.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --wandb-run-path your-org/mjlab/run-id
```


## Development
Run tests:
```
make test          # Run all tests
make test-fast     # Skip slow integration tests
uv run --no-default-groups --group cu128 --group dev pytest
uv run --no-default-groups --group cu128 --group dev pyright
```
Format code:
```
uvx pre-commit install
make format
```