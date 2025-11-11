MyoSuite installation instructions:

1. Install MyoSuite:

```bash
uv pip install -e .
uv pip install "myosuite @ git+https://github.com/MyoHub/myosuite.git@mjx"
```

2. Train a MyoSuite environment:

```bash
uv run scripts/train.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --env.num_envs 4
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
uv run --no-default-groups --group cu128 --group dev pyright
uv run --no-default-groups --group cu128 --group dev pytest
```
Format code:
```
uvx pre-commit install
make format
```