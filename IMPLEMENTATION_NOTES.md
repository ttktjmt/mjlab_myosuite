# Implementation Notes and TODOs

## Current Implementation Status

This is the initial skeleton implementation of the MyoHand Die Reorientation task for mjlab. The core structure follows mjlab patterns, but several key components need refinement for full functionality.

## Key Implementation TODOs

### 1. Die and Goal Body/Geom ID Resolution

**Current State**: Placeholder functions return zeros
**Required**: Extract actual body/geom IDs from the MuJoCo model

```python
# In die_reorient_env_cfg.py, these functions need implementation:
def get_die_orientation(env: ManagerBasedRlEnv) -> torch.Tensor:
    # TODO: Get actual die body quaternion from sim
    # die_bid = env.sim.model.body_name2id("Object")  # or similar
    # quat = env.sim.data.qpos[:, die_qpos_indices]
    pass

def get_goal_orientation(env: ManagerBasedRlEnv) -> torch.Tensor:
    # TODO: Store and retrieve goal orientation from environment state
    pass

def get_die_position(env: ManagerBasedRlEnv) -> torch.Tensor:
    # TODO: Get die center position from sim.data.body_xpos
    pass
```

### 2. Quaternion/Rotation Utilities

**Required**: Implement rotation conversion utilities
- Euler <-> Quaternion conversion
- Quaternion difference/distance calculation
- Rotation vector representation

**Suggested approach**: Import from MyoSuite or implement custom:

```python
from myosuite.utils.quat_math import euler2quat, mat2euler, quat_mul
# Or implement mjlab-compatible versions
```

### 3. Scene Setup and Goal Marker

**Current State**: Basic scene with MyoHand entity
**Required**:
- Add goal visualization marker (green reference die)
- Set initial die position on palm
- Handle goal body pose updates in reset

```python
# In reset_die_orientation event:
# - Update goal body position and quaternion
# - Randomize die initial orientation (if needed)
# - Set visualization markers
```

### 4. Contact Sensor for Drop Detection

**Current State**: Simple Z-position threshold
**Better approach**: Use mjlab ContactSensorCfg

```python
from mjlab.sensor import ContactMatch, ContactSensorCfg

die_ground_contact = ContactSensorCfg(
    name="die_ground_contact",
    primary=ContactMatch(mode="body", pattern="Object", entity="myohand"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="any",
)
cfg.scene.sensors = (die_ground_contact,)

# Then in termination:
def check_die_dropped(env: ManagerBasedRlEnv) -> torch.Tensor:
    sensor = env.scene.sensors["die_ground_contact"]
    return sensor.data.found[:, 0]
```

### 5. Observation Space Refinement

**Current observations** are placeholders. Should include:
- Hand muscle lengths and velocities (proprioception)
- Die position relative to palm
- Die orientation as rotation matrix or quaternion
- Goal orientation
- Optionally: tactile/force feedback

**Reference**: MyoSuite's observation includes:
```python
obs_dict["hand_qpos"]     # Joint positions
obs_dict["hand_qvel"]     # Joint velocities
obs_dict["obj_pos"]       # Object position
obs_dict["obj_rot"]       # Object rotation (euler)
obs_dict["goal_rot"]      # Goal rotation (euler)
obs_dict["pos_err"]       # Position error
obs_dict["rot_err"]       # Rotation error
obs_dict["act"]           # Previous muscle activations
```

### 6. Reward Tuning

**Current weights** are initial guesses. Need to:
- Balance orientation vs position rewards
- Test with actual training runs
- Possibly add:
  - Velocity penalties (for smoother control)
  - Contact force penalties (gentle manipulation)
  - Progress rewards (incremental orientation improvement)

### 7. Model Asset Handling

**Current approach**: Try to load from installed MyoSuite
**Alternative approaches**:
1. Bundle XML in package: `src/mjlab_myosuite/robot/assets/myohand_die.xml`
2. Download on first use
3. Require MyoSuite installation

**If bundling**, update `myohand_constants.py`:
```python
MYOHAND_DIE_XML = Path(__file__).parent / "assets" / "myohand_die.xml"
```

### 8. Testing and Validation

Before training, test:
```bash
# Check environment loads
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent zero

# Check random agent runs
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent random

# Verify observations shape
# Verify actions work
# Check reset randomization
```

### 9. Muscle Actuator Configuration

**Current**: Uses generic XmlMotorActuatorCfg
**Verify**: MyoHand muscle actuators are properly loaded from XML
- Check actuator names match
- Verify action scaling is correct [0, 1] range
- Test muscle activation produces expected movement

### 10. Episode Length and Control Frequency

**Current settings**:
- Sim timestep: 0.002s (500 Hz)
- Decimation: 5 (control at 100 Hz)
- Episode: 6s = 600 control steps

**MyoChallenge original**:
- Episode: 150 steps (likely at 50 Hz = 3s)
- Consider adjusting for consistency

## Next Steps (Priority Order)

1. ✅ Basic file structure and registration
2. ⚠️ **Body/Geom ID extraction** - Critical for functionality
3. ⚠️ **Rotation utilities** - Needed for rewards and observations
4. ⚠️ **Contact sensors** - Better drop detection
5. ⚠️ Test with zero/random agents
6. ⚠️ Refine observation space
7. ⚠️ Add goal visualization
8. ⚠️ Run training test (1k iterations)
9. ⚠️ Tune rewards based on training behavior
10. ⚠️ Document results and hyperparameters

## Useful References

- [MyoSuite reorient_v0.py](https://github.com/MyoHub/myosuite/blob/main/myosuite/envs/myo/myochallenge/reorient_v0.py)
- [mjlab creating tasks guide](https://github.com/mujocolab/mjlab/blob/main/docs/create_new_task.md)
- [MyoChallenge 2022 paper](https://sites.google.com/view/myochallenge)

## Known Limitations

- No visual/depth observations (relies on proprioception)
- No domain randomization yet (Phase 1 focuses on fixed die)
- No curriculum learning
- Simplified reward function compared to original MyoChallenge

## Performance Expectations

Based on MyoChallenge results:
- Training time: ~10-50M environment steps
- Success rate: 60-80% on Phase 1
- Requires careful hyperparameter tuning
