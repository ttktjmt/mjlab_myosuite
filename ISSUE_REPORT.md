# MyoHand Actuator Issue: 39→36 Reduction

## Problem Summary

When using MyoHand with mjlab's Scene system, 39 muscle actuators are reduced to 36 actuators during `spec.attach()`. This causes a keyframe control size mismatch error.

## Root Cause

**MuJoCo 3.4.0's `spec.attach()` has a bug** that incorrectly removes three specific muscle actuators:
- **EDM** (Extensor Digiti Minimi)
- **EPL** (Extensor Pollicis Longus)  
- **FPL** (Flexor Pollicis Longus)

This is **NOT a MyoSuite XML issue**. Extensive investigation confirms:
- MyoSuite's XML files contain **no duplicate definitions** for these actuators
- Entity.spec correctly contains **39 unique actuators** with no duplicates
- Direct XML loading produces a valid model with 39 actuators
- The reduction occurs **only during `spec.attach()`** operation

## How the Issue Occurs

1. **XML loading**: `MjSpec.from_file()` reads myohand_die.xml → 39 actuators ✓
2. **Entity creation**: `Entity(DEFAULT_MYOHAND_CFG)` creates entity.spec → 39 unique actuators ✓
3. **Scene.attach()**: `scene_spec.attach(entity.spec, prefix="myohand/", frame=frame)` → **36 actuators** ✗
4. **Actuators removed**: EDM, EPL, FPL mysteriously disappear during attach
5. **Compilation error**: Keyframe has 39 ctrl values, but model expects 36 → error

## Detailed Evidence

### Test 1: XML Direct Loading
```python
import mujoco as mj
spec = mj.MjSpec.from_file("myohand_die.xml")
model = spec.compile()
print(f"Actuators: {model.nu}")  # Output: 39 ✓

# All three actuators present:
# 21. EDM
# 23. EPL  
# 25. FPL
```

### Test 2: Entity Spec Analysis
```python
from mjlab.entity import Entity
from mjlab_myosuite.robot.myohand_constants import DEFAULT_MYOHAND_CFG

entity = Entity(DEFAULT_MYOHAND_CFG)
actuators = [act.name for act in entity.spec.actuators]
print(f"Total: {len(actuators)}")  # Output: 39 ✓

# Check for duplicates
duplicates = {name for name in actuators if actuators.count(name) > 1}
print(f"Duplicates: {duplicates}")  # Output: set() (no duplicates) ✓
```

### Test 3: spec.attach() Behavior
```python
import mujoco as mj

# Load entity
entity_spec = mj.MjSpec.from_file("myohand_die.xml")
print(f"Before attach: {len(list(entity_spec.actuators))}")  # 39 ✓

# Create empty scene and attach
scene_spec = mj.MjSpec()
frame = scene_spec.worldbody.add_frame()
scene_spec.attach(entity_spec, prefix="myohand/", frame=frame)
print(f"After attach: {len(list(scene_spec.actuators))}")  # 36 ✗

# Find removed actuators
before = [act.name for act in entity_spec.actuators]
after = [act.name.replace("myohand/", "") for act in scene_spec.actuators]
removed = set(before) - set(after)
print(f"Removed: {removed}")  # {'EDM', 'EPL', 'FPL'} ✗
```

### Test 4: XML Verification
```bash
# Verify no duplicates in XML
$ grep -c 'name="EDM"' myohand_assets.xml
1

$ grep -c 'name="EPL"' myohand_assets.xml
1

$ grep -c 'name="FPL"' myohand_assets.xml
1
```

All tests confirm: **No duplicates exist in XML or entity.spec, yet `spec.attach()` removes three actuators.**

## Impact

- Cannot use standard mjlab `XmlMuscleActuatorCfg` + `TendonEffortActionCfg` workflow
- Scene compilation fails with "invalid ctrl size" error
- Requires workaround implementations

## Workaround Solution

Since this is a MuJoCo bug, the current workaround is **correct and should be maintained**:

### Current Implementation (Correct)
```python
# In die_reorient_env_cfg.py
actions = {
    "myohand": mdp.DirectMuscleEffortActionCfg(
        entity_name="myohand",
        scale=1.0,
        offset=0.0,
    )
}
```

### Why This Works
- `DirectMuscleEffortActionCfg` directly controls muscles without relying on articulation config
- Bypasses the problematic `XmlMuscleActuatorCfg` → `TendonEffortActionCfg` pipeline
- Maintains full 39-actuator control

### Configuration
```python
# In myohand_constants.py
DEFAULT_MYOHAND_CFG = EntityCfg(
    spawner=MjcfEntitySpawner(xml_spec_func=get_myohand_spec),
    init_state=EntityInitStateCfg(pos=(0.0, 0.0, 0.0)),
    articulation=EntityArticulationInfoCfg(
        actuator_cfgs={},  # Empty - using DirectMuscleEffortActionCfg
    ),
    collision=COLLISION_CFG,
)
```

## Future Resolution

**Do NOT attempt to fix this by modifying MyoSuite XML** - the XML is correct.

**Recommended next steps:**
1. Report bug to MuJoCo development team with minimal reproduction case
2. Wait for MuJoCo update that fixes `spec.attach()` behavior  
3. After fix is released, migrate to standard mjlab workflow:
   ```python
   # Future implementation (when MuJoCo is fixed)
   articulation=EntityArticulationInfoCfg(
       actuator_cfgs={
           "muscles": XmlMuscleActuatorCfg(
               target_names_expr=(".*_tendon",),
               transmission_type=TransmissionType.TENDON
           ),
       },
   )
   
   actions = {
       "myohand": mdp.TendonEffortActionCfg(
           entity_name="myohand",
           actuator_names=(r".*_tendon",),
           scale=1.0,
           offset=0.0,
       )
   }
   ```

## Environment Details

- **MuJoCo Version**: 3.4.0
- **MyoSuite Version**: Latest (XML verified correct)
- **mjlab**: Latest
- **Issue Status**: MuJoCo bug, workaround implemented
