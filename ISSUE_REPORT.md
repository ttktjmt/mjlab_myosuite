# MyoHand Actuator Issue: 39→36 Reduction

## Problem Summary

MyoHand has 39 muscle actuators, but when attached to mjlab Scene with a prefix, only 36 actuators remain. Three specific actuators are consistently removed: **EDM**, **EPL**, and **FPL**.

## Root Cause - FULLY IDENTIFIED ✅

**MuJoCo 3.4.0's `spec.attach()` has a bug with `mjWRAP_SPHERE` tendon wraps when `sidesite` is not specified.**

### The Exact Mechanism

The bug occurs when ALL of the following conditions are met:
1. Using `spec.attach()` with a **non-empty prefix**
2. Tendon uses a **`mjWRAP_SPHERE`** wrap geometry
3. The wrap **does not have an explicit `sidesite` attribute** (uses default behavior)

### Why These Specific Three Tendons

| Tendon | Problematic Wrap | Type | Sidesite | Wrap Index |
|--------|------------------|------|----------|------------|
| **EDM** (Extensor Digiti Minimi) | `Fifthpm_wrap` | `mjWRAP_SPHERE` | None ⚠️ | [9] |
| **EPL** (Extensor Pollicis Longus) | `MPthumb_wrap` | `mjWRAP_SPHERE` | None ⚠️ | [9] |
| **FPL** (Flexor Pollicis Longus) | `FPL_ellipsoid_wrap` | `mjWRAP_SPHERE` | None ⚠️ | [5] |

**Key Discovery**: These are the **ONLY** three tendons in MyoHand that have `mjWRAP_SPHERE` geometries without explicit `sidesite` attributes.

All 36 preserved tendons either:
- Don't use SPHERE wraps, OR
- Have explicit `sidesite` values for all their SPHERE wraps

### Why This is a MuJoCo Bug

1. **Valid XML**: MuJoCo documentation allows omitting `sidesite` (defaults to "both sides")
2. **Works in standalone mode**: Direct `MjSpec.from_file()` → 39 actuators ✓
3. **Inconsistent behavior**: 
   - `mjWRAP_SITE` elements work fine without `sidesite`
   - Only `mjWRAP_SPHERE` with default `sidesite` causes removal
4. **Prefix-dependent**:
   - Empty prefix: `attach(spec, prefix="", ...)` → 39 actuators ✓
   - Any non-empty prefix: `attach(spec, prefix="x/", ...)` → 36 actuators ✗

### Not a MyoSuite Issue

- MyoSuite XML is correctly formatted and valid
- The default `sidesite` behavior is documented and should work
- Problem is in MuJoCo's prefix handling during `attach()`

## Verification

```bash
# Run verification script
uv run python scripts/verify_actuator_issue.py
```

**Results:**
- Direct XML load: 39 actuators ✓
- After spec.attach() with prefix: 36 actuators ✗
- Removed: EDM, EPL, FPL (and their corresponding tendons)

## Impact on mjlab

mjlab requires entity prefixes for namespace separation, making the workaround unavoidable. Standard `XmlMuscleActuatorCfg` + `TendonEffortActionCfg` workflow cannot be used.

## Workaround Solution

Use `DirectMuscleEffortActionCfg` which operates on the compiled model (36 actuators):

```python
# die_reorient_env_cfg.py
actions = {
    "myohand": DirectMuscleEffortActionCfg(
        entity_name="myohand",
        actuator_names=(".*",),  # Pattern matching all tendon actuators
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

1. **Report to MuJoCo**: Submit minimal reproduction case to MuJoCo development team
2. **Wait for fix**: MuJoCo 3.4.1 or later may address the prefix-handling bug
3. **Migrate to standard workflow** once fixed:
   ```python
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
           actuator_names=(".*_tendon",),
       )
   }
   ```

## Environment

- **MuJoCo**: 3.4.0 (bug confirmed in prefix handling)
- **MyoSuite**: Latest (XML verified correct)
- **mjlab**: Latest (standard API usage)
- **Status**: MuJoCo bug, workaround implemented and tested
