"""Direct muscle actuator action for MyoSuite environments.

This module provides DirectMuscleEffortActionCfg as a workaround for MuJoCo 3.4.0's
spec.attach() bug that incorrectly removes 3 muscle actuators (EDM, EPL, FPL).

By directly controlling muscle actuators without relying on XmlMuscleActuatorCfg,
we bypass the problematic articulation configuration system and maintain full
39-actuator control.

See ISSUE_REPORT.md for detailed explanation of the underlying bug.
"""

from dataclasses import dataclass, MISSING
import torch
from typing import Any

from mjlab.managers.action_manager import ActionTerm
from mjlab.envs.mdp.actions.actions import BaseActionCfg, BaseAction
from mjlab.envs import ManagerBasedRlEnv
from mjlab.actuator.actuator import TransmissionType


class DirectMuscleEffortAction(BaseAction):
    """Direct muscle effort control action.
    
    This action term directly applies effort commands to muscle actuators.
    It extends BaseAction to leverage its actuator finding and scaling capabilities.
    """

    cfg: "DirectMuscleEffortActionCfg"

    def __init__(self, cfg: "DirectMuscleEffortActionCfg", env: ManagerBasedRlEnv):
        """Initialize direct muscle effort action.
        
        Args:
            cfg: Configuration for the action term
            env: The environment instance
        """
        # BaseAction.__init__ handles all the actuator finding and setup
        super().__init__(cfg, env)
        
        print(f"[DirectMuscleEffortAction] Initialized with {self._num_targets} actuators for entity '{cfg.entity_name}'")
        print(f"  Actuator names: {self._target_names[:5]}... (showing first 5)")

    def apply_actions(self) -> None:
        """Apply processed actions to the simulation.
        
        Directly sets ctrl values for muscle actuators.
        """
        # Get ctrl tensor from simulation
        ctrl = self._env.sim.data.ctrl
        
        # Set control values for our actuators
        # Shape: (num_envs, num_actuators)
        ctrl[:, self._target_ids] = self._processed_actions


@dataclass(kw_only=True)
class DirectMuscleEffortActionCfg(BaseActionCfg):
    """Configuration for direct muscle effort action.
    
    This configuration allows direct control of muscle actuators without going
    through the articulation system. It's used as a workaround for MuJoCo 3.4.0's
    spec.attach() bug that removes 3 actuators during scene creation.
    
    Attributes:
        entity_name: Name of the entity to control (e.g., "myohand")
        actuator_names: Actuator names pattern (tuple of regex patterns)
        scale: Scaling factor for actions (default: 1.0)
        offset: Offset added to actions (default: 0.0)
    """

    class_type: type[ActionTerm] = DirectMuscleEffortAction

    # Override actuator_names to make it optional with a default
    actuator_names: tuple[str, ...] | list[str] = (".*",)
    """Actuator names pattern (default: all actuators)."""

    def __post_init__(self):
        """Set transmission type to TENDON for muscle actuators."""
        self.transmission_type = TransmissionType.TENDON

    def build(self, env: Any) -> ActionTerm:
        """Build the action term from this config.
        
        Args:
            env: The environment instance
            
        Returns:
            The instantiated action term
        """
        return self.class_type(self, env)
