"""Direct muscle actuator action for MyoSuite environments.

This module provides DirectMuscleEffortActionCfg as a workaround for MuJoCo 3.4.0's
spec.attach() bug that incorrectly removes 3 muscle actuators (EDM, EPL, FPL).

By directly controlling muscle actuators without relying on XmlMuscleActuatorCfg,
we bypass the problematic articulation configuration system and maintain full
39-actuator control.

See ISSUE_REPORT.md for detailed explanation of the underlying bug.
"""

from dataclasses import MISSING
import torch
from typing import Any

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.envs import ManagerBasedRlEnv


class DirectMuscleEffortAction(ActionTerm):
    """Direct muscle effort control action.
    
    This action term directly applies effort commands to muscle actuators without
    going through the articulation system. This is necessary as a workaround for
    MuJoCo 3.4.0's spec.attach() bug.
    
    The action values are interpreted as effort commands (normalized to [-1, 1] or [0, 1]
    depending on muscle properties) and are scaled/offset before being applied.
    """

    cfg: "DirectMuscleEffortActionCfg"

    def __init__(self, cfg: "DirectMuscleEffortActionCfg", env: ManagerBasedRlEnv):
        """Initialize direct muscle effort action.
        
        Args:
            cfg: Configuration for the action term
            env: The environment instance
        """
        super().__init__(cfg, env)

        # Get all actuator indices for this entity
        entity_name = cfg.entity_name
        model = env.sim.model
        
        # Find all actuators that belong to this entity
        # They should have the prefix "entity_name/"
        self._actuator_indices = []
        for i in range(model.nu):
            act_name = model.actuator(i).name
            # Check if actuator belongs to our entity
            if act_name.startswith(f"{entity_name}/"):
                self._actuator_indices.append(i)
        
        if len(self._actuator_indices) == 0:
            raise ValueError(
                f"No actuators found for entity '{entity_name}'. "
                f"Available actuators: {[model.actuator(i).name for i in range(model.nu)]}"
            )
        
        # Convert to tensor
        self._actuator_indices = torch.tensor(
            self._actuator_indices, dtype=torch.long, device=env.device
        )
        
        # Store scale and offset
        self._scale = cfg.scale
        self._offset = cfg.offset
        
        # Store action dimension
        self._num_actuators = len(self._actuator_indices)
        
        print(f"[DirectMuscleEffortAction] Initialized with {self._num_actuators} actuators for entity '{entity_name}'")

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._num_actuators

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw action commands before processing."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed action commands after scaling and offset."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process and store raw actions.
        
        Args:
            actions: Raw action tensor of shape (num_envs, num_actuators)
        """
        # Store raw actions
        self._raw_actions = actions.clone()
        
        # Apply scaling and offset
        self._processed_actions = self._scale * actions + self._offset

    def apply_actions(self) -> None:
        """Apply processed actions to the simulation.
        
        Directly sets ctrl values for muscle actuators.
        """
        # Get ctrl tensor from simulation
        ctrl = self._env.sim.data.ctrl
        
        # Set control values for our actuators
        # Shape: (num_envs, num_actuators)
        ctrl[:, self._actuator_indices] = self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        # Initialize raw and processed actions to zero
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        
        if not hasattr(self, "_raw_actions"):
            self._raw_actions = torch.zeros(
                (self._env.num_envs, self._num_actuators),
                device=self._env.device
            )
            self._processed_actions = torch.zeros(
                (self._env.num_envs, self._num_actuators),
                device=self._env.device
            )
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


class DirectMuscleEffortActionCfg(ActionTermCfg):
    """Configuration for direct muscle effort action.
    
    This configuration allows direct control of muscle actuators without going
    through the articulation system. It's used as a workaround for MuJoCo 3.4.0's
    spec.attach() bug that removes 3 actuators during scene creation.
    
    Attributes:
        entity_name: Name of the entity to control (e.g., "myohand")
        scale: Scaling factor for actions (default: 1.0)
        offset: Offset added to actions (default: 0.0)
    """

    class_type: type[ActionTerm] = DirectMuscleEffortAction

    entity_name: str = MISSING
    """Name of the entity containing the muscle actuators."""

    scale: float = 1.0
    """Scaling factor applied to action commands."""

    offset: float = 0.0
    """Offset added to action commands after scaling."""
