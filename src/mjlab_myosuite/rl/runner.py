"""RL runner for MyoSuite environments."""

from rsl_rl.runners import OnPolicyRunner

from ..wrapper import MyoSuiteVecEnvWrapper


class MyoSuiteOnPolicyRunner(OnPolicyRunner):
  """On-policy runner for MyoSuite environments.

  This runner extends the base OnPolicyRunner but skips ONNX export
  since MyoSuite environments don't have the ManagerBasedRlEnv structure
  required for ONNX metadata attachment.
  """

  env: MyoSuiteVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save the model and training information.

    For MyoSuite environments, we skip ONNX export since they don't have
    the ManagerBasedRlEnv structure (scene, action_manager, etc.) required
    for ONNX metadata attachment.
    """
    # Just call the base class save method, which saves the model
    # Skip ONNX export since MyoSuite doesn't have the required structure
    super().save(path, infos)
