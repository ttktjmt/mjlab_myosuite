from mjlab.third_party.isaaclab.isaaclab_tasks.utils.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)

# Auto-register MyoSuite environments if available
try:
  import mjlab.tasks.myosuite  # noqa: F401
except ImportError:
  # MyoSuite not installed or not available, skip
  pass
