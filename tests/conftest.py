"""Load the plugin directory as a package (``discord_realtime_voice``) next to an importable Hermes.

Hermes must be importable: install it into the test environment (see README "Development").
"""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = "discord_realtime_voice"

if PKG not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = module
    spec.loader.exec_module(module)
