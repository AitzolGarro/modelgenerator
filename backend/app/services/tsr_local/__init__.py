"""
Local copy of TripoSR with scikit-image marching cubes.

Installs a meta path finder that redirects `import tsr.xxx` to
`import app.services.tsr_local.xxx`. Needed because TripoSR's config.yaml
references classes like "tsr.models.tokenizers.image.TriplaneLearnablePositionalEmbedding"
which get resolved via importlib.import_module at runtime.
"""

import sys
import importlib
import importlib.abc
import importlib.machinery


class _TsrFinder(importlib.abc.MetaPathFinder):
    """Intercepts imports of 'tsr' and 'tsr.*', loads from tsr_local instead."""

    _PREFIX = "tsr"
    _TARGET = "app.services.tsr_local"

    def find_spec(self, fullname, path, target=None):
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
            return importlib.machinery.ModuleSpec(fullname, _TsrLoader(self._PREFIX, self._TARGET))
        return None


class _TsrLoader(importlib.abc.Loader):
    def __init__(self, prefix, target):
        self._prefix = prefix
        self._target = target

    def create_module(self, spec):
        return None  # Use default

    def exec_module(self, module):
        real_name = self._target + module.__name__[len(self._prefix):]

        try:
            real_mod = importlib.import_module(real_name)
        except ImportError as e:
            raise ImportError(f"Cannot import {module.__name__} (mapped to {real_name})") from e

        # Copy all attributes from the real module
        module.__dict__.update(real_mod.__dict__)
        module.__path__ = getattr(real_mod, "__path__", [])
        module.__package__ = module.__name__
        module.__file__ = getattr(real_mod, "__file__", None)

        # Also register under real name for consistency
        sys.modules[real_name] = real_mod


def install():
    """Install the tsr → app.services.tsr_local redirect."""
    if not any(isinstance(f, _TsrFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _TsrFinder())


# Auto-install on import
install()
