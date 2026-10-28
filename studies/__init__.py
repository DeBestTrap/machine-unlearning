"""Expose stable Python imports for study folders with completion timestamps.

For example, ``studies.batch_norm.run`` loads ``run.py`` from the dated
``YYYY-MM-DD_HHMM_batch_norm`` folder without an undated directory alias.
"""

from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path
import re
import sys


def _register_studies():
    folders = {}
    for path in sorted(Path(__file__).resolve().parent.iterdir()):
        match = re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{4}_([a-z][a-z0-9_]*)", path.name)
        if not path.is_dir() or match is None:
            continue
        slug = match.group(1)
        if slug in folders:
            raise ImportError(f"Duplicate study name {slug}: {folders[slug]} and {path}")
        folders[slug] = path
        name = f"{__name__}.{slug}"
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        package = module_from_spec(spec)
        sys.modules[name] = package
        globals()[slug] = package


_register_studies()
