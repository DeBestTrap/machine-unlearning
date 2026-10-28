"""Import a dataset's loader relative to its metadata file, not the working directory."""
from hashlib import sha256
import importlib
import importlib.util
from pathlib import Path
import sys


def load_dataset_module(dataset, metadata):
    directory = Path(dataset).resolve().parent
    loader = metadata["dataloader"]
    if not isinstance(loader, str) or not all(part.isidentifier() for part in loader.split(".")):
        raise ValueError("datasetfile dataloader must be a module name relative to the dataset directory")
    # Use a private package for each directory. This handles absolute paths,
    # spaces/hyphens in parent names, and different datasets with identical loader
    # names, while preserving package-relative imports and Python's module cache.
    package_name = "_sisa_dataset_" + sha256(str(directory).encode()).hexdigest()
    if package_name not in sys.modules:
        initializer = directory / "__init__.py"
        if initializer.is_file():
            spec = importlib.util.spec_from_file_location(
                package_name, initializer, submodule_search_locations=[str(directory)])
        else:
            spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
        package = importlib.util.module_from_spec(spec)
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package
        try:
            if initializer.is_file():
                spec.loader.exec_module(package)
        except BaseException:
            del sys.modules[package_name]
            raise
    return importlib.import_module(f".{loader}", package=package_name)
