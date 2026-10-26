"""Dataset split metadata and guards for SISA containers."""
import json
from pathlib import Path

from dataset_splits import load_manifest
from container_storage import record_directory


def configured_manifest(dataset, **expected):
    dataset = Path(dataset)
    metadata = json.loads(dataset.read_text())
    manifest_path = metadata.get('split_manifest')
    if not manifest_path:
        return None
    manifest = load_manifest(dataset.parent / manifest_path, **expected)
    if metadata['nb_train'] != len(manifest['train_indices']) or metadata.get('nb_val') != len(manifest['val_indices']):
        raise ValueError('datasetfile counts disagree with the shared split manifest')
    return manifest


def validate_container_split(container, dataset):
    directory = record_directory(container)
    manifest = configured_manifest(dataset)
    if manifest is None:
        return
    path = directory / 'dataset_split.json'
    if not path.exists():
        raise ValueError('Container predates verified splits. Create a new container with distribution.py --shards.')
    recorded = json.loads(path.read_text())
    if recorded['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Container was created with a different dataset split. Use a new container.')


def bind_container_split(container, dataset):
    directory = record_directory(container)
    manifest = configured_manifest(dataset)
    if manifest is None:
        return
    if (directory / 'splitfile.npy').exists():
        validate_container_split(container, dataset)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'dataset_split.json').write_text(json.dumps(manifest, sort_keys=True) + '\n')
