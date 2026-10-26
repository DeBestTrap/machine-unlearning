import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_splits import get_manifest, fingerprint_records
from split_support import configured_manifest, bind_container_split, validate_container_split
from sharded import fetchTestBatch


class SharedSplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.original_cwd = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, self.original_cwd)
        labels = [0, 1] * 50
        with patch.dict(os.environ, DATASET_SPLIT_DIR=str(self.directory / 'splits')):
            self.manifest, self.path = get_manifest('fixture', labels, 20, fingerprint_records(labels))
        self.metadata_path = self.directory / 'datasetfile'
        self.metadata = dict(nb_train=80, nb_val=20, nb_test=12, dataloader='fixture_loader', split_manifest=str(self.path))
        self.metadata_path.write_text(json.dumps(self.metadata))

    def test_metadata_counts_must_match_manifest(self):
        self.assertEqual(configured_manifest(self.metadata_path), self.manifest)
        self.metadata['nb_train'] = 100
        self.metadata_path.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, 'counts'):
            configured_manifest(self.metadata_path)

    def test_old_or_different_container_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'predates'):
            validate_container_split('old', self.metadata_path)
        bind_container_split('new', self.metadata_path)
        validate_container_split('new', self.metadata_path)
        marker = self.directory / 'containers/new/dataset_split.json'
        marker.write_text(json.dumps({'manifest_sha256': 'another-split'}))
        with self.assertRaisesRegex(ValueError, 'different dataset split'):
            validate_container_split('new', self.metadata_path)

    def test_distribution_shards_cover_only_logical_training_pool(self):
        result = subprocess.run([sys.executable, str(ROOT / 'distribution.py'), '--shards', '3',
                                 '--dataset', str(self.metadata_path), '--container', 'fixture'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        validate_container_split('fixture', self.metadata_path)
        shards = np.load('containers/fixture/splitfile.npy', allow_pickle=True)
        logical = np.concatenate(shards).astype(int)
        self.assertEqual(sorted(logical.tolist()), list(range(80)))
        source_indices = np.asarray(self.manifest['train_indices'])[logical]
        self.assertFalse(set(source_indices) & set(self.manifest['val_indices']))

    def test_evaluation_batches_use_requested_pool_and_cover_remainder(self):
        for category, size in (('test', 12), ('val', 20)):
            loader = Mock()
            loader.load.side_effect = lambda indices, category: (indices, category)
            with self.subTest(category=category), patch('sharded.importlib.import_module', return_value=loader):
                batches = list(fetchTestBatch(str(self.metadata_path), 7, category))
                self.assertEqual(np.concatenate([batch[0] for batch in batches]).tolist(), list(range(size)))
                self.assertTrue(all(batch[1] == category for batch in batches))


if __name__ == '__main__':
    unittest.main()
