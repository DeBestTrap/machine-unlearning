"""Real tiny-model replay, resume and maintenance-timing checks, without CUDA."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset_splits import get_manifest, fingerprint_records
from deletion_requests import generate_workload
from studies.classification_fixed20_full100 import worker


class WorkerTests(unittest.TestCase):
    def test_s3t_timing_excludes_initial_and_nested_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); journal = root / 'native/comparison_timing'; journal.mkdir(parents=True)
            def save(name, seconds):
                (journal / name).write_text(json.dumps(dict(timing_scope='update_v1', status='complete',
                    measurement_complete=True, update_time_sec=seconds, training_time_sec=0)))
            for step in range(100):
                save(f'delete-{step+1}.json', .1)
            save('initial-s0-q0.json', 999)
            save('g1-s0-q0.json', 10)
            save('publish-1.json', 11)
            result = worker.s3t_maintenance_timing(root)
            self.assertAlmostEqual(result['update_time_sec'], 21)
            self.assertTrue(result['measurement_complete'])

    def test_real_baseline_100_requests_and_resume(self):
        common = Path(__file__).resolve().parents[3] / 'BlockUnlearn/unlearning/studies/2026-11-10_0346_classification_fixed20_full100/study_common.py'
        # The repositories need not be sibling checkouts in production; this
        # offline test uses the repository layout of the shared workspace.
        spec = importlib.util.spec_from_file_location('full100_test_common', common)
        api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with patch.dict(os.environ, DATASET_SPLIT_DIR=str(run / 'splits')):
                parent, _ = get_manifest('cifar100', [i % 2 for i in range(130)], 10,
                                         fingerprint_records(range(130)))
            api.write(run / 'parent_split.json', parent)
            api.write(run / 'requests.json', generate_workload(parent, steps=100))
            api.write(run / 'plan.json', {'fixture': True})
            plan = dict(requests=100, native_model='resnet50', baseline_dropout=0., baseline_model_args={}, evaluation_split='test', selections={'SISA':dict(shards=2,slices=2,epochs=3,learning_rate=.01)})
            records = run / 'records'; records.mkdir()
            job = dict(seed=1,id='baseline-seed1',selection_key='SISA',reported_methods=['sisa','protosisa'])
            class Loader:
                def __init__(self, *args):
                    self.parent=parent;self.train_indices=np.array(parent['train_indices'])
                    self.val_indices=np.array(parent['val_indices']);self.test_set=list(range(6))
                    self.classes=2;self.input_shape=[4];self.test_sha256='fixture-test'
                    self.x=np.random.default_rng(123).normal(size=(120,4)).astype(np.float32)
                    self.y=np.arange(120)%2
                def load(self, indices, category='train'):
                    return self.x[indices],self.y[indices]
            def initialize(path):
                path=Path(path)
                for name in ['cache','outputs','times']:(path/name).mkdir(parents=True,exist_ok=True)
                return path
            def model(config, device):
                return torch.nn.Linear(4,2).to(device)
            with patch.object(worker,'FullLoader',Loader), patch.object(worker,'task_model',model), \
                 patch('container_storage.initialize_container',side_effect=initialize), \
                 patch('protosisa.build_encoder',return_value=torch.nn.Identity()), redirect_stdout(io.StringIO()):
                first=worker.run_baseline(run,plan,job,records,torch.device('cpu'),api)
                second=worker.run_baseline(run,plan,job,records,torch.device('cpu'),api)
            self.assertEqual(first,second)
            self.assertEqual(first['sisa']['requests'],100)
            self.assertTrue(first['protosisa']['measurement_complete'])
            events=[api.signed_read(records/'steps'/f'{step:03d}.json') for step in range(101)]
            self.assertEqual([e['source_ids'] for e in events[1:]],json.loads((run/'requests.json').read_text())['batches'])


if __name__ == '__main__':
    unittest.main()
