#!/usr/bin/env python3
"""Frozen full-pool SISA/ProtoSISA or S3T with 100 shared source-ID requests."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def task_model(config, device):
    import torch
    import protosisa as proto
    model = proto.build_model(config, device)
    scope = config['model_args']['layers_to_finetune']
    names = list(model.model.named_parameters())
    if any(not any(name.startswith(prefix) for name, _ in names) for prefix in scope):
        raise ValueError('A frozen trainable prefix matches no parameters')
    for name, parameter in names:
        parameter.requires_grad_(name.startswith(tuple(scope)))
    torch.nn.init.xavier_uniform_(model.model.heads.head.weight)
    torch.nn.init.zeros_(model.model.heads.head.bias)
    return model


class FullLoader:
    def __init__(self, run, plan):
        import numpy as np
        from torchvision import datasets, transforms
        from dataset_splits import load_manifest, fingerprint_cifar
        self.parent = load_manifest(run / 'parent_split.json')
        normalization = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
        training = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), normalization])
        evaluation = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalization])
        root = plan['baseline_data_root']
        self.train_set = datasets.CIFAR100(root, train=True, download=False, transform=training)
        self.val_set = datasets.CIFAR100(root, train=True, download=False, transform=evaluation)
        self.test_set = datasets.CIFAR100(root, train=False, download=False, transform=evaluation)
        if fingerprint_cifar(self.train_set.data, self.train_set.targets) != self.parent['source_sha256']:
            raise ValueError('Training source differs from frozen manifest')
        self.train_indices = np.asarray(self.parent['train_indices'])
        self.val_indices = np.asarray(self.parent['val_indices'])
        self.test_sha256 = fingerprint_cifar(self.test_set.data, self.test_set.targets)
        self.classes = len(self.train_set.classes)
        if len(self.train_indices) != 45000 or len(self.test_set) != 10000:
            raise ValueError('Unexpected train/test population')

    def load(self, indices, category='train'):
        import numpy as np
        import torch
        if category == 'train':
            dataset, source = self.train_set, self.train_indices
        elif category == 'train_eval':
            dataset, source = self.val_set, self.train_indices
        elif category == 'test':
            dataset, source = self.test_set, np.arange(len(self.test_set))
        else:
            raise ValueError('Only final training and test evaluation are supported')
        values = [dataset[int(source[index])] for index in indices]
        return torch.stack([v[0] for v in values]).numpy(), np.asarray([v[1] for v in values])


def test_features(directory, loader, encoder, config, device, api):
    import numpy as np
    import torch
    import protosisa as proto
    path = directory / 'outputs/test-frozen-features.npz'
    marker = path.with_suffix('.json')
    binding = dict(test_sha256=loader.test_sha256, training_sha256=config['training_sha256'])
    if marker.exists():
        saved = api.signed_read(marker)
        if saved['binding'] != binding or saved['sha256'] != api.file_hash(path):
            raise ValueError('Frozen evaluation features changed')
        values = np.load(path)
        return torch.from_numpy(values['features']), values['labels'].copy()
    features, labels = [], []
    with torch.inference_mode():
        for inputs, target in proto.batches(loader, np.arange(config['dataset_metadata']['nb_test']), config['batch_size'], 'test'):
            features.append(encoder(torch.as_tensor(inputs, device=device)).cpu())
            labels.extend(target.tolist())
    features, labels = torch.cat(features), np.asarray(labels)
    np.savez_compressed(path, features=features.numpy(), labels=labels)
    api.signed_write(marker, dict(binding=binding, sha256=api.file_hash(path)))
    return features, labels


def member_predictions(directory, checkpoint, loader, config, device, api):
    import numpy as np
    import torch
    import protosisa as proto
    key = sha256((str(checkpoint) + loader.test_sha256 + config['training_sha256']).encode()).hexdigest()
    path = directory / 'outputs' / ('prediction-' + key + '.npz')
    marker = path.with_suffix('.json')
    if marker.exists():
        if api.signed_read(marker)['sha256'] != api.file_hash(path):
            raise ValueError('Cached member predictions changed')
        return np.load(path)['scores']
    model = task_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['model'])
    model.eval()
    scores = []
    with torch.inference_mode():
        for inputs, _ in proto.batches(loader, np.arange(config['dataset_metadata']['nb_test']), config['batch_size'], 'test'):
            scores.append(model(torch.as_tensor(inputs, device=device)).cpu().numpy())
    scores = np.concatenate(scores)
    del model
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model predictions')
    np.savez_compressed(path, scores=scores)
    api.signed_write(marker, dict(sha256=api.file_hash(path), checkpoint=str(checkpoint), test_sha256=loader.test_sha256))
    return scores


def run_baseline(run, plan, job, records, device, api):
    import numpy as np
    import torch
    import protosisa as proto
    from container_storage import initialize_container
    from deletion_requests import load_workload, local_batches
    from reporting.timing import total_timing
    loader = FullLoader(run, plan)
    directory = initialize_container(str(records / 'native'))
    metadata = records / 'datasetfile'
    n, test_size = len(loader.train_indices), len(loader.test_set)
    api.write(metadata, dict(nb_train=n, nb_val=len(loader.val_indices), nb_test=test_size, nb_classes=loader.classes,
                            input_shape=[32, 32], split_manifest=str(run / 'parent_split.json')))
    selection = plan['selections']['baseline']
    scope = ['heads', 'encoder.ln', *[f'encoder.layers.encoder_layer_{i}.' for i in range(8, 12)]]
    config = dict(model='vit', dataset=str(metadata), dataset_metadata=json.loads(metadata.read_text()),
        seed=job['seed'], optimizer='adamw', batch_size=32, dropout_rate=0.,
        learning_rate=selection['learning_rate'], shards=selection['shards'], slices=selection['slices'], epochs=selection['epochs'],
        prototype_weighting='safe', prototype_weight=None, eval_split='test',
        model_args=dict(layers_to_finetune=scope, task_head_initialization='xavier_uniform'),
        comparison_run_id=job['id'], plan_sha256=api.file_hash(run / 'plan.json'))
    config['training_sha256'] = proto.digest(config)
    if (directory / 'config.json').exists() and json.loads((directory / 'config.json').read_text()) != config:
        raise ValueError('Frozen baseline configuration changed')
    api.write(directory / 'config.json', config)
    api.write(directory / 'dataset_split.json', loader.parent)
    shards = np.array_split(np.random.default_rng(job['seed']).permutation(n), selection['shards'])
    splitfile = np.empty(len(shards), dtype=object); splitfile[:] = shards
    np.save(directory / 'splitfile.npy', splitfile, allow_pickle=True)
    mapping = {int(local): i for i, shard in enumerate(shards) for local in shard}
    workload = load_workload(run / 'requests.json', loader.parent)
    batches = local_batches(workload, loader.parent, loader.train_indices.tolist())
    paths = [proto.train_member(directory, config, i, shard, [], 0, loader, device, model_factory=task_model)
             for i, shard in enumerate(shards)]
    encoder = proto.build_encoder('vit', device)
    features, labels = test_features(directory, loader, encoder, config, device, api)
    deleted, events = [[] for _ in shards], []
    for step in range(plan['requests'] + 1):
        if step:
            local = batches[step-1][0]
            shard = mapping[local]
            deleted[shard].append(local)
        marker = records / 'steps' / f'{step:03d}.json'
        if marker.exists():
            saved = api.signed_read(marker)
            if saved['plan_sha256'] != config['plan_sha256'] or saved['source_ids'] != ([] if step == 0 else workload['batches'][step-1]):
                raise ValueError('Saved request/plan changed')
            paths = [Path(p) for p in saved['member_paths']]
            events.append(saved)
            continue
        sisa_seconds, measurement_complete = 0., True
        replay_examples = 0
        if step:
            previous = {str(path) for path, _, _ in proto.checkpoint_paths(
                directory, config, shard, shards[shard], deleted[shard][:-1])}
            replay_examples = sum(len(indices) * epochs for path, indices, epochs in proto.checkpoint_paths(
                directory, config, shard, shards[shard], deleted[shard]) if str(path) not in previous)
            paths[shard] = proto.train_member(directory, config, shard, shards[shard], deleted[shard], step,
                                              loader, device, model_factory=task_model)
            timing = total_timing([directory / 'times' / f'shard-{shard}:{step}.json'])
            sisa_seconds = timing['update_time_sec']
            measurement_complete = timing['measurement_complete']
        prototypes = proto.update_prototypes(directory, config, shards, deleted, step, loader, encoder, device)
        prototype_timing = total_timing([directory / 'times' / f'prototypes:{step}.json'])
        ensemble = np.zeros((test_size, loader.classes), dtype=np.float32)
        started = proto.synchronized_time(device)
        for checkpoint in paths:
            ensemble += member_predictions(directory, checkpoint, loader, config, device, api) / len(paths)
        with torch.inference_mode():
            proto_scores = proto.prototype_logits(features.to(device), prototypes).cpu().numpy()
        mixture = proto.mixture_parameters(config, n-step, len(shards))
        mixed = proto.mix_logits(ensemble, proto_scores, mixture['prototype_weight'])
        eval_seconds = proto.synchronized_time(device) - started
        if step in (0, plan['requests']):
            np.savez_compressed(records / f'predictions-step-{step:03d}-test.npz',
                                labels=labels, ensemble=ensemble, protosisa=mixed)
        proto_seconds = prototype_timing['update_time_sec']
        complete = measurement_complete and prototype_timing['measurement_complete']
        event = dict(step=step, source_ids=[] if step == 0 else workload['batches'][step-1],
            plan_sha256=config['plan_sha256'], member_paths=list(map(str, paths)),
            sisa_accuracy=float(np.mean(ensemble.argmax(1) == labels)),
            protosisa_accuracy=float(np.mean(mixed.argmax(1) == labels)),
            sisa_update_sec=sisa_seconds if measurement_complete else None,
            protosisa_update_sec=sisa_seconds + proto_seconds if complete else None,
            measurement_complete=complete, evaluation_seconds=eval_seconds,
            replay_example_presentations=int(replay_examples),
            prototype_feature_examples=0 if step == 0 else len(shards[shard]) - len(deleted[shard]),
            prototype_weight=mixture['prototype_weight'])
        api.signed_write(marker, event); events.append(event)
        print(f'{job["id"]}: step={step}, SISA={event["sisa_accuracy"]:.4f}, ProtoSISA={event["protosisa_accuracy"]:.4f}', flush=True)
    result = {}
    for method in ['sisa', 'protosisa']:
        seconds = [e[method + '_update_sec'] for e in events[1:]]
        result[method] = dict(initial_accuracy=events[0][method + '_accuracy'], final_accuracy=events[-1][method + '_accuracy'],
                              mean_update_sec=statistics.mean(seconds) if all(v is not None for v in seconds) else None,
                              total_replay_example_presentations=sum(e['replay_example_presentations'] for e in events[1:]),
                              prototype_feature_examples=(sum(e['prototype_feature_examples'] for e in events[1:])
                                                          if method == 'protosisa' else 0),
                              requests=plan['requests'], evaluation_split='test', measurement_complete=all(v is not None for v in seconds))
    return result


def s3t_maintenance_timing(records):
    from reporting.timing import total_timing
    journals = records / 'native/comparison_timing'
    delete_journals = sorted(journals.glob('delete-*.json'))
    if len(delete_journals) != 100:
        raise ValueError('Missing S3T deletion timing journals')
    return total_timing(delete_journals + sorted(journals.glob('publish-*.json')))


def run_s3t(run, plan, job, records, device, api):
    from s3t.config import Config
    from s3t.engine import Run
    selection = plan['selections']['S3T']
    config = Config.from_dict(dict(dataset='cifar100', model='vit_b_16', pretrained=True,
        data_root=plan['baseline_data_root'], split_manifest=str(run / 'parent_split.json'),
        **selection, seed=job['seed'], partition_seed=42, sequence_seed=42,
        batch_size=32, eval_batch_size=32, num_workers=4, precision='fp32',
        selection='longest_prefix', aggregation='logits'))
    native = Run.create(records / 'native', config)
    native.train(device)
    native.replay(run / 'requests.json', device=device, split='test', on_exhaustion='retrain')
    state = native.state()
    workload = json.loads((run / 'requests.json').read_text())
    if state['step'] != 100 or [e['source_ids'] for e in state['history']] != workload['batches']:
        raise ValueError('S3T did not execute all 100 shared requests')
    evaluations = {}
    generation_work = {}
    for path in sorted((records / 'native/evaluations').glob('*-test.json')):
        value = json.loads(path.read_text())
        generation_work[value['generation']] = value['generation_training_examples']
        step = value['step']
        if step not in evaluations or value['generation'] >= evaluations[step]['generation']:
            evaluations[step] = value
    if set(evaluations) != set(range(101)):
        raise ValueError('Missing S3T test accuracy measurements')
    api.write(records / 'accuracy_curve.json', evaluations)
    # Native journals separate delete operations and maintenance-generation publication.
    # Publication includes refresh training. Do not double-count its child
    # generation timers, or count initial training as maintenance.
    timing = s3t_maintenance_timing(records)
    return {'S3T': dict(initial_accuracy=evaluations[0]['accuracy'], final_accuracy=evaluations[100]['accuracy'],
        mean_update_sec=timing['update_time_sec'] / 100 if timing['measurement_complete'] else None,
        requests=100, evaluation_split='test', measurement_complete=timing['measurement_complete'],
        total_replay_example_presentations=sum(count for generation, count in generation_work.items() if generation > 0),
        refreshes=state['generation'], active_shards=evaluations[100]['active_shards'])}


def cleanup(plan, job, records, api):
    if (records / 'cleanup.json').exists():
        return
    removed = []
    if plan['discard_models']:
        if job['method'] == 'baseline':
            from container_storage import clear_models
            clear_models([str(records / 'native')])
        else:
            native = records / 'native'
            for path in [native / 'backbone.pt', *sorted((native / 'generations').rglob('*.pt'))]:
                if not path.is_relative_to(native) or path.is_symlink():
                    raise ValueError('Cleanup path is outside the owned S3T run')
                if path.exists():
                    path.unlink(); removed.append(str(path))
    api.write(records / 'cleanup.json', dict(policy='discard_after_success' if plan['discard_models'] else 'keep', removed=removed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run.parents[1]))
    import study_common as api
    plan = api.load_plan(run)
    job = next(j for j in plan['jobs'] if j['id'] == args.job)
    records = Path(job['records']); records.mkdir(parents=True, exist_ok=True)
    result_path = run / 'results' / (job['id'] + '.json')
    if result_path.exists():
        value = api.signed_read(result_path)
        if value['plan_sha256'] != api.file_hash(run / 'plan.json'):
            raise ValueError('Result identity changed')
        cleanup(plan, job, records, api)
        value['cleanup_complete'] = True; api.signed_write(result_path, value)
        return
    import torch
    torch.set_num_threads(plan['threads'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    torch.cuda.set_device(device)
    api.write(records / 'runtime.json', dict(python=sys.executable, python_version=sys.version,
        torch=torch.__version__, cuda=torch.version.cuda, requested_device=args.device,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    methods = (run_baseline if job['method'] == 'baseline' else run_s3t)(run, plan, job, records, device, api)
    value = dict(plan_sha256=api.file_hash(run / 'plan.json'), job=job['id'], methods=methods, cleanup_complete=False)
    api.signed_write(result_path, value)
    cleanup(plan, job, records, api)
    value['cleanup_complete'] = True
    api.signed_write(result_path, value)


if __name__ == '__main__':
    main()
