# Machine Unlearning with SISA
### Lucas Bourtoule, Varun Chandrasekaran, Christopher Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot
This repository contains the core code used in the SISA experiments of our [Machine Unlearning](https://arxiv.org/abs/1912.03817) paper along with some example scripts.

An independent S3T implementation is also available. See [S3T commands and experiment setup](S3T.md)
and [implementation decisions](docs/s3t-decisions.md). It supports ViT on CIFAR-10/CIFAR-100,
RoBERTa on SST-2, and the shared [data splits](SPLITS.md) and [deletion workloads](DELETIONS.md).

You can start runing experiments by having a look at the readme in the purchase example folder at ``example-scripts/purchase-sharding``.

``sisa.py`` is the script that trains a given shard. It should be run as many times as the number of shards.

## SISA checkpoints and retained run data

SISA keeps run records in the local `containers/<name>/` directory. Only its
`cache/` directory points into `saved_models/<name>/cache/`, which can live on
another drive. On this machine, `saved_models/` points to
`/mnt/csdrive/containers/`. New checkouts use a local `saved_models/` directory
unless you create a symlink to your preferred model drive before initialization.

If `containers/` is still a legacy symlink to an external drive, separate it once:

```bash
python container_storage.py separate
```

This copies existing non-cache data locally, dereferences timing/prediction
links, and leaves the original external files untouched. `saved_models/` takes
over the external link. Keep the drive mounted while initializing or training.

The common experiment scripts retain:

- `containers/<name>/config.json`: the experiment settings and dataset metadata.
- `containers/<name>/report.csv`: this run's report, using the existing CSV schema.
- `containers/<name>/checkpoint_storage.json`: the last recorded checkpoint size,
  counting each checkpoint file once, excluding shard symlink aliases.
- `containers/<name>/times/`, `outputs/`, split files, and deletion workloads:
  the data needed to inspect the run or regenerate reports without model files.

Use a new container name for each experiment you want to retain separately.
The usual combined `report-*.csv` files are still written as before.
New reports include complete update timings compatible with BlockUnlearn's
`update_v1` measurements. See [timing boundaries and historical results](TIMING.md).

To delete saved models while keeping run data:

```bash
bash clear_containers.sh my-run  # one container
bash clear_containers.sh         # all local containers
```

Cleanup records checkpoint sizes before deleting `.pt` files and their aliases;
it preserves local records and predictions. You can also delete a run's external
`saved_models/<name>/` directory after the size snapshot has been saved. Existing
predictions remain usable for aggregation, but generating new predictions or
resuming training requires the checkpoints. S3T uses its own `runs/` layout and
is unaffected by this SISA storage change.

## Citing this work

If you use this repository for academic research, you are highly encouraged
(though not required) to cite our paper:

```
@inproceedings{bourtoule2021machine,
  title={Machine Unlearning},
  author={Lucas Bourtoule and Varun Chandrasekaran and Christopher Choquette-Choo and Hengrui Jia and Adelin Travers and Baiwu Zhang and David Lie and Nicolas Papernot},
  booktitle={Proceedings of the 42nd IEEE Symposium on Security and Privacy},
  year={2021}
}
```
