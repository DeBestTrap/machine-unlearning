# Reports

Saved comparison `comparison_v1`:

| Dataset | SISA | S3T |
| --- | --- | --- |
| CIFAR-10 | [CSV](comparison_v1/sisa-cifar10.csv) | [CSV](comparison_v1/s3t-cifar10.csv) |
| CIFAR-100 | [CSV](comparison_v1/sisa-cifar100.csv) | [CSV](comparison_v1/s3t-cifar100.csv) |
| SST-2 | [CSV](comparison_v1/sisa-sst2.csv) | [CSV](comparison_v1/s3t-sst2.csv) |

The associated launchers are [run.sh](../run.sh) and [run.py](../run.py), using
[quick25.json](../configs/quick25.json). These report links point to saved run
outputs without changing the historical CSVs. New runs publish their own `<tag>/`
report index here. See [study details](../README.md) before launching training.

New runs publish `reports/<tag>/comparison/` with shared `steps.csv`,
`summary.csv`, SVG/PNG figures and regeneration settings. Use the associated
[report.py](../report.py) to change the metric or plot axes and refresh both
tables and figures from retained records. The [schema](../../../reporting/README.md)
describes which measurements remain available after checkpoint cleanup.
