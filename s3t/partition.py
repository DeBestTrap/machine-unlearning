"""Persisted source-ID partitions and monotone deletion of sequence suffixes."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass
class Partition:
    shards: list[list[list[int]]]
    sequences: list[tuple[int, ...]]

    def __post_init__(self):
        if not self.shards or not self.shards[0]:
            raise ValueError("Empty partition")
        slices = len(self.shards[0])
        if any(len(shard) != slices or any(not part for part in shard) for shard in self.shards):
            raise ValueError("Each shard must contain the same number of nonempty slices")
        self.locations = {}
        for shard_id, shard in enumerate(self.shards):
            for slice_id, source_ids in enumerate(shard):
                for source_id in source_ids:
                    if type(source_id) is not int or source_id < 0 or source_id in self.locations:
                        raise ValueError("Source IDs must be unique nonnegative integers")
                    self.locations[source_id] = (shard_id, slice_id)
        self.sequences = [tuple(sequence) for sequence in self.sequences]
        if (not self.sequences or len(set(self.sequences)) != len(self.sequences)
                or any(sorted(sequence) != list(range(slices)) for sequence in self.sequences)):
            raise ValueError("Sequences must be distinct permutations of every slice")

    @classmethod
    def create(cls, source_ids, shards, slices, sequences, seed):
        if shards < 1 or slices < 1 or len(source_ids) < shards * slices:
            raise ValueError("Need at least one training example per shard/slice")
        shuffled = np.random.default_rng(seed).permutation(source_ids)
        parts = [[part.astype(int).tolist() for part in np.array_split(shard, slices)]
                 for shard in np.array_split(shuffled, shards)]
        return cls(parts, list(sequences))

    def to_dict(self):
        body = {"schema_version": 1, "shards": self.shards,
                "sequences": [list(sequence) for sequence in self.sequences]}
        return dict(body, partition_sha256=digest(body))

    @classmethod
    def from_dict(cls, value):
        body = {key: value[key] for key in ("schema_version", "shards", "sequences")}
        if body["schema_version"] != 1 or value.get("partition_sha256") != digest(body):
            raise ValueError("Partition checksum/version mismatch")
        return cls(body["shards"], body["sequences"])

    def training_ids(self, shard, sequence, prefix):
        if not 0 <= prefix <= len(self.shards[shard]):
            raise ValueError("Invalid slice prefix")
        return [source_id for sl in self.sequences[sequence][:prefix]
                for source_id in self.shards[shard][sl]]

    def prefixes(self, deleted_ids):
        affected = [set() for _ in self.shards]
        for source_id in deleted_ids:
            if type(source_id) is not int or source_id not in self.locations:
                raise ValueError(f"Deletion ID is outside this training partition: {source_id}")
            shard, sl = self.locations[source_id]
            affected[shard].add(sl)
        return [[next((index for index, sl in enumerate(sequence) if sl in deleted), len(sequence))
                 for sequence in self.sequences] for deleted in affected]

    def select(self, deleted_ids, validation_scores=None):
        result = []
        for shard, prefixes in enumerate(self.prefixes(deleted_ids)):
            candidates = [i for i, prefix in enumerate(prefixes) if prefix > 0]
            if not candidates:
                result.append(None)
                continue
            if validation_scores is None:
                best = max(candidates, key=lambda i: (prefixes[i], -i))
            else:
                best = max(candidates, key=lambda i: (validation_scores[shard][i][prefixes[i] - 1],
                                                       prefixes[i], -i))
            result.append({"shard": shard, "sequence": best, "prefix": prefixes[best]})
        return result
