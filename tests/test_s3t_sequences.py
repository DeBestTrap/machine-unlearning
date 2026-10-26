import itertools
import math
import random
import unittest

from s3t.config import Config
from s3t.partition import Partition
from s3t.sequences import (bms_sequences, conditional_sequences, cyclic_sequences,
                           random_sequences, sequence_score)


class SequenceTests(unittest.TestCase):
    def test_every_small_cyclic_budget_is_exact_distinct_and_valid(self):
        for size in range(1, 6):
            for budget in range(1, math.factorial(size) + 1):
                sequences = cyclic_sequences(size, budget)
                self.assertEqual(len(sequences), budget)
                self.assertEqual(len(set(sequences)), budget)
                self.assertTrue(all(sorted(seq) == list(range(size)) for seq in sequences))
        self.assertEqual(cyclic_sequences(3, 3), [(0, 1, 2), (2, 0, 1), (1, 2, 0)])

    def test_cyclic_and_bms_up_to_L_have_distinct_positions(self):
        for size in range(2, 8):
            probs = [i / sum(range(1, size + 1)) for i in range(1, size + 1)]
            for selector in (lambda: cyclic_sequences(size, size), lambda: bms_sequences(size, size, probs, horizon=3)):
                sequences = selector()
                for position in range(size):
                    self.assertEqual(len({seq[position] for seq in sequences}), size)
                self.assertTrue(all(sorted(seq) == list(range(size)) for seq in sequences))

    def test_seeded_selection_does_not_advance_global_rng(self):
        random.seed(13)
        before = random.getstate()
        for selector in (lambda: random_sequences(4, 24, 4),
                         lambda: conditional_sequences(4, 24, [1.0, 0.0, 0.0, 0.0], 4)):
            first = selector()
            self.assertEqual(first, selector())
            self.assertEqual(set(first), set(itertools.permutations(range(4))))
        self.assertEqual(before, random.getstate())

    def test_bms_uses_paper_score_and_rejects_invalid_budget(self):
        self.assertAlmostEqual(sequence_score((0, 1), [0.1, 0.2, 0.7], 2), 0.9**2 + 2 * 0.7**2)
        with self.assertRaises(ValueError):
            cyclic_sequences(3, 7)
        with self.assertRaises(ValueError):
            bms_sequences(3, 3, [0.5, 0.4, 0.2])
        with self.assertRaises(ValueError):
            Config.from_dict({"budget": 1000})
        with self.assertRaises(ValueError):
            Config.from_dict({"learning_rate": float("nan")})
        with self.assertRaises(ValueError):
            Config.from_dict({"epohs": 1})


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.partition = Partition.create(list(range(11, 64)), 3, 4, cyclic_sequences(4, 4), seed=7)

    def test_remainders_are_kept_and_partition_roundtrips(self):
        self.assertEqual(sorted(self.partition.locations), list(range(11, 64)))
        restored = Partition.from_dict(self.partition.to_dict())
        self.assertEqual(restored.to_dict(), self.partition.to_dict())
        for shard in range(3):
            for sequence in range(4):
                ids = restored.training_ids(shard, sequence, 4)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(set(ids), set(sum(restored.shards[shard], [])))
        corrupt = self.partition.to_dict()
        corrupt["shards"][0][0][0] = 1000
        with self.assertRaises(ValueError):
            Partition.from_dict(corrupt)

    def test_all_deletion_combinations_remove_every_affected_dependency(self):
        partition = self.partition
        for count in range(5):
            for affected in itertools.combinations(range(4), count):
                deleted = [partition.shards[1][sl][0] for sl in affected]
                prefixes = partition.prefixes(deleted)
                self.assertEqual(prefixes[0], [4] * 4)
                self.assertEqual(prefixes[2], [4] * 4)
                for sequence, prefix in enumerate(prefixes[1]):
                    self.assertFalse(set(partition.training_ids(1, sequence, prefix)) & set(deleted))
                    if prefix < 4:
                        self.assertIn(partition.sequences[sequence][prefix], affected)

    def test_deletions_monotonically_shrink_prefixes_and_exhaust_shards(self):
        deleted, previous = [], self.partition.prefixes([])
        for sl in range(4):
            deleted.append(self.partition.shards[0][sl][0])
            current = self.partition.prefixes(deleted)
            self.assertTrue(all(after <= before for old, new in zip(previous, current) for before, after in zip(old, new)))
            previous = current
        self.assertIsNone(self.partition.select(deleted)[0])
        with self.assertRaises(ValueError):
            self.partition.prefixes([0])
        with self.assertRaises(ValueError):
            Partition.create([1, 2], 2, 2, [(0, 1)], 0)

    def test_selection_uses_the_surviving_prefix_score(self):
        partition = Partition([[[10], [11], [12]]], [(0, 1, 2), (2, 0, 1)])
        scores = [[[0.95, 0.2, 0.1], [0.3, 0.99, 1.0]]]
        # Deletion leaves prefixes of length 1 and 2. Validation chooses the latter.
        self.assertEqual(partition.select([11], scores)[0]["sequence"], 1)
        # After deleting 10, only the second sequence's length-one prefix survives.
        self.assertEqual(partition.select([10, 11], scores)[0], {"shard": 0, "sequence": 1, "prefix": 1})


if __name__ == "__main__":
    unittest.main()
