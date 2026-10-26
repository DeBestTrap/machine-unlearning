"""Budgeted sequence selection. No training or global RNG side effects."""
import math
import random

import numpy as np


def validate_probabilities(probs, slices):
    if len(probs) != slices or any(not math.isfinite(p) or p < 0 for p in probs):
        raise ValueError("Deletion probabilities must be finite, nonnegative, and match slices")
    if not math.isclose(sum(probs), 1.0, abs_tol=1e-8):
        raise ValueError("Deletion probabilities must sum to one")


def cyclic_sequences(slices, budget):
    """Algorithm 1 with deterministic round-robin expansion for partial budgets.

    Expand whole-sequence rotations first, then rotate successively shorter
    suffixes. Round-robin over parents keeps partial rounds balanced.
    """
    if slices < 1 or not 1 <= budget <= math.factorial(slices):
        raise ValueError("Require slices >= 1 and 1 <= budget <= slices!")
    root = tuple(range(slices))
    sequences = [root[-k:] + root[:-k] if k else root for k in range(slices)]
    if budget <= slices:
        return sequences[:budget]
    for depth in range(1, slices - 1):
        parents = list(sequences)
        for rotation in range(1, slices - depth):
            for parent in parents:
                suffix = parent[depth:]
                sequences.append(parent[:depth] + suffix[-rotation:] + suffix[:-rotation])
                if len(sequences) == budget:
                    return sequences
    return sequences


def sequence_score(sequence, probabilities, horizon=100):
    """The weighted score printed in paper Eq. (24), not an expectation estimate."""
    cumulative, score = 0.0, 0.0
    for length, index in enumerate(sequence, 1):
        cumulative += probabilities[index]
        score += length * max(0.0, 1.0 - cumulative) ** horizon
    return score


def bms_sequences(slices, budget, probabilities, horizon=100, seed=42):
    """Algorithm 2 for B <= L; conditional sampling for larger budgets."""
    validate_probabilities(probabilities, slices)
    if not 1 <= budget <= math.factorial(slices) or horizon < 1:
        raise ValueError("Invalid sequence budget or score horizon")
    if budget > slices:
        return conditional_sequences(slices, budget, probabilities, seed)
    from scipy.optimize import linear_sum_assignment
    sequences = [(i,) for i in range(slices)]
    for _ in range(1, slices):
        scores = np.full((slices, slices), -np.inf)
        for row, prefix in enumerate(sequences):
            for candidate in range(slices):
                if candidate not in prefix:
                    scores[row, candidate] = sequence_score(prefix + (candidate,), probabilities, horizon)
        rows, columns = linear_sum_assignment(scores, maximize=True)
        for row, column in zip(rows, columns):
            sequences[row] += (int(column),)
    return sorted(sequences, key=lambda seq: (-sequence_score(seq, probabilities, horizon), seq))[:budget]


def _unrank(rank, size):
    remaining, result = list(range(size)), []
    for width in range(size, 0, -1):
        index, rank = divmod(rank, math.factorial(width - 1))
        result.append(remaining.pop(index))
    return tuple(result)


def random_sequences(slices, budget, seed=42):
    total = math.factorial(slices)
    if not 1 <= budget <= total:
        raise ValueError("Invalid sequence budget")
    rng = random.Random(seed)
    # Floyd sampling uses O(B) memory and terminates even at full budget.
    ranks, selected = [], set()
    for j in range(total - budget, total):
        candidate = rng.randrange(j + 1)
        candidate = j if candidate in selected else candidate
        selected.add(candidate)
        ranks.append(candidate)
    rng.shuffle(ranks)
    return [_unrank(rank, slices) for rank in ranks]


def conditional_sequences(slices, budget, probabilities, seed=42):
    """Sample without replacement within a sequence, weighted by 1-p(slice).

    Degenerate/full budgets can exhaust the sampling support. A deterministic
    permutation completion prevents a rejection-sampling infinite loop.
    """
    validate_probabilities(probabilities, slices)
    if not 1 <= budget <= math.factorial(slices):
        raise ValueError("Invalid sequence budget")
    rng, result, seen = random.Random(seed), [], set()
    for _ in range(max(100, 20 * budget)):
        remaining, sequence = list(range(slices)), []
        while remaining:
            weights = [1.0 - probabilities[i] for i in remaining]
            item = rng.choices(remaining, weights=weights if sum(weights) else None, k=1)[0]
            sequence.append(item)
            remaining.remove(item)
        sequence = tuple(sequence)
        if sequence not in seen:
            seen.add(sequence)
            result.append(sequence)
        if len(result) == budget:
            return result
    # Visit ranks deterministically, without materializing L! permutations.
    for rank in range(math.factorial(slices)):
        sequence = _unrank(rank, slices)
        if sequence not in seen:
            result.append(sequence)
            seen.add(sequence)
        if len(result) == budget:
            return result


def select_sequences(config):
    if config.sequence_strategy == "cyclic":
        return cyclic_sequences(config.slices, config.budget)
    if config.sequence_strategy == "random":
        return random_sequences(config.slices, config.budget, config.sequence_seed)
    return bms_sequences(config.slices, config.budget, config.deletion_probabilities,
                         config.score_horizon, config.sequence_seed)
