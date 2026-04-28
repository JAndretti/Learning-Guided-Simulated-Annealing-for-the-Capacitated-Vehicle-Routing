import numpy as np


def cvrplib_rounded_cost(solution: list[int], raw_coords: np.ndarray) -> int:
    """Euclidean distance rounded to nearest integer per edge (CVRPLib Set X/XL convention)."""
    distance = 0
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i + 1]
        distance += int(np.sqrt(np.sum((raw_coords[u] - raw_coords[v]) ** 2)) + 0.5)
    return distance


def exact_euclidean_cost(solution: list[int], raw_coords: np.ndarray) -> float:
    """Exact float Euclidean distance (Queiroga XML convention)."""
    distance = 0.0
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i + 1]
        distance += float(np.sqrt(np.sum((raw_coords[u] - raw_coords[v]) ** 2)))
    return distance


def extract_and_cost(
    solution: list,
    actual_N: int,
    raw_coords: np.ndarray,
    rounded: bool = True,
) -> float:
    """
    Clean up a padded solution then compute its cost.

    Maps ghost nodes (index >= actual_N) back to depot (0), deduplicates
    consecutive depot visits, ensures the tour starts and ends at depot,
    then delegates to cvrplib_rounded_cost or exact_euclidean_cost.
    Used by handler_X_batch for padded heterogeneous instances.
    """
    solution = [v if v < actual_N else 0 for v in solution]
    deduped: list[int] = []
    for val in solution:
        if val == 0 and deduped and deduped[-1] == 0:
            continue
        deduped.append(val)
    if not deduped or deduped[0] != 0:
        deduped.insert(0, 0)
    if deduped[-1] != 0:
        deduped.append(0)
    if rounded:
        return cvrplib_rounded_cost(deduped, raw_coords)
    return exact_euclidean_cost(deduped, raw_coords)
