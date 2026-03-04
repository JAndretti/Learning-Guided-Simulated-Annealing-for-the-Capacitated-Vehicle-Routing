import os
import random
import sys

import numpy as np
import torch


def set_seed(seed):
    """Resets all seeds to ensure the inputs are identical every time."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_operation(deterministic, device):
    seed = 42

    # --- TOGGLE DETERMINISM ---
    if deterministic:
        # 1. Force deterministic algorithms (throws error if an op is non-deterministic)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        # 2. CUDNN settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Default PyTorch behavior
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    results = []

    print(
        f"[{'DETERMINISTIC' if deterministic else 'DEFAULT'}] Running 10 iterations..."
    )

    for i in range(10):
        # Reset seed PER LOOP to ensure we generate the exact same input tensors
        set_seed(seed)

        # 1. Create large random tensors
        # We use a large size to force high parallelism on the GPU
        size = 1000000
        data = torch.randn(size, device=device)

        # Random indices for scatter/gather (this triggers atomic operations)
        indices = torch.randint(0, 1000, (size,), device=device)
        output = torch.zeros(1000, device=device)

        # 2. The Operation: index_add_
        # On GPU, this uses atomicAdd. If not deterministic, the order of summation varies.
        output.index_add_(0, indices, data)

        # We sum the output to get a single checksum value
        checksum = output.sum().item()
        results.append(checksum)

    return results


def compare_results(results):
    first = results[0]
    diffs = [abs(r - first) for r in results]
    max_diff = max(diffs)

    if max_diff == 0:
        print(">> RESULT: PERFECTLY IDENTICAL (Success)")
    else:
        print(">> RESULT: DIFFERENCES FOUND!")
        print(f"   Max deviation from run 1: {max_diff:.20f}")
        print(f"   Unique values count: {len(set(results))}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("This test requires a GPU (CUDA) to demonstrate the issue.")
        sys.exit(0)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("-" * 50)

    # --- TEST 1: OFF (Default) ---
    try:
        results_off = run_operation(deterministic=False, device=device)
        compare_results(results_off)
    except Exception as e:
        print(f"Test 1 failed: {e}")

    print("-" * 50)

    # --- TEST 2: ON (Strict Determinism) ---
    try:
        results_on = run_operation(deterministic=True, device=device)
        compare_results(results_on)
    except Exception as e:
        print(f"Test 2 failed: {e}")
        print("Note: Some operations might not support deterministic mode.")
