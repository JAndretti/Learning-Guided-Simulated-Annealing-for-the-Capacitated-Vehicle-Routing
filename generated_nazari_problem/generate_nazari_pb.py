import torch
from loguru import logger  # Enhanced logging capabilities

# Remove default logger
logger.remove()
# Add custom logger with colored output
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # Timestamp in green
        "<blue>{file}:{line}</blue> | "  # File and line in blue
        "<yellow>{message}</yellow>"  # Message in yellow
    ),
    colorize=True,
)

N_PROBLEMS = 10000
DIM = [10, 20, 50, 100, 500, 1000]
device = "cpu"

for dim in DIM:
    # Generate coordinates (depot + customers)
    coordinates = torch.rand(N_PROBLEMS, dim + 1, 2, device=device)

    # Generate demands (depot has 0 demand, customers have demand 1-9)
    demands = torch.randint(
        1,
        10,
        (N_PROBLEMS, dim + 1),
        device=device,
    )
    demands[:, 0] = 0
    # Set capacity for all problems
    dict_capacity = {10: 20, 20: 30, 50: 40, 100: 50, 500: 50, 1000: 50}
    capacity = dict_capacity.get(dim, 50)
    capacities = torch.full((N_PROBLEMS, 1), capacity, device=device)
    output_path = f"generated_nazari_problem/gen_nazari_{dim}.pt"
    torch.save(
        {
            "capacity": capacities,
            "node_coords": coordinates,
            "demands": demands,
        },
        output_path,
    )
    logger.info(f"Saved generated problems with dimension {dim} to {output_path}.")
