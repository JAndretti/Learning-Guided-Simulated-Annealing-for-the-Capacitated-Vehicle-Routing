import os
import sys
import glob2
import vrplib
import pandas as pd
import argparse

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from algo import or_tools, compute_euclidean_distance_matrix
from tqdm import tqdm

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


def get_opt_cost(sol_file):
    routes = [0]
    with open(sol_file, "r") as f:
        for line in f:
            if line.startswith("Cost"):
                return int(line.strip().split()[1]), routes
            else:
                if line.startswith("Route"):
                    routes.extend(map(int, line.strip().split(":")[1].split()))
                    routes.append(0)  # Return to depot
    return None


def calculate_dist_route(route, distance_matrix):
    """Calculate the total distance of a given route."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--or_tools_time",
        type=float,
        default=1.0,
        help="Time limit for the OR-Tools solver in seconds.",
    )
    args = parser.parse_args()
    cfg = {
        "OR_TOOLS_TIME": args.or_tools_time,
    }
    logger.info(f"Or-Tools solver configuration: {cfg['OR_TOOLS_TIME']}")

    path_opt_cost = "bdd/Vrp-Set-XML100/solutions/"
    instances = glob2.glob("bdd/Vrp-Set-XML100/instances/*.vrp")

    names = []
    dimensions = []
    capacities = []
    node_coords = []
    demands = []
    depots = []
    opt_costs = []
    r_opt_costs = []
    matrixes = []
    opt_routes = []

    for instance_path in tqdm(instances, desc="Getting data", leave=False):
        data = vrplib.read_instance(instance_path)
        name = data["name"]
        names.append(name)
        sol_file = os.path.join(path_opt_cost, f"{name}.sol")
        opt_cost, routes = get_opt_cost(sol_file)
        opt_routes.append(routes)
        opt_costs.append(opt_cost)
        matrix = compute_euclidean_distance_matrix(
            [tuple(coord) for coord in data["node_coord"]]
        )
        matrixes.append(matrix)
        r_opt_costs.append(calculate_dist_route(routes, matrix))
        dimensions.append(data["dimension"])
        capacities.append(data["capacity"])
        node_coords.append(data["node_coord"])
        demands.append(data["demand"])
        depots.append(data["depot"])

    avg_opt_cost = sum(opt_costs) / len(opt_costs)
    avg_r_opt_cost = sum(r_opt_costs) / len(r_opt_costs)
    logger.info(
        "Difference between average opt_costs and r_opt_costs: "
        f"{avg_opt_cost:.2f}, {avg_r_opt_cost:.2f}"
    )

    params = {
        "coords": node_coords,
        "demands": demands,
        "MAX_LOAD": capacities,
        "OR_DIM": dimensions,
        "names": names,
    }
    logger.info("Starting OR-Tools solver...")
    solutions, distances, names = or_tools(params, cfg, mult_thread=True)
    or_cost = [
        calculate_dist_route(solutions[names.index(name)], matrixes[names.index(name)])
        for name in names
    ]
    df_or = pd.DataFrame(
        {
            "name": names,
            "or_tools_cost": or_cost,
        }
    )
    df_opt = pd.DataFrame(
        {
            "name": names,
            "opt_sol": r_opt_costs,
        }
    )

    df = pd.merge(df_or, df_opt, on="name")

    output_path = f"res/Vrp-Set-XML100_res_{cfg['OR_TOOLS_TIME']}.csv"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        logger.warning(
            f"Output directory '{output_dir}' does not exist. "
            "Saving to current directory."
        )
        output_path = f"Vrp-Set-XML100_res_{cfg['OR_TOOLS_TIME']}.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Average or_tools_cost: {df['or_tools_cost'].mean()}")
    logger.info(f"Average opt_sol: {df['opt_sol'].mean()}")
    logger.info(
        "Number of times or_tools_cost < opt_sol: "
        f"{(df['or_tools_cost'] < df['opt_sol']).sum()}"
    )
