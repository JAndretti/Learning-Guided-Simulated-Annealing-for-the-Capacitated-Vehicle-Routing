import pandas as pd
import torch
import os
import warnings
import glob2
import sys
import time  # Add import for time measurement
import argparse

from func import (
    get_HP_for_model,
    set_seed,
    load_model,
    init_problem_parameters,
    init_pb,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from sa import sa_test
from model import CVRPActorPairs, CVRPActor
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

# Suppress warnings if needed
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--INIT",
    choices=["random", "isolate", "sweep", "nearest_neighbor", "Clark_and_Wright"],
    default="random",
    type=str,
    help="Initialization method for CVRP",
)
parser.add_argument(
    "--dim",
    default=100,
    choices=[10, 20, 50, 100, 500, 1000],
    type=int,
    help="Problem dimension",
)
parser.add_argument(
    "--FOLDER",
    type=str,
    help="Path to the trained model",
)
parser.add_argument(
    "--OUTER_STEPS",
    default=10000,
    type=int,
    help="Number of SA steps for the algorithm",
)
parser.add_argument(
    "--DATA",
    default="nazari",
    choices=["nazari", "uchoa"],
    type=str,
    help="Dataset to use",
)
parser.add_argument(
    "--no-baseline",
    dest="BASELINE",  # Store the result in the BASELINE variable
    action="store_false",  # If the argument is present, store False
    default=True,  # Otherwise, the default value is True
    help="Disable the baseline",
)
parser.add_argument(
    "--greedy",
    dest="GREEDY",  # Store the result in the GREEDY variable
    action="store_true",  # If the argument is present, store True
    default=False,  # Otherwise, the default value is False
    help="Enable greedy mode",
)
args = parser.parse_args()
# Configuration
cfg = {
    "PROBLEM_DIM": args.dim,
    "N_PROBLEMS": 10000,
    "OUTER_STEPS": args.OUTER_STEPS,
    "DEVICE": (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "SEED": 0,
    "LOAD_PB": True,
    "INIT": args.INIT,
    "MULTI_INIT": False,
    "DATA": args.DATA,
    "BASELINE": args.BASELINE,
    "GREEDY": args.GREEDY,
}
set_seed(cfg["SEED"])
# PATH and FOLDER setup
FOLDER = args.FOLDER
BASE_PATH = "res/" + FOLDER + "/"
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH, exist_ok=True)
PATH = "wandb/Neural_Simulated_Annealing/"
RESULTS_FILE_ALL_MODEL = BASE_PATH + (f"res_all_model_{cfg['PROBLEM_DIM']}.csv")
RESULTS_FILE = BASE_PATH + f"res_model_{cfg['PROBLEM_DIM']}.csv"
MODEL_NAMES = glob2.glob(os.path.join(PATH, FOLDER, "models", "*"))

PATH_DATA = f"generated_{cfg['DATA']}_problem/gen_{cfg['DATA']}_{cfg['PROBLEM_DIM']}.pt"
indices = torch.randperm(cfg["N_PROBLEMS"], generator=torch.Generator())
bdd = torch.load(PATH_DATA, map_location="cpu")
coords = bdd["node_coords"][indices]
demands = bdd["demands"][indices]
capacities = bdd["capacity"][indices]


def initialize_results_df(columns: list):
    """Initialize or load results DataFrame."""
    new_file = RESULTS_FILE
    if os.path.exists(RESULTS_FILE):
        print(f"Existing results file found at {RESULTS_FILE}, creating a new file.")
        base, ext = os.path.splitext(RESULTS_FILE)
        i = 1
        new_file = f"{base}_{i}{ext}"
        while os.path.exists(new_file):
            i += 1
            new_file = f"{base}_{i}{ext}"
    return pd.DataFrame(columns=columns), new_file


def load_results_models():
    """Load results from all models into a DataFrame."""
    if os.path.exists(RESULTS_FILE_ALL_MODEL):
        df = pd.read_csv(RESULTS_FILE_ALL_MODEL)
        print(f"Loaded existing results from {RESULTS_FILE_ALL_MODEL}")
    else:
        df = pd.DataFrame(
            columns=[
                "model",
                "final_cost",
            ]
        )
        print(f"Created new DataFrame for results at {RESULTS_FILE_ALL_MODEL}")
    return df


def extract_differing_keys(model_names):
    """Extract keys with differing values across HP.yaml files."""
    all_hp_data = []
    for model_name in model_names:
        HP = get_HP_for_model(model_name)
        if HP:
            all_hp_data.append(HP)

    differing_keys = set()
    if all_hp_data:
        keys = all_hp_data[0].keys()
        for key in keys:
            values = {
                tuple(hp.get(key)) if isinstance(hp.get(key), list) else hp.get(key)
                for hp in all_hp_data
            }
            if len(values) > 1:  # Key has differing values
                differing_keys.add(key)
    return differing_keys


def add_hp_to_results(model_name, differing_keys):
    """Extract HP values for the given model and return as a dictionary."""
    HP = get_HP_for_model(model_name)
    if not HP:
        return {key: None for key in differing_keys}
    return {key: HP.get(key, None) for key in differing_keys}


def perform_test(
    model_name: str,
    problem: object,
    init_x: torch.Tensor,
    baseline: bool = True,
):
    """Main execution function."""
    logger.info(f"Processing model: {model_name}")
    # init HP
    HP = init_problem_parameters(model_name, cfg)
    problem.set_heuristic(HP["HEURISTIC"])
    problem.params = HP
    # Initialize models
    if HP["MODEL"] == "pairs":
        actor = CVRPActorPairs(
            embed_dim=HP["EMBEDDING_DIM"],
            c=HP["ENTRY"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if len(HP["HEURISTIC"]) > 1 else False,
            method=HP["UPDATE_METHOD"],
        )
    elif HP["MODEL"] == "seq":
        actor = CVRPActor(
            embed_dim=HP["EMBEDDING_DIM"],
            c=HP["ENTRY"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if len(HP["HEURISTIC"]) > 1 else False,
            method=HP["UPDATE_METHOD"],
        )
    # Load model
    actor = load_model(actor, model_name, "actor")

    # Run simulated annealing with time measurement
    start_time = time.time()

    test = sa_test(
        actor,
        problem,
        init_x,
        HP,
        baseline=False,
        greedy=HP["GREEDY"],
        desc_tqdm="NSA Model",
    )
    execution_time = time.time() - start_time
    init_cost = torch.mean(problem.cost(init_x))
    final_cost = torch.mean(test["min_cost"])
    if baseline:
        step = HP["OUTER_STEPS"]
        HP["OUTER_STEPS"] *= 20
        step_baseline = HP["OUTER_STEPS"]
        start_time = time.time()
        test_baseline = sa_test(
            actor,
            problem,
            init_x,
            HP,
            baseline=True,
            greedy=False,
            desc_tqdm="SA Baseline",
        )
        execution_time_baseline = time.time() - start_time
        HP["OUTER_STEPS"] = step
        final_cost_baseline = torch.mean(test_baseline["min_cost"])
    else:
        final_cost_baseline = torch.tensor(float("nan"))
        execution_time_baseline = torch.tensor(float("nan"))
        step_baseline = torch.tensor(float("nan"))
        step = HP["OUTER_STEPS"]
    # Clear CUDA cache if using GPU
    if cfg["DEVICE"] == "cuda":
        torch.cuda.empty_cache()
    return (
        init_cost,
        final_cost,
        final_cost_baseline,
        execution_time,
        execution_time_baseline,
        step,
        step_baseline,
    )


if __name__ == "__main__":

    # Extract keys with differing values across HP.yaml files
    differing_keys = extract_differing_keys(MODEL_NAMES)

    # Initialize results DataFrame with dynamic columns
    columns = [
        "model",
        "test_data",
        "initial_cost",
        "final_cost",
        "final_cost_baseline",
        "execution_time",
        "execution_time_baseline",
        "NSA_steps",
        "SA_steps",
    ] + list(differing_keys)
    new_df, RESULTS_FILE = initialize_results_df(columns)
    all_models_results_df = load_results_models()
    problem = init_pb(
        cfg,
        coords,
        demands,
        capacities,
    )
    init_x = problem.generate_init_state(cfg["INIT"], False)
    init_cost = torch.mean(problem.cost(init_x))
    logger.info(f"CVRP problem initialized. Initial cost: {init_cost:.4f}")
    for model_name in tqdm(MODEL_NAMES, desc="Processing models"):

        (
            init_cost,
            final_cost,
            final_cost_baseline,
            execution_time,
            execution_time_baseline,
            step,
            step_baseline,
        ) = perform_test(model_name, problem, init_x, cfg["BASELINE"])

        # Extract HP values for the current model
        hp_values = add_hp_to_results(model_name, differing_keys)
        # Add results to DataFrame
        new_df = pd.concat(
            [
                new_df,
                pd.DataFrame(
                    {
                        "model": [model_name.split("/")[-1]],
                        "test_data": [cfg["DATA"]],
                        "initial_cost": [init_cost.item()],
                        "final_cost": [final_cost.item()],
                        "final_cost_baseline": [final_cost_baseline.item()],
                        "execution_time": [execution_time],
                        "execution_time_baseline": [execution_time_baseline],
                        "NSA_steps": [step],
                        "SA_steps": [step_baseline],
                        **hp_values,
                    }
                ),
            ],
            ignore_index=True,
        )
        # Save results for all models
        all_models_results_df = pd.concat(
            [
                all_models_results_df,
                pd.DataFrame(
                    {
                        "model": [FOLDER + "/" + model_name.split("/")[-1]],
                        "final_cost": [final_cost.item()],
                    }
                ),
            ],
            ignore_index=True,
        )

    # Remove duplicate rows based on the 'model' column
    new_df = new_df.drop_duplicates(subset=["model"], keep="first")
    all_models_results_df = all_models_results_df.drop_duplicates(
        subset=["model"], keep="first"
    )

    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)
    logger.info("Results DataFrame:")
    print(new_df.head())

    all_models_results_df.to_csv(RESULTS_FILE_ALL_MODEL, index=False)
    logger.info("All models results DataFrame:")
    print(all_models_results_df.head())
