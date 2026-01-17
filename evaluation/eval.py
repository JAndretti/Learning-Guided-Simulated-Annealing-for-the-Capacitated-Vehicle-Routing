import argparse
import os
import sys
import time
import warnings

import glob2
import pandas as pd
import torch
from tqdm import tqdm

# Adjust imports to match your project structure
# Assuming the functions from init.py are available in 'func' or similar module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from func import (
    get_HP_for_model,
    init_problem_parameters,
    load_model,
    set_seed,
)

from init import test_model, initialize_models, initialize_test_problem
from problem import CVRP
from utils import setup_logging

logger = setup_logging()

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
    dest="BASELINE",
    action="store_false",
    default=True,
    help="Disable the baseline",
)
parser.add_argument(
    "--greedy",
    dest="GREEDY",
    action="store_true",
    default=False,
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
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
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
PATH = "wandb/LGSA/"
RESULTS_FILE_ALL_MODEL = BASE_PATH + (f"res_all_model_{cfg['PROBLEM_DIM']}.csv")
RESULTS_FILE = BASE_PATH + f"res_model_{cfg['PROBLEM_DIM']}.csv"
MODEL_NAMES = glob2.glob(os.path.join(PATH, FOLDER, "models", "*"))


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


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_differing_keys(model_names):
    """Extract keys with differing values across HP.yaml files (handling nested dicts)."""
    all_hp_data = []
    for model_name in model_names:
        HP = get_HP_for_model(model_name)
        if HP:
            # Flatten the config so nested dicts become searchable keys
            # e.g., {'features': {'static': True}} -> {'features.static': True}
            all_hp_data.append(flatten_dict(HP))

    differing_keys = set()
    if all_hp_data:
        # Get a superset of all keys across all models
        keys = set().union(*all_hp_data)

        for key in keys:
            # Create a set of values for this key across all models
            values = set()
            for hp in all_hp_data:
                val = hp.get(key)

                # Handle unhashable types (lists) by converting to tuple
                if isinstance(val, list):
                    val = tuple(val)

                values.add(val)

            if len(values) > 1:  # Key has differing values
                differing_keys.add(key)

    return differing_keys


def add_hp_to_results(model_name, differing_keys):
    """Extract HP values for the given model and return as a dictionary."""
    HP = get_HP_for_model(model_name)
    if not HP:
        return {key: None for key in differing_keys}

    # Flatten here as well so we can find keys like 'features.static'
    flat_hp = flatten_dict(HP)

    return {key: flat_hp.get(key, None) for key in differing_keys}


def perform_test(
    model_name: str,
    problem: CVRP,
    init_x: torch.Tensor,
    baseline: bool = True,
):
    """Main execution function using refactored init methods."""
    logger.info(f"Processing model: {model_name}")

    # 1. Init HP
    HP = init_problem_parameters(model_name, cfg)
    problem.set_heuristic(HP["HEURISTIC"])
    problem.set_feature_flags(HP["features"])
    input_dim = problem.get_input_dim()
    problem.params = HP

    # 2. Initialize Actor using helper from init.py
    actor, _ = initialize_models(
        model_type=HP["MODEL"],
        critic_type="ff",  # Default assumption, irrelevant for inference
        embedding_dim=HP["EMBEDDING_DIM"],
        entry=input_dim,
        num_h_layers=HP["NUM_H_LAYERS"],
        update_method=HP["UPDATE_METHOD"],
        heuristic=HP["HEURISTIC"],
        seed=cfg["SEED"],
        device=HP["DEVICE"],
    )

    # 3. Load weights
    actor = load_model(actor, model_name, "actor")

    # 4. Run Inference (LGSA Model)
    start_time = time.time()

    # Using inf_test_model from init.py
    HP["TEST_OUTER_STEPS"] = HP["OUTER_STEPS"]
    test = test_model(
        actor=actor,
        problem=problem,
        initial_solutions=init_x,
        config=HP,
        baseline=False,
        greedy=HP["GREEDY"],
    )

    execution_time = time.time() - start_time
    init_cost = torch.mean(problem.cost(init_x))
    final_cost = torch.mean(torch.tensor(test["min_cost"]))

    # 5. Run Baseline (if enabled)
    if baseline:
        step = HP["OUTER_STEPS"]
        HP["OUTER_STEPS"] *= 20
        step_baseline = HP["OUTER_STEPS"]

        start_time = time.time()

        # Using inf_test_model for baseline
        test_baseline = test_model(
            actor=actor,
            problem=problem,
            initial_solutions=init_x,
            config=HP,
            baseline=True,
            greedy=False,
        )

        execution_time_baseline = time.time() - start_time
        HP["OUTER_STEPS"] = step
        final_cost_baseline = torch.mean(torch.tensor(test_baseline["min_cost"]))
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
        "LGSA_steps",
        "SA_steps",
    ] + list(differing_keys)

    new_df, RESULTS_FILE = initialize_results_df(columns)
    all_models_results_df = load_results_models()

    # --- PROBLEM INITIALIZATION ---
    problem, init_x = initialize_test_problem(
        config=cfg,
        test_dim=cfg["PROBLEM_DIM"],
        n_test_problems=cfg["N_PROBLEMS"],
        init_method=cfg["INIT"],
        data=cfg["DATA"],
        device=cfg["DEVICE"],
    )

    init_cost = torch.mean(problem.cost(init_x))
    logger.info(f"CVRP problem initialized. Initial cost: {init_cost:.4f}")

    # Process Models
    for model_name in tqdm(MODEL_NAMES, desc="Processing models", leave=False):
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
                        "LGSA_steps": [step],
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
