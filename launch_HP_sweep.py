import random
import subprocess
from itertools import product
from multiprocessing import Process, Queue
from time import sleep

import yaml
from loguru import logger

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<blue>{file}:{line}</blue> | "
        "<yellow>{message}</yellow>"
    ),
    colorize=True,
)

# ----------------------------------------------------
# Global Parameters
# ----------------------------------------------------
GPU_AVAILABLES = [0]

SWEEP_MODE = "random"  # grid, random

TRAINING_SCRIPT_PATH = "src/main.py"
HYPERPARAMETERS_PATH = "src/HyperParameters/HP_sweep.yaml"
# ----------------------------------------------------


def read_YAML_hyperparameters_sweep(hp_path):
    """
    Read all hyperparampeters used for the sweep from a YAML file

    Inputs
        - hp_path (str) : Path to the YAML file
        containing the hyperparameter sweep values

    Outputs
        - hyperparameter_list  (List): list of possible
        combinations of hyperparameters values
        - hyperparameter_names (List): list of possible
        combinations of hyperparameters names
    """

    with open(hp_path, "r") as f:
        hyperparameters = yaml.safe_load(f)

    hyperparameter_names = list(hyperparameters.keys())
    hyperparameter_list = list(product(*hyperparameters.values()))

    return hyperparameter_names, hyperparameter_list


def run_training_script(gpu_id, hyperparameter_names, hyperparameter_values):
    """
    Constructs a command string to launch a training script with given hyperparameters
    on a specified GPU and runs it

    Inputs
        - gpu_id                (int):  id of the GPU to run the training script on
        - hyperparameter_names  (List): List of hyperparameter names
        - hyperparameter_values (List): List of hyperparameter values

    """

    cmd_str = f"CUDA_VISIBLE_DEVICES={gpu_id} python3 {TRAINING_SCRIPT_PATH}"

    for hp_name, hp_val in zip(hyperparameter_names, hyperparameter_values):
        cmd_str += f" --{hp_name} \"{hp_val}\""  # cmd_str += f" --{hp_name} \"{hp_val}\""

    subprocess.run(cmd_str, shell=True)


def execute_process(gpu_queue, hyperparameter_names, hyperparameter_queue):
    """
    Launch a training script with a given hyperparameter configuration
    when a GPU is available

    Inputs
        - gpu_queue             (Queue): Shared Queue with all available GPU ids
        - hyperparameter_names  (List):  List of hyperparameter names
        - hyperparameter_values (List):  List of hyperparameter values

    """

    while True:
        if not gpu_queue.empty():
            gpu_id = gpu_queue.get()  # get a gpu_id from the queue

            if not hyperparameter_queue.empty():
                hyperparameter_values = (
                    hyperparameter_queue.get()
                )  # get an hyperparameter configuration to process
                # Log the current hyperparameter configuration
                logger.info(f"Starting training on GPU {gpu_id}")
                for name, value in zip(hyperparameter_names, hyperparameter_values):
                    logger.info(f"  {name}: {value}")
                try:
                    run_training_script(
                        gpu_id, hyperparameter_names, hyperparameter_values
                    )  # execute the function with the hyperparameter and the gpu_id
                except Exception as e:
                    print(f"GPU {gpu_id} error :")
                    print(e, "\n")

                gpu_queue.put(gpu_id)  # put the gpu_id back in the queue
            else:
                break  # if there are no more hyperparameters, exit the loop

        sleep(1)


if __name__ == "__main__":
    # Create a queue to store the gpu_ids
    gpu_queue = Queue()
    for gpu_id in GPU_AVAILABLES:
        gpu_queue.put(gpu_id)

    # Create a queue to store all hyperparameters combinations
    hyperparameter_names, hyperparameter_list = read_YAML_hyperparameters_sweep(
        HYPERPARAMETERS_PATH
    )

    # Print the number of configurations to be tested
    logger.info(f"Number of configurations to be tested: {len(hyperparameter_list)}")

    # Shuffle the parameter configurations for the random search
    if SWEEP_MODE == "random":
        random.shuffle(hyperparameter_list)

    # Add the configurations to the sharred queue
    hyperparameter_queue = Queue()
    for hyperparameter_config in hyperparameter_list:
        hyperparameter_queue.put(hyperparameter_config)

    # Create a list of processes to execute
    processes = [
        Process(
            target=execute_process,
            args=(gpu_queue, hyperparameter_names, hyperparameter_queue),
        )
        for _ in GPU_AVAILABLES
    ]

    # Start the processes
    for p in processes:
        p.start()

    # Wait for the processes to finish
    for p in processes:
        p.join()
