import datetime
import logging
import os
import shutil
from typing import Callable, Optional

import yaml

import wandb
from model import SAModel

from .HP import _HP


def save_HP(path, cfg):
    """
    Saves the hyperparameters used for the current run
    """
    path = os.path.join(path, "HP.yaml")
    if not os.path.exists(path):
        with open(path, "w") as fichier:
            yaml.dump(cfg, fichier, default_flow_style=False, allow_unicode=True)
    return path


def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")


WANDB_DIRECTORY = os.path.join(os.getcwd(), "wandb")
create_folder(WANDB_DIRECTORY)


# Weights & Biases #
class WandbLogger:
    _instance = None

    def __init__(
        self, code_dir: Optional[str], num_models_to_keep: int, HP: "_HP"
    ) -> None:
        project_name = str(HP["PROJECT"]) if HP.get("PROJECT") else "default_project"
        wandb_project_path = os.path.join(WANDB_DIRECTORY, project_name)

        os.makedirs(wandb_project_path, exist_ok=True)
        with open("src/key.txt", "r") as key_file:
            key = key_file.read().strip()

        wandb.login(key=key)
        wandb.init(
            project=HP["PROJECT"],
            group=HP["GROUP"],
            name=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            dir=wandb_project_path,
            save_code=False,
            config=HP.get_config(),
        )

        self.run_dir = wandb.run.dir

        self.num_models_to_keep = num_models_to_keep
        self.model_dir = os.path.join(
            WANDB_DIRECTORY,
            HP["PROJECT"],
            HP["GROUP"],
            "models",
            f"{wandb.run.name}_{wandb.run.id}",
        )
        self.sorted_queue = []
        os.makedirs(self.model_dir, exist_ok=True)

        self.code_dir = code_dir
        self.code_extensions = ["py", "yaml"]
        self.HP = HP
        save_HP(self.model_dir, self.HP)

    @classmethod
    def get_model_dir(cls) -> str:
        return cls._instance.model_dir

    @classmethod
    def init(cls, code_dir, num_models_to_keep, HP):
        cls._instance = WandbLogger(code_dir, num_models_to_keep, HP)

    @classmethod
    def log(cls, logs: dict):
        logging.info(logs)
        if cls._instance is None:
            logging.warning("No Wandb logging")
            return
        wandb.log(logs)

    @classmethod
    def get_image_logging_path(
        cls, figure_name: str, epoch: int, figure_subname=""
    ) -> str:
        """Returns the suitable path to save a figure for the currnt epoch

        Args:
            figure_name (str): name (folder) of the figure
            epoch (int): current epoch
            figure_subname (str, optional): Suffix to the figure. Defaults to "".

        Returns:
            str: Path to save the figure
        """
        if cls._instance is None:
            logging.warn("No Wandb logging")
            return "tmp.png"
        path = os.path.join(
            cls._instance.run_dir,
            "images",
            figure_name,
            f"{epoch:03d}_{figure_subname}.png",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @classmethod
    def close(cls):
        if cls._instance is not None:
            wandb.finish()
            cls._instance = None

    # def __del__(self):
    #     self.close()

    def log_code(self):
        """
        Logs the code used for the current run by copying specified files
        to a new directory.

        Inputs
            run_dir (str): The directory path where the code will be saved.
            code_dir (str): The directory path of the source code.
            extensions (list): A list of file extensions to copy.
        """

        # Create the directory to save the code
        os.makedirs(os.path.join(self.run_dir, "code", self.code_dir), exist_ok=True)

        # Loop through all files in the specified directory with the given extensions
        for file in os.listdir(self.code_dir):
            ext = file.split(".")[-1]
            if ext in self.code_extensions:
                # Copy the file to the new directory
                shutil.copyfile(
                    os.path.join(self.code_dir, file),
                    os.path.join(self.run_dir, "code", self.code_dir, file),
                )

    @classmethod
    def log_model(
        cls,
        save_func: Callable[[str], None],
        model: SAModel,
        val_loss: float,
        epoch: int,
        model_name: str,
    ):
        """
        Saves the given model to the given file path if it's among the best models

        Inputs
            - model     (keras.Model): The model to be saved
            - val_loss  (float):       The validation loss of the model

        Returns:
            - bool: True if a model was saved, False otherwise
        """
        if cls._instance is None:
            logging.warning("No Wandb logging")
            return False, None
        if len(cls._instance.sorted_queue) < cls._instance.num_models_to_keep:
            if model_name is not None:
                model_name += "_"
            model_name += f"epoch_{epoch}_loss_{val_loss:.6f}.pt"
            file_path = os.path.join(cls._instance.model_dir, model_name)

            cls._instance.sorted_queue.append((val_loss, model_name))
            cls._instance.sorted_queue = sorted(
                cls._instance.sorted_queue, key=lambda x: x[0]
            )

            save_func(file_path, model)
            # save_HP(cls._instance.model_dir, cls._instance.HP)
            return True, file_path

        else:
            if val_loss < cls._instance.sorted_queue[-1][0]:
                _, file_suffix_to_delete = cls._instance.sorted_queue.pop()

                try:
                    for filename in os.listdir(cls._instance.model_dir):
                        if filename.endswith(file_suffix_to_delete):
                            os.remove(os.path.join(cls._instance.model_dir, filename))
                except Exception:
                    logging.error("Error : Can't delete file %s", file_suffix_to_delete)

                saved, path = cls.log_model(
                    save_func, model, val_loss, epoch, model_name
                )
                return saved, path
            return False, None
