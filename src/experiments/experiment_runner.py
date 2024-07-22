""" This file is to run the experiments end-to-end including preprocessing. """
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

from data.preprocess_dataset import preprocess_dataset
from model.model import InstantUV
from model.train import Trainer

# Append src/
sys.path.append(str(Path(__file__).parent.parent))
# Change into instant-uv
os.chdir(Path(__file__).parent.parent.parent)
# Append scripts DIR
SCRIPTS_DIR = str(Path(__file__).parent.parent / "tiny-cuda-nn/scripts")
sys.path.append(SCRIPTS_DIR)
# # # DO NOT CHANGE ABOVE # # #


DEFAULTS_HUMAN = "config/human/config_human_defaults.yaml"
DEFAULTS_CAT = "config/cat/config_cat_defaults.yaml"

from util.utils import load_config, load_yaml


def preprocess_if_required(config):
    dataset_path = config["data"]["raw_data_path"]
    out_dir = config["data"]["preproc_data_path"]
    mesh_path = config["data"]["mesh_path"]
    data_split = load_yaml(config["data"]["data_split"])
    mesh_views_list_train = data_split[f"mesh_views_list_train"]
    preprocess_dataset(
        split="train",
        dataset_path=dataset_path,
        path_to_mesh=mesh_path,
        out_dir=out_dir,
        mesh_views_list_train=mesh_views_list_train,
        config=config
    )


def train_with_config(config):
    model = InstantUV(config)

    if config["training"]["device"].lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device chosen: {device}")
    else:
        device = config["training"]["device"]

    trainer = Trainer(model, config, device)
    trainer.train()


def train(config):
    print("Checking status of preprocessing.")
    preprocess_if_required(config)

    print("Starting training.")
    train_with_config(config)


def run_human_gt_blender():
    config_path = "config/human/config_human_gt.yaml"
    config = load_config(config_path, DEFAULTS_HUMAN)
    train(config)


def run_human_xatlas():
    config_path = "config/human/config_human_xatlas.yaml"
    config = load_config(config_path, DEFAULTS_HUMAN)
    train(config)


def run_human_blender():
    config_path = "config/human/config_human_blender.yaml"
    config = load_config(config_path, DEFAULTS_HUMAN)
    train(config)


def run_cat_gt_blender():
    config_path = "config/human/config_cat_gt.yaml"
    config = load_config(config_path, DEFAULTS_CAT)
    train(config)


def run_cat_xatlas():
    config_path = "config/human/config_cat_xatlas.yaml"
    config = load_config(config_path, DEFAULTS_CAT)
    train(config)


def run_cat_blender():
    config_path = "config/human/config_cat_blender.yaml"
    config = load_config(config_path, DEFAULTS_CAT)
    train(config)


def load_env():
    load_dotenv("src/.env")


if __name__ == "__main__":
    load_env()
    run_human_gt_blender()
