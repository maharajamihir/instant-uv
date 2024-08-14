import torch
import numpy as np

import wandb

import os
import json
import yaml
import sys
from pathlib import Path
import argparse

# Append src/
sys.path.append("src/")
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

# Change into instant-uv
os.chdir(Path(__file__).parent.parent.parent)

# Append scripts DIR
SCRIPTS_DIR = str(Path(__file__).parent.parent / "tiny-cuda-nn/scripts")
sys.path.append(SCRIPTS_DIR)

from data.dataset import InstantUVDataset, InstantUVDataLoader
from util.render import ImageRenderer
from util.utils import compute_psnr, load_config, export_uv
from model import InstantUV
from train import Trainer


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    parser.add_argument("config_path", nargs="?", default="config/human/config_human.yaml",
                        help="YAML config for our training stuff")
    parser.add_argument("tiny_nn_config", nargs="?", default="src/tiny-cuda-nn/data/config_hash.json",
                        help="JSON config for tiny-cuda-nn")
    parser.add_argument("n_steps", nargs="?", type=int, default=1000000, help="Number of training steps")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    config = load_config(args.config_path)

    with open(args.tiny_nn_config) as tiny_nn_config_file:
        tiny_nn_config = json.load(tiny_nn_config_file)

    # start a new wandb run to track this script
    wandb.init(
        project="instant-uv",
    )
    tiny_nn_config["encoding"]["n_levels"] = wandb.config["hash_n_levels"]
    tiny_nn_config["encoding"]["log_2_hashmap_size"] = wandb.config["log_2_hashmap_size"]
    config["training"]["lr"] = wandb.config["lr"]
    config["training"]["weight_decay"] = wandb.config["weight_decay"]

    print("=============================================================")
    print(tiny_nn_config)
    print("=============================================================")
    print(config)

    model = InstantUV(tiny_nn_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, config, tiny_nn_config, device)
    trainer.train()

if __name__ == "__main__":
    sweep_configuration = {
        "method": "random",
        "parameters": {
            "lr": {"values": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]},
            "hash_n_levels": {"values": [2, 4,6,8,10,12,15]},
            "log_2_hashmap_size": {"values": [1,3,6,9,12,15]},
            "weight_decay": {"values": [1e-4, 5e-4, 5e-5, 1e-5]}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="instant-uv")
    print(sweep_id)
    wandb.agent(sweep_id, function=main, count=10)

