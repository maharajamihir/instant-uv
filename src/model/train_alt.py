import torch
import numpy as np

import wandb

import os
import json
import yaml
import sys
from pathlib import Path
import argparse

from model.train import Trainer

# Append src/
sys.path.append(str(Path(__file__).parent.parent))

# Change into instant-uv
os.chdir(Path(__file__).parent.parent.parent)

# Append scripts DIR
SCRIPTS_DIR = str(Path(__file__).parent.parent / "tiny-cuda-nn/scripts")
sys.path.append(SCRIPTS_DIR)

# Remove src/model from path (matters when we debug train.py directly)
while str(Path(__file__).parent) in sys.path:
    sys.path.remove(str(Path(__file__).parent))

from data.dataset import InstantUVDataset, InstantUVDataLoader, InstantUVDataset2
from util.render import ImageRenderer, downscale_image
from util.utils import compute_psnr, load_config, export_uv, export_reference_image, load_np, load_mesh
from util.enums import CacheFilePaths
from model.model import InstantUV


class TrainerAlt(Trainer):
    def load_data(self, split):  # TODO: Perhaps different file
        base_dir = str(Path(self.config["data"]["preproc_data_path"]) / split)
        uv = load_np(f"{base_dir}/uvs.npy", allow_not_exists=False)
        rgb = load_np(f"{base_dir}/rgbs.npy", allow_not_exists=False)
        fids = load_np(f"{base_dir}/fids.npy", allow_not_exists=False)
        barys = load_np(f"{base_dir}/bs.npy", allow_not_exists=False)

        return uv, rgb, barys, fids

    def experimental_seam_loss_init(self, uv_path, vids_of_hit_faces):
        # BLENDER BASED VMAPPINGS ONLY!!!
        if self.uv_backend == "xatlas":
            raise NotImplementedError
        vmapping = np.load(uv_path, allow_pickle=True)
        d = {}
        for v in vmapping.values():
            for k, vv in v.items():
                arr = d.setdefault(k, [])
                if list(vv) not in arr:
                    arr.append(list(vv))
        vertex_ids_seams = {k: v for k, v in d.items() if len(v) != 1}
        has_seam_vertices = [any([vertex_ids_seams.get(xx, False) for xx in x]) for x in vids_of_hit_faces]
        self.loss_pairs = []
        for l in vertex_ids_seams.values():
            if len(l) == 2:
                self.loss_pairs.append(list(l))
            if len(l) == 3:
                self.loss_pairs.append(list(l[0:2]))
                self.loss_pairs.append(list(l[1:3]))
        self.loss_pairs = torch.from_numpy(np.array(self.loss_pairs)).to("cuda")

    def __init__(self, model, config, device):
        """
        Initialize the Trainer with the given model and configuration.

        Args:
            model: The model to be trained.
            config: Configuration settings for training, such as optimizer, learning rate, epochs, etc.
            device: torch.device on what to train
        """
        self.device = device
        self.model = model.to(self.device)
        self.config = config
        self.name = self.config["experiment"]["name"].replace(" ", "_")
        self.use_wandb = self.config["training"]["use_wandb"]
        self.seam_factor = self.config["model"]["seam_loss"]  # TODO: Determine if this is good or not

        if self.use_wandb:
            self.assert_wandb()

        # Load train data
        (uv, rgb, barys, fids) = self.load_data("train")

        self.uv_resolution = (1024 * 2, 1024 * 2)  # TODO: Make this config arg
        self.uv_preprocess_resolution = (1024 * 2, 1024 * 2)  # (1536, 1536)

        # FIXME: PREPROCESSING!!! PT2
        # FIXME: Note to mihir. I don't know if this does anything, feel free to experiment with commenting in/out
        # train_expected_rgbs = self.preprocess_dataset(uv=train_uv_coords, rgb=train_expected_rgbs)

        self.train_data = InstantUVDataset2(uv=uv, rgb=rgb)

        # Load val data
        (uv_val, rgb_val, barys_val, fids_val) = self.load_data("train")

        self.val_data = InstantUVDataset2(uv=uv_val, rgb=rgb_val)
        data_split_path = config.get("data", {}).get("data_split")
        if data_split_path:
            with open(data_split_path, "rb") as f:
                self.data_split = yaml.safe_load(f)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(self.config["training"]["lr"]),
                                          betas=(float(self.config["training"]["beta1"]),
                                                 float(self.config["training"]["beta2"])),
                                          eps=float(self.config["training"]["epsilon"]),
                                          weight_decay=float(self.config["training"]["weight_decay"])

                                          )  # FIXME get this from config

        # FIXME would it also work if we just used nn.MSELoss?
        # or does the normalizing improve this significantly?
        self.loss = self.config["model"].get("loss", "L2").lower()
        assert self.loss in ["l1", "l2", "c"], "Loss must be either L1 or L2 or c"
        if self.loss == "l2":
            self.loss_fn = lambda pred, target: (
                    (pred - target.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)).mean()
        elif self.loss == "l1":
            self.loss_fn = lambda pred, target: torch.abs(pred - target.to(pred.dtype)).mean()
        elif self.loss == "c":
            self.gamma = 1  # TODO: ARGS
            self.loss_fn = lambda pred, target: torch.log(
                1 + ((pred - target.to(pred.dtype)) / self.gamma) ** 2).mean()

        xatlas_path = str(Path(config["data"]["preproc_data_path"]) / "xatlas.obj")

        self.uv_backend = self.config["preprocessing"]["uv_backend"].lower()
        # TODO: Refactor all this shit its HORRIBLE
        if self.uv_backend == "blender" or self.uv_backend == "gt":
            uv_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "blender_uv.pkl")
            path_to_mesh = self.config["data"]["mesh_path"]
        else:
            uv_path = str(Path(config["data"]["preproc_data_path"]) / "uv.pkl")
            path_to_mesh = xatlas_path

        self.image_renderer = ImageRenderer(
            path_to_mesh=path_to_mesh,
            dataset_path=config["data"]["raw_data_path"],
            uv_path=uv_path,
            verbose=True,
        )

        self.render_scale = self.config["training"]["render_scale"]
        self.save_validation_images = self.config["training"]["save_validation_images"]

    def _train_step(self, batch):
        """
        Perform a single training step with the given batch of data.

        Args:
            batch: A batch of data from the training DataLoader.

        Returns:
            loss: The loss value for the training step.
        """
        self.optimizer.zero_grad()
        uv, target_rgb = batch["uv"].to(self.device), batch["rgb"].to(self.device)
        pred_rgb = self.model(uv)

        loss = self.loss_fn(pred_rgb, target_rgb)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate_step(self, batch):
        """
        Perform a single validation step with the given batch of data.

        Args:
            batch: A batch of data from the validation DataLoader.

        Returns:
            loss: The loss value for the validation step.
        """
        uv, target_rgb = batch["uv"].to(self.device), batch["rgb"].to(self.device)

        pred_rgb = self.model(uv)
        loss = self.loss_fn(pred_rgb, target_rgb)

        return loss.item()


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    parser.add_argument("config_path", nargs="?", default="config/human/config_human_gt.yaml",
                        help="YAML config for our training stuff")
    parser.add_argument("n_steps", nargs="?", type=int, default=1000000, help="Number of training steps")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from dotenv import load_dotenv

    args = get_args()
    DEFAULTS_HUMAN = "config/human/config_human_defaults.yaml"
    cfg = load_config(args.config_path, DEFAULTS_HUMAN)
    load_dotenv("src/.env")

    mdl = InstantUV(cfg)

    if cfg["training"]["device"].lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = cfg["training"]["device"]
    trainer = Trainer(mdl, cfg, device)
    trainer.train()
