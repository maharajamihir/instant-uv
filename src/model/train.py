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


class Trainer:
    """
    TODO update docstring

    A generic Trainer class for training.

    Attributes:
        model: The model to be trained.
        config: Configuration settings for training, such as optimizer, learning rate, epochs, etc.
        optimizer: The optimizer used for training.
        loss_fn: The loss function used for training.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """

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

        # Load train data
        train_uv_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "uv_coords.npy")
        train_rgb_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "expected_rgbs.npy")
        train_bary_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "barycentric_coords.npy")
        train_uv_coords = np.load(train_uv_path)
        train_expected_rgbs = np.load(train_rgb_path)
        train_bary_coords = np.load(train_bary_path)
        self.train_data = InstantUVDataset(uv=train_uv_coords, rgb=train_expected_rgbs, points_xyz=train_bary_coords)

        # Load val data 
        val_uv_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "uv_coords.npy")
        val_rgb_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "expected_rgbs.npy")
        val_bary_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "barycentric_coords.npy")
        val_uv_coords = np.load(val_uv_path)
        val_expected_rgbs = np.load(val_rgb_path)
        val_bary_coords = np.load(val_bary_path)
        self.val_data = InstantUVDataset(uv=val_uv_coords, rgb=val_expected_rgbs, points_xyz=val_bary_coords)
        data_split_path = config.get("data", {}).get("data_split")
        if data_split_path:
            with open(data_split_path, "rb") as f:
                self.data_split = yaml.safe_load(f)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=float(self.config["training"]["lr"]),
                                            weight_decay=float(self.config["training"]["weight_decay"]))  # FIXME get this from config

        # FIXME would it also work if we just used nn.MSELoss? 
        # or does the normalizing improve this significantly?
        self.loss = self.config["model"].get("loss", "L2").lower()
        assert self.loss in ["l1", "l2"], "Loss must be either L1 or L2"
        if self.loss == "l2":
            self.loss_fn = lambda pred, target: ((pred - target.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)).mean()
        elif self.loss == "l1":
            self.loss_fn = lambda pred, target: torch.abs(pred - target.to(pred.dtype)).mean()

        uv_pkl_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "uv.pkl")
        xatlas_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "xatlas.obj")

        self.image_renderer = ImageRenderer(
            xatlas_path,
            dataset_path=config["data"]["raw_data_path"],
            uv_path=uv_pkl_path
        )

        self.uv_resolution = (1024, 1024)  # TODO: Make this config arg

    def train(self):
        """
        Train the model for a specified number of epochs.

        This function iterates over the number of epochs defined in the configuration, 
        performing training and validation steps.
        """
        best_val = 10000.
        best_val_psnr = 0.
        self.train_loader = InstantUVDataLoader(
            self.train_data, batch_size=self.config["training"]["batch_size"], shuffle=True
        )
        self.val_loader = InstantUVDataLoader(
            self.val_data, batch_size=self.config["training"]["batch_size"], shuffle=False
        )
        num_epochs = self.config["training"].get('epochs', 10)
        print("Length train loader:", len(self.train_loader))
        print("Length val loader:", len(self.val_loader))

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self._train_epoch()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss}")
            wandb.log({"epoch": epoch, "train_loss": train_loss})

            # Validation
            if epoch % self.config["training"].get("eval_every", 1000) == 0:
                val_loss, val_psnr = self._validate_epoch()
                print("Validation Loss:", val_loss, "Validation PSNR:", val_psnr)
                wandb.log({"epoch": epoch, "val_loss": val_loss, "val_psnr": val_psnr})
                if val_loss < best_val:
                    best_val = val_loss
                    print("Saving model best (val_error)...")
                    # FIXME: Note i changed most of this to self.model (check if that was ok)
                    torch.save(self.model.state_dict(), "model.pt")
                    export_uv(self.model, "best_uv.png", resolution=self.uv_resolution)
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    print("Saving model best (val_psnr)...")
                    torch.save(self.model.state_dict(), "model_psnr.pt")
                    export_uv(self.model, "best_uv_psnr.png", resolution=self.uv_resolution)

    def _train_epoch(self):
        """
        Perform one epoch of training.

        This function iterates over the training data loader, performing training steps and logging progress.

        Returns:
            Mean training loss over this epoch
        """
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            loss = self._train_step(batch)
            running_loss += loss

        return running_loss / len(self.train_loader)

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

    def _validate_epoch(self):
        """
        Perform one epoch of validation.

        This function iterates over the validation data loader, performing validation steps. 
        It also renders the val images and saves them in the `reports` directory

        Returns:
            Mean val loss and psnr over this epoch
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._validate_step(batch)
                running_loss += loss

        torch.cuda.synchronize()
        # compute psnr
        images_np, gts, masks = self.image_renderer.render_views(
            self.model,
            mesh_views_list=self.data_split["mesh_views_list_val"],
            save_validation_images=False  # TODO: Make this a config arg (Saving takes ~2.2 of total 2.6 seconds
        )
        val_psnrs = np.zeros(len(images_np), dtype=np.float32)

        for i, (image_pred, image_gt, mask) in enumerate(list(zip(images_np, gts, masks))):
            val_psnrs[i] = compute_psnr(
                image_gt[mask].astype("int16") / 255.0,  # FIXME: utype8 would mess up the calculation (CHECK) / Note from Mihir: yes it messes it up, need int16
                image_pred[mask].astype("int16") / 255.0
            )

        val_psnr = np.mean(val_psnrs)
        val_loss = running_loss / len(self.val_loader)
        return val_loss, val_psnr

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

    parser.add_argument("config_path", nargs="?", default="config/human/config_human.yaml",
                        help="YAML config for our training stuff")
    parser.add_argument("tiny_nn_config", nargs="?", default="src/tiny-cuda-nn/data/config_hash.json",
                        help="JSON config for tiny-cuda-nn")
    parser.add_argument("n_steps", nargs="?", type=int, default=1000000, help="Number of training steps")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)

    with open(args.tiny_nn_config) as tiny_nn_config_file:
        tiny_nn_config = json.load(tiny_nn_config_file)

    model = InstantUV(tiny_nn_config)


    # start a new wandb run to track this script
    wandb.init(
        project="instant-uv",
        # track hyperparameters and run metadata
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, config, device)
    trainer.train()
