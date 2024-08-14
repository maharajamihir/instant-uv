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
sys.path.append(str(Path(__file__).parent.parent))

# Change into instant-uv
os.chdir(Path(__file__).parent.parent.parent)

# Append scripts DIR
SCRIPTS_DIR = str(Path(__file__).parent.parent / "tiny-cuda-nn/scripts")
sys.path.append(SCRIPTS_DIR)

# Remove src/model from path (matters when we debug train.py directly)
while str(Path(__file__).parent) in sys.path:
    sys.path.remove(str(Path(__file__).parent))

from data.dataset import InstantUVDataset, InstantUVDataLoader, WeightedDataLoader
from util.render import ImageRenderer, downscale_image
from util.utils import compute_psnr, load_config, export_uv, export_reference_image, load_np
from util.enums import CacheFilePaths
from model.model import InstantUV


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

    def preprocess_dataset(self, uv, rgb):
        """
        COPIED FROM sample_train where we calculate GT IMAGE
        Now we want to smoothen the UV map / GT values!!
        """

        # Get predictions for all that we have in dataset
        input = torch.from_numpy(uv).to("cpu")
        pixel_xy = (input * torch.tensor(self.uv_preprocess_resolution, device="cpu")).long()
        gt = torch.from_numpy(rgb).to("cpu")
        # Multiply predictions by 255 and convert to int in one step
        scaled_gt = (gt * 255).type(torch.uint8)

        # Flattened indices
        indices_px = pixel_xy[:, 1] * self.uv_preprocess_resolution[0] + pixel_xy[:, 0]

        # Initialize accumulators and count arrays
        summed_values = torch.zeros((self.uv_preprocess_resolution[0] * self.uv_preprocess_resolution[1], 3),
                                    dtype=torch.float32, device=pixel_xy.device)
        counts = torch.zeros((self.uv_preprocess_resolution[0] * self.uv_preprocess_resolution[1],), dtype=torch.int32,
                             device=pixel_xy.device)

        # Use scatter_add to sum values and count occurrences
        summed_values.index_add_(0, indices_px, scaled_gt.float())
        counts.index_add_(0, indices_px, torch.ones_like(indices_px, dtype=torch.int32))

        # Avoid division by zero
        nonzero_mask = counts > 0
        summed_values[nonzero_mask] = summed_values[nonzero_mask] / counts[nonzero_mask].unsqueeze(1)
        summed_values = summed_values.type(torch.uint8).reshape(*self.uv_preprocess_resolution, 3)

        # TODO: Return after_querying_the_averaged_img = conv_image[pixel_xy[:, 1], pixel_xy[:, 0]] / 255
        # TODO: And make things below its own function. (maybe)

        """ NOW ADD CONVOLUTION"""
        # Make mask
        image_valid_mask = counts.reshape(*self.uv_preprocess_resolution) > 0

        # Conv
        conv_kernel = torch.ones((1, 1, 2, 2), dtype=torch.float32)

        # Convert
        image_tensor = torch.tensor(summed_values, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        padded_down_right_img = torch.nn.functional.pad(image_tensor, (0, 1, 0, 1))

        mask_tensor = torch.tensor(image_valid_mask, dtype=torch.float32).unsqueeze(0)
        padded_down_right_mask = torch.nn.functional.pad(mask_tensor, (0, 1, 0, 1))

        # Perform convolution on image and mask
        r = torch.nn.functional.conv2d(padded_down_right_img[:, 0:1, :, :], conv_kernel, padding=0)
        g = torch.nn.functional.conv2d(padded_down_right_img[:, 1:2, :, :], conv_kernel, padding=0)
        b = torch.nn.functional.conv2d(padded_down_right_img[:, 2:3, :, :], conv_kernel, padding=0)
        conv_image = torch.concat((r, g, b), dim=1)[0].permute(1, 2, 0)  # Shape: (1, 3, 1023, 1023)
        conv_mask = torch.nn.functional.conv2d(padded_down_right_mask, conv_kernel, padding=0)[0]

        nonzero_mask = conv_mask > 0
        # perform division
        conv_image[nonzero_mask] = conv_image[nonzero_mask] / conv_mask[nonzero_mask].unsqueeze(-1)

        """ QUERY IMAGE !!"""
        # VERY BIG NOTE FIRST [:, 1] and then [:, 0]!!!
        # after_querying_the_averaged_img = summed_values[pixel_xy[:, 1], pixel_xy[:, 0]] / 255
        after_querying_the_averaged_img = conv_image[pixel_xy[:, 1], pixel_xy[:, 0]] / 255

        """COMPARISON"""
        # black_image = torch.zeros((*self.uv_resolution, 3), dtype=torch.uint8, device="cpu")
        # # Use advanced indexing to assign values
        # black_image[pixel_xy[:, 1], pixel_xy[:, 0], :] = scaled_gt

        # Assign the new averaged rgbs
        diff = (rgb - after_querying_the_averaged_img.numpy())
        changed_rgbs_mask = diff.sum(axis=1) > 0
        changed_rgbs = changed_rgbs_mask.sum()
        avg_color_change = (diff[changed_rgbs_mask] * 255).mean(axis=0).round(2)
        print(
            f"#changed_rgbs: {changed_rgbs} of total {len(rgb)} ({changed_rgbs / len(rgb) * 100:.2f}%) with avg_color_change: {avg_color_change}")
        return after_querying_the_averaged_img.numpy()

    def assert_wandb(self):
        assert os.environ.get("WANDB_API_KEY") is not None, ("You trying to use wandb. Thus "
                                                             "WANDB_API_KEY environment variable must be set."
                                                             "See README.md")

    def init_wandb(self):
        try:
            wandb.init(
                project="instant-uv",
                # track hyperparameters and run metadata
                config=self.config
            )
        except wandb.errors.UsageError as e:
            print("WANDB_API_KEY environment variable is not configured correctly. See README.md")
            raise e

    def load_data(self, split):  # TODO: Perhaps different file
        base_dir = str(Path(self.config["data"]["preproc_data_path"]) / split)
        uv = load_np(f"{base_dir}/{CacheFilePaths.UV_COORDS.value}", allow_not_exists=False)
        rgb = load_np(f"{base_dir}/{CacheFilePaths.EXPECTED_RGBS.value}", allow_not_exists=False)
        bary = load_np(f"{base_dir}/{CacheFilePaths.BARYCENTRIC_COORDS.value}", allow_not_exists=False)
        vids = load_np(f"{base_dir}/{CacheFilePaths.VIDS_OF_HIT_FACES.value}", allow_not_exists=False)

        should_exist_angles = self.config["preprocessing"]["export_angles"]
        angles = load_np(f"{base_dir}/{CacheFilePaths.ANGLES.value}", allow_not_exists=should_exist_angles) \
            if should_exist_angles else None
        angles2 = load_np(f"{base_dir}/{CacheFilePaths.ANGLES2.value}", allow_not_exists=should_exist_angles) \
            if should_exist_angles else None

        should_exist_c3d = self.config["preprocessing"]["export_coords3d"]
        c3d = load_np(f"{base_dir}/{CacheFilePaths.COORDS3D.value}", allow_not_exists=should_exist_c3d) \
            if should_exist_c3d else None

        return uv, rgb, bary, vids, angles, angles2, c3d

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

        if False and self.use_wandb:
            self.assert_wandb()

        # Load train data
        (train_uv_coords,
         train_expected_rgbs,
         train_bary_coords,
         vids_of_hit_faces,
         train_angles,
         train_angles2,
         coords3d
         ) = self.load_data("train")

        self.uv_resolution = (1024 * 2, 1024 * 2)  # TODO: Make this config arg
        self.uv_preprocess_resolution = (1024 * 2, 1024 * 2)  # (1536, 1536)

        # FIXME: PREPROCESSING!!! PT2
        # FIXME: Note to mihir. I don't know if this does anything, feel free to experiment with commenting in/out
        # train_expected_rgbs = self.preprocess_dataset(uv=train_uv_coords, rgb=train_expected_rgbs)

        self.train_data = InstantUVDataset(uv=train_uv_coords, rgb=train_expected_rgbs, points_xyz=train_bary_coords,
                                           angles=None)

        # Load val data
        (val_uv_coords,
         val_expected_rgbs,
         val_bary_coords,
         val_vids,
         val_angles,
         val_angles2,
         val_c3d) = self.load_data("val")

        self.val_data = InstantUVDataset(
            uv=val_uv_coords, rgb=val_expected_rgbs, points_xyz=val_bary_coords, angles=val_angles, coords3d=coords3d
        )
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
        assert self.loss in ["l1", "l2", "l2r", "c"], "Loss must be either L1 or L2 or l2n or c"
        if self.loss == "l2r":
            self.loss_fn = lambda pred, target: (
                    (pred - target.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)
            ).mean()
        elif self.loss == "l2":
            self.loss_fn = lambda pred, target: torch.mean((pred - target.to(pred.dtype)) ** 2)
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

        if self.seam_factor > 0:
            self.experimental_seam_loss_init(uv_path, vids_of_hit_faces)

        self.image_renderer = ImageRenderer(
            path_to_mesh=path_to_mesh,
            dataset_path=config["data"]["raw_data_path"],
            uv_path=uv_path,
            verbose=True,
        )

        self.render_scale = self.config["training"]["render_scale"]
        self.save_validation_images = self.config["training"]["save_validation_images"]

    def train(self):
        """
        Train the model for a specified number of epochs.

        This function iterates over the number of epochs defined in the configuration,
        performing training and validation steps.
        """
        if self.use_wandb:
            self.init_wandb()  # TODO: DECLARE IN CONFIG IF WE USE WANDB OR NOT /// OR CALL IN __INIT__

        best_val = 10000.
        best_val_psnr = 0.
        self.train_loader = InstantUVDataLoader(
            self.train_data, batch_size=self.config["training"]["batch_size"], shuffle=True
        )
        # self.train_loader = WeightedDataLoader(
        #     self.train_data, batch_size=self.config["training"]["batch_size"]
        # )
        bs_val = min(self.config["training"]["batch_size_val"], len(self.val_data))
        self.val_loader = InstantUVDataLoader(
            self.val_data, batch_size=bs_val, shuffle=False
        )
        num_epochs = self.config["training"].get('epochs', 10)
        print("Length train loader:", len(self.train_loader))
        print("Length val loader:", len(self.val_loader))

        # Create reference image of dataset
        export_reference_image(dataset=self.train_data, path="reference.png", resolution=self.uv_resolution)

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self._train_epoch()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss}")
            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss})

            # Validation
            if epoch % self.config["training"].get("eval_every", 1000) == 0:
                val_loss, val_psnr = self._validate_epoch()
                print("Validation Loss:", val_loss, "Validation PSNR:", val_psnr)
                if self.use_wandb:
                    wandb.log({"epoch": epoch, "val_loss": val_loss, "val_psnr": val_psnr})

                # TODO: REMOVE
                export_uv(self.model, f"random.png", resolution=(512,512))

                os.makedirs("models", exist_ok=True)
                os.makedirs("uvs", exist_ok=True)
                if val_loss < best_val:
                    best_val = val_loss
                    print("Saving model best (val_error)...")
                    # FIXME: Note i changed most of this to self.model (check if that was ok)
                    torch.save(self.model.state_dict(), f"models/model{'_' + self.name}.pt")
                    # if self.config["training"]["save_uvs"]:
                    #     export_uv(self.model, f"uvs/best_uv{'_' + self.name}.png", resolution=self.uv_resolution)
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    print("Saving model best (val_psnr)...")
                    torch.save(self.model.state_dict(), f"models/model{'_' + self.name}_psnr.pt")
                    if self.config["training"]["save_uvs"]:
                        export_uv(self.model, f"uvs/best_uv{'_' + self.name}_psnr.png", resolution=self.uv_resolution)

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

    def _calcualte_seam_loss(self):
        if self.seam_factor > 0:
            loss_tensor = torch.tensor(0.0, requires_grad=True)
            seam1 = self.model(self.loss_pairs[:, 0, :])
            seam2 = self.model(self.loss_pairs[:, 1, :])

            # Use L1 Loss
            loss_tensor = loss_tensor + (
                    self.seam_factor *
                    (lambda pred, target: torch.abs(pred - target.to(pred.dtype)).mean())(seam1, seam2)
            )
            return loss_tensor
        return 0.0

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
        if "angles" in batch.keys():
            angles = batch["angles"].to(self.device)
            uv = torch.cat((uv, angles.unsqueeze(1)), dim=1)
        pred_rgb = self.model(uv)

        loss = self.loss_fn(pred_rgb, target_rgb)

        # Experimental, can be removed if required
        seam_loss = self._calcualte_seam_loss()
        loss += seam_loss

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
        images_np, gts, masks, hit_counts = self.image_renderer.render_views(
            self.model,
            mesh_views_list=self.data_split["mesh_views_list_val"],
            save_validation_images=self.save_validation_images,
            scale=self.render_scale
        )

        if self.render_scale != 1:
            images_np = [downscale_image(i, h, kernel_size=self.render_scale) for (i, h) in zip(images_np, hit_counts)]
        val_psnrs = np.zeros(len(images_np), dtype=np.float32)

        for i, (image_pred, image_gt, mask) in enumerate(list(zip(images_np, gts, masks))):
            val_psnrs[i] = compute_psnr(
                image_gt[mask].astype("int16") / 255.0,
                # FIXME: utype8 would mess up the calculation (CHECK) / Note from Mihir: yes it messes it up, need int16
                image_pred[mask].astype("int16") / 255.0
            )

        # fromarray(np.abs((image_gt.astype("int16") - image_pred.astype("int16"))).mean(axis=2).astype("uint8"), "L").show()

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
        if "angles" in batch.keys():
            angles = batch["angles"].to(self.device)
            uv = torch.cat((uv, angles.unsqueeze(1)), dim=1)

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
