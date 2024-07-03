import torch
import numpy as np

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
from util.render import ImageRenderer, downscale_image
from util.utils import compute_psnr, load_config, export_uv, export_reference_image
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

        # vertices_of_hit_faces_old = np.array(mesh_view_pre_proc.mesh.vertices[np.stack(mesh_view_pre_proc.cache_vertex_idxs_of_hit_faces)])
        # coords_3d_old = np.sum(np.stack(mesh_view_pre_proc.cache_barycentric_coords)[:, :, np.newaxis] * vertices_of_hit_faces_old, axis=1)
        # import trimesh
        # trimesh.PointCloud(vertices=coords_3d_old, colors=np.stack(mesh_view_pre_proc.cache_expected_rgbs) * 255).show(
        #     line_settings={'point_size': 5}
        # )

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

        self.uv_resolution = (1024 * 2, 1024 * 2)  # TODO: Make this config arg
        self.uv_preprocess_resolution = (1024 * 2, 1024 * 2)  # (1536, 1536)

        # FIXME: PREPROCESSING!!! PT2
        # FIXME: Note to mihir. I don't know if this does anything, feel free to experiment with commenting in/out
        train_expected_rgbs = self.preprocess_dataset(uv=train_uv_coords, rgb=train_expected_rgbs)

        self.train_data = InstantUVDataset(uv=train_uv_coords, rgb=train_expected_rgbs, points_xyz=train_bary_coords)

        """ DEBUG END """
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # FIXME get this from config

        # FIXME would it also work if we just used nn.MSELoss? 
        # or does the normalizing improve this significantly?
        self.loss = self.config["model"].get("loss", "L2").lower()
        assert self.loss in ["l1", "l2"], "Loss must be either L1 or L2"
        if self.loss == "l2":
            self.loss_fn = lambda pred, target: (
                    (pred - target.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)).mean()
        elif self.loss == "l1":
            self.loss_fn = lambda pred, target: torch.abs(pred - target.to(pred.dtype)).mean()

        xatlas_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "xatlas.obj")

        self.uv_backend = self.config["training"].get("uv_backend", "blender").lower()
        # TODO: Refactor all this shit its HORRIBLE
        if self.uv_backend == "blender":
            uv_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "blender_uv.npy")
            path_to_mesh = xatlas_path  # self.config["data"]["mesh_path"]
        else:
            uv_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "uv.pkl")
            path_to_mesh = xatlas_path

        self.image_renderer = ImageRenderer(
            path_to_mesh=path_to_mesh,
            dataset_path=config["data"]["raw_data_path"],
            uv_path=uv_path
        )
        self.render_scale = self.config["training"].get("render_scale", 2)
        self.save_validation_images = self.config["training"].get("save_validation_images", False)

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

        # Create reference image of dataset
        export_reference_image(dataset=self.train_data, path="reference.png", resolution=self.uv_resolution)

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self._train_epoch()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss}")

            # Validation
            if epoch % self.config["training"].get("eval_every", 1000) == 0:
                val_loss, val_psnr = self._validate_epoch()
                print("Validation Loss:", val_loss, "Validation PSNR:", val_psnr)
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
        images_np, gts, masks, hit_counts = self.image_renderer.render_views(
            self.model,
            mesh_views_list=self.data_split["mesh_views_list_val"],
            save_validation_images=self.save_validation_images,
            scale=self.render_scale
        )

        if self.render_scale != 1:
            images_np = [downscale_image(i, h, kernel_size=self.render_scale) for (i, h) in zip(images_np, hit_counts)]
        val_psnrs = np.zeros(len(images_np), dtype=np.float32)

        # FIXME this loop is super slow!!! fix this / Note from moritz: its not slow lol
        for i, (image_pred, image_gt, mask) in enumerate(list(zip(images_np, gts, masks))):
            val_psnrs[i] = compute_psnr(
                image_gt[mask].astype("int16") / 255.0,  # FIXME: utype8 would mess up the calculation (CHECK)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, config, device)
    trainer.train()
