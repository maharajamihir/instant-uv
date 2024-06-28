import os
from pathlib import Path

import numpy as np
import torch
import argparse
from tqdm import tqdm
import imageio.v2 as imageio
import sys

sys.path.append(str(Path(__file__).parent.parent))

from util.mesh import MeshViewPreProcessor
from util.utils import load_obj_mask, load_cameras, load_config

os.chdir(Path(__file__).parent.parent.parent)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("--config_path", type=str,
                        default=Path(__file__).parent.parent.parent / "config/human/config_human.yaml")
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test'],
                        help="Dataset split [train, val, test]", default="train")
    args = parser.parse_args()
    return args


def preprocess_views(mesh_view_pre_proc, mesh_views_list_train, dataset_path):
    """
    Preprocess mesh views and cache the processed data.

    Args:
        mesh_view_pre_proc (MeshViewPreProcessor): The preprocessor for mesh views.
        mesh_views_list_train (list): List of mesh view file names to process.
        dataset_path (str): Path to the dataset directory.
    """
    for mesh_view in tqdm(mesh_views_list_train):
        mesh_view_path = os.path.join(dataset_path, mesh_view)
        camCv2world, K = load_cameras(mesh_view_path)

        # Load depth map for building a mask
        obj_mask = load_obj_mask(mesh_view_path, as_numpy=False)

        # Load image
        img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))

        img[~obj_mask] = [255, 255, 255]
        imageio.imwrite(os.path.join(mesh_view_path, "image", "001.png"), img)
        img = torch.from_numpy(img).to(dtype=torch.float32)
        img /= 255.

        # Preprocess and cache the current view
        mesh_view_pre_proc.cache_single_view(camCv2world, K, obj_mask, img)

    mesh_view_pre_proc.write_to_disk()


def preprocess_dataset(split, dataset_path, path_to_mesh, out_dir, mesh_views_list_train, check_depth=False, config=None):
    """
    Preprocess the entire dataset for a given split.

    Args:
        split (str): The dataset split to process [train, val, test].
        dataset_path (str): Path to the dataset directory.
        path_to_mesh (str): Path to the mesh file.
        out_dir (str): Directory to save the preprocessed data.
        mesh_views_list_train (list): List of mesh view file names to process.
    """
    split_out_dir = os.path.join(out_dir, split)

    # if os.path.exists(split_out_dir):
    # raise RuntimeError(f"Error: You are trying to overwrite the following directory: {split_out_dir}")
    os.makedirs(split_out_dir, exist_ok=True)

    mesh_view_pre_proc = MeshViewPreProcessor(path_to_mesh, split_out_dir, config=config, split=split)

    preprocess_views(mesh_view_pre_proc, mesh_views_list_train, dataset_path)


def main():
    args = parse_args()
    config = load_config(args.config_path)
    dataset_path = config["data"]["raw_data_path"]
    out_dir = config["data"]["preproc_data_path"]
    mesh_path = config["data"]["mesh_path"]
    data_split = load_config(config["data"]["data_split"])
    mesh_views_list_train = data_split[f"mesh_views_list_{args.split}"]

    if "all" in mesh_views_list_train:
        # NOTE:
        # This is for debugging only. TODO: Delete this later again

        # Quick and dirty way to select all strings that look like humanXXX
        mesh_views_list_train = [f for f in os.listdir("data/raw/human_dataset_v2_tiny/") if len(f) == 8 and
                                 "human" in f]
    print(f"Preprocessing dataset: {args.split}")
    preprocess_dataset(
        split=args.split,
        dataset_path=dataset_path,
        path_to_mesh=mesh_path,
        out_dir=out_dir,
        mesh_views_list_train=mesh_views_list_train,
        config=config
    )


if __name__ == "__main__":
    main()
