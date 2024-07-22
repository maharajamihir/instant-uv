import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import argparse
from tqdm import tqdm
import imageio.v2 as imageio
import sys

sys.path.append(str(Path(__file__).parent.parent))

from util.mesh import MeshViewPreProcessor
from util.utils import load_obj_mask, load_cameras, load_config, load_yaml

os.chdir(Path(__file__).parent.parent.parent)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("--config_path", type=str,
                        default=Path(__file__).parent.parent.parent / "config/human/config_human_gt.yaml")
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

        #
        # """ DEBUG ONLY"""
        # """ DEBUG ONLY"""
        #
        # uv_coords = np.stack(mesh_view_pre_proc.cache_uv_coords)
        # expected_rgbs = np.stack(mesh_view_pre_proc.cache_expected_rgbs)
        #
        # # Lets try only the legs
        # vertices_of_hit_faces = mesh_view_pre_proc.mesh.vertices[np.stack(mesh_view_pre_proc.cache_vertex_idxs_of_hit_faces)]
        # coords_3d = np.sum(np.stack(mesh_view_pre_proc.cache_barycentric_coords)[:, :, np.newaxis] * vertices_of_hit_faces,
        #                    axis=1)  # TODO might need to integrate this into the pipeline as well
        #
        # mask = coords_3d[:, 2] < 0
        # coords_3d = coords_3d[mask]
        # uv_coords = uv_coords[mask]
        # expected_rgbs = expected_rgbs[mask]
        #
        # # Maybe we need to flip???
        # # uv_coords = np.flip(uv_coords, axis=1).copy()
        # """ DEBUG END"""
        #
        # import trimesh
        # # Sanity-Check visualization
        # trimesh.PointCloud(vertices=coords_3d, colors=expected_rgbs * 255).show(
        # line_settings={'point_size': 5}
        # )

    mesh_view_pre_proc.write_to_disk()


def create_validation_hash_from_params(relevant_params):
    # Convert the dictionary to a JSON string
    params_str = json.dumps(relevant_params, sort_keys=True)

    # Create a hash of the JSON string using sha256
    return hashlib.sha256(params_str.encode()).hexdigest()


def check_if_preprocessing_required(validation_hash_path, validation_hash, force_preprocessing):
    if force_preprocessing:
        return True
    else:
        # Check the hashes to see if we need to redo the preprocessing.
        if os.path.exists(validation_hash_path):
            with open(validation_hash_path, 'r') as file:
                validation_hash_file_content = file.read()
        else:
            validation_hash_file_content = None

        if validation_hash_file_content != validation_hash:
            print(f"Validation hash found in file validation_hash.txt ({validation_hash_file_content}) "
                  f"does not equal current configuration hash ({validation_hash}).")
            return True
        else:
            print(f"Validation hash found in file validation_hash.txt ({validation_hash_file_content}) "
                  f"matches current configuration. Skipping the preprocessing.")
            return False


def calculate_validation_hash(config, split, mesh_views_list_train, dataset_path, path_to_mesh):
    hash_relevant_params = {
        "mesh_views_list_train": mesh_views_list_train,
        "path_to_mesh": path_to_mesh,
        "dataset_path": dataset_path,
        "split": split,
        "preprocessing": deepcopy(config["preprocessing"])
    }
    if hash_relevant_params["preprocessing"]["uv_backend"].lower() == "blender":
        hash_relevant_params["preprocessing"]["uv_backend_options"].pop("xatlas", None)
    if hash_relevant_params["preprocessing"]["uv_backend"].lower() == "xatlas":
        hash_relevant_params["preprocessing"]["uv_backend_options"].pop("blender", None)
    if hash_relevant_params["preprocessing"]["uv_backend"].lower() == "gt":
        hash_relevant_params["preprocessing"]["uv_backend_options"].pop("blender", None)
        hash_relevant_params["preprocessing"]["uv_backend_options"].pop("xatlas", None)

    validation_hash = create_validation_hash_from_params(hash_relevant_params)
    return validation_hash


def preprocess_dataset(split, dataset_path, path_to_mesh, out_dir, mesh_views_list, config, force_preprocessing):
    """
    Preprocess the entire dataset for a given split.

    Args:
        split (str): The dataset split to process [train, val, test].
        dataset_path (str): Path to the dataset directory.
        path_to_mesh (str): Path to the mesh file.
        out_dir (str): Directory to save the preprocessed data.
        mesh_views_list (list): List of mesh view file names to process.
    """
    split_out_dir = os.path.join(out_dir, split)
    os.makedirs(split_out_dir, exist_ok=True)

    # Validation hash calculation
    validation_hash_path = os.path.join(out_dir, split, "validation_hash.txt")
    validation_hash = calculate_validation_hash(
        config=config,
        split=split,
        mesh_views_list_train=mesh_views_list,
        dataset_path=dataset_path,
        path_to_mesh=path_to_mesh
    )

    # Now check if we need to do the preprocessing
    do_preprocessing = check_if_preprocessing_required(
        validation_hash_path=validation_hash_path, validation_hash=validation_hash,
        force_preprocessing=force_preprocessing
    )

    if do_preprocessing:
        mesh_view_pre_proc = MeshViewPreProcessor(path_to_mesh, split_out_dir, config=config, split=split)
        preprocess_views(mesh_view_pre_proc, mesh_views_list, dataset_path)
        print("Saving validation hash.")
        with open(validation_hash_path, 'w') as file:
            file.write(validation_hash)
        print("Preprocessing finished.")


def main():
    args = parse_args()
    config = load_config(args.config_path, "config/human/config_human_defaults.yaml")
    dataset_path = config["data"]["raw_data_path"]
    out_dir = config["data"]["preproc_data_path"]
    mesh_path = config["data"]["mesh_path"]
    data_split = load_yaml(config["data"]["data_split"])
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
        mesh_views_list=mesh_views_list_train,
        config=config,
        force_preprocessing=True
    )


if __name__ == "__main__":
    main()
