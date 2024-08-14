import torch
import numpy as np

import wandb
import lpips
import json
import yaml
import os
import sys
from pathlib import Path

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
from util.utils import compute_psnr, load_config, export_uv, compute_ssim, time_method
from model.model import InstantUV

DEFAULTS_HUMAN = "config/human/config_human_defaults.yaml"
DEFAULTS_CAT = "config/cat/config_cat_defaults.yaml"


def evaluate_qualitative(model, config):
    model.eval()
    print(config)
    #breakpoint()
    # Load data split
    data_split_path = config.get("data", {}).get("data_split")
    if data_split_path:
        with open(data_split_path, "rb") as f:
            data_split = yaml.safe_load(f)

    # initialize renderer
    xatlas_path = str(Path(config["data"]["preproc_data_path"]) / "xatlas.obj")
    uv_backend = config["preprocessing"]["uv_backend"].lower()
    # TODO: Refactor all this shit its HORRIBLE
    if uv_backend == "blender" or uv_backend == "gt":
        uv_path = str(Path(config["data"]["preproc_data_path"]) / "train" / "blender_uv.pkl")
        path_to_mesh = config["data"]["mesh_path"]
    else:
        uv_path = str(Path(config["data"]["preproc_data_path"]) / "uv.pkl")
        path_to_mesh = xatlas_path

    image_renderer = ImageRenderer(
        path_to_mesh=path_to_mesh,
        dataset_path=config["data"]["raw_data_path"],
        uv_path=uv_path,
        verbose=True
    )

    # compute psnr, dssim, lpips
    images_np, gts, masks, hit_counts = image_renderer.render_views(
        model,
        mesh_views_list=data_split["mesh_views_list_val"],
        save_validation_images=False,
        scale=1,
    )

    val_psnrs = np.zeros(len(images_np), dtype=np.float32)
    val_ssims = np.zeros(len(images_np), dtype=np.float32)
    val_lpipss = np.zeros(len(images_np), dtype=np.float32)

    lpips_eval = lpips.LPIPS(net='alex', version='0.1')
    print("calculating psnr, dssim and lpips...")
    for i, (image_pred, image_gt, mask) in enumerate(list(zip(images_np, gts, masks))):
        val_psnrs[i] = compute_psnr(
            image_gt[mask].astype("int16") / 255.0,
            image_pred[mask].astype("int16") / 255.0
        )
        val_ssims[i] = compute_ssim(
            image_gt.astype("int16") / 255.0,
            image_pred.astype("int16") / 255.0
        )
        val_lpipss[i] = lpips_eval(
            torch.from_numpy((image_gt / 127.5 - 1.0).astype('float32').transpose(2, 0, 1)).unsqueeze(0),
            torch.from_numpy((image_pred / 127.5 - 1.0).astype('float32').transpose(2, 0, 1)).unsqueeze(0)
        )

    val_psnr = np.mean(val_psnrs)
    val_dssim = (1 - np.mean(val_ssims)) / 2
    val_lpips = np.mean(val_lpipss)
    return {"psnr": val_psnr, "dssim": val_dssim * 100, "lpips": val_lpips * 100}

def speed_test(model, config):
    dummy_inp = {"points_uv": torch.rand(2**15, 2)}
    eval_time = time_method(model, dummy_inp)
    print(eval_time)



if __name__ == "__main__":
    weights_path = "models/model_GT_Human_psnr.pt"  # FIXME sorry for hardcoding
    weights_path = "models/model_XATLAS_CAT_psnr.pt" # FIXME sorry for hardcoding
    config_path = "config/cat/config_cat_xatlas.yaml" # FIXME sorry for hardcode
    config = load_config(config_path, DEFAULTS_CAT)

    model = InstantUV(config)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    # print(model)



    stats = evaluate_qualitative(model, config)
    print("\033[1m" + str(stats) + "\033[0m")
    speed_test(model, config)
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)
