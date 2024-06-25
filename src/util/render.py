import os
import pickle
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm

from util.cameras import DistortionTypes, undistort_pixels_meshroom_radial_k3
from util.mesh import get_ray_mesh_intersector, ray_mesh_intersect
from util.utils import load_mesh, load_cameras, load_obj_mask_as_tensor


class ImageRenderer:

    # TODO: CHANGE THIS DEFAULT DATASET_PATH ITS JUST FOR DEBUGGING
    def __init__(self, path_to_mesh, dataset_path="data/raw/human_dataset_v2_tiny/",
                 uv_path="data/preprocessed/human_dataset_v2_tiny/train/uv.pkl"):
        # self.out_dir = out_directory
        self.mesh = load_mesh(path_to_mesh)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)
        # self.config = config
        self.dataset_path = dataset_path

        # TODO: Note we can also just use np.load and np.dump since uv is a np array :D instead of pickle
        with open(uv_path, 'rb') as f:
            self.uv = pickle.load(f)

    def render_views(self, model, mesh_views_list):
        # TODO: This function name does not really reflect what it does since it does more than that
        images = []
        gts = []
        masks = []
        for mesh_view in tqdm(mesh_views_list):
            try:
                # TODO:
                distortion_coeffs = None  # TODO: Fill
                distortion_type = None  # TODO: Fill

                mesh_view_path = os.path.join(self.dataset_path, mesh_view)
                camCv2world, K = load_cameras(mesh_view_path)

                # Load depth map for building a mask
                obj_mask = load_obj_mask_as_tensor(mesh_view_path)
                masks.append(obj_mask)

                # Load image
                img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))
                img[~obj_mask] = [0, 0, 0]
                gts.append(img)

                img = np.zeros(img.shape, dtype=np.int16)

                # Set to -1 for now
                img[~obj_mask] = [0, 0, 0]
                xys = torch.nonzero(obj_mask)

                """ NOTE: COPIED FROM create_ray_origins_and_directions"""
                # TODO: Refactor methods to remove duplicates!!!!
                # If the views are distorted, remove the distortion from the 2D pixel coordinates

                # IMPORTANT!!!! WE HAVE TO FLIP THE COORDS!!!!
                selected_coord2d = xys[..., [1, 0]]
                L = len(xys)

                if distortion_type is not None:
                    assert distortion_coeffs is not None
                    if distortion_type == DistortionTypes.MESHROOM_RADIAL_K3:
                        selected_coord2d = undistort_pixels_meshroom_radial_k3(selected_coord2d.numpy(), K.numpy(),
                                                                               distortion_coeffs)
                        selected_coord2d = torch.from_numpy(selected_coord2d).to(torch.float32)
                    else:
                        raise ValueError(f"Unknown distortion type: {distortion_type}")

                # Get the 3D world coordinates of the ray origins as well as the 3D unit direction vector

                # Origin of the rays of the current view (it is already in 3D world coordinates)
                ray_origins = camCv2world[:, 3].unsqueeze(0).expand(L, -1)  # L x 3

                # Transform 2d coordinates into homogeneous coordinates.
                selected_coord2d = torch.cat((selected_coord2d, torch.ones((L, 1))), dim=-1)  # L x 3
                # Calculate the ray direction: R (K^-1_{3x3} [u v 1]^T)
                ray_dirs = camCv2world[:3, :3].matmul(K[:3, :3].inverse().matmul(selected_coord2d.T)).T  # L x 3
                unit_ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
                assert unit_ray_dirs.dtype == torch.float32

                """ ----------------------------------------------- Refactor above! ^ ^ ^ ^ ^ """

                # Then, we can compute the ray-mesh-intersections
                # TODO: Fix private method issues

                """ NOTE: THIS WAS COPIED FROM MeshViewPreProcessor._ray_mesh_intersect"""
                vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_mask = ray_mesh_intersect(
                    self.ray_mesh_intersector,
                    self.mesh,
                    ray_origins,
                    unit_ray_dirs,
                    return_depth=False,
                    return_hit_mask=True
                )

                uv_vertices_of_hit_faces = np.array(self.uv[vertex_idxs_of_hit_faces])
                coords_2d_normalized = np.sum(barycentric_coords.numpy()[:, :, np.newaxis] * uv_vertices_of_hit_faces,
                                              axis=1)

                # TODO: Use proper methods
                with torch.no_grad():
                    # CLAMP IMPORTANT!!
                    rgbs_normalized = model(
                        torch.from_numpy(coords_2d_normalized).to(model.params.device).detach()).clamp(0.0, 1.0)
                    rgbs_scaled = (rgbs_normalized * 255).type(torch.int16).cpu().numpy()

                    if len(vertex_idxs_of_hit_faces) != len(ray_origins):
                        # Reconstruct original list and fill non_hit_rays with color 0,0,0

                        # Create the full array filled with 0
                        padded_rgbs_scaled = np.full((len(hit_mask), 3), 0, dtype="uint8")

                        # Insert original values into filled_array
                        padded_rgbs_scaled[hit_mask] = rgbs_scaled

                        # Replace
                        rgbs_scaled = padded_rgbs_scaled

                    img[obj_mask] = rgbs_scaled

                images.append(img)
            except AssertionError as e:
                print(f"Assertion error on image: {mesh_view}. Ignoring the image")
                continue

        return images, gts, masks


# TODO: THIS IS ONLY FOR DEBUGGING
if __name__ == "__main__":
    FILEPATH = Path(__file__)

    # Change into instant-uv folder
    os.chdir(str(Path(__file__).parent.parent.parent))
    sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

    # NOTE: This must be the XATLAS.obj!!! Aka new_mesh
    ir = ImageRenderer(
        path_to_mesh="data/preprocessed/human_dataset_v2_tiny/train/xatlas.obj",
        dataset_path="data/raw/human_dataset_v2_tiny/",
    )
    # TODO: From config
    ir.render_views(
        ...,
        ["human_val000",
         "human_val001",
         "human_val002",
         "human_val003", ]
    )
