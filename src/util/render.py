import os
import pickle
import sys
from pathlib import Path

import imageio as imageio
import numpy as np
import torch
from tqdm import tqdm

from util.cameras import DistortionTypes, undistort_pixels_meshroom_radial_k3
from util.mesh import get_ray_mesh_intersector, ray_mesh_intersect_np
from util.utils import load_mesh, load_cameras, load_obj_mask, LRUCache


class ImageRenderer:

    # TODO: CHANGE THIS DEFAULT DATASET_PATH ITS JUST FOR DEBUGGING
    def __init__(self, path_to_mesh, uv_path, dataset_path="data/raw/human_dataset_v2_tiny/",
                 cache_capacity=150,
                 verbose=False):
        self.mesh = load_mesh(path_to_mesh)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)
        self.dataset_path = dataset_path

        self.verbose = verbose

        self.gt_image_cache = LRUCache(capacity=cache_capacity)
        self.uv_cache = LRUCache(capacity=cache_capacity)
        self.obj_mask_cache = LRUCache(capacity=cache_capacity)
        self.hit_mask_cache = LRUCache(capacity=cache_capacity)

        self.angles_cache = LRUCache(capacity=cache_capacity)

        # TODO: Make this prettier
        if os.path.splitext(uv_path)[-1] == ".npy":
            self.uv = np.load(uv_path)
        else:
            with open(uv_path, 'rb') as f:
                self.uv = pickle.load(f)

    @staticmethod
    def calculate_rays(mesh_view_path: str, obj_mask: torch.Tensor, distortion_type=None, distortion_coeffs=None):
        assert torch.is_tensor(obj_mask), "obj_mask should be a tensor."
        camCv2world, K = load_cameras(mesh_view_path, as_numpy=False)

        xys = torch.nonzero(obj_mask)

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

        return ray_origins, unit_ray_dirs

    @staticmethod
    def calculate_rays_np(mesh_view_path: str, obj_mask: np.ndarray, distortion_type=None, distortion_coeffs=None,
                          scale=1):
        assert isinstance(obj_mask, np.ndarray), "obj_mask should be a numpy array."
        camCv2world, K = load_cameras(mesh_view_path, as_numpy=True)

        # Scale
        if scale > 1:
            K[0, 0] *= scale  # fx
            K[1, 1] *= scale  # fy
            K[0, 2] *= scale  # cx
            K[1, 2] *= scale  # cy

        xys = np.transpose(np.nonzero(obj_mask))

        # IMPORTANT!!!! WE HAVE TO FLIP THE COORDS!!!!
        selected_coord2d = xys[..., [1, 0]]
        L = len(xys)

        # If the views are distorted, remove the distortion from the 2D pixel coordinates
        if distortion_type is not None:
            assert distortion_coeffs is not None
            if distortion_type == DistortionTypes.MESHROOM_RADIAL_K3:
                selected_coord2d = undistort_pixels_meshroom_radial_k3(selected_coord2d.numpy(), K.numpy(),
                                                                       distortion_coeffs)
                selected_coord2d = selected_coord2d.astype(np.float32)
            else:
                raise ValueError(f"Unknown distortion type: {distortion_type}")

        # Get the 3D world coordinates of the ray origins as well as the 3D unit direction vector

        # Origin of the rays of the current view (it is already in 3D world coordinates)
        column_3_np = camCv2world[:, 3].reshape(1, -1)  # shape: (1, 4)
        ray_origins_np = np.tile(column_3_np, (L, 1))[:, :3]  # L x 3

        # Transform 2d coordinates into homogeneous coordinates.
        ones_column = np.ones((L, 1))
        selected_coord2d = np.concatenate((selected_coord2d, ones_column), axis=1)

        # Calculate the ray direction: R (K^-1_{3x3} [u v 1]^T)
        ray_dirs = (camCv2world[:3, :3] @ np.linalg.inv(K[:3, :3]) @ selected_coord2d.T.astype(np.float32)).T
        unit_ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)
        assert unit_ray_dirs.dtype == np.float32

        return ray_origins_np, unit_ray_dirs

    def render_views(self, model, mesh_views_list, save_validation_images, scale=1):
        # TODO: This function name does not really reflect what it does since it does more than that
        images = []
        gts = []
        masks = []
        scaled_ray_hit_counts = []

        # Define iterator
        iterator = tqdm(mesh_views_list) if self.verbose else mesh_views_list

        for mesh_view in iterator:
            mesh_view_path = os.path.join(self.dataset_path, mesh_view)

            # Load depth map for building a mask
            obj_mask = self.obj_mask_cache.get(mesh_view)
            if obj_mask is None:
                obj_mask = load_obj_mask(mesh_view_path, as_numpy=True)
                self.obj_mask_cache.put(mesh_view, obj_mask)

            if scale:
                obj_mask_scaled = np.repeat(np.repeat(obj_mask, scale, axis=0), scale, axis=1)
            else:
                obj_mask_scaled = obj_mask

            # Load gt image
            gt_img = self.gt_image_cache.get(mesh_view)
            if gt_img is None:
                gt_img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))
                gt_img[~obj_mask] = [0, 0, 0]
                self.gt_image_cache.put(mesh_view, gt_img)

            # Init the rendered image
            img = np.zeros((gt_img.shape[0] * scale, gt_img.shape[1] * scale, gt_img.shape[2]), dtype=np.int16)
            img[~obj_mask_scaled] = [0, 0, 0]
            scaled_ray_hit_count = np.zeros((gt_img.shape[0] * scale, gt_img.shape[1] * scale), dtype=np.int16)

            # Calculate the rays if we don't have uv_coords in the cache
            coords_2d_normalized = self.uv_cache.get(mesh_view)
            hit_mask = self.hit_mask_cache.get(mesh_view)
            angles_normalized_0_1 = self.angles_cache.get(mesh_view)
            if coords_2d_normalized is None or hit_mask is None or angles_normalized_0_1 is None:
                ray_origins, unit_ray_dirs = self.calculate_rays_np(mesh_view_path, obj_mask_scaled, scale=scale)

                vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_mask = ray_mesh_intersect_np(
                    self.ray_mesh_intersector,
                    self.mesh,
                    ray_origins,
                    unit_ray_dirs,
                    return_depth=False,
                    return_hit_mask=True
                )

                is_blender_based_vmapping = True if isinstance(self.uv, dict) else False  # TODO: THIS IS TEMPORARY!!!!!!!!!

                if is_blender_based_vmapping:  # TODO: THIS IS ALL TEMPORARY!!!
                    # NOTE: COPIED FROM map_uv_to_blender!!!!!!! BUT DOES NOT WORK BECAUSE HERE WE HAVE ONLY NPARRAY!!
                    uvs = []
                    for fid, vids, in zip(face_idxs, vertex_idxs_of_hit_faces):
                        mapping = self.uv.get(int(fid))
                        uv = np.array([mapping.get(int(v)) for v in vids])
                        assert None not in uv, "something went wrong."
                        uvs.append(uv)

                    uv_vertices_of_hit_faces = np.stack(uvs)
                    # Note: we can simply use the same barycentric coords since its all linear
                    # FIXME: torch->np->torch is shit
                    coords_2d_normalized = np.sum(barycentric_coords[:, :, np.newaxis] * uv_vertices_of_hit_faces,
                                                  axis=1)
                else:
                    uv_vertices_of_hit_faces = self.uv[vertex_idxs_of_hit_faces]
                    coords_2d_normalized = np.sum(
                        barycentric_coords[:, :, np.newaxis] * uv_vertices_of_hit_faces,
                        axis=1
                    )

                """NORMALS"""  # TODO: CACHE FIXME: COPIED FROM PREPROCESSING!!! MAKE FUNCTIONS!!!
                vertices_of_hit_faces = np.array(self.mesh.vertices[vertex_idxs_of_hit_faces])
                coords_3d = np.sum(barycentric_coords[:, :, np.newaxis] * vertices_of_hit_faces, axis=1)

                camera_center = ray_origins[0]
                intersect_to_cam = camera_center - coords_3d
                lengths = np.linalg.norm(intersect_to_cam, axis=1)
                lengths[lengths == 0] = 1
                lengths = lengths[:, np.newaxis]
                intersect_to_cam = intersect_to_cam / lengths
                normals = self.mesh.face_normals[face_idxs]

                dp = np.einsum('ij,ij->i', normals, intersect_to_cam)
                angles = np.arccos(dp)
                angles_degrees = np.degrees(angles)  # Unused
                angles_normalized_0_1 = angles / (np.pi / 2)

                self.uv_cache.put(mesh_view, coords_2d_normalized)
                self.hit_mask_cache.put(mesh_view, hit_mask)
                self.angles_cache.put(mesh_view, angles_normalized_0_1)

            # TODO: Use proper methods
            with torch.no_grad():
                # CLAMP IMPORTANT!!
                if model.model.n_input_dims != 2:  # TODO: THIS IS TEMPORARY
                    coords_2d_normalized = np.concatenate(
                        (coords_2d_normalized, angles_normalized_0_1.reshape(-1, 1)),
                        axis=1
                    )
                rgbs_normalized = model(
                    torch.from_numpy(coords_2d_normalized).to(model.params.device).detach()
                ).clamp(0.0, 1.0)

                rgbs_scaled = (rgbs_normalized * 255).type(torch.int16).cpu().numpy().clip(0, 255)

                # Start with assumption that all rays hit
                ray_hit_count = np.ones(hit_mask.shape[0])

                # Pad if not all rays hit
                if len(coords_2d_normalized) != len(hit_mask):
                    # Reconstruct original list and fill non_hit_rays with color 0,0,0
                    padded_rgbs_scaled = np.full((len(hit_mask), 3), 0, dtype="int16")
                    padded_rgbs_scaled[hit_mask] = rgbs_scaled
                    rgbs_scaled = padded_rgbs_scaled

                    # Mark rays as hit accordingly
                    ray_hit_count[~hit_mask] = 0

                # Fill the image
                img[obj_mask_scaled] = rgbs_scaled
                img = img.astype(np.uint8)
                scaled_ray_hit_count[obj_mask_scaled] = ray_hit_count

            if save_validation_images:
                os.makedirs("reports/human", exist_ok=True)
                imageio.imwrite(f"reports/human/{mesh_view}.png", img)

            # Append outputs
            images.append(img)
            masks.append(obj_mask)
            gts.append(gt_img)
            scaled_ray_hit_counts.append(scaled_ray_hit_count)

        return images, gts, masks, scaled_ray_hit_counts


def downscale_image(image, hit_count, kernel_size=2):
    conv_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    hit_count_tensor = torch.tensor(hit_count, dtype=torch.float32).unsqueeze(0)

    mask = torch.nn.functional.conv2d(hit_count_tensor, conv_kernel, stride=kernel_size)[0]
    mask[mask == 0] = 1  # For division

    # Perform convolution
    r = torch.nn.functional.conv2d(image_tensor[:, 0:1, :, :], conv_kernel, stride=kernel_size)[0][0] / mask
    g = torch.nn.functional.conv2d(image_tensor[:, 1:2, :, :], conv_kernel, stride=kernel_size)[0][0] / mask
    b = torch.nn.functional.conv2d(image_tensor[:, 2:3, :, :], conv_kernel, stride=kernel_size)[0][0] / mask

    return torch.stack((r, g, b)).permute(1, 2, 0).numpy().astype(np.uint8)
