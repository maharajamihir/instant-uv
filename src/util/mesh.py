######################################################################################################
# Code mainly borrowed from https://github.com/tum-vision/intrinsic-neural-fields/blob/main/mesh.py  #
######################################################################################################

import numpy as np
import torch
import igl
import trimesh
from trimesh import triangles
import os
import gc
import scipy as sp
import warnings

from util.enums import CacheFilePaths

warnings.filterwarnings('ignore')

from util.cameras import undistort_pixels_meshroom_radial_k3, DistortionTypes
from util.utils import tensor_mem_size_in_bytes, load_mesh, map_to_UV, get_mapping, get_mapping_blender, \
    map_to_UV_blender, normalize_values


def get_ray_mesh_intersector(mesh):
    try:
        import pyembree
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except ImportError:
        print("Warning: Could not find pyembree, the ray-mesh intersection will be significantly slower.")
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


def create_ray_origins_and_directions(camCv2world, K, mask_1d, *, H, W, distortion_coeffs=None, distortion_type=None):
    # Let L be the number of pixels where the object is seen in the view
    L = mask_1d.sum()

    try:
        # This does not work for older PyTorch versions.
        coord2d_x, coord2d_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    except TypeError:
        # Workaround for older versions. Simulate indexing='xy'
        coord2d_x, coord2d_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
        coord2d_x, coord2d_y = coord2d_x.T, coord2d_y.T

    coord2d = torch.cat([coord2d_x[..., None], coord2d_y[..., None]], dim=-1).reshape(-1, 2)  # N*M x 2
    selected_coord2d = coord2d[mask_1d]  # L x 2

    # If the views are distorted, remove the distortion from the 2D pixel coordinates
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


def ray_mesh_intersect(ray_mesh_intersector, mesh, ray_origins, ray_directions, return_depth=False, camCv2world=None,
                       return_hit_mask=False):
    # Compute the intersection points between the mesh and the rays

    # Note: It might happen that M <= N where M is the number of returned hits
    intersect_locs, hit_ray_idxs, face_idxs = \
        ray_mesh_intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)

    if return_hit_mask:
        hit_mask = ray_mesh_intersector.intersects_any(ray_origins, ray_directions)

    # Next, we need to determine the barycentric coordinates of the hit points.

    vertex_idxs_of_hit_faces = torch.from_numpy(mesh.faces[face_idxs]).reshape(-1)  # M*3
    hit_triangles = mesh.vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3)  # M x 3 x 3

    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1, 3)  # M x 3

    barycentric_coords = triangles.points_to_barycentric(hit_triangles, intersect_locs,
                                                         method='cramer')  # M x 3

    if return_depth:
        assert camCv2world is not None
        camCv2world = camCv2world.cpu().numpy()
        camCv2world = np.concatenate([camCv2world, np.array([[0., 0, 0, 1]], dtype=camCv2world.dtype)], 0)

        vertices_world = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[:, :1])], -1)  # V, 4

        camWorld2Cv = np.linalg.inv(camCv2world)
        vertices_cam = np.dot(vertices_world, camWorld2Cv.T)
        z_vals = vertices_cam[:, 2][vertex_idxs_of_hit_faces]
        assert np.all(z_vals > 0)

        assert z_vals.shape == barycentric_coords.shape
        assert np.allclose(np.sum(barycentric_coords, -1), 1)

        hit_depth = np.sum(z_vals * barycentric_coords, -1)
        hit_depth = torch.from_numpy(hit_depth)

    barycentric_coords = torch.from_numpy(barycentric_coords).to(dtype=torch.float32)  # M x 3

    hit_ray_idxs = torch.from_numpy(hit_ray_idxs)
    face_idxs = torch.from_numpy(face_idxs).to(dtype=torch.int64)

    if return_depth and return_hit_mask:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth, hit_mask
    if return_depth:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth
    if return_hit_mask:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_mask
    return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs


def ray_mesh_intersect_np(ray_mesh_intersector, mesh, ray_origins, ray_directions, return_depth=False, camCv2world=None,
                          return_hit_mask=False):
    # Compute the intersection points between the mesh and the rays

    # NOTE: intersects_location converts to float64 and applies unitize to ray_directions
    # intersect_any does not! Thus we need to apply it else we get different results!
    ray_origins, ray_directions = ray_origins.astype(np.float64), ray_directions.astype(np.float64)

    # Note: It might happen that M <= N where M is the number of returned hits
    intersect_locs, hit_ray_idxs, face_idxs = \
        ray_mesh_intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)

    if return_hit_mask:
        hit_mask = ray_mesh_intersector.intersects_any(ray_origins, trimesh.unitize(ray_directions))

    # Next, we need to determine the barycentric coordinates of the hit points.

    vertex_idxs_of_hit_faces = mesh.faces[face_idxs].reshape(-1)  # M*3
    hit_triangles = mesh.vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3)  # M x 3 x 3

    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1, 3)  # M x 3

    barycentric_coords = triangles.points_to_barycentric(hit_triangles, intersect_locs, method='cramer')  # M x 3

    if return_depth:
        assert camCv2world is not None
        camCv2world = camCv2world.cpu().numpy()
        camCv2world = np.concatenate([camCv2world, np.array([[0., 0, 0, 1]], dtype=camCv2world.dtype)], 0)

        vertices_world = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[:, :1])], -1)  # V, 4

        camWorld2Cv = np.linalg.inv(camCv2world)
        vertices_cam = np.dot(vertices_world, camWorld2Cv.T)
        z_vals = vertices_cam[:, 2][vertex_idxs_of_hit_faces]
        assert np.all(z_vals > 0)

        assert z_vals.shape == barycentric_coords.shape
        assert np.allclose(np.sum(barycentric_coords, -1), 1)

        hit_depth = np.sum(z_vals * barycentric_coords, -1)
        hit_depth = hit_depth

    barycentric_coords = barycentric_coords.astype(np.float32)  # M x 3

    hit_ray_idxs = hit_ray_idxs
    face_idxs = face_idxs.astype(np.int64)

    if return_depth and return_hit_mask:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth, hit_mask
    if return_depth:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth
    if return_hit_mask:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_mask
    return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs


def ray_mesh_intersect_batched(ray_mesh_intersector, mesh, ray_origins, ray_directions):
    batch_size = 1 << 18
    num_rays = ray_origins.shape[0]
    idxs = np.arange(0, num_rays)
    batch_idxs = np.split(idxs, np.arange(batch_size, num_rays, batch_size), axis=0)

    total_vertex_idxs_of_hit_faces = []
    total_barycentric_coords = []
    total_hit_ray_idxs = []
    total_face_idxs = []

    total_hits = 0
    hit_ray_idx_offset = 0
    for cur_idxs in batch_idxs:
        cur_ray_origins = ray_origins[cur_idxs]
        cur_ray_dirs = ray_directions[cur_idxs]

        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect(ray_mesh_intersector,
                                                                                                   mesh,
                                                                                                   cur_ray_origins,
                                                                                                   cur_ray_dirs)

        # Correct the hit_ray_idxs
        hit_ray_idxs += hit_ray_idx_offset

        num_hits = vertex_idxs_of_hit_faces.shape[0]

        # Append results to output
        if num_hits > 0:
            total_vertex_idxs_of_hit_faces.append(vertex_idxs_of_hit_faces)
            total_barycentric_coords.append(barycentric_coords)
            total_hit_ray_idxs.append(hit_ray_idxs)
            total_face_idxs.append(face_idxs)

        hit_ray_idx_offset += cur_idxs.shape[0]
        total_hits += num_hits

    # Concatenate results
    out_vertex_idxs_of_hit_faces = torch.zeros((total_hits, 3), dtype=torch.int64)
    out_barycentric_coords = torch.zeros((total_hits, 3), dtype=torch.float32)
    out_hit_ray_idxs = torch.zeros(total_hits, dtype=torch.int64)
    out_face_idxs = torch.zeros(total_hits, dtype=torch.int64)

    offset = 0
    for i in range(len(total_vertex_idxs_of_hit_faces)):
        hits_of_batch = total_vertex_idxs_of_hit_faces[i].shape[0]
        low = offset
        high = low + hits_of_batch

        out_vertex_idxs_of_hit_faces[low:high] = total_vertex_idxs_of_hit_faces[i]
        out_barycentric_coords[low:high] = total_barycentric_coords[i]
        out_hit_ray_idxs[low:high] = total_hit_ray_idxs[i]
        out_face_idxs[low:high] = total_face_idxs[i]

        offset = high

    return out_vertex_idxs_of_hit_faces, out_barycentric_coords, out_hit_ray_idxs, out_face_idxs


def ray_tracing_xyz(ray_mesh_intersector,
                    mesh,
                    vertices,
                    camCv2world,
                    K,
                    obj_mask_1d=None,
                    *,
                    H,
                    W,
                    batched=True,
                    distortion_coeffs=None,
                    distortion_type=None):
    if obj_mask_1d is None:
        mask = torch.tensor([True]).expand(H * W)
    else:
        mask = obj_mask_1d
    ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world,
                                                                   K,
                                                                   mask,
                                                                   H=H,
                                                                   W=W,
                                                                   distortion_coeffs=distortion_coeffs,
                                                                   distortion_type=distortion_type)
    if batched:
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect_batched(
            ray_mesh_intersector,
            mesh,
            ray_origins,
            unit_ray_dirs)
    else:
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect(ray_mesh_intersector,
                                                                                                   mesh,
                                                                                                   ray_origins,
                                                                                                   unit_ray_dirs)

    # Calculate the xyz hit points using the barycentric coordinates    
    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1)  # M*3
    face_vertices = torch.tensor(vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3), dtype=torch.float32)  # M x 3 x 3
    hit_points_xyz = torch.einsum('bij,bi->bj', face_vertices, barycentric_coords)  # M x 3

    return barycentric_coords, hit_ray_idxs, unit_ray_dirs[hit_ray_idxs], face_idxs, hit_points_xyz


class MeshViewPreProcessor:
    def __init__(self, path_to_mesh, out_directory, config, split):
        self.out_dir = out_directory
        self.mesh = load_mesh(path_to_mesh)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)
        self.config = config
        self.split = split
        self.pp_config = self.config["preprocessing"]

        self.cache_vertex_idxs_of_hit_faces = []
        self.cache_barycentric_coords = []
        self.cache_expected_rgbs = []
        self.cache_unit_ray_dirs = []
        self.cache_face_idxs = []
        self.cache_uv_coords = []
        self.cache_coords3d = []
        self.cache_angles = []
        self.cache_angles2 = []

    def _ray_mesh_intersect(self, ray_origins, ray_directions, return_depth=False, camCv2world=None):
        # TODO instead of doing simple ray mesh intersection, we are interested in the x,y,z coordinates
        # TODO change to ray_tracing_xyz and retrieve x,y,z coordinates
        return ray_mesh_intersect(self.ray_mesh_intersector,
                                  self.mesh,
                                  ray_origins,
                                  ray_directions,
                                  return_depth=return_depth,
                                  camCv2world=camCv2world)

    def calculate_angles(self, barycentric_coords, vertex_idxs_of_hit_faces, face_idxs, ray_origins):
        """ NORMALS """
        vertices_of_hit_faces = np.array(self.mesh.vertices[vertex_idxs_of_hit_faces])
        coords_3d = np.sum(barycentric_coords.numpy()[:, :, np.newaxis] * vertices_of_hit_faces, axis=1)

        camera_center = ray_origins[0].numpy()
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

        # 2_angle
        # Project vectors onto the x-y plane
        u_xy = np.array([normals[:, 0], normals[:, 1], np.zeros(len(normals))]).transpose()
        v_xy = np.array([intersect_to_cam[:, 0], intersect_to_cam[:, 1], np.zeros(len(intersect_to_cam))]).transpose()

        # Calculate the dot products and cross products
        dot_product_xy = np.einsum('ij,ij->i', u_xy, v_xy)  # Dot product of projections
        cross_product_xy = np.cross(u_xy, v_xy)
        azimuth = np.arctan2(np.linalg.norm(cross_product_xy, axis=1), dot_product_xy)
        elevation = np.arcsin(
            np.einsum('ij,j->i', np.cross(normals, intersect_to_cam), np.array([0, 0, 1]))
        )
        azimuth_n = normalize_values(azimuth, -np.pi, np.pi)
        elevation_n = normalize_values(elevation, -np.pi / 2, np.pi / 2)

        return angles_degrees, angles_normalized_0_1, azimuth_n, elevation_n

    def calculate_coords3d(self, barycentric_coords, vertex_idxs_of_hit_faces):
        vertices_of_hit_faces = np.array(self.mesh.vertices[vertex_idxs_of_hit_faces])
        return np.sum(barycentric_coords.numpy()[:, :, np.newaxis] * vertices_of_hit_faces, axis=1)

    def perform_ray_mesh_intersection(self, camCv2world, K, mask, distortion_coeffs, distortion_type):
        H, W = mask.shape
        mask = mask.reshape(-1)  # H*W

        # Get the ray origins and unit directions
        ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world,
                                                                       K,
                                                                       mask,
                                                                       H=H,
                                                                       W=W,
                                                                       distortion_coeffs=distortion_coeffs,
                                                                       distortion_type=distortion_type)

        # Then, we can compute the ray-mesh-intersections
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = (
            self._ray_mesh_intersect(ray_origins, unit_ray_dirs)
        )
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, unit_ray_dirs, ray_origins

    def get_uv_coords(self, barycentric_coords, face_idxs, vertex_idxs_of_hit_faces):
        uv_backend = self.pp_config["uv_backend"].lower()

        if uv_backend == "blender":
            # get uv coordinates
            mapping = get_mapping_blender(self.mesh, self.split, self.config)
            uv_coords = map_to_UV_blender(barycentric_coords, face_idxs, vertex_idxs_of_hit_faces, mapping)

        elif uv_backend == "xatlas":
            # get uv coordinates
            mapping = get_mapping(self.mesh, self.split, self.config)

            # FIXME: mapping, we must get the right object directly form get_mapping
            uv_coords = map_to_UV(barycentric_coords, face_idxs, mapping)

        else:
            raise Exception("Unsupported uv_backend: {}".format(uv_backend))
        return uv_coords

    def cache_single_view(self, camCv2world, K, mask, img, distortion_coeffs=None, distortion_type=None):
        # Perform ray mesh intersection
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, unit_ray_dirs, ray_origins = (
            self.perform_ray_mesh_intersection(
                camCv2world=camCv2world,
                K=K,
                mask=mask,
                distortion_coeffs=distortion_coeffs,
                distortion_type=distortion_type
            )
        )

        H, W = mask.shape
        mask = mask.reshape(-1)  # H*W
        img = img.reshape(H * W, -1)  # H*W x 3
        expected_rgbs = img[mask]  # L x 3
        """
        TODO use the retrieved values from ray mesh intersection to obtain our preferred dataset format

        We have:
            - Face IDs for each hit face
            - Barycentric coordinates for each hitpoint
            - Vertex IDs of each vertex of each hit face
            - RGB values for each point
        We want:
            - 3D coodinates 
            - 2D coordinates (which can be bijectively mapped to the 3D coordinates)
            - [DONE] RGB values for each point 

        The idea here is to 
            - Use the ray_tracing_xyz function to retrieve 3D coordinates
            - Use the map_to_UV function from `utils/utils.py` to obtain a 2D points

        """

        uv_coords = self.get_uv_coords(
            barycentric_coords=barycentric_coords,
            face_idxs=face_idxs,
            vertex_idxs_of_hit_faces=vertex_idxs_of_hit_faces
        )

        # Choose the correct GTs and viewing directions for the hits.
        num_hits = hit_ray_idxs.size()[0]
        expected_rgbs = expected_rgbs[hit_ray_idxs]
        unit_ray_dirs = unit_ray_dirs[hit_ray_idxs]

        if self.pp_config["export_angles"]:
            angles_degrees, angles_normalized_0_1, azimuth_n, elevation_n = self.calculate_angles(
                barycentric_coords=barycentric_coords,
                vertex_idxs_of_hit_faces=vertex_idxs_of_hit_faces,
                face_idxs=face_idxs,
                ray_origins=ray_origins
            )
            angles_normalized_0_1 = torch.from_numpy(angles_normalized_0_1).to(torch.float32)
            azimuth_n = torch.from_numpy(azimuth_n).to(torch.float32)
            elevation_n = torch.from_numpy(elevation_n).to(torch.float32)
            for idx in range(num_hits):
                self.cache_angles.append(angles_normalized_0_1[idx])
                self.cache_angles2.append(torch.from_numpy(np.array([azimuth_n[idx], elevation_n[idx]])))

        if self.pp_config["export_coords3d"]:
            coords_3d = self.calculate_coords3d(
                barycentric_coords=barycentric_coords,
                vertex_idxs_of_hit_faces=vertex_idxs_of_hit_faces
            )
            for idx in range(num_hits):
                self.cache_coords3d.append(torch.from_numpy(coords_3d[idx]))

        # Some clean up to free memory
        del ray_origins, hit_ray_idxs, mask, img
        gc.collect()  # Force garbage collection

        # Cast the indices down to int32 to save memory. Usually indices have to be int64, however, we assume that
        # the indices from 0 to 2^31-1 are sufficient. Therefore, we can safely cast down

        assert torch.all(face_idxs <= (2 << 31) - 1)
        face_idxs = face_idxs.to(torch.int32)
        assert torch.all(vertex_idxs_of_hit_faces <= (2 << 31) - 1)
        vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.to(torch.int32)
        barycentric_coords = barycentric_coords.to(torch.float32)
        expected_rgbs = expected_rgbs.to(torch.float32)
        unit_ray_dirs = unit_ray_dirs.to(torch.float32)
        uv_coords = uv_coords.to(torch.float32)

        # And finally, we store the results in the cache
        for idx in range(num_hits):
            self.cache_face_idxs.append(face_idxs[idx])
            self.cache_vertex_idxs_of_hit_faces.append(vertex_idxs_of_hit_faces[idx])
            self.cache_barycentric_coords.append(barycentric_coords[idx])
            self.cache_expected_rgbs.append(expected_rgbs[idx])
            self.cache_unit_ray_dirs.append(unit_ray_dirs[idx])
            self.cache_uv_coords.append(uv_coords[idx])

    def _stack_and_write_to_disk(self, array, filename: str):
        stacked = torch.stack(array)
        name = filename.split(".")[0]
        print(f"{name}: dim={stacked.size()}, "
              f"mem_size={tensor_mem_size_in_bytes(stacked)}B,"
              f"dtype={stacked.dtype}")
        np.save(os.path.join(self.out_dir, filename), stacked, allow_pickle=False)
        del array
        del stacked
        gc.collect()  # Force garbage collection

    def write_to_disk(self):
        print("Starting to write to disk...")

        # Write the cached eigenfuncs and cached expected RGBs to disk
        os.makedirs(self.out_dir, exist_ok=True)

        # write to disk and free memory
        self._stack_and_write_to_disk(self.cache_face_idxs, CacheFilePaths.FACE_IDXS.value)
        self._stack_and_write_to_disk(self.cache_vertex_idxs_of_hit_faces, CacheFilePaths.VIDS_OF_HIT_FACES.value)
        self._stack_and_write_to_disk(self.cache_barycentric_coords, CacheFilePaths.BARYCENTRIC_COORDS.value)
        self._stack_and_write_to_disk(self.cache_uv_coords, CacheFilePaths.UV_COORDS.value)
        self._stack_and_write_to_disk(self.cache_expected_rgbs, CacheFilePaths.EXPECTED_RGBS.value)
        self._stack_and_write_to_disk(self.cache_unit_ray_dirs, CacheFilePaths.UNIT_RAY_DIRS.value)

        if self.pp_config["export_angles"]:
            self._stack_and_write_to_disk(self.cache_angles, CacheFilePaths.ANGLES.value)
            self._stack_and_write_to_disk(self.cache_angles2, CacheFilePaths.ANGLES2.value)

        if self.pp_config["export_coords3d"]:
            self._stack_and_write_to_disk(self.cache_coords3d, CacheFilePaths.COORDS3D.value)
