######################################################################################################
# Code mainly borrowed from https://github.com/tum-vision/intrinsic-neural-fields/blob/main/mesh.py  #
######################################################################################################

import numpy as np
import torch
import igl
import trimesh
import os
import gc
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
from PIL.Image import fromarray

from util.cameras import undistort_pixels_meshroom_radial_k3, DistortionTypes
from util.utils import tensor_mem_size_in_bytes, load_mesh, map_to_UV, get_mapping

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
            selected_coord2d = undistort_pixels_meshroom_radial_k3(selected_coord2d.numpy(), K.numpy(), distortion_coeffs)
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


def ray_mesh_intersect(ray_mesh_intersector, mesh, ray_origins, ray_directions, return_depth=False, camCv2world=None):
    # Compute the intersection points between the mesh and the rays

    # Note: It might happen that M <= N where M is the number of returned hits
    intersect_locs, hit_ray_idxs, face_idxs = \
        ray_mesh_intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)

    # Next, we need to determine the barycentric coordinates of the hit points.

    vertex_idxs_of_hit_faces = torch.from_numpy(mesh.faces[face_idxs]).reshape(-1)  # M*3
    hit_triangles = mesh.vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3)  # M x 3 x 3

    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1, 3)  # M x 3

    barycentric_coords = trimesh.triangles.points_to_barycentric(hit_triangles, intersect_locs, method='cramer')  # M x 3

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

    if return_depth:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth
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
    def __init__(self, path_to_mesh, out_directory, config=None):
        self.out_dir = out_directory
        self.mesh = load_mesh(path_to_mesh)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)
        self.config = config

        self.cache_vertex_idxs_of_hit_faces = []
        self.cache_barycentric_coords = []
        self.cache_expected_rgbs = []
        self.cache_unit_ray_dirs = []
        self.cache_face_idxs = []
        self.cache_uv_coords = []
        

    def _ray_mesh_intersect(self, ray_origins, ray_directions, return_depth=False, camCv2world=None):

        # TODO instead of doing simple ray mesh intersection, we are interested in the x,y,z coordinates 
        # TODO change to ray_tracing_xyz and retrieve x,y,z coordinates
        return ray_mesh_intersect(self.ray_mesh_intersector, 
                                  self.mesh, 
                                  ray_origins, 
                                  ray_directions, 
                                  return_depth=return_depth, 
                                  camCv2world=camCv2world)

    def cache_single_view(self, camCv2world, K, mask, img, distortion_coeffs=None, distortion_type=None):
        H, W = mask.shape

        mask = mask.reshape(-1)  # H*W
        img = img.reshape(H * W, -1)  # H*W x 3

        # Let L be the number of pixels where the object is seen in the view

        # Get the expected RGB value of the intersection points with the mesh
        expected_rgbs = img[mask]  # L x 3

        # Get the ray origins and unit directions
        ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world, 
                                                                       K, 
                                                                       mask, 
                                                                       H=H, 
                                                                       W=W, 
                                                                       distortion_coeffs=distortion_coeffs, 
                                                                       distortion_type=distortion_type)

        # Then, we can compute the ray-mesh-intersections
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = self._ray_mesh_intersect(ray_origins, unit_ray_dirs)

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
        
        # get uv coordinates
        mapping = get_mapping(self.mesh, self.config)
        uv_coords = map_to_UV(barycentric_coords, vertex_idxs_of_hit_faces, mapping)

        #########################################################
        #print(uv_coords.shape)
        # breakpoint()
        uv_coords = np.clip(uv_coords, 0, 1)

        # Convert UV coordinates to pixel positions
        image_width = 512 #1024  # Adjust as needed
        image_height = 512 #1024  # Adjust as needed

        pixel_x = (uv_coords[:, 0] * (image_width - 1)).astype(np.int32)
        pixel_y = (uv_coords[:, 1] * (image_height - 1)).astype(np.int32)

        # Create an empty image
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Populate the image with the RGB values
        for i in range(len(expected_rgbs)):
            x = pixel_x[i]
            y = pixel_y[i]
            image[y, x] = (expected_rgbs[i] * 255).astype(np.uint8)  # Convert RGB to 0-255 range

        # Convert numpy array to PIL image
        image_pil = fromarray(image)

        # Save the image
        image_pil.save('uv_image.png')
        #####################################################



        # Choose the correct GTs and viewing directions for the hits.
        num_hits = hit_ray_idxs.size()[0]
        expected_rgbs = expected_rgbs[hit_ray_idxs]
        unit_ray_dirs = unit_ray_dirs[hit_ray_idxs]

        # Some clean up to free memory
        del ray_origins, hit_ray_idxs, mask, img
        gc.collect()  # Force garbage collection

        # Cast the indices down to int32 to save memory. Usually indices have to be int64, however, we assume that
        # the indices from 0 to 2^31-1 are sufficient. Therefore, we can safely cast down

        assert torch.all(face_idxs <= (2<<31)-1)
        face_idxs = face_idxs.to(torch.int32)
        assert torch.all(vertex_idxs_of_hit_faces <= (2<<31)-1)
        vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.to(torch.int32)
        barycentric_coords = barycentric_coords.to(torch.float32)
        expected_rgbs = expected_rgbs.to(torch.float32)
        unit_ray_dirs = unit_ray_dirs.to(torch.float32)
        uv_coords = uv_coords.to(torch.float32) # FIXME check if this works

        # And finally, we store the results in the cache
        for idx in range(num_hits):
            self.cache_face_idxs.append(face_idxs[idx])
            self.cache_vertex_idxs_of_hit_faces.append(vertex_idxs_of_hit_faces[idx])
            self.cache_barycentric_coords.append(barycentric_coords[idx])
            self.cache_expected_rgbs.append(expected_rgbs[idx])
            self.cache_unit_ray_dirs.append(unit_ray_dirs[idx])
            self.cache_uv_coords.append(uv_coords[idx])


    def write_to_disk(self):
        print("Starting to write to disk...")

        # Write the cached eigenfuncs and cached expected RGBs to disk
        os.makedirs(self.out_dir, exist_ok=True)

        # Stack the results, write to disk, and then free up memory

        self.cache_face_idxs = torch.stack(self.cache_face_idxs)
        print(
            f"Face Idxs: dim={self.cache_face_idxs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_face_idxs)}B, dtype={self.cache_face_idxs.dtype}")
        np.save(os.path.join(self.out_dir, "face_idxs.npy"), self.cache_face_idxs, allow_pickle=False)
        del self.cache_face_idxs
        gc.collect()  # Force garbage collection

        self.cache_vertex_idxs_of_hit_faces = torch.stack(self.cache_vertex_idxs_of_hit_faces)
        print(
            f"Vertex Idxs of Hit Faces: dim={self.cache_vertex_idxs_of_hit_faces.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_vertex_idxs_of_hit_faces)}B, dtype={self.cache_vertex_idxs_of_hit_faces.dtype}")
        np.save(os.path.join(self.out_dir, "vids_of_hit_faces.npy"), self.cache_vertex_idxs_of_hit_faces,
                allow_pickle=False)
        del self.cache_vertex_idxs_of_hit_faces
        gc.collect()  # Force garbage collection

        self.cache_barycentric_coords = torch.stack(self.cache_barycentric_coords)
        print(
            f"Barycentric Coords: dim={self.cache_barycentric_coords.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_barycentric_coords)}B, dtype={self.cache_barycentric_coords.dtype}")
        np.save(os.path.join(self.out_dir, "barycentric_coords.npy"), self.cache_barycentric_coords, allow_pickle=False)
        del self.cache_barycentric_coords
        gc.collect()  # Force garbage collection

        self.cache_uv_coords = torch.stack(self.cache_uv_coords)
        print(
            f"UV Coords: dim={self.cache_uv_coords.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_uv_coords)}B, dtype={self.cache_uv_coords.dtype}")
        np.save(os.path.join(self.out_dir, "uv_coords.npy"), self.cache_uv_coords, allow_pickle=False)
        del self.cache_uv_coords
        gc.collect()  # Force garbage collection

        self.cache_expected_rgbs = torch.stack(self.cache_expected_rgbs)
        print(
            f"Expected RGBs: dim={self.cache_expected_rgbs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_expected_rgbs)}B, dtype={self.cache_expected_rgbs.dtype}")
        np.save(os.path.join(self.out_dir, "expected_rgbs.npy"), self.cache_expected_rgbs, allow_pickle=False)
        del self.cache_expected_rgbs
        gc.collect()  # Force garbage collection

        self.cache_unit_ray_dirs = torch.stack(self.cache_unit_ray_dirs)
        print(
            f"Unit Ray Dirs: dim={self.cache_unit_ray_dirs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_unit_ray_dirs)}B, dtype={self.cache_unit_ray_dirs.dtype}")
        np.save(os.path.join(self.out_dir, "unit_ray_dirs.npy"), self.cache_unit_ray_dirs, allow_pickle=False)
        del self.cache_unit_ray_dirs
        gc.collect()  # Force garbage collection
