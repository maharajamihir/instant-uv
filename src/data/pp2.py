import gc
import os
from pathlib import Path

import numpy as np
import torch
import argparse
from tqdm import tqdm
import imageio.v2 as imageio
import sys

from data.intersection import any_intersection

sys.path.append(str(Path(__file__).parent.parent))

from util.mesh import MeshViewPreProcessor, get_ray_mesh_intersector
from util.utils import load_obj_mask, load_cameras, load_config, load_mesh, tensor_mem_size_in_bytes, \
    get_mapping_gt_via_blender, map_to_UV_blender, get_mapping, map_to_UV

os.chdir(Path(__file__).parent.parent.parent)

"""Triangle Meshes to Point Clouds"""
import numpy as np


def calculate_normals_vectorized(faces):
    # Extract the vertices of the faces
    v0 = faces[:, 0, :]
    v1 = faces[:, 1, :]
    v2 = faces[:, 2, :]

    # Calculate the edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Calculate the cross product of the edge vectors to get the face normal
    normals = np.cross(edge1, edge2, axis=1)

    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1
    normalized_normals = normals / magnitudes
    # Normalize the normals to unit vectors

    return normalized_normals


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    def _calc_triangle_area(c1, c2, c3):
        v1 = c2 - c1
        v2 = c3 - c1

        cross_product = np.cross(v1, v2)
        area_times2 = np.linalg.norm(cross_product)
        return 0.5 * area_times2

    def _sample_from_triangle_coords(c1, c2, c3):
        r1, r2 = np.random.rand(2)

        sqrt_r1 = np.sqrt(r1)

        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = r2 * sqrt_r1

        return u * c1 + v * c2 + w * c3, np.array([u, v, w])

    total_surface = 0.0
    surfaces = []
    for f in faces:
        c1, c2, c3 = vertices[int(f[0])], vertices[int(f[1])], vertices[int(f[2])]
        area = _calc_triangle_area(c1, c2, c3)
        surfaces.append(area)
        total_surface += area
    probabilities = [s / total_surface for s in surfaces]

    indices_to_be_sampled = np.random.choice(range(len(probabilities)), size=n_points, p=probabilities)

    points = []
    barys = []
    for i in indices_to_be_sampled:
        f = faces[i]
        c1, c2, c3 = vertices[int(f[0])], vertices[int(f[1])], vertices[int(f[2])]
        point, bary = _sample_from_triangle_coords(c1, c2, c3)
        points.append(point)
        barys.append(bary)

    return np.array(points), np.array(barys), np.array(indices_to_be_sampled)


def project_vertices(vertices, K, RT):
    # Extract rotation and translation
    R = RT[0:3, 0:3]
    t = RT[0:3, 3]

    # Compute the inverse
    R_inv = R.T
    t_inv = -R_inv @ t

    # Construct the world-to-camera matrix
    world_to_camera = np.eye(4)
    world_to_camera[0:3, 0:3] = R_inv
    world_to_camera[0:3, 3] = t_inv

    # Transform vertices from world coordinates to camera coordinates
    ones = np.ones((vertices.shape[0], 1))
    vertices_homogeneous = np.hstack((vertices, ones))

    vertices_cam_homogeneous = (world_to_camera @ vertices_homogeneous.T).T

    # Drop the homogeneous coordinate for further processing
    vertices_cam = vertices_cam_homogeneous[:, :3]  # Shape: (N, 3)
    vertices_depth = vertices_cam_homogeneous[:, 2]  # Shape: (N,)

    # Project vertices onto the image plane
    vertices_proj_homogeneous = np.dot(vertices_cam, K.T)  # Shape: (N, 3)

    # Normalize by the third coordinate (homogeneous coordinate)
    vertices_proj = vertices_proj_homogeneous[:, :2] / vertices_proj_homogeneous[:, 2, np.newaxis]  # Shape: (N, 2)

    # import trimesh
    # trimesh.PointCloud(vertices=np.vstack([vertices_homogeneous[:, 0:3], vertices_cam])).show(
    #     line_settings={'point_size': 5}
    # )

    return vertices_proj, vertices_depth


def backface_culling(vertices, normals, camera_pos):
    # Compute the view direction from each vertex to the camera
    view_directions = camera_pos - vertices

    # Normalize the view direction vectors
    view_directions /= np.linalg.norm(view_directions, axis=1, keepdims=True)

    # Compute the dot product between the vertex normals and view directions
    dot_products = np.einsum('ij,ij->i', normals, view_directions)

    # Determine which vertices are facing the camera (dot product > 0)
    visible = dot_products > 0
    return visible


def get_pixel_indices(vertices_proj, image_size):
    """
    Convert projected 2D coordinates to pixel indices and ensure they are within the image bounds.

    :param vertices_proj: numpy array of shape (N, 2) representing the projected 2D coordinates.
    :param image_size: Tuple (height, width) representing the size of the image.
    :return: numpy array of shape (N, 2) with pixel indices (y, x).
    """
    height, width = image_size

    # Clip coordinates to be within the image bounds
    x = np.clip(vertices_proj[:, 0].astype(int), 0, width - 1)
    y = np.clip(vertices_proj[:, 1].astype(int), 0, height - 1)

    # Combine y and x into a single array of pixel indices
    pixel_indices = np.stack((y, x), axis=-1)  # Shape: (N, 2)

    return pixel_indices


def _stack_and_write_to_disk(array, out_dir, filename: str):
    name = filename.split(".")[0]
    print(f"{name}: Saving...")
    np.save(os.path.join(out_dir, filename), array, allow_pickle=True)
    del array
    gc.collect()  # Force garbage collection


def preprocess_views(mesh, points, barys, face_ids, normals, mesh_views_list_train, dataset_path, config):
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)

    rgbs = []  # shape num_viewsx3
    bs = []  # shape num_viewsxpointsx3
    fids = []  # shape num_viewsxpointsx1
    uvs = []  # shape num_viewsxpointsx2

    for mesh_view in tqdm(mesh_views_list_train):
        copy_points = np.copy(points)
        copy_barys = np.copy(barys)
        copy_face_ids = np.copy(face_ids)
        copy_normals = np.copy(normals)

        mesh_view_path = os.path.join(dataset_path, mesh_view)
        camCv2world, K = load_cameras(mesh_view_path)

        camCv2world = camCv2world.cpu().numpy()
        camCv2world = np.concatenate([camCv2world, np.array([[0., 0, 0, 1]], dtype=camCv2world.dtype)], 0)
        K = K.cpu().numpy()[:, :3]

        cam_position = camCv2world[:3, 3]

        # Filter points that are seen from the back.
        visibility = backface_culling(copy_points, copy_normals, cam_position)
        copy_points = copy_points[visibility]
        copy_normals = copy_normals[visibility]
        copy_face_ids = copy_face_ids[visibility]

        projected, depth = project_vertices(copy_points, K, camCv2world)

        pixel_indices = np.round(projected).astype(int)
        ray_dirs = cam_position - copy_points
        magnitudes = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
        magnitudes[magnitudes == 0] = 1.0
        ray_dirs = ray_dirs / magnitudes

        # TODO: HERE PYEMBREE
        # intersector = get_ray_mesh_intersector(mesh)

        any_inters = any_intersection(copy_points, ray_dirs, mesh_vertices[mesh_faces])
        no_intersection = ~any_inters
        # Filter
        copy_points = copy_points[no_intersection]
        copy_normals = copy_normals[no_intersection]
        copy_face_ids = copy_face_ids[no_intersection]
        pixel_indices = pixel_indices[no_intersection]

        # Load depth map for building a mask
        obj_mask = load_obj_mask(mesh_view_path, as_numpy=False)

        # Load image
        img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))

        # IMPORTANT FLIP!!!!
        flipped_pixels = pixel_indices[:, [1, 0]]
        rgb_array = img[flipped_pixels[:, 0], flipped_pixels[:, 1]] / 255.0
        # import trimesh
        # trimesh.PointCloud(vertices=points, colors=rgb_array).show(
        #     line_settings={'point_size': 5}
        # )

        # GROUP!!!
        # Create a structured array for easier unique operations
        dtype = [('x', 'i4'), ('y', 'i4')]
        structured_coords = np.array(list(map(tuple, flipped_pixels)), dtype=dtype)

        # Find unique structured coordinates and their indices
        _, inverse_indices = np.unique(structured_coords, return_inverse=True)

        # Create the mapping from (x, y) to indices
        unique_coords = np.unique(flipped_pixels, axis=0)
        px_to_indices = [np.nonzero(inverse_indices == idx)[0]
                         for idx, coord in enumerate(unique_coords)]

        for datapoint in px_to_indices:
            rgbs.append(rgb_array[datapoint])
            b = copy_barys[datapoint]
            f = copy_face_ids[datapoint]
            bs.append(b)
            fids.append(f)

            uv_backend = "gt"  # MANUAL !!!!
            barycentric_coords, face_idxs, vertex_idxs_of_hit_faces = torch.from_numpy(b), torch.from_numpy(f), torch.from_numpy(mesh_faces[f])
            if uv_backend == "gt":
                mapping = get_mapping_gt_via_blender(mesh, "train", config)
                uv_coords = map_to_UV_blender(barycentric_coords, face_idxs, vertex_idxs_of_hit_faces, mapping)

            elif uv_backend == "xatlas":
                # get uv coordinates
                mapping = get_mapping(mesh, "train", config)

                # FIXME: mapping, we must get the right object directly form get_mapping
                uv_coords = map_to_UV(barycentric_coords, face_idxs, mapping)
            else:
                raise Exception("WTF YOU DOING!")

            uvs.append(uv_coords)

        img[~obj_mask] = [255, 255, 255]
        imageio.imwrite(os.path.join(mesh_view_path, "image", "001.png"), img)
        img = torch.from_numpy(img).to(dtype=torch.float32)
        img /= 255.

    return np.array(rgbs), np.array(bs), np.array(fids), np.array(uvs)


def preprocess_dataset(split, dataset_path, path_to_mesh, out_dir, mesh_views_list_train, check_depth=False,
                       config=None):
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
    os.makedirs(split_out_dir, exist_ok=True)

    mesh = load_mesh(path_to_mesh)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    points, barys, face_ids = sample_point_cloud(vertices, faces, 5000)
    normals = calculate_normals_vectorized(vertices[faces[face_ids]])

    rgbs, bs, fids, uv_coords = preprocess_views(mesh, points, barys, face_ids, normals, mesh_views_list_train, dataset_path, config)

    print("SAVING DATASET")
    _stack_and_write_to_disk(rgbs, out_dir, "rgbs.npy")
    _stack_and_write_to_disk(bs, out_dir, "bs.npy")
    _stack_and_write_to_disk(fids, out_dir, "fids.npy")
    _stack_and_write_to_disk(uv_coords, out_dir, "uvs.npy")


def main():
    config = load_config("config/human/config_human_gt.yaml", "config/human/config_human_defaults.yaml", )
    dataset_path = config["data"]["raw_data_path"]
    out_dir = config["data"]["preproc_data_path"]
    mesh_path = config["data"]["mesh_path"]
    mesh_views_list_train = ["human023", "human010", "human013", "human004", "human012"]

    preprocess_dataset(
        split="train",
        dataset_path=dataset_path,
        path_to_mesh=mesh_path,
        out_dir=out_dir,
        mesh_views_list_train=mesh_views_list_train,
        config=config
    )


if __name__ == "__main__":
    main()
