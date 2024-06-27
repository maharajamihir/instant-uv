import argparse
import commentjson as json
import os
import sys
import numpy as np
import torch
import time
import pickle

from pathlib import Path

import trimesh
import xatlas
import yaml
from PIL.Image import fromarray

from trimesh import visual

from data.dataset import InstantUVDataset
from util.utils import load_mesh

# Append src/
sys.path.append(str(Path(__file__).parent))

# Change into instant-uv
os.chdir(Path(__file__).parent.parent)

# Append scripts DIR
SCRIPTS_DIR = str(Path(__file__).parent / "tiny-cuda-nn/scripts")
sys.path.append(SCRIPTS_DIR)

try:
    import tinycudann as tcnn
except ImportError:
    print("This sample requires the tiny-cuda-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-cuda-nn$ cd bindings/torch")
    print("tiny-cuda-nn/bindings/torch$ python setup.py install")
    print("============================================================")
    sys.exit()

from common import read_image, write_image, ROOT_DIR

DATA_DIR = os.path.join("data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


class Dataset(torch.nn.Module):
    def __init__(self, filename, device):
        super(Dataset, self).__init__()
        self.data = read_image(filename)
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

    def forward(self, xs):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.shape

            xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[1] - 1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0] - 1)
            x1 = (x0 + 1).clamp(max=shape[1] - 1)
            y1 = (y0 + 1).clamp(max=shape[0] - 1)

            # Note from moritz:
            # They appear to "spread the gradients" when calculating the color of a pixel. So they basically calculate
            # it through the four neighboring pixels
            # I assume it is beneficial for learning to use four pixels instead of one
            return (
                    self.data[y0, x0] * (1.0 - lerp_weights[:, 0:1]) * (1.0 - lerp_weights[:, 1:2]) +
                    self.data[y0, x1] * lerp_weights[:, 0:1] * (1.0 - lerp_weights[:, 1:2]) +
                    self.data[y1, x0] * (1.0 - lerp_weights[:, 0:1]) * lerp_weights[:, 1:2] +
                    self.data[y1, x1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
            )


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    # parser.add_argument("image", nargs="?", default="data/images/albert.jpg", help="Image to match")
    parser.add_argument("our_config", nargs="?", default="config/human/config_human.yaml",
                        help="YAML config for our training stuff")
    parser.add_argument("tiny_nn_config", nargs="?", default="src/tiny-cuda-nn/data/config_hash.json",
                        help="JSON config for tiny-cuda-nn")
    parser.add_argument("n_steps", nargs="?", type=int, default=100, help="Number of training steps")
    # parser.add_argument("result_filename", nargs="?", default="", help="Number of training steps")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("================================================================")
    print(" TODO DESCRIPTION!")
    print(" TODO DESCRIPTION! Bla bla using tiny-cuda-nn's PyTorch extension.")
    print("================================================================")

    print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

    device = torch.device("cuda")
    args = get_args()

    with open(args.tiny_nn_config) as tiny_nn_config_file:
        tiny_nn_config = json.load(tiny_nn_config_file)

    # TODO: use function from util
    with open(args.our_config) as our_config_file:
        our_config = yaml.safe_load(our_config_file)

    # Load preprocessed data
    preproc_data_path = Path(our_config["data"]["preproc_data_path"]) / "train"

    barycentric_coords = np.load(preproc_data_path / "barycentric_coords.npy")
    expected_rgbs = np.load(preproc_data_path / "expected_rgbs.npy")
    face_idxs = np.load(preproc_data_path / "face_idxs.npy")
    unit_ray_dirs = np.load(preproc_data_path / "unit_ray_dirs.npy")
    vids_of_hit_faces = np.load(preproc_data_path / "vids_of_hit_faces.npy")

    """ LONG TEST """

    mesh_file = our_config["data"]["mesh_path"]

    # Pre sanity check
    """ DELETE THIS"""
    mesh_old = trimesh.load(mesh_file)
    vertices_of_hit_faces_old = np.array(mesh_old.vertices[vids_of_hit_faces])
    coords_3d_old = np.sum(barycentric_coords[:, :, np.newaxis] * vertices_of_hit_faces_old, axis=1)
    # trimesh.PointCloud(vertices=coords_3d_old, colors=expected_rgbs * 255).show(
        # line_settings={'point_size': 0.005}
    # )
    """ DELETE THIS END"""


    # def extract_uv_mapping(obj_file):
    #     mesh = trimesh.load(obj_file)
    #     if 'visual' in mesh.metadata:
    #         uv = mesh.visual.uv
    #         return mesh, uv
    #     else:
    #         raise ValueError("The mesh does not contain UV mapping.")
    #
    #
    # def map_3d_to_uv(mesh, uv_coords, point_3d):
    #     distances, face_index = mesh.nearest.on_surface([point_3d])
    #     face_index = face_index[0]
    #     face_vertices = mesh.vertices[mesh.faces[face_index]]
    #     uv_vertices = uv_coords[mesh.faces[face_index]]
    #
    #     def barycentric_coords(p, a, b, c):
    #         v0, v1, v2 = b - a, c - a, p - a
    #         d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    #         d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    #         denom = d00 * d11 - d01 * d01
    #         v, w = (d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom
    #         u = 1 - v - w
    #         return u, v, w
    #
    #     u, v, w = barycentric_coords(point_3d, *face_vertices)
    #     uv_point = u * uv_vertices[0] + v * uv_vertices[1] + w * uv_vertices[2]
    #     return uv_point


    # Usage example
    mesh = load_mesh(mesh_file)

    xatlas_path = str(Path(our_config["data"]["preproc_data_path"]) / "train" / "xatlas.obj")
    v_path = str(Path(our_config["data"]["preproc_data_path"]) / "train" / "new_vertex_id_to_old_vertex_id.npy")
    rv_path = str(Path(our_config["data"]["preproc_data_path"]) / "train" / "old_vertex_to_new_vertexes.pkl")
    uv_path = str(Path(our_config["data"]["preproc_data_path"]) / "train" / "uv.pkl")

    if not os.path.isfile(xatlas_path):
        # Extract with xatlas
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        np.save(v_path, vmapping, allow_pickle=False)

        # maps original vertex to new vertex_id (in case we need it)
        reverse_vmapping = {}
        for left, right in zip(range(len(vmapping)), vmapping):
            reverse_vmapping.setdefault(right, []).append(left)
        with open(rv_path, "wb") as f:
            pickle.dump(reverse_vmapping, f)

        """ TODO CHECK AND DELETE"""
        # face mapping (for every index there we map it using vmapping)
        # TODO: CHECK VALIDITY (AND THEN REMOVE)
        indices_translated_into_old_mesh = [list(map(lambda x: vmapping[x], f)) for f in indices]
        # We check that the old_indices match the new face indices
        assert indices_translated_into_old_mesh == mesh.faces.view(np.ndarray).tolist()
        """ -------------------- """ 

        # Turn into trimesh and save
        # Process=False will keep the duplicates
        new_mesh = trimesh.Trimesh(vertices=mesh.vertices[vmapping], faces=indices, process=False)
        new_mesh.export(xatlas_path)

        # save uv seperately
        with open(uv_path, "wb") as f:
            pickle.dump(uvs, f)

    else:
        # Load data
        new_mesh = load_mesh(xatlas_path)
        with open(rv_path, 'rb') as f:
            reverse_vmapping = pickle.load(f)
        with open(uv_path, 'rb') as f:
            uvs = pickle.load(f)
        vmapping = np.load(v_path)
        indices = new_mesh.faces
        new_mesh.visual = visual.texture.TextureVisuals(uv=uvs)

    # Sanity check: Translation of preprocessed data
    barycentric_coords = barycentric_coords  # Keep the same
    expected_rgbs = expected_rgbs  # Keep the same
    face_idxs = face_idxs  # Stays the same because apparently xatlas does not change the order
    unit_ray_dirs = unit_ray_dirs  # Stays the same
    vids_of_hit_faces = indices[face_idxs]


    # NOTE: Check for proof.
    # mesh_old.vertices[mesh_old.faces[face_idxs]], new_mesh.vertices[indices[face_idxs]], vids_of_hit_faces, mesh_old.vertices[vids_of_hit_faces], new_mesh.faces.tolist().index([reverse_vmapping.get(v)[0] for v in vids_of_hit_faces[0]]),

    # NOTE:
    # mesh_old.vertices[mesh_old.faces[face_idxs]] == new_mesh.vertices[indices[face_idxs]]  HOLDS!!!

    # NOTE:
    # mesh_old.vertices[vids_of_hit_faces] !!!!=== mesh_old.vertices[mesh_old.faces[face_idxs]]

    uv_vertices_of_hit_faces = np.array(new_mesh.visual.uv[vids_of_hit_faces])
    # TODO: SOMEONE PLS VERIFY THIS CALCULATION
    # Note: we can simply use the same barycentric coords since its all linear
    uv_coords = np.sum(barycentric_coords[:, :, np.newaxis] * uv_vertices_of_hit_faces, axis=1)

    # --------------------------------------

    vertices_of_hit_faces = np.array(new_mesh.vertices[vids_of_hit_faces])
    coords_3d = np.sum(barycentric_coords[:, :, np.newaxis] * vertices_of_hit_faces, axis=1)

    """ DEBUG ONLY"""
    # Lets try only the legs
    # mask = coords_3d[:, 2] < 0
    # coords_3d = coords_3d[mask]
    # uv_coords = uv_coords[mask]
    # expected_rgbs = expected_rgbs[mask]

    # Maybe we need to flip???
    # uv_coords = np.flip(uv_coords, axis=1).copy()
    """ DEBUG END"""

    # Sanity-Check visualization
    # trimesh.PointCloud(vertices=coords_3d, colors=expected_rgbs * 255).show(
        # line_settings={'point_size': 0.005}
    # )
    """ LONG TEST END"""

    print("Loading data")
    dataset = InstantUVDataset(uv=uv_coords, rgb=expected_rgbs, points_xyz=coords_3d)
    n_channels = dataset.rgb.shape[1]
    print("Channels:", n_channels)
    print("Initializing model")
    model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=n_channels,
                                          encoding_config=tiny_nn_config["encoding"],
                                          network_config=tiny_nn_config["network"]).to(device)
    print(model)

    # ===================================================================================================
    # The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
    # tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
    # ===================================================================================================
    # encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
    # network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
    # model = torch.nn.Sequential(encoding, network)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Variables for saving/displaying image results
    # NOTE: JUST FOR DEBUGGING LOL
    xs = new_mesh.visual.uv[:, 0]
    ys = new_mesh.visual.uv[:, 1]
    min_x = new_mesh.visual.uv[:, 0].min()
    max_x = new_mesh.visual.uv[:, 0].max()
    min_y = new_mesh.visual.uv[:, 1].min()
    max_y = new_mesh.visual.uv[:, 1].max()
    # Also normalized with aspect ratio 1:1

    resolution = (500, 500)  # For testing
    img_shape = resolution + torch.Size([n_channels])
    n_pixels = resolution[0] * resolution[1]

    half_dx = 0.5 / resolution[0]
    half_dy = 0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((yv.flatten(), xv.flatten())).t()

    # path = f"reference.jpg"
    # print(f"Writing '{path}'... ", end="")
    # write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
    # print("done.")

    prev_time = time.perf_counter()

    # batch_size = 2 ** 18
    # batch_size = min(len(dataset), 2**19)
    batch_size = len(dataset)
    interval = 10

    print(f"Beginning optimization with {args.n_steps} training steps.")

    # try:
    #     batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
    #     # Tracing to optimize the calculation / forward pass of the batch calculation
    #     traced_image = torch.jit.trace(image, batch)
    # except:
    #     # If tracing causes an error, fall back to regular execution
    #     print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
    #     traced_image = image

    for i in range(args.n_steps):
        # For now lets just randomize
        random_indices = torch.randperm(len(dataset))[:batch_size]
        batch = torch.from_numpy(dataset.uv)[random_indices].to(device)
        targets = torch.from_numpy(dataset.rgb)[random_indices].to(device)

        output = model(batch)

        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (output.detach() ** 2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0:
            loss_val = loss.item()
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time * 1000000)}[µs]")

            path = f"{i}.jpg"
            print(f"Writing '{path}'... ", end="")
            with torch.no_grad():
                write_image(path, np.flipud(model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()))
            print("done.")

            """Second image only on dataset"""
            path = f"{i}_ds.jpg"
            print(f"Writing '{path}'... ", end="")
            with torch.no_grad():
                # Define grey image
                grey_image = torch.ones((*resolution, 3), dtype=torch.uint8, device=device) * 128

                # Get predictions for all that we have in dataset
                input = torch.from_numpy(dataset.uv).to(device)
                pixel_xy = (input * torch.tensor(resolution, device=device)).long()
                predictions = model(input).clamp(0.0, 1.0).detach()

                # Multiply predictions by 255 and convert to int in one step
                scaled_predictions = (predictions * 255).type(torch.uint8)

                # Use advanced indexing to assign values
                grey_image[pixel_xy[:, 1], pixel_xy[:, 0], :] = scaled_predictions

                # Note: If we flip here we are in the right orientation for the output map if we visualize!!
                fromarray(np.flipud(grey_image.cpu().numpy()), "RGB").save(path)
            print("done.")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10

    if args.result_filename:
        print(f"Writing '{args.result_filename}'... ", end="")
        with torch.no_grad():
            write_image(args.result_filename, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
        print("done.")

    tcnn.free_temporary_memory()
