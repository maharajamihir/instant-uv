import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
import os
import imageio
import yaml
import sys
import igl
import trimesh
from trimesh import visual
from collections import OrderedDict
import xatlas
from pathlib import Path
import pickle
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append("src/models/")

# Make sure loading .exr works for imageio
try:
    imageio.plugins.freeimage.download()
except FileExistsError:
    # Ignore
    pass


def load_preprocessed_data(preproc_data_path):
    data = {}

    vertex_idxs_of_hit_faces = np.load(os.path.join(preproc_data_path, "vids_of_hit_faces.npy"))
    data["vertex_idxs_of_hit_faces"] = torch.from_numpy(vertex_idxs_of_hit_faces).to(dtype=torch.int64)

    barycentric_coords = np.load(os.path.join(preproc_data_path, "barycentric_coords.npy"))
    data["barycentric_coords"] = torch.from_numpy(barycentric_coords).to(dtype=torch.float32)

    expected_rgbs = np.load(os.path.join(preproc_data_path, "expected_rgbs.npy"))
    data["expected_rgbs"] = torch.from_numpy(expected_rgbs).to(dtype=torch.float32)

    unit_ray_dirs_path = os.path.join(preproc_data_path, "unit_ray_dirs.npy")
    face_idxs_path = os.path.join(preproc_data_path, "face_idxs.npy")
    if os.path.exists(unit_ray_dirs_path) and os.path.exists(face_idxs_path):
        unit_ray_dirs = np.load(unit_ray_dirs_path)
        data["unit_ray_dirs"] = torch.from_numpy(unit_ray_dirs).to(dtype=torch.float32)

        face_idxs = np.load(face_idxs_path)
        data["face_idxs"] = torch.from_numpy(face_idxs).to(dtype=torch.int64)

    return data


def tensor_mem_size_in_bytes(x):
    return sys.getsizeof(x.untyped_storage())


def load_trained_model(weights_path, device):
    model = torch.load(weights_path)
    return model


def load_mesh(path):
    # Note: We load using libigl because trimesh does some unwanted preprocessing and vertex
    # reordering (even if process=False and maintain_order=True is set). Hence, we load it
    # using libigl and then convert the loaded mesh into a Trimesh object.
    v, f = igl.read_triangle_mesh(path)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)

    assert np.array_equal(v, mesh.vertices) and np.array_equal(f, mesh.faces)
    return mesh


def load_cameras(view_path, as_numpy=False):
    cameras = np.load(os.path.join(view_path, "depth", "cameras.npz"))

    if as_numpy:
        return cameras["world_mat_0"].astype(np.float32), cameras["camera_mat_0"].astype(np.float32)

    # Torch version
    camCv2world = torch.from_numpy(cameras["world_mat_0"]).to(dtype=torch.float32)
    K = torch.from_numpy(cameras["camera_mat_0"]).to(dtype=torch.float32)
    return camCv2world, K


def model_summary(model, data):
    data_batch = next(iter(data["train"]))
    summary(model, input_data=[data_batch])


def load_obj_mask(view_path, as_numpy=False):
    if view_path.endswith(".npy"):
        return np.load(view_path)

    depth_path = os.path.join(view_path, "depth", "depth_0000.exr")
    if os.path.exists(depth_path):
        depth_map = imageio.imread(depth_path)[..., 0]

        mask_value = 1.e+10
        obj_mask = depth_map != mask_value
    else:
        mask_path = os.path.join(view_path, "depth", "mask.png")
        assert os.path.exists(mask_path), "Must have depth or mask"
        mask = imageio.imread(mask_path)
        obj_mask = mask != 0  # 0 is invalid

    if as_numpy:
        return obj_mask

    return torch.from_numpy(obj_mask)


def load_depth_as_numpy(view_path):
    depth_path = os.path.join(view_path, "depth", "depth_0000.exr")
    assert os.path.exists(depth_path)
    depth_map = imageio.imread(depth_path)[..., 0]

    return depth_map


def batchify_dict_data(data_dict, input_total_size, batch_size):
    idxs = np.arange(0, input_total_size)
    batch_idxs = np.split(idxs, np.arange(batch_size, input_total_size, batch_size), axis=0)
    batches = []
    for cur_idxs in batch_idxs:
        data = {}
        for key in data_dict.keys():
            data[key] = data_dict[key][cur_idxs]
        batches.append(data)
    return batches


def fill_defaults(from_d, to_d, key):
    # We fill on the following conditions:
    # 1. Key does not exist
    if key not in to_d:
        to_d[key] = from_d[key]
    # 2. Key value is "DEFAULT"
    if isinstance(to_d[key], str):
        if to_d[key].lower() == "default":
            to_d[key] = from_d[key]

    # Lastly, we recurse if key value is a dict.
    if isinstance(from_d[key], dict):
        assert isinstance(to_d[key], dict), "???"
        for k_key in from_d[key].keys():
            fill_defaults(from_d[key], to_d[key], k_key)


def load_yaml(path):
    with open(path, "r") as f1:
        content = yaml.safe_load(f1)
    return content


def load_config(path, defaults):
    defaults = load_yaml(defaults)
    config = load_yaml(path)

    for key in defaults.keys():
        fill_defaults(defaults, config, key)
    return config


def get_loss_fn(loss_type):
    # L1 loss
    if loss_type == "L1":
        return nn.L1Loss()
    elif loss_type == "L2":
        return nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented")


def get_combine_res_fn(combine_res_type):
    if combine_res_type == "cat":
        return torch.cat
    elif combine_res_type == "sum":
        return torch.sum
    else:
        raise NotImplementedError(f"Combine res type {combine_res_type} not implemented")


##########################################################################################
# The following is taken from:
# https://github.com/tum-vision/tandem/blob/master/cva_mvsnet/utils.py
##########################################################################################


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, **kwargs):
        if isinstance(vars, list):
            return [wrapper(x, **kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, **kwargs) for k, v in vars.items()}
        else:
            return func(vars, **kwargs)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def to_device(x, *, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, str):
        return x
    else:
        raise NotImplementedError(f"Invalid type for to_device: {type(x)}")


##########################################################################################

class LossWithGammaCorrection(torch.nn.Module):
    def __init__(self, loss_type='L1'):
        super().__init__()

    def forward(self, x, y):
        x = linear2sRGB(x)
        y = linear2sRGB(y)

        return self.criterion(x, y)


def linear2sRGB(linear, eps=None):
    """
    Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB.
    From https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/image.py#L48
    """
    if eps is None:
        eps = torch.tensor(torch.finfo(torch.float32).eps)

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def compute_psnr(x, y):
    # x and y must be numpy arrays in the range [0.0 1.0] with shape [H, W, C]
    return psnr(x, y, data_range=1.0)


def compute_ssim(x, y):
    # x and y must be numpy arrays in the range [0.0 1.0] with shape [H, W, C]
    ssim_val = ssim(
        x,
        y,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        data_range=1.0,
        channel_axis=2
    )
    return ssim_val


class Metrics:
    def __init__(self):
        self.lpips_eval = lpips.LPIPS(net='alex', version='0.1')

    def compute_metrics(self, renderings):
        # ######### Compute psnr and ssim
        psnrs_linear = []
        psnrs_srgb = []
        ssims = []

        # Loop over images
        for im_linear, im_gt_linear, im_srgb, im_gt_srgb in zip(
                renderings['images_linear'],
                renderings['images_gt_linear'],
                renderings['images_srgb'],
                renderings['images_gt_srgb']
        ):
            psnrs_linear.append(compute_psnr(im_linear.numpy(), im_gt_linear.numpy()))
            psnrs_srgb.append(compute_psnr(im_srgb.numpy(), im_gt_srgb.numpy()))
            ssims.append(compute_ssim(im_srgb.numpy(), im_gt_srgb.numpy()))

        psnrs_linear = np.stack(psnrs_linear)
        psnrs_srgb = np.stack(psnrs_srgb)
        ssims = np.stack(ssims)

        # ################ Compute lpips
        with torch.no_grad():
            lpips = self.lpips_eval(
                2 * torch.stack(renderings['images_srgb']).permute(0, 3, 1, 2) - 1.0,
                2 * torch.stack(renderings['images_gt_srgb']).permute(0, 3, 1, 2) - 1.0
            ).squeeze()

        result = {
            'psnrs_linear': psnrs_linear,
            'psnrs_srgb': psnrs_srgb,
            'ssims': ssims,
            'lpips': lpips
        }

        return result


def time_method(model, dummy_input, repetitions=300):
    """
    Model and dummy_input must be on the target device
    dummy_input must be such that model(**dummy_input) can be evaluated
    Taken from here: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    """

    model.eval()
    with torch.no_grad():
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            tmp_out = model(**dummy_input)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(**dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_eval_time = np.sum(timings) / repetitions

    return mean_eval_time


#################################################
def get_mapping_blender(mesh, split, config):
    import bpy

    # Replace with the path to your OBJ file
    # obj_file = "/home/morkru/Desktop/Github/instant-uv/data/raw/human/RUST_3d_Low1.obj"
    # obj_file = str(Path(config["data"]["preproc_data_path"]) / split / "xatlas.obj")
    obj_file = "data/gts/human/human_triangle_mesh_gt.obj"
    # obj_file = "/home/morkru/Desktop/Github/instant-uv/data/preprocessed/triangle_CAT.obj"
    # new_mesh = load_mesh(obj_file)

    # Hier das war vorher
    # tmp_path = str(Path(config["data"]["preproc_data_path"]) / split / "mesh_tmp.obj")
    # We must export because else blender will import non-triangle mesh
    # mesh.export(tmp_path)

    # THESE ASSERTIONS MUST HOLD!!!
    # mesh2 = load_mesh(tmp_path)
    # assert (np.array(mesh2.vertices) == np.array(mesh.vertices)).sum() == mesh.vertices.shape[0] * mesh.vertices.shape[1]
    # assert (np.array(mesh2.faces) == np.array(mesh.faces)).sum() == mesh.faces.shape[0] * mesh.faces.shape[1]

    blender_path = str(Path(config["data"]["preproc_data_path"]) / split / "blender_uv.pkl")
    if not os.path.isfile(blender_path):

        # Clear existing scene data
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import OBJ file
        bpy.ops.wm.obj_import(filepath=obj_file)

        # Select all objects
        bpy.ops.object.select_all(action='SELECT')

        # Switch to Edit mode
        bpy.ops.object.mode_set(mode='EDIT')

        # Select all faces
        bpy.ops.mesh.select_all(action='SELECT')

        # Unwrap UVs
        assert False, ("here we should define what to do. either we can use blender smart uv project or the plugin "
                       "Unwrap Me. Which i tested but we need to extract the proper commands first.")
        # bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.2)
        # Lets try transform to triangles
        # bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

        # Switch back to Object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        me = bpy.context.object.data
        uv_layer = me.uv_layers.active.data

        face_id_to_uv_mapping = {}
        face_mapping = {}
        for poly in me.polygons:
            print("Polygon", poly.index)

            # Find out what face we are
            face = sorted([v for v in poly.vertices])

            # For each particular face we get its own mapping
            vertex_to_uv_mini = {}
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                vertex_to_uv_mini[vi] = np.array(uv_layer[li].uv)
            face_mapping[str(face)] = vertex_to_uv_mini

        for i, f in enumerate(mesh.faces):
            key = str(sorted(f))
            face_id_to_uv_mapping[i] = face_mapping[key]

        blender_uv = face_id_to_uv_mapping
        with open(blender_path, 'wb') as f:
            pickle.dump(blender_uv, f)
        # Exit Blender
        bpy.ops.wm.quit_blender()
    else:
        with open(blender_path, 'rb') as f:
            blender_uv = pickle.load(f)
    return blender_uv


def get_mapping_gt_via_blender(mesh, split, config):
    import bpy

    obj_file = config["data"]["mesh_path"]
    blender_path = str(Path(config["data"]["preproc_data_path"]) / split / "blender_uv.pkl")
    if not os.path.isfile(blender_path):

        # Clear existing scene data
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import OBJ file
        bpy.ops.wm.obj_import(filepath=obj_file)

        # Parse uv
        # TODO: Technically everything below here is a duplicate
        me = bpy.context.object.data
        uv_layer = me.uv_layers.active.data

        face_id_to_uv_mapping = {}
        face_mapping = {}
        for poly in me.polygons:
            print("Polygon", poly.index)

            # Find out what face we are
            face = sorted([v for v in poly.vertices])

            # For each particular face we get its own mapping
            vertex_to_uv_mini = {}
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                vertex_to_uv_mini[vi] = np.array(uv_layer[li].uv)
            face_mapping[str(face)] = vertex_to_uv_mini

        for i, f in enumerate(mesh.faces):
            key = str(sorted(f))
            face_id_to_uv_mapping[i] = face_mapping[key]

        blender_uv = face_id_to_uv_mapping
        with open(blender_path, 'wb') as f:
            pickle.dump(blender_uv, f)
        # Exit Blender
        bpy.ops.wm.quit_blender()
    else:
        with open(blender_path, 'rb') as f:
            blender_uv = pickle.load(f)
    return blender_uv


def get_mapping(mesh, split, config):
    """
    TODO: Implement the function to unwrap the mesh and return the mapping for UV coordinates.

    This function is supposed to unwrap the mesh using an appropriate library (e.g., xatlas-python)
    and return the mapping. The mapping can then be stored and later used in the `map_to_UV` function.

    Args:
        mesh: The mesh object to be unwrapped. This could be in a format compatible with the chosen
              unwrapping library, such as a Trimesh object.

    Returns:
        mapping: The UV mapping of the mesh. This could be a data structure that associates each vertex or
                 face of the mesh with its corresponding UV coordinates. Preferably, the mapping should be
                 in a format that can be easily used in the `map_to_UV` function.
    """
    xatlas_path = str(Path(config["data"]["preproc_data_path"]) / "xatlas.obj")
    v_path = str(Path(config["data"]["preproc_data_path"]) / "new_vertex_id_to_old_vertex_id.npy")
    rv_path = str(Path(config["data"]["preproc_data_path"]) / "old_vertex_to_new_vertexes.pkl")
    uv_path = str(Path(config["data"]["preproc_data_path"]) / "uv.pkl")
    if not os.path.isfile(xatlas_path):
        # Extract with xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(mesh.vertices, mesh.faces)

        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        pack_options.padding = config["preprocessing"]["uv_backend_options"]["xatlas"]["padding"]

        print("Generating chart...")
        atlas.generate(chart_options, pack_options)
        print("Complete.")

        vmapping, indices, uvs = atlas[0]
        # vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        np.save(v_path, vmapping, allow_pickle=False)

        # maps original vertex to new vertex_id (in case we need it)
        reverse_vmapping = {}
        for left, right in zip(range(len(vmapping)), vmapping):
            reverse_vmapping.setdefault(right, []).append(left)
        with open(rv_path, "wb") as f:
            pickle.dump(reverse_vmapping, f)

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
    return new_mesh  # return mesh object with the uv mapping


def map_to_UV_blender(point_barys, face_idxs, vertex_idxs_of_hit_faces, mapping_dict):
    """
    TODO: Implement the function to map bary points (point_bary) to 2D UV coordinates.

    This function should take a 3D point or a list of 3D points represented in
    Cartesian coordinates (x, y, z) and map them to 2D UV coordinates. The exact
    transformation will depend on the specifics of the UV mapping, which might
    involve perspective projection, orthographic projection, or another method
    suited to the particular application.

    Args:
        point_xyz (tuple or list of tuples): A tuple (x, y, z) representing a 3D point or a list of such tuples.
        Preferably torch tensors or numpy arrays!
        mapping (TODO define): The mapping in some way. Might be a function or an array or so that maps xyz points to 2d coordinates. Need to check how we implement this

    Returns:
        tuple or list of tuples: A tuple (u, v) representing the 2D UV coordinates or a list of such tuples.
        Return torch tensors if possible
    """

    uvs = []
    for fid, vids, in zip(face_idxs, vertex_idxs_of_hit_faces):
        mapping = mapping_dict.get(int(fid))
        uv = np.array([mapping.get(int(v)) for v in vids])
        assert None not in uv, "something went wrong."
        uvs.append(uv)

    uv_vertices_of_hit_faces = np.stack(uvs)
    # Note: we can simply use the same barycentric coords since its all linear
    # FIXME: torch->np->torch is shit
    uv_coords = torch.from_numpy(np.sum(point_barys.numpy()[:, :, np.newaxis] * uv_vertices_of_hit_faces, axis=1))

    return uv_coords


def map_to_UV(point_barys, face_idxs, mesh_with_mapping):
    """
    TODO: Implement the function to map bary points (point_bary) to 2D UV coordinates.

    This function should take a 3D point or a list of 3D points represented in
    Cartesian coordinates (x, y, z) and map them to 2D UV coordinates. The exact
    transformation will depend on the specifics of the UV mapping, which might
    involve perspective projection, orthographic projection, or another method
    suited to the particular application.

    Args:
        point_xyz (tuple or list of tuples): A tuple (x, y, z) representing a 3D point or a list of such tuples.
        Preferably torch tensors or numpy arrays!
        mapping (TODO define): The mapping in some way. Might be a function or an array or so that maps xyz points to 2d coordinates. Need to check how we implement this

    Returns:
        tuple or list of tuples: A tuple (u, v) representing the 2D UV coordinates or a list of such tuples.
        Return torch tensors if possible
    """

    indices = mesh_with_mapping.faces
    vids_of_hit_faces = indices[face_idxs]
    texture_visual = mesh_with_mapping.visual
    uv_vertices_of_hit_faces = np.array(texture_visual.uv[vids_of_hit_faces])
    # Note: we can simply use the same barycentric coords since its all linear
    # FIXME: torch->np->torch is shit
    uv_coords = torch.from_numpy(np.sum(point_barys.numpy()[:, :, np.newaxis] * uv_vertices_of_hit_faces, axis=1))

    return uv_coords


def export_uv(model, path, resolution=(700, 700), n_channels=3, device="cuda"):
    img_shape = resolution + torch.Size([n_channels])

    half_dx = 0.5 / resolution[0]
    half_dy = 0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((yv.flatten(), xv.flatten())).t()

    print(f"Writing uv: '{path}'... ", end="")
    with torch.no_grad():
        if model.model.n_input_dims != 2:  # TODO: THIS IS TEMPORARY
            xy = torch.cat((xy, (torch.ones(len(xy)) * 0.5).to(xy.device).unsqueeze(1)), dim=1)
        rgbs = np.flipud(model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
        rgbs_scaled = (rgbs * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite(path, rgbs_scaled)

    print("done.")


def export_reference_image(dataset, path, resolution, device="cpu"):
    # Get predictions for all that we have in dataset
    input = dataset.uv.to(device)
    pixel_xy = (input * torch.tensor(resolution, device=device)).long()
    gt = dataset.rgb.to(device)
    # Multiply predictions by 255 and convert to int in one step
    scaled_gt = (gt * 255).type(torch.uint8)

    # Flattened indices
    indices_px = pixel_xy[:, 1] * resolution[0] + pixel_xy[:, 0]

    # Initialize accumulators and count arrays
    summed_values = torch.zeros((resolution[0] * resolution[1], 3), dtype=torch.float32, device=pixel_xy.device)
    counts = torch.zeros((resolution[0] * resolution[1],), dtype=torch.int32, device=pixel_xy.device)

    # Use scatter_add to sum values and count occurrences
    summed_values.index_add_(0, indices_px, scaled_gt.float())
    counts.index_add_(0, indices_px, torch.ones_like(indices_px, dtype=torch.int32))

    # Avoid division by zero
    nonzero_mask = counts > 0
    summed_values[nonzero_mask] = summed_values[nonzero_mask] / counts[nonzero_mask].unsqueeze(1)
    summed_values = summed_values.type(torch.uint8).reshape(*resolution, 3)

    # Note: If we flip here we are in the right orientation for the output map if we visualize!!
    imageio.imwrite(path, np.flipud(summed_values.cpu().numpy()))


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def normalize_values(values, min_val, max_val):
    """ Normalize values to the range [0, 1] """
    return (values - min_val) / (max_val - min_val)


def load_np(path, allow_not_exists):
    try:
        return np.load(path)
    except FileNotFoundError as e:
        if allow_not_exists:
            return None
        raise e
