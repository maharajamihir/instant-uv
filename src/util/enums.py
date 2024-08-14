from enum import Enum


class CacheFilePaths(Enum):
    FACE_IDXS = "face_idxs.npy"
    VIDS_OF_HIT_FACES = "vids_of_hit_faces.npy"
    BARYCENTRIC_COORDS = "barycentric_coords.npy"
    UV_COORDS = "uv_coords.npy"
    EXPECTED_RGBS = "expected_rgbs.npy"
    UNIT_RAY_DIRS = "unit_ray_dirs.npy"
    ANGLES = "angles.npy"
    ANGLES2 = "angles2.npy"
    COORDS3D = "coords3d.npy"
