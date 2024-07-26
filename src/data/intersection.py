from typing import Tuple

import numpy as np


from numba import jit


@jit
def any_intersection(
        ray_origins: np.ndarray, ray_directions: np.ndarray, triangles_vertices: np.ndarray
) -> np.ndarray:
    """
    MODIFIED BY MORITZ:!!!!


    Möller–Trumbore intersection algorithm for calculating whether the ray intersects the triangle
    and for which t-value. Based on: https://github.com/kliment/Printrun/blob/master/printrun/stltool.py,
    which is based on:
    http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    Parameters
    ----------
    ray_origin : np.ndarray(3)
        Origin coordinate (x, y, z) from which the ray is fired
    ray_directions : np.ndarray(n, 3)
        Directions (dx, dy, dz) in which the rays are going
    triangle_vertices : np.ndarray(m, 3, 3)
        3D vertices of multiple triangles

    Returns
    -------
    tuple[np.ndarray<bool>(n, m), np.ndarray(n, m)]
        The first array indicates whether or not there was an intersection, the second array
        contains the t-values of the intersections
    """

    output_shape = (len(ray_directions))

    all_rays_intersected = np.full(output_shape, True)

    v1 = triangles_vertices[:, 0]
    v2 = triangles_vertices[:, 1]
    v3 = triangles_vertices[:, 2]

    eps = 0.000001

    edge1 = v2 - v1
    edge2 = v3 - v1

    for i, ray in enumerate(ray_directions):
        all_t = np.zeros((len(triangles_vertices)))
        intersected = np.full((len(triangles_vertices)), True)

        pvec = np.cross(ray, edge2)

        det = np.sum(edge1 * pvec, axis=1)

        non_intersecting_original_indices = np.absolute(det) < eps

        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        inv_det = 1.0 / det

        ray_origin = ray_origins[i]
        tvec = ray_origin - v1

        u = np.sum(tvec * pvec, axis=1) * inv_det

        non_intersecting_original_indices = (u < 0.0) + (u > 1.0)
        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        qvec = np.cross(tvec, edge1)

        v = np.sum(ray * qvec, axis=1) * inv_det

        non_intersecting_original_indices = (v < 0.0) + (u + v > 1.0)

        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        t = (
                np.sum(
                    edge2 * qvec,
                    axis=1,
                )
                * inv_det
        )

        non_intersecting_original_indices = t < eps
        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        intersecting_original_indices = np.invert(non_intersecting_original_indices)
        all_t[intersecting_original_indices] = t[intersecting_original_indices]

        all_rays_intersected[i] = np.any(intersected)

    return all_rays_intersected


# from typing import Tuple
#
# import numpy as np
#
#
# # from numba import jit
# #
# #
# # @jit
# def any_intersection(
#         ray_origins: np.ndarray, ray_directions: np.ndarray, triangles_vertices: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     MODIFIED BY MORITZ:!!!!
#
#
#     Möller–Trumbore intersection algorithm for calculating whether the ray intersects the triangle
#     and for which t-value. Based on: https://github.com/kliment/Printrun/blob/master/printrun/stltool.py,
#     which is based on:
#     http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
#
#     Parameters
#     ----------
#     ray_origin : np.ndarray(3)
#         Origin coordinate (x, y, z) from which the ray is fired
#     ray_directions : np.ndarray(n, 3)
#         Directions (dx, dy, dz) in which the rays are going
#     triangle_vertices : np.ndarray(m, 3, 3)
#         3D vertices of multiple triangles
#
#     Returns
#     -------
#     tuple[np.ndarray<bool>(n, m), np.ndarray(n, m)]
#         The first array indicates whether or not there was an intersection, the second array
#         contains the t-values of the intersections
#     """
#
#     output_shape = (len(ray_directions), len(triangles_vertices))
#
#     all_rays_t = np.zeros(output_shape)
#     all_rays_intersected = np.full(output_shape, True)
#
#     v1 = triangles_vertices[:, 0]
#     v2 = triangles_vertices[:, 1]
#     v3 = triangles_vertices[:, 2]
#
#     eps = 0.000001
#
#     edge1 = v2 - v1
#     edge2 = v3 - v1
#
#     for i, ray in enumerate(ray_directions):
#         all_t = np.zeros((len(triangles_vertices)))
#         intersected = np.full((len(triangles_vertices)), True)
#
#         pvec = np.cross(ray, edge2)
#
#         det = np.sum(edge1 * pvec, axis=1)
#
#         non_intersecting_original_indices = np.absolute(det) < eps
#
#         all_t[non_intersecting_original_indices] = np.nan
#         intersected[non_intersecting_original_indices] = False
#
#         inv_det = 1.0 / det
#
#         ray_origin = ray_origins[i]
#         tvec = ray_origin - v1
#
#         u = np.sum(tvec * pvec, axis=1) * inv_det
#
#         non_intersecting_original_indices = (u < 0.0) + (u > 1.0)
#         all_t[non_intersecting_original_indices] = np.nan
#         intersected[non_intersecting_original_indices] = False
#
#         qvec = np.cross(tvec, edge1)
#
#         v = np.sum(ray * qvec, axis=1) * inv_det
#
#         non_intersecting_original_indices = (v < 0.0) + (u + v > 1.0)
#
#         all_t[non_intersecting_original_indices] = np.nan
#         intersected[non_intersecting_original_indices] = False
#
#         t = (
#                 np.sum(
#                     edge2 * qvec,
#                     axis=1,
#                 )
#                 * inv_det
#         )
#
#         non_intersecting_original_indices = t < eps
#         all_t[non_intersecting_original_indices] = np.nan
#         intersected[non_intersecting_original_indices] = False
#
#         intersecting_original_indices = np.invert(non_intersecting_original_indices)
#         all_t[intersecting_original_indices] = t[intersecting_original_indices]
#
#         all_rays_t[i] = all_t
#         all_rays_intersected[i] = intersected
#
#     return all_rays_intersected, all_rays_t



""" TEST """
# triangle_vertices = np.array([
#     [1, 1, 0],  # Vertex 1
#     [4, 1, 0],  # Vertex 2
#     [1, 5, 0]  # Vertex 3
# ]).reshape(1, 3, 3)  # Reshaping to (1, 3, 3)
#
# # Define the ray origin slightly behind the triangle and direction pointing towards the XY plane
# ray_origins = np.array([[2, 2, -1]])  # Shape (1, 3)
# ray_directions = np.array([[0, 0, 1]])  # Shape (1, 3)
#
# any_intersection(mesh_vertices[mesh_faces[0]][0:1], ray_dirs[0:1], mesh_vertices[mesh_faces[0:2]])