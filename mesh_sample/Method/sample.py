import numpy as np
import open3d as o3d
from math import floor
from typing import Union
from scipy.spatial import Delaunay

from mesh_sample.Config.constant import K
from mesh_sample.Method.normal import normalize
from mesh_sample.Method.rotate import getRAndT


def sampleEdgePoints(
    p1: np.ndarray, p2: np.ndarray, dist_max: float
) -> Union[np.ndarray, None]:
    edge_length = np.linalg.norm(p2 - p1)
    if edge_length == 0:
        return None

    sample_point_num = floor(edge_length / dist_max)
    if sample_point_num < 1:
        return None

    return np.linspace(p1, p2, sample_point_num + 2)[1:-1]


def sampleBoundPoints(vertices: np.ndarray, dist_max: float) -> Union[np.ndarray, None]:
    edge_points_1 = sampleEdgePoints(vertices[0], vertices[1], dist_max)
    edge_points_2 = sampleEdgePoints(vertices[1], vertices[2], dist_max)
    edge_points_3 = sampleEdgePoints(vertices[2], vertices[0], dist_max)

    valid_edge_points = []
    if edge_points_1 is not None:
        valid_edge_points.append(edge_points_1)
    if edge_points_2 is not None:
        valid_edge_points.append(edge_points_2)
    if edge_points_3 is not None:
        valid_edge_points.append(edge_points_3)

    if len(valid_edge_points) == 0:
        return None

    if len(valid_edge_points) == 1:
        return valid_edge_points[0]

    bound_points = np.vstack(valid_edge_points)
    return bound_points


def toTriangleArea(v1: np.ndarray, v2: np.ndarray) -> float:
    cross = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross)
    return float(area)


def sampleInnerPoints(vertices: np.ndarray, dist_max: float) -> Union[np.ndarray, None]:
    if vertices.shape[0] != 3:
        print("[ERROR][sample::toSubdivMesh]")
        print("\t vertices.shape != [3, 3]!")
        print("\t vertices.shape:", vertices.shape)
        return None

    area = toTriangleArea(vertices[1] - vertices[0], vertices[2] - vertices[0])
    assert area > 0

    sample_point_num = floor(area / K / dist_max / dist_max)
    if sample_point_num < 1:
        return None

    center = np.mean(vertices, axis=0)

    if sample_point_num == 1:
        return center.reshape(1, 3)

    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    move_vectors = center - vertices
    move_dists = np.linalg.norm(move_vectors, axis=1, keepdims=True)
    max_move_dists = np.max(move_dists)
    if max_move_dists <= dist_max:
        return None

    valid_move_dists = np.minimum(move_dists, dist_max)

    scaled_vertices = vertices + normalize(move_vectors) * valid_move_dists
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    inner_pcd = mesh.sample_points_poisson_disk(sample_point_num)
    inner_points = np.asarray(inner_pcd.points)
    return inner_points


def toSubdivMesh(
    vertices: np.ndarray, dist_max: float
) -> Union[o3d.geometry.TriangleMesh, None]:
    bound_points = sampleBoundPoints(vertices, dist_max)
    inner_points = sampleInnerPoints(vertices, dist_max)

    merge_points = []
    if bound_points is not None:
        merge_points.append(bound_points)
    if inner_points is not None:
        merge_points.append(inner_points)

    if len(merge_points) == 0:
        subdiv_mesh = o3d.geometry.TriangleMesh()
        subdiv_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        subdiv_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        return subdiv_mesh

    merge_points.append(vertices)

    if len(merge_points) == 1:
        merge_points = merge_points[0]

    merge_points = np.vstack(merge_points)

    R, T = getRAndT(vertices)

    uni_points = (merge_points + T) @ R.T

    tri = Delaunay(uni_points[:, :2])
    triangles = tri.simplices

    subdiv_mesh = o3d.geometry.TriangleMesh()
    subdiv_mesh.vertices = o3d.utility.Vector3dVector(merge_points)
    subdiv_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return subdiv_mesh
