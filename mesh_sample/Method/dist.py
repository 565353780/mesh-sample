import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def toMinVertexDist(mesh: o3d.geometry.TriangleMesh) -> float:
    vertices = np.asarray(mesh.vertices)

    tree = cKDTree(vertices)

    distances, indices = tree.query(vertices, k=2)

    min_nonzero_distances = distances[:, 1]

    min_distance = np.min(min_nonzero_distances)
    return min_distance
