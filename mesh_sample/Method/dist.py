import numpy as np
from scipy.spatial import cKDTree


def toMinNeighboorDist(points: np.ndarray) -> float:
    tree = cKDTree(points)

    distances, _ = tree.query(points, k=2)

    min_nonzero_distances = distances[:, 1]

    min_distance = np.min(min_nonzero_distances)
    return min_distance
