import numpy as np
from typing import Tuple


def toUniqueEdgesWithTriangleEdgeMap(
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num_triangles = triangles.shape[0]

    # (M, 3, 2)
    tri_edges = np.stack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1
    )

    # (M*3, 2)
    all_edges = tri_edges.reshape(-1, 2)

    all_edges_sorted = np.sort(all_edges, axis=1)

    edges_unique, inverse_indices = np.unique(
        all_edges_sorted, axis=0, return_inverse=True
    )

    # (M, 3)
    edge_indices_per_triangle = inverse_indices.reshape(num_triangles, 3)

    return edges_unique, edge_indices_per_triangle
