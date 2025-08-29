import numpy as np
import open3d as o3d


def normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)

        if norm == 0:
            return vectors

        return vectors / norm

    if vectors.ndim == 2:
        norm = np.linalg.norm(vectors, axis=1)
        normed_vectors = np.zeros_like(vectors)

        valid_norm_idxs = np.where(norm > 0)[0]

        normed_vectors[valid_norm_idxs] = vectors[valid_norm_idxs] / norm[
            valid_norm_idxs
        ].reshape(-1, 1)

        return normed_vectors

    print("[ERROR][normal::normalize]")
    print("\t vectors dim not valid!")
    print("\t vectors.shape:", vectors.shape)
    return np.array([])


def updateVertexNormals(mesh: o3d.geometry.TriangleMesh) -> bool:
    mesh.compute_vertex_normals()
    return True


def updateTriangleNormals(mesh: o3d.geometry.TriangleMesh) -> bool:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    normal_vectors_1 = np.cross(e0, -e2)
    normal_vectors_2 = np.cross(e1, -e0)
    normal_vectors_3 = np.cross(e2, -e1)

    norm_1 = np.linalg.norm(normal_vectors_1, axis=1)
    norm_2 = np.linalg.norm(normal_vectors_2, axis=1)
    norm_3 = np.linalg.norm(normal_vectors_3, axis=1)

    norms = np.stack([norm_1, norm_2, norm_3], axis=1)

    max_indices = np.argmax(norms, axis=1)

    stacked = np.stack([normal_vectors_1, normal_vectors_2, normal_vectors_3], axis=1)
    max_normal_vectors = stacked[np.arange(normal_vectors_1.shape[0]), max_indices]

    norm = np.linalg.norm(max_normal_vectors, axis=1)
    print(np.min(norm))

    normals = normalize(max_normal_vectors)

    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
    return True
