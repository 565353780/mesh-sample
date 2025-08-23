import open3d as o3d


def toMaxBound(mesh: o3d.geometry.TriangleMesh) -> float:
    aabb = mesh.get_axis_aligned_bounding_box()
    max_bound = aabb.get_max_bound()
    min_bound = aabb.get_min_bound()
    bound = max_bound - min_bound
    max_bound = max(bound)

    return max_bound
