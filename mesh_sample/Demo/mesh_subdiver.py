import os
import open3d as o3d

from mesh_sample.Method.scale import toMaxBound
from mesh_sample.Method.render import renderMeshEdges
from mesh_sample.Module.mesh_subdiver import MeshSubdiver


def demo():
    mesh_file_path = "/Users/chli/chLi/Dataset/BitAZ/mesh/BitAZ.ply"
    dist_max = 0.1

    if not os.path.exists(mesh_file_path):
        print("mesh file not exist!")
        print("mesh_file_path:", mesh_file_path)
        return False

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    max_bound = toMaxBound(mesh)

    mesh_subdiver = MeshSubdiver(mesh, max_bound * dist_max)
    subdiv_mesh = mesh_subdiver.createSubdivMesh()

    renderMeshEdges(subdiv_mesh)
    return True
