import os
import open3d as o3d

from mesh_sample.Method.render import renderMeshEdges, renderTriangleNormals
from mesh_sample.Module.mesh_subdiver import MeshSubdiver


def demo():
    mesh_file_path = "/Users/chli/chLi/Dataset/Famous/bunny-v2.ply"
    # mesh_file_path = "/Users/chli/chLi/Dataset/BitAZ/mesh/BitAZ.ply"
    dist_max = 1.0 / 100
    # dist_max = float("inf")
    print_progress = True
    save_mesh_file_path = "/Users/chli/chLi/Dataset/BitAZ/subdiv_mesh/BitAZ.ply"
    save_mesh_file_path = None

    if not os.path.exists(mesh_file_path):
        print("mesh file not exist!")
        print("mesh_file_path:", mesh_file_path)
        return False

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    mesh_subdiver = MeshSubdiver(mesh, dist_max, print_progress)
    subdiv_mesh = mesh_subdiver.createSubdivMesh(save_mesh_file_path)

    renderMeshEdges(subdiv_mesh)
    renderTriangleNormals(subdiv_mesh)
    return True
