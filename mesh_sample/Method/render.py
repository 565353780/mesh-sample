import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from mesh_sample.Method.rotate import getRAndT


def renderSubdivMesh(subdiv_mesh: o3d.geometry.TriangleMesh) -> bool:
    vertices = np.asarray(subdiv_mesh.vertices)
    triangles = np.asarray(subdiv_mesh.triangles)

    R, T = getRAndT(vertices)

    uni_v = (vertices + T) @ R.T

    # -------- Step 5: 2D 可视化 --------
    plt.figure(figsize=(6, 6))
    plt.triplot(uni_v[:, 0], uni_v[:, 1], triangles, color="black", linewidth=0.8)
    plt.plot(uni_v[:, 0], uni_v[:, 1], "o", color="red", markersize=3)
    plt.title("Delaunay Triangulation in a Single Triangle")
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.show()
    return True
