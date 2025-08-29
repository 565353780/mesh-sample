import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from mesh_sample.Method.rotate import getRAndT


def meshToLineSet(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
    edges = set()
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edges.add(edge)

    edge_list = np.array(list(edges))
    lineset = o3d.geometry.LineSet()
    lineset.points = mesh.vertices
    lineset.lines = o3d.utility.Vector2iVector(edge_list)
    return lineset


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


def renderMeshEdges(mesh: o3d.geometry.TriangleMesh) -> bool:
    mesh.compute_vertex_normals()

    lineset = meshToLineSet(mesh)

    o3d.visualization.draw_geometries([mesh, lineset])

    return True


def renderTriangleNormals(
    mesh: o3d.geometry.TriangleMesh,
    normal_length: float = 0.01,
) -> bool:
    mesh.compute_triangle_normals()

    # 获取三角形的中心点
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_centers = np.mean(vertices[triangles], axis=1)

    # 获取法向量
    triangle_normals = np.asarray(mesh.triangle_normals)

    # 创建法向箭头（小线段）用于可视化
    line_set = o3d.geometry.LineSet()

    points = []
    lines = []
    colors = []

    for i in range(len(triangle_centers)):
        start = triangle_centers[i]
        end = start + triangle_normals[i] * normal_length
        points.append(start)
        points.append(end)
        lines.append([2 * i, 2 * i + 1])
        colors.append([1, 0, 0])  # 红色箭头

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 可视化：网格 + 法向线段
    o3d.visualization.draw_geometries([mesh, line_set])
    return True
