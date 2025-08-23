import numpy as np
import matplotlib.pyplot as plt

from mesh_sample.Method.rotate import getRotationAndTranslate
from mesh_sample.Method.sample import toSubdivMesh

dist_max = 0.1

vertices = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.2, 0.0],
    ]
)

subdiv_mesh = toSubdivMesh(vertices, dist_max)
assert subdiv_mesh is not None, "toSubdivMesh failed!"

vertices = np.asarray(subdiv_mesh.vertices)
triangles = np.asarray(subdiv_mesh.triangles)

R, T = getRotationAndTranslate(vertices)

uni_v = (vertices + T) @ R.T

# -------- Step 5: 2D 可视化 --------
plt.figure(figsize=(6, 6))
plt.triplot(uni_v[:, 0], uni_v[:, 1], triangles, color="black", linewidth=0.8)
plt.plot(uni_v[:, 0], uni_v[:, 1], "o", color="red", markersize=3)
plt.title("Delaunay Triangulation in a Single Triangle")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.show()
