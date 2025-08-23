import numpy as np

from mesh_sample.Module.mesh_subdiver import MeshSubdiver
from mesh_sample.Method.sample import toSubdivMesh
from mesh_sample.Method.render import renderMeshEdges

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

mesh_subdiver = MeshSubdiver(subdiv_mesh, dist_max * 0.1)
subdiv_mesh = mesh_subdiver.createSubdivMesh()

renderMeshEdges(subdiv_mesh)
