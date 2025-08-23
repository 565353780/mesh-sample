import numpy as np
import open3d as o3d
from tqdm import trange
from typing import Union

from mesh_sample.Method.sample import sampleInnerPoints


class InnerPoints(object):
    def __init__(
        self,
        mesh: Union[o3d.geometry.TriangleMesh, None] = None,
        dist_max: float = 0.1,
    ) -> None:
        self.dist_max = dist_max

        self.inner_points = np.array([])
        self.point_triangle_idxs = np.array([])

        if mesh is not None:
            self.loadMesh(mesh, dist_max)
        return

    def loadMesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        dist_max: Union[float, None] = None,
    ) -> bool:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if dist_max is not None:
            self.dist_max = dist_max

        inner_points = []
        point_triangle_idxs = []

        print("[INFO][InnerPoints::loadMesh]")
        print("\t start create inner points...")
        for i in trange(triangles.shape[0]):
            triangle_vertices = vertices[triangles[i]]

            curr_inner_points = sampleInnerPoints(triangle_vertices, self.dist_max)
            if curr_inner_points is None:
                continue

            inner_points.append(curr_inner_points)
            point_triangle_idxs.append(
                np.ones(curr_inner_points.shape[0], dtype=np.int32) * i
            )

        if len(inner_points) == 0:
            return True

        self.inner_points = np.vstack(inner_points)
        self.point_triangle_idxs = np.hstack(point_triangle_idxs)

        return True

    @property
    def innerPointNum(self) -> int:
        return self.inner_points.shape[0]

    def getTriangleInnerPointIdxs(self, triangle_idx: int) -> Union[np.ndarray, None]:
        if self.inner_points.shape[0] == 0:
            print("[ERROR][InnerPoints::getTriangleInnerPointIdxs]")
            print("\t inner_points is empty! please load mesh first!")
            return None

        triangle_inner_point_idxs = np.where(self.point_triangle_idxs == triangle_idx)[
            0
        ]
        if triangle_inner_point_idxs.shape[0] == 0:
            return None

        return triangle_inner_point_idxs

    def getTriangleInnerPoints(self, triangle_idx: int) -> Union[np.ndarray, None]:
        triangle_inner_point_idxs = self.getTriangleInnerPointIdxs(triangle_idx)
        if triangle_inner_point_idxs is None:
            return None

        triangle_inner_points = self.inner_points[triangle_inner_point_idxs]

        return triangle_inner_points
