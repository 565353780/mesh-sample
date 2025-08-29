import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union, Tuple
from joblib import Parallel, delayed

from mesh_sample.Method.sample import sampleInnerPoints
from mesh_sample.Module.tqdm_joblib import tqdm_joblib


class InnerPoints(object):
    def __init__(
        self,
        mesh: Union[o3d.geometry.TriangleMesh, None] = None,
        dist_max: float = 0.1,
        print_progress: bool = True,
    ) -> None:
        self.dist_max = dist_max
        self.print_progress = print_progress

        self.vertices = np.array([])
        self.triangles = np.array([])

        self.inner_points = np.array([])
        self.point_triangle_idxs = np.array([])

        if mesh is not None:
            self.loadMesh(mesh, dist_max)
        return

    def createTriangleInnerPoints(
        self, triangle_idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        triangle_vertices = self.vertices[self.triangles[triangle_idx]]

        curr_inner_points = sampleInnerPoints(triangle_vertices, self.dist_max)

        if curr_inner_points is None:
            return None, None

        curr_point_triangle_idxs = (
            np.ones(curr_inner_points.shape[0], dtype=np.int32) * triangle_idx
        )
        return curr_inner_points, curr_point_triangle_idxs

    def createInnerPoints(self) -> bool:
        inner_points = []
        point_triangle_idxs = []

        for_data = range(self.triangles.shape[0])
        if self.print_progress:
            for_data = tqdm(for_data)
            print("[INFO][InnerPoints::createInnerPoints]")
            print("\t start create inner points...")
        for i in for_data:
            curr_inner_points, curr_point_triangle_idxs = (
                self.createTriangleInnerPoints(i)
            )

            if curr_inner_points is None:
                continue

            inner_points.append(curr_inner_points)
            point_triangle_idxs.append(curr_point_triangle_idxs)

        if len(inner_points) == 0:
            return True

        self.inner_points = np.vstack(inner_points)
        self.point_triangle_idxs = np.hstack(point_triangle_idxs)

        return True

    def createInnerPointsJoblib(self, n_jobs: int = os.cpu_count()) -> bool:
        inner_points = []
        point_triangle_idxs = []

        if self.print_progress:
            print("[INFO][InnerPoints::createInnerPointsMP]")
            print("\t start create inner points...")
            with tqdm(total=self.triangles.shape[0]) as progress:
                with tqdm_joblib(progress):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(self.createTriangleInnerPoints)(i)
                        for i in range(self.triangles.shape[0])
                    )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.createTriangleInnerPoints)(i)
                for i in range(self.triangles.shape[0])
            )

        for result in results:
            if result is None:
                continue

            curr_inner_points, curr_point_triangle_idxs = result

            if curr_inner_points is None:
                continue

            inner_points.append(curr_inner_points)
            point_triangle_idxs.append(curr_point_triangle_idxs)

        if len(inner_points) == 0:
            return True

        self.inner_points = np.vstack(inner_points)
        self.point_triangle_idxs = np.hstack(point_triangle_idxs)

        return True

    def loadMesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        dist_max: Union[float, None] = None,
        n_jobs: int = os.cpu_count(),
    ) -> bool:
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)

        if dist_max is not None:
            self.dist_max = dist_max

        # self.createInnerPoints()
        self.createInnerPointsJoblib(n_jobs)

        return True

    @property
    def innerPointNum(self) -> int:
        return self.inner_points.shape[0]

    def getTriangleInnerPointIdxs(self, triangle_idx: int) -> Union[np.ndarray, None]:
        if self.inner_points.shape[0] == 0:
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
