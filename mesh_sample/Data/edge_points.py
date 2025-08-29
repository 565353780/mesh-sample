import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union

from mesh_sample.Data.edges import Edges
from mesh_sample.Method.sample import sampleEdgePoints


class EdgePoints(object):
    def __init__(
        self,
        mesh: Union[o3d.geometry.TriangleMesh, None] = None,
        dist_max: float = 0.1,
        print_progress: bool = True,
    ) -> None:
        self.dist_max = dist_max
        self.print_progress = print_progress

        self.vertices = np.array([])

        self.edge_points = np.array([])
        self.point_edge_idxs = np.array([])

        self.edges = Edges()

        if mesh is not None:
            self.loadMesh(mesh, dist_max)
        return

    def createEdgePoints(
        self, edge_idx: int, dist_max: float
    ) -> Union[np.ndarray, None]:
        if self.vertices is None:
            print("[ERROR][EdgePoints::createEdgePoints]")
            print("\t vertices is None! please load mesh first!")
            return None

        edge_point_idxs = self.edges.getEdgePointIdxs(edge_idx)
        if edge_point_idxs is None:
            print("[ERROR][EdgePoints::createEdgePoints]")
            print("\t getEdgePointIdxs failed:")
            print("\t edge_idx:", edge_idx)
            return None

        edge_points = self.vertices[edge_point_idxs]

        sampled_edge_points = sampleEdgePoints(edge_points[0], edge_points[1], dist_max)

        return sampled_edge_points

    def loadMesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        dist_max: Union[float, None] = None,
    ) -> bool:
        if dist_max is not None:
            self.dist_max = dist_max

        self.vertices = np.asarray(mesh.vertices)

        triangles = np.asarray(mesh.triangles)
        self.edges.loadTriangles(triangles)

        edge_points = []
        point_edge_idxs = []

        for_data = range(self.edges.edgeNum)
        if self.print_progress:
            for_data = tqdm(for_data)
            print("[INFO][EdgePoints::loadMesh]")
            print("\t start create edge points...")
        for edge_idx in for_data:
            curr_edge_points = self.createEdgePoints(edge_idx, self.dist_max)

            if curr_edge_points is None:
                continue

            edge_points.append(curr_edge_points)

            point_edge_idxs.append(
                np.ones(curr_edge_points.shape[0], dtype=np.int32) * edge_idx
            )

        if len(edge_points) == 0:
            return True

        self.edge_points = np.vstack(edge_points)
        self.point_edge_idxs = np.hstack(point_edge_idxs)
        return True

    @property
    def edgePointNum(self) -> int:
        return self.edge_points.shape[0]

    def getEdgePointIdxs(self, edge_idx: int) -> Union[np.ndarray, None]:
        edge_point_idxs = np.where(self.point_edge_idxs == edge_idx)[0]
        if edge_point_idxs.shape[0] == 0:
            return None

        return edge_point_idxs

    def getEdgePoints(self, edge_idx: int) -> Union[np.ndarray, None]:
        edge_point_idxs = self.getEdgePointIdxs(edge_idx)
        if edge_point_idxs is None:
            return None

        return self.edge_points[edge_point_idxs]

    def getTriangleEdgePointIdxs(self, triangle_idx: int) -> Union[np.ndarray, None]:
        triangle_edge_idxs = self.edges.getTriangleEdgeIdxs(triangle_idx)
        if triangle_edge_idxs is None:
            print("[ERROR][EdgePoints::getTriangleEdgePointIdxs]")
            print("\t getTriangleEdgeIdxs failed!")
            return None

        triangle_edge_point_idxs = []
        for edge_idx in triangle_edge_idxs:
            edge_point_idxs = self.getEdgePointIdxs(edge_idx)
            if edge_point_idxs is None:
                continue

            triangle_edge_point_idxs.append(edge_point_idxs)

        if len(triangle_edge_point_idxs) == 0:
            return None

        triangle_edge_point_idxs = np.hstack(triangle_edge_point_idxs)
        return triangle_edge_point_idxs

    def getTriangleEdgePoints(self, triangle_idx: int) -> Union[np.ndarray, None]:
        triangle_edge_point_idxs = self.getTriangleEdgePointIdxs(triangle_idx)
        if triangle_edge_point_idxs is None:
            return None

        triangle_edge_points = self.edge_points[triangle_edge_point_idxs]
        return triangle_edge_points
