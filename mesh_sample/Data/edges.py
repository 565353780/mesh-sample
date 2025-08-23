import numpy as np
from typing import Union

from mesh_sample.Method.edge import toUniqueEdgesWithTriangleEdgeMap


class Edges(object):
    def __init__(
        self,
        triangles: Union[np.ndarray, None] = None,
    ) -> None:
        self.unique_edges = np.array([])
        self.triangle_edge_map = np.array([])

        if triangles is not None:
            self.loadTriangles(triangles)
        return

    def loadTriangles(self, triangles: np.ndarray) -> bool:
        self.unique_edges, self.triangle_edge_map = toUniqueEdgesWithTriangleEdgeMap(
            triangles
        )
        return True

    @property
    def edgeNum(self) -> int:
        return self.unique_edges.shape[0]

    def getEdgePointIdxs(self, edge_idx: int) -> Union[np.ndarray, None]:
        if self.unique_edges.shape[0] == 0:
            print("[ERROR][Edges::getEdgePointIdxs]")
            print("\t unique edges is empty! please load triangles first!")
            return None

        if edge_idx < 0 or edge_idx >= self.unique_edges.shape[0]:
            print("[ERROR][Edges::getEdgePointIdxs]")
            print("\t edge idx out of range!")
            print("\t edge_idx:", edge_idx)
            return None

        edge_point_idxs = self.unique_edges[edge_idx]
        return edge_point_idxs

    def getTriangleEdgeIdxs(self, triangle_idx: int) -> Union[np.ndarray, None]:
        if self.triangle_edge_map.shape[0] == 0:
            print("[ERROR][Edges::getTriangleEdgeIdxs]")
            print("\t triangle edge map is empty! please load triangles first!")
            return None

        if triangle_idx < 0 or triangle_idx >= self.triangle_edge_map.shape[0]:
            print("[ERROR][Edges::getTriangleEdgeIdxs]")
            print("\t triangle idx out of range!")
            print("\t triangle_idx:", triangle_idx)
            return None

        return self.triangle_edge_map[triangle_idx]

    def getTriangleEdgePointIdxs(self, triangle_idx: int) -> Union[np.ndarray, None]:
        triangle_edge_idxs = self.getTriangleEdgeIdxs(triangle_idx)
        if triangle_edge_idxs is None:
            print("[ERROR][Edges::getTriangleEdgePointIdxs]")
            print("\t getTriangleEdgeIdxs failed!")
            return None

        if self.unique_edges is None:
            print("[ERROR][Edges::getTriangleEdgeIdxs]")
            print("\t unique edges is None! please load triangles first!")
            return None

        triangle_edge_point_idxs = self.unique_edges[triangle_edge_idxs]
        return triangle_edge_point_idxs
