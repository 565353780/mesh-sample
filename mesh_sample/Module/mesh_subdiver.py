import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from scipy.spatial import Delaunay
from joblib import Parallel, delayed

from mesh_sample.Data.edge_points import EdgePoints
from mesh_sample.Data.inner_points import InnerPoints
from mesh_sample.Method.normal import toTriangleAreas, updateTriangleNormals, updateVertexNormals
from mesh_sample.Method.path import createFileFolder, renameFile
from mesh_sample.Method.dist import toMinNeighboorDist
from mesh_sample.Method.scale import toMaxBound
from mesh_sample.Method.rotate import getRAndT
from mesh_sample.Module.tqdm_joblib import tqdm_joblib

"""
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)
"""

class MeshSubdiver(object):
    def __init__(
        self,
        mesh: Union[o3d.geometry.TriangleMesh, None] = None,
        dist_max: float = 0.1,
        print_progress: bool = True,
    ) -> None:
        self.print_progress = print_progress

        self.edge_points = EdgePoints(print_progress=self.print_progress)
        self.inner_points = InnerPoints(print_progress=self.print_progress)

        self.triangles = np.array([])

        self.merge_vertices = np.array([])

        if mesh is not None:
            self.loadMesh(mesh, dist_max)
        return

    @property
    def vertices(self) -> np.ndarray:
        return self.edge_points.vertices

    @property
    def verticesNum(self) -> int:
        return self.vertices.shape[0]

    @property
    def triangleNum(self) -> int:
        return self.triangles.shape[0]

    @property
    def edgePointNum(self) -> int:
        return self.edge_points.edgePointNum

    @property
    def innerPointNum(self) -> int:
        return self.inner_points.innerPointNum

    @property
    def mergeVerticesNum(self) -> int:
        return self.merge_vertices.shape[0]

    @property
    def distMax(self) -> float:
        return self.edge_points.dist_max

    def loadMesh(self, mesh: o3d.geometry.TriangleMesh, dist_max: float) -> bool:
        self.triangles = np.asarray(mesh.triangles)

        max_bound = toMaxBound(mesh)
        weighted_dist_max = dist_max * max_bound
        # min_vertex_dist = toMinNeighboorDist(mesh)
        # weighted_dist_max = dist_max * max_bound + (1.0 - dist_max) * min_vertex_dist

        self.edge_points.loadMesh(mesh, weighted_dist_max)
        self.inner_points.loadMesh(mesh, weighted_dist_max)
        return True

    def createMergeVertices(self) -> bool:
        edge_points = self.edge_points.edge_points
        inner_points = self.inner_points.inner_points

        merge_vertices = [self.vertices]

        if edge_points.shape[0] > 0:
            merge_vertices.append(edge_points)
        if inner_points.shape[0] > 0:
            merge_vertices.append(inner_points)

        if len(merge_vertices) == 1:
            self.merge_vertices = self.vertices
            return True

        self.merge_vertices = np.vstack(merge_vertices)
        return True

    def createSubdivTriangles(self, triangle_idx: int) -> Union[np.ndarray, None]:
        if self.triangles.size == 0:
            print("[ERROR][MeshSubdiver::createSubdivTriangle]")
            print("\t triangles is empty! please load mesh first!")
            return None

        if triangle_idx < 0 or triangle_idx >= self.triangles.shape[0]:
            print("[ERROR][MeshSubdiver::createSubdivTriangle]")
            print("\t triangle idx out of range!")
            print("\t triangle_idx:", triangle_idx)
            return None

        triangle_vertex_idxs = self.triangles[triangle_idx]

        triangle_edge_point_idxs = self.edge_points.getTriangleEdgePointIdxs(
            triangle_idx
        )
        triangle_inner_point_idxs = self.inner_points.getTriangleInnerPointIdxs(
            triangle_idx
        )

        merge_vertex_idxs = [triangle_vertex_idxs]

        if triangle_edge_point_idxs is not None:
            triangle_edge_point_idxs += self.verticesNum
            merge_vertex_idxs.append(triangle_edge_point_idxs)

        if triangle_inner_point_idxs is not None:
            triangle_inner_point_idxs += self.verticesNum + self.edgePointNum
            merge_vertex_idxs.append(triangle_inner_point_idxs)

        if len(merge_vertex_idxs) == 1:
            return merge_vertex_idxs[0].reshape(1, 3)

        merge_vertex_idxs = np.hstack(merge_vertex_idxs)
        merge_vertices = self.merge_vertices[merge_vertex_idxs]

        R, T = getRAndT(merge_vertices)

        unit_vertices = (merge_vertices + T) @ R.T

        tri = Delaunay(unit_vertices[:, :2])
        triangles = tri.simplices

        triangle_areas = toTriangleAreas(merge_vertices, triangles, True)
        valid_triangle_mask = triangle_areas > 1e-12

        valid_triangles = triangles[valid_triangle_mask]

        unique_triangles = merge_vertex_idxs[valid_triangles]

        return unique_triangles

    def createAllSubdivTriangles(self) -> np.ndarray:
        subdiv_triangles = []

        for_data = range(self.triangleNum)
        if self.print_progress:
            for_data = tqdm(for_data)
            print("[INFO][MeshSubdiver::createAllSubdivTriangles]")
            print("\t start subdiv triangles...")
            for i in for_data:
                curr_subdiv_triangles = self.createSubdivTriangles(i)
                subdiv_triangles.append(curr_subdiv_triangles)
        else:
            for i in range(self.triangleNum):
                curr_subdiv_triangles = self.createSubdivTriangles(i)
                subdiv_triangles.append(curr_subdiv_triangles)

        subdiv_triangles = np.vstack(subdiv_triangles)

        return subdiv_triangles

    def createAllSubdivTrianglesJoblib(
        self,
        n_jobs: int = os.cpu_count(),
    ) -> bool:
        subdiv_triangles = []

        if self.print_progress:
            print("[INFO][MeshSubdiver::createAllSubdivTrianglesJoblib]")
            print("\t start subdiv triangles...")
            with tqdm(total=self.triangleNum) as progress:
                with tqdm_joblib(progress):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(self.createSubdivTriangles)(i)
                        for i in range(self.triangleNum)
                    )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.createSubdivTriangles)(i) for i in range(self.triangleNum)
            )

        subdiv_triangles = np.vstack(results)

        return subdiv_triangles

    def createSubdivMesh(
        self,
        save_mesh_file_path: Union[str, None] = None,
        n_jobs: int = os.cpu_count(),
    ) -> o3d.geometry.TriangleMesh:
        self.createMergeVertices()
        assert self.merge_vertices.shape[0] > 0

        # subdiv_triangles = self.createAllSubdivTriangles()
        subdiv_triangles = self.createAllSubdivTrianglesJoblib(n_jobs)

        subdiv_mesh = o3d.geometry.TriangleMesh()
        subdiv_mesh.vertices = o3d.utility.Vector3dVector(self.merge_vertices)
        subdiv_mesh.triangles = o3d.utility.Vector3iVector(subdiv_triangles)

        updateVertexNormals(subdiv_mesh)
        updateTriangleNormals(subdiv_mesh)

        if save_mesh_file_path is not None:
            createFileFolder(save_mesh_file_path)

            tmp_save_mesh_file_path = (
                save_mesh_file_path[:-4] + "_tmp" + save_mesh_file_path[-4:]
            )
            o3d.io.write_triangle_mesh(
                tmp_save_mesh_file_path, subdiv_mesh, write_ascii=True
            )

            renameFile(tmp_save_mesh_file_path, save_mesh_file_path)

        return subdiv_mesh
