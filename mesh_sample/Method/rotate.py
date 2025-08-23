import numpy as np
from typing import Tuple


def normalize(v):
    return v / np.linalg.norm(v)


def compute_rotation_matrix(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)

    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)

    if np.allclose(v, 0):  # 两个向量平行（或反向）
        if c > 0:
            return np.eye(3)  # 已对齐
        else:
            # 反向：选择任意垂直轴旋转180度
            axis = (
                np.array([1, 0, 0])
                if not np.allclose(from_vec, [1, 0, 0])
                else np.array([0, 1, 0])
            )
            return compute_rotation_matrix(from_vec, -axis) @ compute_rotation_matrix(
                -axis, to_vec
            )

    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R


def getRAndT(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.array(v)  # v = [v0, v1, v2]

    # 1. 重心
    G = np.mean(v, axis=0)

    # 2. 平移到重心为原点
    v_centered = v - G

    # 3. 法向量
    n = np.cross(v_centered[1] - v_centered[0], v_centered[2] - v_centered[0])
    n = normalize(n)

    # 4. 旋转矩阵，使法向量旋转到 [0, 0, 1]
    R = compute_rotation_matrix(n, np.array([0, 0, 1]))

    return R, -G
