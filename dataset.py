import json
from pathlib import Path

import torch
import torchvision
import trimesh
import numpy as np
from torch.utils.data import Dataset

from misc import CameraParams


def _load_additional_attributes(mesh: trimesh.Trimesh, n_connect_types: int) -> trimesh.Trimesh:
    n    = len(mesh.vertices)
    meta = mesh.metadata

    def _vec3(key: str) -> np.ndarray:
        val = meta.get(key)
        return np.asarray(val, dtype=np.float32).reshape(n, 3) if val is not None else np.zeros((n, 3), dtype=np.float32)

    def _int1(key: str) -> np.ndarray:
        val = meta.get(key)
        return np.asarray(val, dtype=np.int32).reshape(n) if val is not None else np.zeros(n, dtype=np.int32)

    mesh.vertex_attributes["connect_male"]   = _vec3("connect_male")
    mesh.vertex_attributes["connect_female"] = _vec3("connect_female")

    ct        = _int1("connect_type")
    ct_onehot = np.zeros((n, n_connect_types), dtype=np.float32)
    mask      = ct > 0
    ct_onehot[mask, np.clip(ct[mask] - 1, 0, n_connect_types - 1)] = 1.0  # 0 → zero vec; k → one-hot[k-1]
    mesh.vertex_attributes["connect_type"] = ct_onehot

    return mesh


def _sample_sdf(
    mesh:      trimesh.Trimesh,
    n_surface: int,
    n_uniform: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    has_attrs = "connect_male" in mesh.vertex_attributes
    if has_attrs:
        # Grab references before mesh.copy() so we keep the originals
        va_male   = mesh.vertex_attributes["connect_male"]    # (V, 3)
        va_female = mesh.vertex_attributes["connect_female"]  # (V, 3)
        va_type   = mesh.vertex_attributes["connect_type"]    # (V, K)

    mesh = mesh.copy()
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

    half = n_surface // 2
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
    near    = surface_pts[:half]           + np.random.randn(half,             3) * 0.005
    mid     = surface_pts[half:]           + np.random.randn(n_surface - half, 3) * 0.05
    lo, hi  = mesh.bounds
    uniform = np.random.uniform(lo, hi, size=(n_uniform, 3))
    points  = np.concatenate([near, mid, uniform], axis=0).astype(np.float32)

    _, distances, triangle_ids = trimesh.proximity.closest_point(mesh, points)
    inside = mesh.contains(points)
    sdf    = np.where(inside, -distances, distances).astype(np.float32)
    sdf_t  = torch.from_numpy(sdf)

    if not has_attrs:
        return torch.from_numpy(points), sdf_t

    # Nearest vertex per point: check the 3 vertices of the closest triangle
    face_verts    = mesh.faces[triangle_ids]                                  # (N, 3)
    verts_pos     = mesh.vertices[face_verts]                                 # (N, 3, 3)
    d_to_verts    = np.linalg.norm(points[:, None, :] - verts_pos, axis=-1)  # (N, 3)
    nearest_local = d_to_verts.argmin(axis=1)                                 # (N,)
    nearest_vert  = face_verts[np.arange(len(points)), nearest_local]         # (N,)

    sampled = []
    for arr in (va_male, va_female, va_type):
        s = arr[nearest_vert].copy()
        s[n_surface:] = 0.0   # uniform (far-from-surface) points carry no connect info
        sampled.append(torch.from_numpy(s.astype(np.float32)))

    labels = torch.cat([sdf_t.unsqueeze(-1), *sampled], dim=-1)  # (N, 1+3+3+K)
    return torch.from_numpy(points), labels


class MeshSDFDataset(Dataset):
    """
    folder/
        parts.json          {"code": "shape_A", "filename": "shape_A.obj"}, ...}
        shape_A.obj
        ...

    __getitem__ returns:
        (code, points (N,3), sdf (N,))                             when n_connect_types == 0
        (code, points (N,3), labels (N, 1+3+3+n_connect_types))   when n_connect_types  > 0
            labels = [sdf | connect_male | connect_female | connect_type_onehot]
    """

    def __init__(
        self,
        folder:          str | Path,
        n_surface:       int = 100_000,
        n_uniform:       int = 50_000,
        n_connect_types: int = 0,
    ):
        self.folder          = Path(folder)
        self.n_surface       = n_surface
        self.n_uniform       = n_uniform
        self.n_connect_types = n_connect_types

        parts_path = self.folder / "parts.json"
        with open(parts_path) as f:
            parts: list[dict] = json.load(f)

        self.codes = [p["code"]                  for p in parts]
        self.paths = [self.folder / p["filename"] for p in parts]

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        mesh = trimesh.load(self.paths[idx], force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{self.paths[idx]} did not load as a single Trimesh")
        if self.n_connect_types > 0:
            mesh = _load_additional_attributes(mesh, self.n_connect_types)
        points, sdf = _sample_sdf(mesh, self.n_surface, self.n_uniform)
        return self.codes[idx], points, sdf


class MultiViewDataset(Dataset):
    """
    folder/
        views.json   [{"file": "view_000.png", "translation": [x,y,z], "rot_vec": [x,y,z]}, ...]
        view_000.png
        ...

    __getitem__ returns (image (H, W, 3) in [0, 1], CameraParams)
    """

    def __init__(self, folder: str | Path, views_filename: str = "views.json"):
        self.folder = Path(folder)

        with open(self.folder / views_filename) as f:
            views: list[dict] = json.load(f)["views"]

        self.files        = [self.folder / v["file"]            for v in views]
        self.translations = [torch.tensor(v["translation"], dtype=torch.float32)          for v in views]
        self.rot_vecs      = [torch.tensor(v.get("rot_vec", [0., 0., 0.]), dtype=torch.float32) for v in views]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, CameraParams]:
        image = torchvision.io.read_image(str(self.files[idx])).transpose(0, -1).transpose(0, 1) / 255
        cam = CameraParams(translation=self.translations[idx], rot_vec=self.rot_vecs[idx])
        return image, cam


def multiview_collate(batch: list[tuple[torch.Tensor, CameraParams]]) -> tuple[torch.Tensor, list[CameraParams]]:
    """Stacks images into a batch; keeps camera poses as a plain list (CameraParams isn't tensor-stackable)."""
    images = torch.stack([img for img, _ in batch], dim=0)   # (N, H, W, 3)
    cams   = [cam for _, cam in batch]
    return images, cams
