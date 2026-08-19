import torch
from dataclasses import dataclass, field

def _rotvec_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector(s) (..., 3) → rotation matrix/matrices (..., 3, 3) via
    matrix exponential. Shape-agnostic (works for a single (3,) vector or any
    batch of them, e.g. (V, H, W, 3)) since torch.linalg.matrix_exp batches
    over all leading dims on its own."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    O = torch.zeros_like(x)
    skew = torch.stack([O, -z, y, z, O, -x, -y, x, O], dim=-1).reshape(*r.shape[:-1], 3, 3)
    return torch.linalg.matrix_exp(skew)


@dataclass
class SceneObject:
    """One placed instance of a shape in the scene."""
    embedding:   torch.Tensor                       # (latent_dim,)
    translation: torch.Tensor                       # (3,)
    rot_vec:     torch.Tensor = field(default=None) # (3,) axis-angle, defaults to zero (identity)
    color:       torch.Tensor = field(default=None) # (3,) RGB in [0, 1], defaults to white

    def __post_init__(self):
        if self.rot_vec is None:
            self.rot_vec = torch.zeros(3, device=self.translation.device,
                                       dtype=self.translation.dtype)
        if self.color is None:
            self.color = torch.ones(3, device=self.translation.device,
                                    dtype=self.translation.dtype)

    @property
    def rotation(self) -> torch.Tensor:
        """(3, 3) rotation matrix derived from the axis-angle vector."""
        return _rotvec_to_matrix(self.rot_vec)

    def world_to_local(self, points: torch.Tensor) -> torch.Tensor:
        """points: (N, 3) world-space  →  (N, 3) object-local space"""
        return (points - self.translation) @ self.rotation

    def flatten(self) -> torch.Tensor:
        """Concatenates all fields into a single 1-D tensor: [embedding | translation | rot_vec | color]."""
        return torch.cat([self.embedding, self.translation, self.rot_vec, self.color])

    @classmethod
    def inflate(cls, tensor: torch.Tensor) -> "SceneObject":
        """Inverse of flatten; infers latent_dim from tensor length (total - 9)."""
        d = tensor.shape[0] - 9   # 3 translation + 3 rot_vec + 3 color
        return cls(
            embedding=tensor[:d],
            translation=tensor[d:d + 3],
            rot_vec=tensor[d + 3:d + 6],
            color=tensor[d + 6:d + 9],
        )


@dataclass
class CameraParams():
    """Camera pose stored as 6 independent values: axis-angle rotation + translation."""
    translation: torch.Tensor                       # (3,)
    rot_vec:     torch.Tensor = field(default=None) # (3,) axis-angle, defaults to zero (identity)

    def __post_init__(self):
        if self.rot_vec is None:
            self.rot_vec = torch.zeros(3, device=self.translation.device,
                                       dtype=self.translation.dtype)

    def to_c2w(self) -> torch.Tensor:
        """Builds the (4, 4) camera-to-world matrix."""
        R    = _rotvec_to_matrix(self.rot_vec)
        c2w  = torch.eye(4, device=self.translation.device, dtype=self.translation.dtype)
        c2w[:3, :3] = R
        c2w[:3,  3] = self.translation
        return c2w

    @classmethod
    def from_c2w(cls, c2w: torch.Tensor) -> "CameraParams":
        """Decomposes a (4, 4) c2w matrix into the compact representation."""
        # torch.linalg has no matrix_log, so extract axis-angle directly via the
        # closed-form inverse of _rotvec_to_matrix's Rodrigues formula.
        R = c2w[:3, :3]
        cos_angle = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2
        angle = torch.acos(cos_angle.clamp(-1.0, 1.0))
        axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        sin_angle = torch.sin(angle)
        small = sin_angle.abs() < 1e-6   # angle ~ 0: axis is undefined, rotation is ~identity
        safe_sin = torch.where(small, torch.ones_like(sin_angle), sin_angle)
        rot_vec = torch.where(small, torch.zeros_like(axis), axis * (angle / (2 * safe_sin)))
        return cls(translation=c2w[:3, 3].clone(), rot_vec=rot_vec)

    def flatten(self) -> torch.Tensor:
        """Returns a (6,) tensor: [rot_vec | translation]."""
        return torch.cat([self.rot_vec, self.translation])

    @classmethod
    def inflate(cls, tensor: torch.Tensor) -> "CameraParams":
        """Inverse of flatten; expects a (6,) tensor."""
        return cls(rot_vec=tensor[:3], translation=tensor[3:6])
