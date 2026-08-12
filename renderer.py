import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NeuralSDF, SceneSDF
from misc import SceneObject, CameraParams



class SDFRenderer(nn.Module):
    """
    Differentiable sphere-tracing renderer for a scene made of neural SDF instances.

    forward(sdf_net, objects, c2w) → (H, W, 3) image

    Sphere tracing is done without gradients for efficiency.
    Normals at hit points are recomputed with autograd, so gradients flow
    back through the embeddings and object positions.
    """

    def __init__(
        self,
        image_h:   int   = 256,
        image_w:   int   = 256,
        fov_deg:   float = 60.0,
        near:      float = 0.1,
        far:       float = 10.0,
        n_steps:   int   = 128,
        hit_eps:   float = 5e-4,
        soft_beta: float = 0.01,  # sigmoid sharpness for soft silhouette
    ):
        super().__init__()
        self.H, self.W = image_h, image_w
        self.fov_deg   = fov_deg
        self.near, self.far = near, far
        self.n_steps   = n_steps
        self.hit_eps   = hit_eps
        self.soft_beta = soft_beta

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_rays(self, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        c2w: (4, 4) camera-to-world matrix (OpenCV convention: +x right, +y down, +z forward)
        Returns:
            origins: (H*W, 3)
            dirs:    (H*W, 3) unit vectors in world space
        """
        device = c2w.device
        focal  = 0.5 * self.W / math.tan(math.radians(self.fov_deg * 0.5))

        i, j = torch.meshgrid(
            torch.arange(self.W, device=device, dtype=torch.float32),
            torch.arange(self.H, device=device, dtype=torch.float32),
            indexing="xy",
        )
        dirs_cam = torch.stack(
            [(i - self.W * 0.5) / focal,
             -(j - self.H * 0.5) / focal,
             -torch.ones_like(i)],
            dim=-1,
        ).reshape(-1, 3)                                   # (H*W, 3)

        dirs_world = F.normalize(dirs_cam @ c2w[:3, :3].T, dim=-1)  # (H*W, 3)
        origins    = c2w[:3, 3].unsqueeze(0).expand_as(dirs_world)  # (H*W, 3)
        return origins, dirs_world

    @torch.no_grad()
    def _sphere_trace(
        self,
        origins: torch.Tensor,
        dirs:    torch.Tensor,
        scene_sdf: SceneSDF,
        objects: list[SceneObject],
        scale:   torch.Tensor,          # (1,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            t        (N,) marched distance per ray
            hit_mask (N,) bool
        """
        t   = torch.full((origins.shape[0],), self.near, device=origins.device)
        hit = torch.zeros_like(t, dtype=torch.bool)

        for _ in range(self.n_steps):
            pts = origins + t.unsqueeze(-1) * dirs
            sdf = scene_sdf(pts, objects, scale)[..., 0].min(dim=0).values

            hit  |= sdf.abs() < self.hit_eps
            alive = ~hit & (t < self.far)
            if not alive.any():
                break
            t = torch.where(alive, t + sdf.clamp(min=self.hit_eps * 0.5), t)

        return t, hit

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        scene_sdf: SceneSDF,
        objects:   list[SceneObject],
        c2w:       torch.Tensor | CameraParams,
        light_dir: torch.Tensor | None = None,
        bg_color:  float = 1.0,
        scale:     torch.Tensor | None = None,  # (1,) uniform scale applied to all objects; None = 1
    ) -> torch.Tensor:
        """
        sdf_net:   trained NeuralSDF (shared across all objects)
        objects:   scene instances — each carries its embedding + pose
        c2w:       (4, 4) camera-to-world matrix, or a CameraParams instance
        light_dir: (3,) unit vector toward the light (world space);
                   defaults to the camera's forward axis
        bg_color:  scalar fill value for rays that miss all geometry
        scale:     (1,) tensor — uniform scale applied to all objects before world placement
        Returns:   (H, W, 3) Lambertian-shaded image in [0, 1],
                   differentiable w.r.t. embeddings, translations, and scale
        """
        if isinstance(c2w, CameraParams):
            c2w = c2w.to_c2w()
        device = c2w.device
        H, W   = self.H, self.W

        if light_dir is None:
            light_dir = F.normalize(c2w[:3, 2], dim=0)   # toward camera = away from scene

        if scale is None:
            scale = torch.ones(1, device=device)

        origins, dirs = self._make_rays(c2w)

        # ── Step 1: sphere trace (no grad) — finds approximate surface positions
        t, hit = self._sphere_trace(origins, dirs, scene_sdf, objects, scale)

        # Marching stops the instant |sdf| < hit_eps, so sdf_val there is barely
        # negative/positive — nowhere near saturating the alpha sigmoid below.
        # Step hit rays a bit further inward so solid surfaces read as fully
        # opaque instead of blending ~50% with the background.
        t = torch.where(hit, t + self.hit_eps + 3 * self.soft_beta, t)

        # ── Step 2: soft differentiable rendering over all rays ───────────────
        # Detach t so no grad flows through the marching steps themselves;
        # gradients enter via scene_sdf(all_pts, objects, scale) which depends
        # on objects.translation / scale through the SDF evaluation.
        with torch.enable_grad():
            all_pts = (origins + t.unsqueeze(-1) * dirs).detach().requires_grad_(True)

            sdfs             = scene_sdf(all_pts, objects, scale)       # (n_obj, N, C)
            sdf_val, sdf_idx = sdfs[..., 0].min(dim=0)                  # (N,), (N,)

            # Soft alpha: ~1 inside surface, ~0 outside — differentiable silhouette
            alpha = torch.sigmoid(-sdf_val / self.soft_beta)             # (N,)

            normals = torch.autograd.grad(
                sdf_val.sum(), all_pts, create_graph=True
            )[0]
        normals = F.normalize(normals, dim=-1)

        diffuse   = (normals @ F.normalize(light_dir, dim=0)).clamp(min=0.0)
        ambient   = 0.2
        shading   = (ambient + (1.0 - ambient) * diffuse)            # (N,)

        obj_color     = torch.stack([obj.color for obj in objects], dim=0)[sdf_idx]  # (N, 3)
        surface_color = (shading.unsqueeze(-1) * obj_color).clamp(0.0, 1.0)          # (N, 3)

        # Soft composite: alpha-blend surface over background
        image = alpha.unsqueeze(-1) * surface_color + (1.0 - alpha).unsqueeze(-1) * bg_color

        # Segmentation: object index per pixel, -1 for background
        hit = alpha > 0.5
        seg = torch.where(hit, sdf_idx, torch.full_like(sdf_idx, -1))

        # Return image, background probability, and segmentation mask
        return image.reshape(H, W, 3), (1.0 - alpha).reshape(H, W), seg.reshape(H, W)

if __name__ == "__main__":
    from PIL import Image
    from model import ShapeEmbedding

    saved_dict = torch.load("neural_sdf.pt", weights_only=False)

    shape_emb = ShapeEmbedding(saved_dict["codes"], 256)
    shape_emb.load_state_dict(saved_dict["shape_emb"])
    shape_emb.eval()

    sdf_net = NeuralSDF(1, 256, 512, 8)
    sdf_net.load_state_dict(saved_dict["sdf_net"])
    sdf_net.eval()

    obj = SceneObject(
        embedding=shape_emb[saved_dict["codes"][0]],
        translation=torch.tensor([0.0, 0.0, 0.0]),
        rot_vec=torch.randn(3),
        color=torch.tensor([0.8, 0.3, 0.2]),   # warm red
    )

    # Camera at (0, 0, 3) looking toward the origin, identity rotation
    cam = CameraParams(translation=torch.tensor([0.0, 0.0, 3.0]))

    renderer = SDFRenderer()
    image = renderer(sdf_net, [obj], cam)

    pixels = (image.detach().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(pixels).save("render.png")
    print("Saved render.png")

