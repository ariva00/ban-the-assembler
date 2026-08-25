import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from torch.utils.data import DataLoader

from renderer import SDFRenderer
from misc import SceneObject, CameraParams
from model import ShapeEmbedding, SceneSDF, NeuralParts, VisionModel, MultiViewTransformer
from dataset import MultiViewDataset, multiview_collate
from losses import color_histogram_loss
from utils import CSVLogger
from tqdm import tqdm
from PIL import Image

N_OBJECTS = 5
N_CONNECT_TYPES = 1
LATENT_DIM   = 8
HIDDEN_DIM   = 256
NUM_LAYERS   = 8
TRANSFORMER_HIDDEN_DIM = 512   # MultiViewTransformer's internal hidden dim
EPOCHS = 500
LR = 1e-4   # fine-tuning LR for the pretrained checkpoint (well below a from-scratch LR)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_POINTS = 10000
SAMPLE_BOUND = 3.0   # sampling cube half-extent: points drawn from [-SAMPLE_BOUND, SAMPLE_BOUND]^3
INTERSECTION_STEEPNESS = 1000   # softsign steepness for the intersection-loss step approximation
RENDER_STEPS = 128   # sphere-trace step count
HIT_EPS = 0.01   # sphere-trace hit threshold
BACKGROUND_COLOR = torch.tensor([1.0,1.0,1.0], device=DEVICE)
MASK_THRESHOLD = 0.005   # background isclose tolerance
# commitment_loss pulls pred_embs toward a valid codebook value.
COMMIT_LOSS_WEIGHT_START = 1.0
COMMIT_LOSS_WEIGHT_END   = 3.0
COMMIT_LOSS_WEIGHT_RAMP_FRAC = 0.5   # fraction of EPOCHS over which the ramp completes
FRUSTUM_LOSS_WEIGHT = 0.1
INTERSECTION_LOSS_WEIGHT = 1.0   # intersection_loss is normalized to a per-point mean below
MASK_LOSS_WEIGHT = 0.1
CONNECT_LOSS_WEIGHT = 0.0   # connectivity loss weight (disabled)
HIST_LOSS_WEIGHT = 0.3   # foreground color-histogram loss weight
MULTIVIEW_FOLDER = "multiview_target"
VISION_HIDDEN_DIM = 512   # must match the hidden_dim vision_model.pt was trained with
MESH_FOLDER = "data"
RENDER_RES = 128   # render resolution (rays = RENDER_RES^2); targets downsampled to match


def save_scene_ply(objs: list[SceneObject], scale: torch.Tensor, obj_codes: list[str], mesh_folder: str, out_path: str):
    """Assembles the trained ply parts (mesh_folder/parts.json + mesh_folder/ply/*)
    into one world-space mesh, placed/rotated/scaled/colored per SceneObject."""
    folder = Path(mesh_folder)
    with open(folder / "parts.json") as f:
        parts = json.load(f)
    code_to_file = {p["code"]: folder / p["filename"] for p in parts}

    scale_val = scale.item()
    meshes = []
    for obj, code in zip(objs, obj_codes):
        mesh = trimesh.load(code_to_file[code], force="mesh")
        # Same normalization as dataset.py's _sample_sdf, so vertices land in the
        # canonical space the SDF net (and thus this object's pose) was trained on.
        mesh.vertices -= mesh.bounding_box.centroid
        mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

        # Inverse of SceneObject.world_to_local + SceneSDF's /scale: canonical vertex
        # -> world space is (v * scale) @ R.T + t (R.T since world_to_local uses
        # (p - t) @ R, and R is orthogonal so R^-1 = R.T).
        R = obj.rotation.detach().cpu().numpy()
        t = obj.translation.detach().cpu().numpy()
        mesh.vertices = (mesh.vertices * scale_val) @ R.T + t

        color = (obj.color.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        mesh.visual.vertex_colors = np.tile(np.append(color, 255), (len(mesh.vertices), 1))

        meshes.append(mesh)

    trimesh.util.concatenate(meshes).export(out_path)
    print(f"Saved {out_path}")


def main():

    Path("iterations").mkdir(exist_ok=True)

    saved_dict = torch.load("neural_sdf.pt", weights_only=False)
    codes = saved_dict["codes"]

    shape_emb = ShapeEmbedding(saved_dict["codes"], LATENT_DIM).to(DEVICE)
    shape_emb.load_state_dict(saved_dict["shape_emb"])
    shape_emb.eval()

    sdf_net = NeuralParts(n_connect_types=N_CONNECT_TYPES, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    sdf_net.load_state_dict(saved_dict["sdf_net"])
    sdf_net.eval()

    vision_model = VisionModel(hidden_dim=VISION_HIDDEN_DIM).to(DEVICE)
    vision_model.load_state_dict(torch.load("vision_model.pt", map_location=DEVICE, weights_only=True))
    vision_model.eval()
    for p in vision_model.parameters():
        p.requires_grad_(False)

    renderer = SDFRenderer(n_steps=RENDER_STEPS, hit_eps=HIT_EPS, image_h=RENDER_RES, image_w=RENDER_RES).to(DEVICE)
    scene_sdf = SceneSDF(sdf_net)

    # ── Load the posed multi-view image collection ─────────────────────────────
    dataset = MultiViewDataset(MULTIVIEW_FOLDER)
    loader  = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=multiview_collate)
    targets, cams = next(iter(loader))
    targets = targets.to(DEVICE)
    cams    = [CameraParams(c.translation.to(DEVICE), c.rot_vec.to(DEVICE)) for c in cams]

    target_masks = (targets.isclose(BACKGROUND_COLOR, MASK_THRESHOLD)).all(dim=-1).float()   # (V, H, W)

    targets      = F.interpolate(targets.permute(0, 3, 1, 2), size=(RENDER_RES, RENDER_RES),
                                  mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    target_masks = F.interpolate(target_masks.unsqueeze(1), size=(RENDER_RES, RENDER_RES),
                                  mode="bilinear", align_corners=False, antialias=True).squeeze(1)

    with torch.no_grad():
        image_embs = vision_model(targets)   # (V, VISION_HIDDEN_DIM)

    d = LATENT_DIM
    seeds = (torch.rand(N_OBJECTS, d + 9, device=DEVICE) * 2) - 1
    seeds[..., d + 3:d + 6] = seeds[..., d + 3:d + 6] * torch.pi          # [-1, 1] → [-π, π]
    seeds[..., d + 6:d + 9] = (seeds[..., d + 6:d + 9] + 1) / 2           # [-1, 1] → [0, 1]
    seeds = [SceneObject.inflate(seed) for seed in seeds.unbind()]

    multiview_transformer = MultiViewTransformer(latent_dim=LATENT_DIM, hidden_dim=TRANSFORMER_HIDDEN_DIM, image_emb_dim=VISION_HIDDEN_DIM).to(DEVICE)
    multiview_transformer.load_state_dict(torch.load("multiview_transformer.pt", map_location=DEVICE, weights_only=True))

    optim = torch.optim.Adam(multiview_transformer.parameters(), lr=LR)

    n_views = len(cams)
    tan_hf  = math.tan(math.radians(renderer.fov_deg / 2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(
        f"logs/optimize_scene_multiview_{timestamp}.csv",
        columns=["epoch", "recon", "commit", "frustum", "intersec", "connect", "mask", "hist", "scale", "commit_weight"],
    )
    ramp_epochs = max(1, int(EPOCHS * COMMIT_LOSS_WEIGHT_RAMP_FRAC))

    for epoch in tqdm(range(EPOCHS)):
        commit_weight = COMMIT_LOSS_WEIGHT_START + min(1.0, epoch / ramp_epochs) * (COMMIT_LOSS_WEIGHT_END - COMMIT_LOSS_WEIGHT_START)

        optim.zero_grad()
        log = dict(recon=0.0, commit=0.0, frustum=0.0, intersec=0.0, connect=0.0, mask=0.0, hist=0.0)
        first_view_pixels = None
        last_scale = None

        for v, cam in enumerate(cams):
            output = multiview_transformer(seeds, image_embs, cams)
            objs   = output["objects"]
            scale  = output["scale"]
            last_scale = scale

            # ── STE quantization ──────────────────────────────────────────────
            valid_embs  = shape_emb.embedding.weight.detach()            # (n_codes, 256)
            pred_embs   = torch.stack([o.embedding for o in objs])       # (N, 256)
            nearest_idx = torch.cdist(pred_embs, valid_embs).argmin(dim=1)
            quantized   = valid_embs[nearest_idx]                        # (N, 256) — no grad
            quantized_st = pred_embs + (quantized - pred_embs).detach()

            quant_objs = [
                SceneObject(quantized_st[i], o.translation, o.rot_vec, o.color)
                for i, o in enumerate(objs)
            ]

            # ── Intersection and connection loss ────────────────────────────────
            points = (torch.rand((N_POINTS, 3)).to(DEVICE) * 2 - 1) * SAMPLE_BOUND
            sdfs = scene_sdf(points, quant_objs, scale)
            intersection_loss = (torch.nn.functional.softsign(sdfs[..., 0]*(-INTERSECTION_STEEPNESS)).relu().sum(dim=0)-1).relu().sum() / N_POINTS
            connect_male = torch.cat((sdfs[..., 1:4], sdfs[..., 7:]), dim=-1).mean(dim=0)
            connect_female = sdfs[..., 4:].mean(dim=0)
            connect_loss = torch.nn.functional.mse_loss(connect_male, connect_female, reduction="mean")
            commitment_loss = F.mse_loss(pred_embs, quantized.detach())  # pulls encoder toward codebook

            # ── Render + per-view losses ─────────────────────────────────────────
            image, bg_prob, _ = renderer(scene_sdf, quant_objs, cam, scale=scale)
            recon_loss = F.mse_loss(image, targets[v])
            # bg_prob = 1 - alpha from renderer: 1 where background, 0 where object
            mask_loss = F.binary_cross_entropy(bg_prob.clamp(1e-6, 1-1e-6), target_masks[v])
            hist_loss = color_histogram_loss(image, 1.0 - bg_prob, targets[v], 1.0 - target_masks[v])

            c2w          = cam.to_c2w().detach()
            R, t_cam     = c2w[:3, :3], c2w[:3, 3]
            translations = torch.stack([o.translation for o in objs])    # (N, 3)
            p_cam        = (translations - t_cam) @ R                    # (N, 3) camera space
            depth        = -p_cam[:, 2]
            frustum_loss = (
                F.relu(renderer.near - depth) +
                F.relu(depth - renderer.far)  +
                F.relu(p_cam[:, 0].abs() / depth.clamp(min=1e-6) - tan_hf) +
                F.relu(p_cam[:, 1].abs() / depth.clamp(min=1e-6) - tan_hf)
            ).mean()

            view_loss = (
                recon_loss +
                commit_weight * commitment_loss +
                FRUSTUM_LOSS_WEIGHT * frustum_loss +
                INTERSECTION_LOSS_WEIGHT * intersection_loss +
                CONNECT_LOSS_WEIGHT * connect_loss +
                MASK_LOSS_WEIGHT * mask_loss +
                HIST_LOSS_WEIGHT * hist_loss
            ) / n_views

            view_loss.backward()

            log["recon"]    += recon_loss.item() / n_views
            log["commit"]   += commitment_loss.item() / n_views
            log["frustum"]  += frustum_loss.item() / n_views
            log["intersec"] += intersection_loss.item() / n_views
            log["connect"]  += connect_loss.item() / n_views
            log["mask"]     += mask_loss.item() / n_views
            log["hist"]     += hist_loss.item() / n_views

            if v == 0:
                first_view_pixels = (image.detach().cpu().numpy() * 255).astype("uint8")

        obj_codes = [codes[i] for i in nearest_idx.tolist()]
        save_scene_ply(quant_objs, scale, obj_codes, MESH_FOLDER, "output_scene.ply")
        optim.step()
        logger.write([epoch, log["recon"], log["commit"], log["frustum"],
                      log["intersec"], log["connect"], log["mask"], log["hist"], last_scale.item(), commit_weight])
        print(f"\nrec: {log['recon']:.4f} | commit: {log['commit']:.4f} | frustum: {log['frustum']:.4f} | intersec: {log['intersec']:.4f} | connect: {log['connect']:.4f} | mask: {log['mask']:.4f} | hist: {log['hist']:.4f} | commit_weight: {commit_weight:.3f}\nscale: {last_scale.item()}")
        Image.fromarray(first_view_pixels).save(f"iterations/{epoch:04d}.png")

    # ── Final scene export ──────────────────────────────────────────────────────
    multiview_transformer.eval()
    with torch.no_grad():
        output = multiview_transformer(seeds, image_embs, cams)
        objs   = output["objects"]
        scale  = output["scale"]

        valid_embs  = shape_emb.embedding.weight.detach()
        pred_embs   = torch.stack([o.embedding for o in objs])
        nearest_idx = torch.cdist(pred_embs, valid_embs).argmin(dim=1)
        quant_objs = [
            SceneObject(valid_embs[nearest_idx[i]], o.translation, o.rot_vec, o.color)
            for i, o in enumerate(objs)
        ]
        obj_codes = [codes[i] for i in nearest_idx.tolist()]

    save_scene_ply(quant_objs, scale, obj_codes, MESH_FOLDER, "output_scene.ply")

if __name__ == "__main__":
    main()
