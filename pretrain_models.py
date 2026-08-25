import itertools
import math
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ShapeEmbedding, NeuralParts, SceneSDF, VisionModel, MultiViewTransformer
from renderer import SDFRenderer
from misc import SceneObject, CameraParams
from dataset import MeshSDFDataset
from losses import scene_attribute_loss, probe_occupancy_loss
from utils import CSVLogger
from tqdm import tqdm

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MESH_FOLDER  = "data"
N_CONNECT_TYPES = 1
LATENT_DIM   = 8
HIDDEN_DIM   = 256
NUM_LAYERS   = 8
POINT_BATCH  = 16_384    # points per gradient step (within one mesh)
LR_NETWORK   = 1e-4
LR_LATENT    = 1e-3
LR_VISION_MODEL    = 1e-3
LR_MULTIVIEW_TRANSFORMER = 1e-3
EPOCHS       = 1000
EPOCHS_VISION_MULTIVIEW = 500   # epoch count for train_vision_multiview()
SCENES_PER_STEP = 16   # independent scenes averaged into one gradient step
OCCUPANCY_LOSS_WEIGHT = 0.1   # weight on the probe-grid auxiliary occupancy loss
LR_DECAY_SPLITS = 4   # StepLR halves the LR this many times over the run
LR_DECAY_GAMMA = 0.5
CLAMP_DIST   = 0.1
LATENT_REG   = 1e-4
N_SURFACE    = 100_000 // 4
N_UNIFORM    = 50_000 // 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = 0
N_OBJECTS    = 5
N_VIEWS_MULTIVIEW = 8
VISION_HIDDEN_DIM = 512   # must match the hidden_dim vision_model.pt was trained with
RENDER_RES   = 128   # render resolution (rays = RENDER_RES^2)
# ─────────────────────────────────────────────────────────────────────────────


def _random_camera(radius_min: float, radius_max: float, device) -> CameraParams:
    """Uniform-on-sphere camera position looking at the origin (same scheme as
    render_views.py), built as a torch c2w matrix and converted via CameraParams.from_c2w."""
    radius = random.uniform(radius_min, radius_max)
    theta  = random.uniform(0.0, 2 * math.pi)
    z      = random.uniform(-1.0, 1.0)          # uniform-on-sphere, not phi ~ U(0, pi)
    phi    = math.acos(z)

    pos = torch.tensor([
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    ], device=device)

    forward = F.normalize(-pos, dim=0)                      # points from pos toward origin
    up_hint = torch.tensor([0., 0., 1.], device=device)
    if forward.abs().dot(up_hint).abs() > 0.99:              # near-parallel guard
        up_hint = torch.tensor([0., 1., 0.], device=device)
    right   = F.normalize(torch.cross(forward, up_hint, dim=0), dim=0)
    true_up = torch.cross(right, forward, dim=0)

    c2w = torch.eye(4, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward          # camera looks down local -Z, matches renderer.py's convention
    c2w[:3, 3] = pos
    return CameraParams.from_c2w(c2w)


def iter_point_batches(points: torch.Tensor, sdf: torch.Tensor, batch_size: int):
    """Yield (points, sdf) mini-batches shuffled within one mesh."""
    perm = torch.randperm(points.shape[0])
    for start in range(0, points.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        yield points[idx].to(DEVICE), sdf[idx].to(DEVICE)


def train_parts():
    dataset = MeshSDFDataset(MESH_FOLDER, n_surface=N_SURFACE, n_uniform=N_UNIFORM, n_connect_types=N_CONNECT_TYPES)
    # One mesh per worker iteration; mesh loading + sampling happens inside workers
    loader  = DataLoader(dataset, batch_size=1, shuffle=True,
                         num_workers=NUM_WORKERS, collate_fn=lambda x: x[0])

    shape_emb = ShapeEmbedding(dataset.codes, LATENT_DIM).to(DEVICE)
    sdf_net   = NeuralParts(N_CONNECT_TYPES, LATENT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)


    optimizer = torch.optim.Adam([
        {"params": shape_emb.parameters(), "lr": LR_LATENT},
        {"params": sdf_net.parameters(),   "lr": LR_NETWORK},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    loss_fn = nn.L1Loss()

    for epoch in tqdm(range(1, EPOCHS + 1), "Training SDF"):
        shape_emb.train(); sdf_net.train()
        total_loss, total_steps = 0.0, 0

        for code, points, sdf_gt in loader:   # one mesh at a time
            for pts_b, sdf_b in iter_point_batches(points, sdf_gt, POINT_BATCH):
                emb  = shape_emb([code]).expand(pts_b.shape[0], -1)  # (B, latent_dim)
                pred = sdf_net(emb, pts_b).squeeze(-1)

                if CLAMP_DIST is not None:
                    pred  = torch.clamp(pred,  -CLAMP_DIST, CLAMP_DIST)
                    sdf_b = torch.clamp(sdf_b, -CLAMP_DIST, CLAMP_DIST)

                loss = loss_fn(pred, sdf_b) + LATENT_REG * shape_emb.embedding.weight.norm(dim=-1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss  += loss.item()
                total_steps += 1

        scheduler.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS}  loss={total_loss / total_steps:.6f}")

    torch.save({
        "shape_emb": shape_emb.state_dict(),
        "sdf_net":   sdf_net.state_dict(),
        "codes":     dataset.codes,
    }, "neural_sdf.pt")
    print("Saved neural_sdf.pt")

def train_vision_multiview():
    """Trains vision_model and multiview_transformer jointly on rendered multi-view scenes."""
    saved_dict = torch.load("neural_sdf.pt", weights_only=False)

    shape_emb = ShapeEmbedding(saved_dict["codes"], LATENT_DIM).to(DEVICE)
    shape_emb.load_state_dict(saved_dict["shape_emb"])
    shape_emb.eval()

    sdf_net = NeuralParts(n_connect_types=N_CONNECT_TYPES, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    sdf_net.load_state_dict(saved_dict["sdf_net"])
    sdf_net.eval()

    renderer = SDFRenderer(n_steps=128, hit_eps=0.01, image_h=RENDER_RES, image_w=RENDER_RES).to(DEVICE)
    scene_sdf = SceneSDF(sdf_net)

    vision_model = VisionModel(hidden_dim=VISION_HIDDEN_DIM).to(DEVICE)
    multiview_transformer = MultiViewTransformer(latent_dim=LATENT_DIM, hidden_dim=512, image_emb_dim=VISION_HIDDEN_DIM).to(DEVICE)

    optimizer = torch.optim.Adam([
        {"params": vision_model.parameters(),         "lr": LR_VISION_MODEL},
        {"params": multiview_transformer.parameters(), "lr": LR_MULTIVIEW_TRANSFORMER},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, EPOCHS_VISION_MULTIVIEW // LR_DECAY_SPLITS),
        gamma=LR_DECAY_GAMMA,
    )

    codes = saved_dict["codes"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(
        f"logs/pretrain_vision_multiview_{timestamp}.csv",
        columns=["epoch", "loss", "obj", "scale", "occupancy", "lr"],
    )

    def sample_scene_losses():
        """One random scene, rendered from N_VIEWS_MULTIVIEW cameras. Returns
        (obj_loss, scale_loss, occupancy_loss), not yet backward()'d, so the caller
        can accumulate several of these into one averaged gradient step."""
        n_obj = torch.randint(1, N_OBJECTS + 1, (1,)).item()
        with torch.no_grad():
            gt_objs = []
            for i in range(n_obj):
                embedding = shape_emb[codes[torch.randint(0, len(codes), (1,)).item()]].detach()
                translation = (torch.rand(3, device=DEVICE) * 2 - 1)
                rot_vec = (torch.rand(3, device=DEVICE) * 2 - 1) * math.pi
                color = torch.rand(3, device=DEVICE)
                gt_objs.append(SceneObject(embedding, translation, rot_vec, color))
            scale_gt = torch.empty((), device=DEVICE).uniform_(0.5, 1.5)

            cams = [_random_camera(2.5, 4.0, DEVICE) for _ in range(N_VIEWS_MULTIVIEW)]
            images = torch.stack([renderer(scene_sdf, gt_objs, cam, scale=scale_gt)[0] for cam in cams])   # (V, H, W, 3)

            # Ground-truth occupancy for the transformer's fixed probe grid: union (min)
            # over objects of the frozen sdf_net's distance channel, same convention the
            # renderer itself uses (renderer.py's sdfs[..., 0].min(dim=0)).
            probe_sdf = scene_sdf(multiview_transformer.probe_coords, gt_objs, scale_gt)[..., 0]   # (n_obj, P)
            occupancy_gt = (probe_sdf.min(dim=0).values < 0).float()                                # (P,)

        image_embs = vision_model(images)   # (V, VISION_HIDDEN_DIM)

        # ── multiview_transformer: predict objects+scale from image_embs ────────
        seeds = [SceneObject.inflate(torch.randn(LATENT_DIM + 9, device=DEVICE)) for _ in range(n_obj)]
        output = multiview_transformer(seeds, image_embs, cams)
        pred_objs  = output["objects"]
        pred_scale = output["scale"]
        pred_occupancy = output["probe_occupancy"]

        # ── Combinatorial nearest-neighbor match ────────────────────────────────
        # No positional encoding on the seeds, so predicted object i has no inherent
        # correspondence to ground-truth object j — match via lowest-cost permutation.
        pred_flat = torch.stack([o.flatten() for o in pred_objs])   # (n_obj, D)
        gt_flat   = torch.stack([o.flatten() for o in gt_objs])     # (n_obj, D)
        cost = torch.cdist(pred_flat, gt_flat.detach()).detach().cpu().numpy()

        best_perm, best_cost = None, float("inf")
        for perm in itertools.permutations(range(n_obj)):
            c = sum(cost[i, perm[i]] for i in range(n_obj))
            if c < best_cost:
                best_cost, best_perm = c, perm

        matched_gt_flat = gt_flat[list(best_perm)]

        obj_loss   = scene_attribute_loss(pred_flat, matched_gt_flat, LATENT_DIM)
        scale_loss = F.mse_loss(pred_scale, scale_gt.reshape(pred_scale.shape))
        occupancy_loss = probe_occupancy_loss(pred_occupancy, occupancy_gt)

        return obj_loss, scale_loss, occupancy_loss

    for epoch in tqdm(range(1, EPOCHS_VISION_MULTIVIEW + 1), "Training Vision + MultiView jointly"):
        optimizer.zero_grad()

        obj_total, scale_total, occupancy_total = 0.0, 0.0, 0.0
        for _ in range(SCENES_PER_STEP):
            obj_loss, scale_loss, occupancy_loss = sample_scene_losses()
            scene_loss = (obj_loss + 0.1 * scale_loss + OCCUPANCY_LOSS_WEIGHT * occupancy_loss) / SCENES_PER_STEP
            scene_loss.backward()

            obj_total       += obj_loss.item() / SCENES_PER_STEP
            scale_total     += scale_loss.item() / SCENES_PER_STEP
            occupancy_total += occupancy_loss.item() / SCENES_PER_STEP

        optimizer.step()
        scheduler.step()

        loss_total = obj_total + 0.1 * scale_total + OCCUPANCY_LOSS_WEIGHT * occupancy_total
        logger.write([epoch, loss_total, obj_total, scale_total, occupancy_total, scheduler.get_last_lr()[0]])

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS_VISION_MULTIVIEW}  loss={loss_total:.6f}  "
                  f"obj={obj_total:.6f}  scale={scale_total:.6f}  occupancy={occupancy_total:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(vision_model.state_dict(), "vision_model.pt")
    torch.save(multiview_transformer.state_dict(), "multiview_transformer.pt")


if __name__ == "__main__":
    #train_parts()
    train_vision_multiview()
    pass
