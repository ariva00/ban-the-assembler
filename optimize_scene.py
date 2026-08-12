import math
import torch
import torch.nn.functional as F
import torchvision

from renderer import SDFRenderer
from misc import SceneObject, CameraParams
from model import NeuralSDF, ShapeEmbedding, SceneTransformer, SceneSDF, Scene, NeuralParts, VisionModel
from tqdm import tqdm
from PIL import Image

N_OBJECTS = 1
N_CONNECT_TYPES = 1
LATENT_DIM   = 8
HIDDEN_DIM   = 256
NUM_LAYERS   = 8
EPOCHS = 500
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_POINTS = 10000
BACKGROUND_COLOR = torch.tensor([1.0,1.0,1.0], device=DEVICE)
VISION_LOSS_WEIGHT = 0.1

def main():

    saved_dict = torch.load("neural_sdf.pt", weights_only=False)

    shape_emb = ShapeEmbedding(saved_dict["codes"], LATENT_DIM).to(DEVICE)
    shape_emb.load_state_dict(saved_dict["shape_emb"])
    shape_emb.eval()

    #sdf_net = NeuralSDF(1 + 6 * int(N_CONNECT_TYPES>0) + N_CONNECT_TYPES, LATENT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    sdf_net = NeuralParts(n_connect_types=N_CONNECT_TYPES, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    sdf_net.load_state_dict(saved_dict["sdf_net"])
    sdf_net.eval()

    vision_model = VisionModel(output_dim=LATENT_DIM + 3 + 3 + 3, hidden_dim=512).to(DEVICE)
    vision_model.load_state_dict(torch.load("vision_model.pt", map_location=DEVICE, weights_only=True))
    vision_model.eval()
    for p in vision_model.parameters():
        p.requires_grad_(False)

    codes = saved_dict["codes"]
    objs = []
    for i in range(N_OBJECTS):
        embedding = shape_emb[codes[i % len(codes)]].detach()
        objs.append(SceneObject(embedding.to(DEVICE), torch.rand((3,)).to(DEVICE)))

    scene_transformer = Scene(N_OBJECTS, latent_dim=LATENT_DIM).to(DEVICE)
    renderer = SDFRenderer(n_steps=1024, hit_eps=0.01).to(DEVICE)
    cam = CameraParams(translation=torch.tensor([0.0, 0.0, 3.0]).to(DEVICE))

    scene_sdf = SceneSDF(sdf_net)

    test_cube = SceneObject(shape_emb['cube'], torch.tensor((0.,0.,0.)).to(DEVICE), color=torch.tensor((1.,0.,0.)).to(DEVICE))

    
    image, _, _ = renderer(scene_sdf, [test_cube,], cam)
    pixels = (image.detach().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(pixels).save(f"test_world.png")

    target = torchvision.io.read_image("target.png").to(DEVICE).transpose(0,-1).transpose(0,1)/255
    target_mask = (target.isclose(BACKGROUND_COLOR, 0.005)).all(dim=-1).float()
    pixels = (target.detach().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(pixels).save(f"actual_target.png")

    with torch.no_grad():
        target_feat, _ = vision_model(target.unsqueeze(0))
        target_feat = target_feat.squeeze(0)

    optim = torch.optim.Adam(scene_transformer.parameters(), lr=0.1)

    for epoch in tqdm(range(EPOCHS)):
        optim.zero_grad()
        output = scene_transformer()
        objs   = output["objects"]
        scale  = output["scale"]


        # ── STE quantization ──────────────────────────────────────────────────
        valid_embs  = shape_emb.embedding.weight.detach()            # (n_codes, 256)
        pred_embs   = torch.stack([o.embedding for o in objs])       # (N, 256)
        nearest_idx = torch.cdist(pred_embs, valid_embs).argmin(dim=1)
        quantized   = valid_embs[nearest_idx]                        # (N, 256) — no grad
        # Forward = nearest valid embedding; backward = straight through pred_embs
        quantized_st = pred_embs + (quantized - pred_embs).detach()

        quant_objs = [
            SceneObject(quantized_st[i], o.translation, o.rot_vec, o.color)
            for i, o in enumerate(objs)
        ]

        image, bg_prob, _ = renderer(scene_sdf, quant_objs, cam, scale=1.0)

        recon_loss      = F.mse_loss(image, target)
        pred_feat, _    = vision_model(image.unsqueeze(0))
        pred_feat       = pred_feat.squeeze(0)
        vision_loss     = F.mse_loss(pred_feat, target_feat)
        commitment_loss = F.mse_loss(pred_embs, quantized.detach())  # pulls encoder toward codebook
        # bg_prob = 1 - alpha from renderer: 1 where background, 0 where object
        mask_loss       = F.binary_cross_entropy(bg_prob.clamp(1e-6, 1-1e-6), target_mask)

        # ── Intersection and connection loss ──────────────────────────────────
        points = (torch.rand((N_POINTS, 3)).to(DEVICE) * 6) - 3
        sdfs = scene_sdf(points, quant_objs, 1.0)
        intersection_loss = (torch.nn.functional.softsign(sdfs[..., 0]*(-1000)).relu().sum(dim=0)-1).relu().sum()
        connect_male = torch.cat((sdfs[..., 1:4], sdfs[..., 7:]), dim=-1).mean(dim=0)
        connect_female = sdfs[..., 4:].mean(dim=0)
        connect_loss = torch.nn.functional.mse_loss(connect_male, connect_female, reduction="mean")

        # ── Frustum loss ──────────────────────────────────────────────────────
        c2w          = cam.to_c2w().detach()
        R, t_cam     = c2w[:3, :3], c2w[:3, 3]
        translations = torch.stack([o.translation for o in objs])    # (N, 3)
        p_cam        = (translations - t_cam) @ R                    # (N, 3) camera space
        depth        = -p_cam[:, 2]
        tan_hf       = math.tan(math.radians(renderer.fov_deg / 2))
        frustum_loss = (
            F.relu(renderer.near - depth) +
            F.relu(depth - renderer.far)  +
            F.relu(p_cam[:, 0].abs() / depth.clamp(min=1e-6) - tan_hf) +
            F.relu(p_cam[:, 1].abs() / depth.clamp(min=1e-6) - tan_hf)
        ).mean()

        loss = (
            recon_loss +
            VISION_LOSS_WEIGHT * vision_loss +
            0.25 * commitment_loss +
            0.1 * frustum_loss +
            intersection_loss +
            0.5 * connect_loss +
            0.1 * mask_loss +
            0
        )

        loss.backward()
        optim.step()
        print(f"\nrec: {recon_loss.item():.4f} | vision: {vision_loss.item():.4f} | commit: {commitment_loss.item():.4f} | frustum: {frustum_loss.item():.4f} | intersec: {intersection_loss.item():.4f} | connect: {connect_loss.item():.4f} | mask: {mask_loss.item():.4f}\nscale: {scale.item()}")
        objs = [SceneObject.inflate(o.flatten().detach()) for o in objs]
        pixels = (image.detach().cpu().numpy() * 255).astype("uint8")
        Image.fromarray(pixels).save(f"iterations/{epoch:04d}.png")

if __name__ == "__main__":
    main()