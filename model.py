import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import SceneObject, CameraParams
from transformer import Transformer as CrossAttentionTransformer


class ShapeEmbedding(nn.Module):
    """Maps string shape codes to learned embedding vectors."""

    def __init__(self, codes: list[str], latent_dim: int = 256):
        super().__init__()
        self.code_to_idx = {code: i for i, code in enumerate(codes)}
        self.embedding = nn.Embedding(len(codes), latent_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, codes: list[str]) -> torch.Tensor:
        """codes: list of B string codes  →  (B, latent_dim)"""
        device = self.embedding.weight.device
        indices = torch.tensor(
            [self.code_to_idx[c] for c in codes],
            dtype=torch.long, device=device
        )
        return self.embedding(indices)

    def __getitem__(self, code: str) -> torch.Tensor:
        """Convenience: single code → (latent_dim,)"""
        return self.forward([code]).squeeze(0)


class NeuralSDF(nn.Module):
    """
    Given a shape embedding and query points, outputs SDF values.
    The embedding is provided externally — this module has no knowledge of shapes.
    """

    def __init__(self, out_dim:int = 1, latent_dim: int = 256, hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        input_dim = latent_dim + 3  # embedding || xyz

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            if i == num_layers // 2:
                in_dim += input_dim  # skip connection re-injects the original input
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.skip_at = num_layers // 2
        self.num_layers = num_layers

    def forward(self, embeddings: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, latent_dim)
        points:     (B, 3)
        returns:    (B, out_dim)
        """
        x = torch.cat([embeddings, points], dim=-1)
        skip = x

        layer_idx = 0
        for i in range(self.num_layers):
            if i == self.skip_at:
                x = torch.cat([x, skip], dim=-1)
            x = self.layers[layer_idx](x)
            x = self.layers[layer_idx + 1](x)
            layer_idx += 2

        return self.out(x)


class NeuralParts(nn.Module):
    """
    Given a shape embedding and query points, outputs SDF values.
    The embedding is provided externally — this module has no knowledge of shapes.
    """

    def __init__(self, n_connect_types:int = 1, latent_dim: int = 256, hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        self.sdf = NeuralSDF(out_dim=hidden_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.sdf_linear = torch.nn.Linear(hidden_dim, 1)
        self.direction_male_linear = torch.nn.Linear(hidden_dim, 3)
        self.direction_female_linear = torch.nn.Linear(hidden_dim, 3)
        self.connect_linear = torch.nn.Linear(hidden_dim, n_connect_types)

    def forward(self, embeddings: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, latent_dim)
        points:     (B, 3)
        returns:    (B, out_dim)
        """

        sdf_out = self.sdf(embeddings, points)
        distance = self.sdf_linear(sdf_out)
        direction_male = F.normalize(self.direction_male_linear(sdf_out))
        direction_female = F.normalize(self.direction_female_linear(sdf_out))
        connection_type = self.connect_linear(sdf_out).softmax(dim=-1)
        return torch.cat((distance, direction_male, direction_female, connection_type), dim=-1)

class SceneSDF(nn.Module):
    def __init__(self, sdf_net: NeuralSDF):
        super().__init__()
        self.sdf_net = sdf_net

    def forward(
        self,
        points:  torch.Tensor,
        objects: torch.Tensor | list[SceneObject],
        scale:   torch.Tensor,
    ) -> torch.Tensor:
        """
        points: (N, 3)
        Returns (N, K) scene SDF = smooth union (min) over all objects.
        """
        N    = points.shape[0]
        if isinstance(objects, torch.Tensor):
            objects = [SceneObject.inflate(o) for o in objects.unbind(0)]
        O = len(objects)

        # Batched over objects instead of a Python for-loop issuing O separate
        # sdf_net forward passes: this is called once per sphere-trace step, so
        # O sequential kernel launches per step adds up to O x n_steps x n_views
        # x n_epochs sequential dispatches. Stacking objects into one bigger
        # batch call is the same math, ~O fewer sequential GPU round-trips.
        translations = torch.stack([obj.translation for obj in objects], dim=0)  # (O, 3)
        rotations    = torch.stack([obj.rotation for obj in objects], dim=0)     # (O, 3, 3)
        embeddings   = torch.stack([obj.embedding for obj in objects], dim=0)    # (O, latent_dim)

        local = (points.unsqueeze(0) - translations.unsqueeze(1)) @ rotations    # (O, N, 3)
        local = local / scale                                                    # shrink query into scaled space
        emb   = embeddings.unsqueeze(1).expand(-1, N, -1)                        # (O, N, latent_dim)

        sdfs = self.sdf_net(emb.reshape(O * N, -1), local.reshape(O * N, 3))     # (O*N, K)
        sdfs = sdfs.reshape(O, N, -1) * scale                                     # The scale is global so resizing the whole out vector is not a problem, if each object had its scale we should be way more carefull
        return sdfs                                                              # (O, N, K)


class Scene(nn.Module):
    def __init__(self, n_objects: int = 10, latent_dim: int = 256):
        super().__init__()
        self.embedding = torch.nn.Parameter((torch.rand((n_objects, latent_dim)) * 2) - 1)
        self.translate = torch.nn.Parameter((torch.rand((n_objects, 3)) * 2) - 1)
        self.rotation = torch.nn.Parameter((torch.rand((n_objects, 3)) * 2) - 1)
        self.color = torch.nn.Parameter((torch.rand((n_objects, 3)) * 2) - 1)
        self.scale = torch.nn.Parameter(torch.tensor((1.,)))

    def forward(self, return_as_list = True):
        emb   = torch.nn.functional.tanh(self.embedding)
        trans = torch.nn.functional.tanh(self.translate)
        rot   = torch.nn.functional.tanh(self.rotation) * torch.pi          # [-1, 1] → [-π, π]
        color = (torch.nn.functional.tanh(self.color) + 1) / 2               # [-1, 1] → [0, 1]
        out   = torch.cat([emb, trans, rot, color], dim=-1)

        return {
            "objects" : [SceneObject.inflate(o) for o in out.unbind(0)] if return_as_list else out,
            "scale"   : F.softplus(self.scale),            # always positive
        }

def _group_norm(channels: int) -> torch.nn.GroupNorm:
    """Largest divisor of `channels` that's <= 8, so GroupNorm always gets a
    valid, reasonably-sized number of groups regardless of the channel count
    at a given layer (channel counts here are powers of two, so this lands on
    8 almost everywhere, but stays correct even if hidden_dim/n_layers change)."""
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return torch.nn.GroupNorm(groups, channels)


class VisionModel(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 123, hidden_dim: int = 512, n_layers: int = 4):
        assert hidden_dim % (2**n_layers) == 0, "hidden_dim must be divisible by 2^n_layers"
        super().__init__()
        step = hidden_dim // (2**n_layers)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_conv = torch.nn.Conv2d(3, step, kernel_size=3, padding=1)
        self.input_norm = _group_norm(step)
        self.output_conv = torch.nn.Conv2d(max(step * 2, output_dim), output_dim, kernel_size=1)
        conv_down = []
        norm_down = []
        conv_up   = []
        norm_up   = []
        for i in range(n_layers):
            conv_down.append(torch.nn.Conv2d(step * (2 ** i), step * (2 ** (i+1)), 3, padding=1))
            conv_down.append(torch.nn.Conv2d(step * (2 ** (i+1)), step * (2 ** (i+1)), 3, padding=1))
            conv_down.append(torch.nn.Conv2d(step * (2 ** (i+1)), step * (2 ** (i+1)), 3, padding=1))
            norm_down += [_group_norm(step * (2 ** (i+1)))] * 3
            if i > 0:
                in_ch  = step * (2**(n_layers + 1 - i))
                out_ch = step * (2**(n_layers - i))
                up_in, up_out = max(in_ch, output_dim), max(out_ch, output_dim)
                conv_up.append(torch.nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2))
                conv_up.append(torch.nn.Conv2d(max(2 * out_ch, out_ch + output_dim), up_out, kernel_size=3, padding=1))
                conv_up.append(torch.nn.Conv2d(up_out, up_out, kernel_size=3, padding=1))
                norm_up += [_group_norm(up_out)] * 3
        self.conv_down = torch.nn.ModuleList(conv_down)
        self.norm_down = torch.nn.ModuleList(norm_down)
        self.conv_up   = torch.nn.ModuleList(conv_up)
        self.norm_up   = torch.nn.ModuleList(norm_up)

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        x= x.transpose(-1,-2).transpose(-2, -3)
        x = F.adaptive_avg_pool2d(x, (self.hidden_dim, self.hidden_dim))
        # Conv -> GroupNorm -> ReLU throughout (except output_conv, which stays a
        # raw linear projection to the regression targets). VisionModel is a ~21
        # conv layer encoder-decoder trained from scratch with no normalization
        # at all previously -- a well-known recipe for training to stall early,
        # which matched what the pretraining logs showed (dense_loss, a plain
        # unambiguous per-pixel regression, plateaued almost immediately).
        x = self.input_norm(self.input_conv(x)).relu()
        conv_down_outs = []
        for i in range(self.n_layers):
            x = self.norm_down[(3 * i)](self.conv_down[(3 * i)](x)).relu()
            x = self.norm_down[(3 * i) + 1](self.conv_down[(3 * i) + 1](x)).relu()
            x = self.norm_down[(3 * i) + 2](self.conv_down[(3 * i) + 2](x)).relu()
            conv_down_outs.append(x)
            if i < self.n_layers - 1:
                x = F.max_pool2d(x, kernel_size=2)

        bottleneck = conv_down_outs[-1]
        signature = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)

        for i in range(self.n_layers - 1):
            x = self.norm_up[(3 * i)](self.conv_up[(3 * i)](x)).relu()
            x = self.norm_up[(3 * i) + 1](self.conv_up[(3 * i) + 1](torch.cat((conv_down_outs[self.n_layers - 2 - i], x), dim=-3))).relu()
            x = self.norm_up[(3 * i) + 2](self.conv_up[(3 * i) + 2](x)).relu()

        x = self.output_conv(x)
        x = F.adaptive_avg_pool2d(x, (H, W))
        x = x.transpose(-3, -2).transpose(-2, -1)
        return x, signature



class SceneTransformer(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = torch.nn.Parameter(torch.tensor((1.,)))
        self.lin_in = torch.nn.Linear(latent_dim + 9, hidden_dim)
        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim + 9),
            torch.nn.Tanh(),
        )
        self.transformer = torch.nn.Transformer(hidden_dim, batch_first=True)
        self.pooling = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(256),
            torch.nn.Linear(256, 256),
            torch.nn.AdaptiveAvgPool1d(128),
            torch.nn.Linear(128, 128),
            torch.nn.AdaptiveAvgPool1d(64),
            torch.nn.Linear(64, 64),
            torch.nn.AdaptiveAvgPool1d(32),
            torch.nn.Linear(32, 32),
            torch.nn.AdaptiveAvgPool1d(16),
            torch.nn.Linear(16, 16),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.scene_params = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, objects: torch.Tensor | list[SceneObject]):
        if not isinstance(objects, torch.Tensor):
            objects = torch.cat([o.flatten().unsqueeze(0) for o in objects], dim=0)
            return_as_list = True
        out = self.lin_in(objects)
        out = self.transformer(out, out)
        #params = self.pooling(out.transpose(-1,-2)).squeeze(-1)
        #params = self.scene_params(params)
        out = self.lin_out(out)   # all values in [-1, 1] from Tanh

        d = self.latent_dim
        emb   = out[..., :d]
        trans = out[..., d:d + 3]
        rot   = out[..., d + 3:d + 6] * torch.pi          # [-1, 1] → [-π, π]
        color = (out[..., d + 6:d + 9] + 1) / 2           # [-1, 1] → [0, 1]
        out   = torch.cat([emb, trans, rot, color], dim=-1)

        return {
            "objects" : [SceneObject.inflate(o) for o in out.unbind(0)] if return_as_list else out,
            "scale"   : F.softplus(self.scale),            # always positive
        }


class MultiViewTransformer(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, image_emb_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = torch.nn.Parameter(torch.tensor((1.,)))

        self.seed_in  = torch.nn.Linear(latent_dim + 9, hidden_dim)
        self.pose_in  = torch.nn.Linear(6, hidden_dim)
        self.image_in = torch.nn.Linear(image_emb_dim, hidden_dim)

        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim + 9),
            torch.nn.Tanh(),
        )
        # Pure cross-attention, no self-attention among the N seed tokens: a
        # control test (debug_pretanh.py --no-attention) showed trans/emb only
        # collapse to identical values across objects when attention is in the
        # loop — bypassing it entirely preserved (and even grew) per-object
        # diversity. nn.TransformerDecoderLayer bundles self-attention over the
        # seeds together with cross-attention to the image context in one block
        # with no way to disable just the self-attention half, so this uses
        # transformer.py's Transformer, which only ever does attn(to_q(x), to_k(y),
        # to_v(y)) — no x-attends-to-x pass.
        self.transformer = CrossAttentionTransformer(hidden_dim, num_heads=8, num_layers=2)

    def forward(self, seeds: torch.Tensor | list[SceneObject], image_emb: torch.Tensor, image_pose: list[CameraParams]):
        """
        seeds:      (N, latent_dim + 9) or list[SceneObject] — object queries to refine
        image_emb:  (n_images, image_emb_dim) — vision-model signatures, one per view
        image_pose: list of n_images CameraParams — camera pose for each view
        """
        return_as_list = False
        if not isinstance(seeds, torch.Tensor):
            seeds = torch.cat([s.flatten().unsqueeze(0) for s in seeds], dim=0)
            return_as_list = True

        pose = torch.stack([p.flatten() for p in image_pose], dim=0)   # (n_images, 6)

        seed_tokens  = self.seed_in(seeds)                             # (N, hidden_dim)
        image_tokens = self.image_in(image_emb) + self.pose_in(pose)   # (n_images, hidden_dim)

        # cross attention: seeds (query/x) attend to the per-view image context (key-value/y).
        # transformer.py's layers expect a batch dim (batch, seq, embed_dim); these
        # tokens are unbatched (seq, embed_dim), so add/drop a size-1 batch dim.
        out = self.transformer(seed_tokens.unsqueeze(0), image_tokens.unsqueeze(0)).squeeze(0)
        out = self.lin_out(out)   # (N, latent_dim + 9), all values in [-1, 1] from Tanh

        d = self.latent_dim
        emb   = out[..., :d]
        trans = out[..., d:d + 3]
        rot   = out[..., d + 3:d + 6] * torch.pi          # [-1, 1] → [-π, π]
        color = (out[..., d + 6:d + 9] + 1) / 2           # [-1, 1] → [0, 1]
        out   = torch.cat([emb, trans, rot, color], dim=-1)

        return {
            "objects" : [SceneObject.inflate(o) for o in out.unbind(0)] if return_as_list else out,
            "scale"   : F.softplus(self.scale),            # always positive
        }

