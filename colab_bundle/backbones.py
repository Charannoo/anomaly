#!/usr/bin/env python3
"""Frozen backbones for XMV-AD-H, fully self-contained.

  * DINOBackbone  : DINO ViT-B/8 (`vit_base_patch8_224_dino`, timm).
  * PointTransformer : the exact Point-MAE architecture from
        third_party/m3dm/models/models.py (same weights), rebuilt with PURE
        PyTorch grouping (masked FPS + masked KNN) so the ancient nvidia
        `pointnet2_ops` / `knn_cuda` CUDA extensions are never compiled.

Grouping semantics match the repo: centers are chosen by FPS over the exact
same valid point cloud, neighbours by exact KNN over the same cloud; padded
batch positions are masked out so they can never be selected.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tifffile


# ----------------------------------------------------------------------
# Point operators (pure PyTorch ports of third_party/m3dm/models/pointnet2_utils.py)
# ----------------------------------------------------------------------

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def masked_farthest_point_sample(xyz, valid, npoint):
    """FPS over valid points only; invalid (padded) rows can never win."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    low = torch.full_like(xyz[..., 0], -1e9) if False else None
    del low
    distance = torch.full((B, N), 1e10, device=device)
    distance = torch.where(valid, distance, torch.full_like(distance, -1e9))
    farthest = valid.float().argmax(dim=1)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = torch.where(valid, dist, torch.full_like(dist, -1e9))
        mask = dist < distance
        distance[mask] = dist[mask]
        distance = torch.where(valid, distance, torch.full_like(distance, -1e9))
        farthest = torch.max(distance, -1)[1]
    return centroids


def masked_knn(xyz, center, valid, k, center_chunk=256):
    """Exact K-nearest valid neighbours per centroid (brute force, chunked)."""
    B, P, _ = xyz.shape
    G = center.shape[1]
    inf = torch.full((1,), float("inf"), dtype=xyz.dtype, device=xyz.device)
    outs = []
    for b in range(B):
        parts = []
        for s in range(0, G, center_chunk):
            c = center[b, s:s + center_chunk]
            d = torch.cdist(c, xyz[b])
            d = torch.where(valid[b].unsqueeze(0), d, inf.to(d.dtype))
            _, nidx = torch.topk(d, k=k, largest=False, dim=1)
            parts.append(nidx)
        outs.append(torch.cat(parts, dim=0).unsqueeze(0))
    return torch.cat(outs, dim=0)


class Group(nn.Module):
    """Same interface and output as repo Group (models/models.py) minus CUDA deps."""

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, valid):
        B, num_points, _ = xyz.shape
        center_idx = masked_farthest_point_sample(xyz, valid, self.num_group)
        center = index_points(xyz, center_idx)
        idx = masked_knn(xyz, center, valid, self.group_size)
        ori_idx = idx
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(B * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(B, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        pg = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(pg.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        dpr = drop_path_rate if isinstance(drop_path_rate, list) else [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    """Point-MAE transformer: architecture identical to models/models.py."""

    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = group_size
        self.num_group = num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(embed_dim=self.trans_dim, depth=self.depth,
                                         drop_path_rate=dpr, num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location="cpu", weights_only=False)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith("MAE_encoder"):
                base_ckpt[k[len("MAE_encoder."):]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith("base_model"):
                base_ckpt[k[len("base_model."):]] = base_ckpt[k]
                del base_ckpt[k]
        incompatible = self.load_state_dict(base_ckpt, strict=False)
        print("[PointTransformer] loaded %s  missing=%d unexpected=%d"
              % (bert_ckpt_path, len(incompatible.missing_keys), len(incompatible.unexpected_keys)))

    def forward(self, pts, valid=None):
        """pts: (B, 3, N)   valid: (B, N) bool mask of real points."""
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)
        if valid is None:
            valid = torch.ones((B, N), dtype=torch.bool, device=pts.device)
        neighborhood, center, ori_idx, center_idx = self.group_divider(pts, valid)
        group_input_tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)
        x = group_input_tokens
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat(feature_list, dim=1)
        return x, center, ori_idx, center_idx


# ----------------------------------------------------------------------
# DINO RGB backbone (timm)
# ----------------------------------------------------------------------

class DINOBackbone(nn.Module):
    def __init__(self, name="vit_base_patch8_224_dino", device="cuda"):
        super().__init__()
        import timm
        self.backbone = timm.create_model(name, pretrained=True)
        self.backbone.eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x):
        m = self.backbone
        try:
            feat = m.forward_features(x)
        except Exception:
            try:
                feat = m.forward(x)
            except Exception:
                feat = m.get_intermediate_layers(x, n=1)[0]
        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            elif "x_norm_patchtokens" in feat:
                feat = feat["x_norm_patchtokens"]
            else:
                feat = next(iter(feat.values()))
        if feat.ndim == 4:
            return feat
        if feat.ndim == 3:
            prefix_tokens = int(getattr(m, "num_prefix_tokens", 1))
            if feat.shape[1] == DINO_TOKENS + prefix_tokens:
                feat = feat[:, prefix_tokens:]
            assert feat.shape[1] == DINO_TOKENS, "unexpected DINO token count %d" % feat.shape[1]
        B, N, C = feat.shape
        return feat.permute(0, 2, 1).reshape(B, C, DINO_GRID, DINO_GRID)


# ----------------------------------------------------------------------
# Cloud I/O (kept in lock-step with the repo canonical extractor)
# ----------------------------------------------------------------------

RGB_SIZE = 224
GRID_PIX = RGB_SIZE * RGB_SIZE
DINO_GRID = RGB_SIZE // 8
DINO_TOKENS = DINO_GRID * DINO_GRID


def read_tiff_organized_pc(path):
    return tifffile.imread(path)


def chunked_interpolating_points(xyz1, xyz2, points2, chunk_size=30000):
    """xyz1 (B,3,N), xyz2 (B,3,S), points2 (B,D,S) -> (B,D,N)."""
    xyz1_p = xyz1.permute(0, 2, 1)     # (B, N, 3)
    B, N, _ = xyz1_p.shape
    xyz2_p = xyz2.permute(0, 2, 1)
    points2_p = points2.permute(0, 2, 1)
    _, _, S = xyz2.shape
    _, _, D = points2.shape

    def _square_distance(src, dst):
        b, n, _ = src.shape
        _, m, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(b, n, 1)
        dist += torch.sum(dst ** 2, -1).view(b, 1, m)
        return dist

    out_parts = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xyz1_chunk = xyz1_p[:, start:end, :]
        dists = _square_distance(xyz1_chunk, xyz2_p)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        pts_chunk = torch.sum(index_points(points2_p, idx) * weight.view(B, end - start, 3, 1), dim=2)
        out_parts.append(pts_chunk)
    return torch.cat(out_parts, dim=1).permute(0, 2, 1)


def load_xyz_batch(tiff_paths, device):
    """Return (clouds(B,P,3), valids(B,P), nz_lists) on device; P = max length."""
    clouds, nz_lists = [], []
    for p in tiff_paths:
        organized_pc = read_tiff_organized_pc(p)
        tensor = torch.from_numpy(organized_pc).permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(tensor, size=(RGB_SIZE, RGB_SIZE), mode="nearest")
        org = resized.squeeze(0).permute(1, 2, 0).numpy()
        unorganized = org.reshape(GRID_PIX, 3)
        nonzero = np.nonzero(np.all(unorganized != 0, axis=1))[0]
        if nonzero.shape[0] < 1024:
            if nonzero.shape[0] == 0:
                cloud = np.zeros((1024, 3), dtype=np.float32)
                nz_grid = np.arange(1024, dtype=np.int64)
            else:
                rep = int(np.ceil(1024 / nonzero.shape[0]))
                cloud = np.tile(unorganized[nonzero, :], (rep, 1))[:1024]
                nz_grid = nonzero.astype(np.int64)
        else:
            cloud = unorganized[nonzero, :]
            nz_grid = nonzero.astype(np.int64)
        clouds.append(torch.from_numpy(np.ascontiguousarray(cloud, dtype=np.float32)))
        nz_lists.append(nz_grid)

    P = max(int(c.shape[0]) for c in clouds)
    padded = torch.zeros((len(clouds), P, 3), dtype=torch.float32)
    valid = torch.zeros((len(clouds), P), dtype=torch.bool)
    for b, c in enumerate(clouds):
        padded[b, :c.shape[0]] = c
        valid[b, :c.shape[0]] = True
    return padded.to(device), valid.to(device), nz_lists


if __name__ == "__main__":
    import sys
    print("backbones.py OK (load:%s cuda:%s)" % (os.path.basename(sys.argv[0]),
                                                 torch.cuda.is_available()))
