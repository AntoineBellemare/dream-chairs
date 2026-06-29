"""Embed shapes and text into a shared space for cosine-similarity matching.

Default backend `clip_multiview`: render the point cloud from several angles to 2D
images, encode each with an OpenCLIP image encoder, and average — placing shapes in
the same CLIP space as the symbol descriptor text. This mirrors Antoine's existing
CLIP descriptor-matching approach, extended to 3D.

`openshape` backend: scaffolded for native point-cloud<->text matching (Antoine's
proposal to Mark). See README before enabling.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .generate import Shape
from .utils import l2_normalize


# ---------------------------------------------------------------------------
# Multi-view rasterizer (pure numpy -> PIL; no OpenGL, Windows-friendly)
# ---------------------------------------------------------------------------
def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def render_point_views(points: np.ndarray, colors: np.ndarray | None,
                       num_views: int, image_size: int, point_size: int) -> list[Image.Image]:
    pts = points.astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pts, axis=1).max() + 1e-6
    pts = pts / scale

    col = None
    if colors is not None:
        col = np.clip(colors.astype(np.float32), 0.0, 1.0)

    images: list[Image.Image] = []
    size = image_size
    margin = 0.12
    tilt = _rot_x(np.deg2rad(20.0))  # slight downward tilt for a 3/4 view

    for i in range(num_views):
        rot = _rot_y(2 * np.pi * i / num_views) @ tilt
        p = pts @ rot.T
        x, y, z = p[:, 0], p[:, 1], p[:, 2]

        u = ((x * (1 - margin) + 1) / 2 * (size - 1)).astype(np.int32)
        v = ((1 - (y * (1 - margin) + 1) / 2) * (size - 1)).astype(np.int32)

        order = np.argsort(z)  # far (small z) first so near points overwrite
        u, v, zo = u[order], v[order], z[order]

        if col is not None:
            c = col[order]
        else:
            zr = (zo - zo.min()) / (np.ptp(zo) + 1e-6)  # 0 far .. 1 near
            shade = 0.15 + 0.6 * zr                     # near = darker text on white
            c = np.stack([1.0 - shade] * 3, axis=1)

        canvas = np.ones((size, size, 3), dtype=np.float32)
        r = point_size
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                uu = np.clip(u + dx, 0, size - 1)
                vv = np.clip(v + dy, 0, size - 1)
                canvas[vv, uu] = c  # last write wins -> near occludes far

        images.append(Image.fromarray((canvas * 255).astype(np.uint8)))
    return images


# ---------------------------------------------------------------------------
# Embedder backends
# ---------------------------------------------------------------------------
class CLIPMultiViewEmbedder:
    def __init__(self, device: str, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", num_views: int = 8,
                 image_size: int = 224, point_size: int = 2, num_points: int = 4096):
        import open_clip
        import torch  # noqa: F401

        self.device = device
        self.num_views = num_views
        self.image_size = image_size
        self.point_size = point_size
        self.num_points = num_points

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_shape(self, shape: Shape) -> np.ndarray:
        import torch

        points = shape.points
        if points is None and shape.mesh is not None:
            import trimesh
            points, _ = trimesh.sample.sample_surface(shape.mesh, self.num_points)
            points = np.asarray(points)
        imgs = render_point_views(points, shape.colors, self.num_views,
                                  self.image_size, self.point_size)
        batch = torch.stack([self.preprocess(im) for im in imgs]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vec = feats.mean(dim=0)
            vec = vec / vec.norm()
        return vec.cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


class OpenShapeEmbedder:
    """Native point-cloud <-> text embedding via OpenShape (no rendering).

    The vendored OpenShape PPAT encoder maps a point cloud [1, 6, N] (xyz+rgb,
    centered + unit-sphere) into a CLIP-aligned space; text is encoded with the
    matching CLIP, so cosine similarity compares 3D and text directly in one space.

      vitl14 / vitb32 -> OpenAI CLIP text encoder   (clip_backend='openai')
      vitg14          -> OpenCLIP ViT-bigG-14 text  (clip_backend='openclip')
    """

    def __init__(self, device: str, checkpoint: str = "openshape-pointbert-vitl14-rgb",
                 clip_backend: str = "openai", clip_model_name: str = "ViT-L/14",
                 clip_pretrained: str = "laion2b_s39b_b160k", num_points: int = 10000,
                 swap_yz=None):
        import torch  # noqa: F401
        from .openshape_vendor import load_pc_encoder, model_spec

        self.device = device
        self.num_points = num_points
        spec = model_spec.get(checkpoint, {})
        # OpenShape g14 wants z-up; b32/l14 want y-up. We treat our clouds as y-up and
        # swap to z-up only when the encoder needs it (override via swap_yz).
        self.swap_yz = (spec.get("gravity") == "z") if swap_yz is None else bool(swap_yz)

        self.pc_encoder = load_pc_encoder(checkpoint)
        self.pc_encoder.eval()

        self.clip_backend = clip_backend
        if clip_backend == "openai":
            import clip
            self.clip_model, _ = clip.load(clip_model_name, device=device)
            self._tokenize = clip.tokenize
        else:
            import open_clip
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=clip_pretrained, device=device)
            self._tokenize = open_clip.get_tokenizer(clip_model_name)
        self.clip_model.eval()

    def _prep_points(self, points: np.ndarray, colors: np.ndarray | None) -> np.ndarray:
        pts = points.astype(np.float32)
        pts = pts - pts.mean(axis=0, keepdims=True)
        scale = np.sqrt((pts ** 2).sum(axis=1)).max() + 1e-6
        pts = pts / scale
        n = pts.shape[0]
        idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
        pts = pts[idx]
        if colors is not None:
            col = np.clip(colors.astype(np.float32), 0.0, 1.0)[idx]
        else:
            col = np.full((self.num_points, 3), 0.4, dtype=np.float32)
        feat = np.concatenate([pts, col], axis=1)  # [N, 6] -> x,y,z,r,g,b
        if self.swap_yz:
            feat = feat[:, [0, 2, 1, 3, 4, 5]]
        return feat

    def embed_shape(self, shape: Shape) -> np.ndarray:
        import torch

        points = shape.points
        if points is None and shape.mesh is not None:
            import trimesh
            points, _ = trimesh.sample.sample_surface(shape.mesh, self.num_points)
            points = np.asarray(points)
        feat = self._prep_points(points, shape.colors)
        x = torch.from_numpy(np.ascontiguousarray(feat.T[None])).float().to(self.device)
        with torch.no_grad():
            v = self.pc_encoder(x)
            v = v / v.norm(dim=-1, keepdim=True)
        return v[0].float().cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        tokens = self._tokenize(list(texts)).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().astype(np.float32)


def build_embedder(cfg, device: str):
    emb = cfg.embedding
    backend = emb.get("backend", "clip_multiview")
    num_points = int(cfg.generation.get("num_points", 4096))
    if backend == "clip_multiview":
        clip = emb.get("clip", {})
        mv = emb.get("multiview", {})
        return CLIPMultiViewEmbedder(
            device=device,
            model_name=clip.get("model_name", "ViT-B-32"),
            pretrained=clip.get("pretrained", "laion2b_s34b_b79k"),
            num_views=int(mv.get("num_views", 8)),
            image_size=int(mv.get("image_size", 224)),
            point_size=int(mv.get("point_size", 2)),
            num_points=num_points,
        )
    if backend == "openshape":
        os_cfg = emb.get("openshape", {})
        return OpenShapeEmbedder(
            device=device,
            checkpoint=os_cfg.get("checkpoint", "openshape-pointbert-vitl14-rgb"),
            clip_backend=os_cfg.get("clip_backend", "openai"),
            clip_model_name=os_cfg.get("clip_model_name", "ViT-L/14"),
            clip_pretrained=os_cfg.get("clip_pretrained", "laion2b_s39b_b160k"),
            num_points=int(os_cfg.get("num_points", 10000)),
            swap_yz=os_cfg.get("swap_yz", None),
        )
    raise ValueError(f"Unknown embedding backend: {backend!r}")
