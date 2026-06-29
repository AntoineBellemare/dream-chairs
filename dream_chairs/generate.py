"""Text -> 3D shape generation via Point-E or Shap-E.

Heavy imports (point_e / shap_e / torch) happen lazily inside each generator's
__init__ so that importing this module (and running --dry-run) never requires the
generation stack to be installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Shape:
    """A generated shape, normalized to a point cloud for downstream embedding."""

    id: str
    prompt: str
    points: np.ndarray              # (N, 3) float32
    colors: np.ndarray | None = None  # (N, 3) in 0..1, optional
    mesh: Any = None                # trimesh.Trimesh, optional (Shap-E)


class PointEGenerator:
    """OpenAI Point-E: text -> colored point cloud."""

    def __init__(self, device: str, base_model: str = "base40M-textvec",
                 upsample: bool = True, guidance_scale: float = 3.0, num_points: int = 4096):
        import torch  # noqa: F401
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
        from point_e.diffusion.sampler import PointCloudSampler
        from point_e.models.configs import MODEL_CONFIGS, model_from_config
        from point_e.models.download import load_checkpoint

        self.device = device
        base = model_from_config(MODEL_CONFIGS[base_model], device)
        base.eval()
        base_diff = diffusion_from_config(DIFFUSION_CONFIGS[base_model])
        base.load_state_dict(load_checkpoint(base_model, device))

        models = [base]
        diffusions = [base_diff]
        if upsample:
            up = model_from_config(MODEL_CONFIGS["upsample"], device)
            up.eval()
            up_diff = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
            up.load_state_dict(load_checkpoint("upsample", device))
            models.append(up)
            diffusions.append(up_diff)
            self.sampler = PointCloudSampler(
                device=device, models=models, diffusions=diffusions,
                num_points=[1024, max(num_points - 1024, 0)],
                aux_channels=["R", "G", "B"],
                guidance_scale=[guidance_scale, 0.0],
                model_kwargs_key_filter=("texts", ""),
            )
        else:
            self.sampler = PointCloudSampler(
                device=device, models=models, diffusions=diffusions,
                num_points=[num_points],
                aux_channels=["R", "G", "B"],
                guidance_scale=[guidance_scale],
                model_kwargs_key_filter=("texts",),
            )

    def generate(self, dream_id: str, prompt: str) -> Shape:
        samples = None
        for x in self.sampler.sample_batch_progressive(
            batch_size=1, model_kwargs=dict(texts=[prompt])
        ):
            samples = x
        pc = self.sampler.output_to_point_clouds(samples)[0]
        points = np.asarray(pc.coords, dtype=np.float32)
        colors = None
        if all(k in pc.channels for k in ("R", "G", "B")):
            colors = np.stack([pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=1)
            colors = np.asarray(colors, dtype=np.float32)
            if colors.max() > 1.5:
                colors = colors / 255.0
        return Shape(id=dream_id, prompt=prompt, points=points, colors=colors)


class ShapEGenerator:
    """OpenAI Shap-E: text -> mesh, sampled to a point cloud for embedding."""

    def __init__(self, device: str, guidance_scale: float = 15.0,
                 karras_steps: int = 64, use_fp16: bool = True, num_points: int = 4096):
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.diffusion.sample import sample_latents
        from shap_e.models.download import load_config, load_model
        from shap_e.util.notebooks import decode_latent_mesh

        self.device = device
        self.guidance_scale = guidance_scale
        self.karras_steps = karras_steps
        self.use_fp16 = use_fp16
        self.num_points = num_points
        self._sample_latents = sample_latents
        self._decode = decode_latent_mesh
        self.xm = load_model("transmitter", device=device)
        self.model = load_model("text300M", device=device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    def generate(self, dream_id: str, prompt: str) -> Shape:
        import trimesh

        latents = self._sample_latents(
            batch_size=1, model=self.model, diffusion=self.diffusion,
            guidance_scale=self.guidance_scale, model_kwargs=dict(texts=[prompt]),
            progress=True, clip_denoised=True, use_fp16=self.use_fp16,
            use_karras=True, karras_steps=self.karras_steps,
            sigma_min=1e-3, sigma_max=160, s_churn=0,
        )
        tri = self._decode(self.xm, latents[0]).tri_mesh()
        verts = np.asarray(tri.verts, dtype=np.float32)
        faces = np.asarray(tri.faces, dtype=np.int64)

        vcol = None
        channels = getattr(tri, "vertex_channels", {}) or {}
        if all(c in channels for c in ("R", "G", "B")):
            vcol = np.stack([np.asarray(channels[c]) for c in ("R", "G", "B")], axis=1)
            vcol = np.clip(vcol.astype(np.float32), 0.0, 1.0)

        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_colors=((vcol * 255).astype(np.uint8) if vcol is not None else None),
            process=False,
        )
        points, fidx = trimesh.sample.sample_surface(mesh, self.num_points)
        colors = vcol[faces[fidx]].mean(axis=1).astype(np.float32) if vcol is not None else None
        return Shape(id=dream_id, prompt=prompt, points=np.asarray(points, dtype=np.float32),
                     colors=colors, mesh=mesh)


def build_generator(cfg, device: str):
    gen = cfg.generation
    backend = gen.get("backend", "point_e")
    num_points = int(gen.get("num_points", 4096))
    if backend == "point_e":
        pe = gen.get("point_e", {})
        return PointEGenerator(
            device=device,
            base_model=pe.get("base_model", "base40M-textvec"),
            upsample=bool(pe.get("upsample", True)),
            guidance_scale=float(pe.get("guidance_scale", 3.0)),
            num_points=num_points,
        )
    if backend == "shap_e":
        se = gen.get("shap_e", {})
        return ShapEGenerator(
            device=device,
            guidance_scale=float(se.get("guidance_scale", 15.0)),
            karras_steps=int(se.get("karras_steps", 64)),
            use_fp16=bool(se.get("use_fp16", True)),
            num_points=num_points,
        )
    raise ValueError(f"Unknown generation backend: {backend!r}")


def random_shape(dream_id: str, prompt: str, num_points: int = 4096, seed: int | None = None) -> Shape:
    """A blobby random point cloud — used by --dry-run to exercise embed/classify
    without downloading the generation models."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(num_points, 3)).astype(np.float32)
    pts *= rng.uniform(0.5, 1.5, size=(1, 3)).astype(np.float32)  # anisotropic scale
    return Shape(id=dream_id, prompt=prompt, points=pts, colors=None)
