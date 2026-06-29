"""Presentation-quality 3D rendering (matplotlib) — the Point-E notebook look:
a bounding-box cube, coolwarm depth/height coloring, anti-aliased perspective.

Used for figures and `render_dreams.py --matplotlib`. Independent of the fast splat
rasterizer in embed.py (which exists only to feed CLIP and stays as-is).
"""
from __future__ import annotations

import io

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def _normalize(points: np.ndarray) -> np.ndarray:
    p = points.astype(float)
    p = p - p.mean(axis=0)
    return p / (np.abs(p).max() + 1e-6)


def _fig_to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_points_3d(points: np.ndarray, colors: np.ndarray | None = None,
                     color_mode: str = "height", cmap: str = "coolwarm",
                     elev: float = 18, azim: float = -62, px: int = 360,
                     point_size: float = 7.0, dpi: int = 100) -> Image.Image:
    """Render a point cloud as a coolwarm-by-height 3D scatter inside a cube.
    color_mode: 'height' (default, legible regardless of model RGB) or 'rgb'."""
    p = _normalize(points)
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    if color_mode == "rgb" and colors is not None:
        kw = dict(c=np.clip(colors, 0, 1))
    else:
        kw = dict(c=p[:, 1], cmap=cmap)  # y = up
    # plot (x, z) horizontal, y vertical
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=point_size, depthshade=True,
               linewidths=0, **kw)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor((0.65, 0.65, 0.65, 1.0))
        axis.line.set_color((0.65, 0.65, 0.65, 1.0))
    return _fig_to_image(fig)


def render_mesh_3d(mesh, n: int = 30000, **kw) -> Image.Image:
    """Render a mesh by densely sampling its surface — smooth, OpenGL-free."""
    try:
        pts = np.asarray(mesh.sample(n))
    except Exception:
        pts = np.asarray(mesh.vertices)
    kw.setdefault("point_size", 2.5)
    return render_points_3d(pts, color_mode="height", **kw)
