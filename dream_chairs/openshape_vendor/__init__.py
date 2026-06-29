# Vendored from OpenShape (Apache-2.0):
#   https://huggingface.co/OpenShape/openshape-demo-support  (openshape/__init__.py)
# Loads the released OpenShape point-cloud encoders (PPAT/PointBERT) which embed a
# point cloud [B, 6, N] (xyz+rgb) into a CLIP-aligned space, enabling direct
# point-cloud <-> text cosine similarity with no rendering. DGL dependency removed
# (see pointnet_util.py).
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from .ppat_rgb import Projected, PointPatchTransformer


def module(state_dict: dict, name):
    return {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if k.startswith(name + '.')}


def G14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 512 * 3, 256, 384, 0.2, 64, 6),
        nn.Linear(512, 1280)
    )
    model.load_state_dict(module(s['state_dict'], 'module'))
    return model


def L14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6),
        nn.Linear(512, 768)
    )
    model.load_state_dict(module(s, 'pc_encoder'))
    return model


def B32(s):
    model = PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6)
    model.load_state_dict(module(s, 'pc_encoder'))
    return model


model_list = {
    "openshape-pointbert-vitb32-rgb": B32,
    "openshape-pointbert-vitl14-rgb": L14,
    "openshape-pointbert-vitg14-rgb": G14,
}

# Embedding dim + the CLIP space each encoder is aligned to (for the text side).
model_spec = {
    "openshape-pointbert-vitb32-rgb": {"dim": 512, "clip": "ViT-B/32", "gravity": "y"},
    "openshape-pointbert-vitl14-rgb": {"dim": 768, "clip": "ViT-L/14", "gravity": "y"},
    "openshape-pointbert-vitg14-rgb": {"dim": 1280, "clip": "ViT-bigG-14", "gravity": "z"},
}


def load_pc_encoder(name):
    # weights_only=True restricts unpickling to tensors/primitives, removing the
    # arbitrary-code-execution risk of loading an external pickle. OpenShape
    # checkpoints are plain tensor state-dicts, so this is sufficient.
    path = hf_hub_download("OpenShape/" + name, "model.pt")
    s = torch.load(path, map_location='cpu', weights_only=True)
    model = model_list[name](s).eval()
    if torch.cuda.is_available():
        model.cuda()
    return model
