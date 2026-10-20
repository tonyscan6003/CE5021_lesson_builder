"""Pre-trained detection and segmentation models (torchvision), for Section 3.

    classify(img)          ResNet-50, ImageNet: top-k labels
    detect(img, model)     Faster R-CNN / RetinaNet / SSD300, COCO: boxes, labels, scores
    instance_masks(img)    Mask R-CNN, COCO: boxes, labels, scores, masks
    semantic(img)          DeepLabV3, the 20 Pascal VOC classes + background

Every call is cached in data/_torch/ through _torchvis.cached, keyed on the
model, the image key and the settings, so a build reuses the stored outputs.
Images are passed by their credit key so the cache key names the input.
Boxes are (x1, y1, x2, y2) in the pixel coordinates of the image passed in.
"""
from __future__ import annotations

import numpy as np

from . import _torchvis as TV
from ._detmetrics import iou  # noqa: F401  (re-exported for the engines)

_MODELS: dict = {}

DETECTORS = {
    "faster_rcnn": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights"),
    "retinanet": ("retinanet_resnet50_fpn_v2", "RetinaNet_ResNet50_FPN_V2_Weights"),
    "ssd300": ("ssd300_vgg16", "SSD300_VGG16_Weights"),
    "mask_rcnn": ("maskrcnn_resnet50_fpn_v2", "MaskRCNN_ResNet50_FPN_V2_Weights"),
}


def _load(name):
    if name not in _MODELS:
        import torch
        from torchvision.models import detection, segmentation
        import torchvision.models as models
        if name in DETECTORS:
            fn, w = DETECTORS[name]
            weights = getattr(detection, w).DEFAULT
            net = getattr(detection, fn)(weights=weights, box_score_thresh=0.0)
        elif name == "deeplab":
            weights = segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            net = segmentation.deeplabv3_resnet50(weights=weights)
        elif name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)
        for p in net.parameters():
            p.requires_grad_(False)
        _MODELS[name] = (net.eval().to(TV.device()), weights)
    return _MODELS[name]


def categories(model: str) -> list[str]:
    """Class names, index = label id, without loading the network weights."""
    from torchvision.models import detection, segmentation
    import torchvision.models as models
    if model in DETECTORS:
        return getattr(detection, DETECTORS[model][1]).DEFAULT.meta["categories"]
    if model == "deeplab":
        return segmentation.DeepLabV3_ResNet50_Weights.DEFAULT.meta["categories"]
    return models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]


def _tensor(img):
    import torch
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().to(TV.device())


def classify(key: str, img: np.ndarray, k=5) -> dict:
    def run():
        import torch
        net, w = _load("resnet50")
        with torch.no_grad():
            p = net(w.transforms()(_tensor(img))[None]).softmax(1)[0]
        s, i = p.topk(k)
        return dict(labels=i.cpu().numpy(), scores=s.cpu().numpy())
    return TV.cached(f"classify_{key}", dict(key=key, shape=img.shape, k=k), run)


def detect(key: str, img: np.ndarray, model="faster_rcnn", score_min=0.05) -> dict:
    """All detections above `score_min`, sorted by score (NMS already applied
    by the model, as in deployment)."""
    def run():
        import torch
        net, _ = _load(model)
        with torch.no_grad():
            out = net([_tensor(img)])[0]
        keep = out["scores"] >= score_min
        r = dict(boxes=out["boxes"][keep].cpu().numpy().astype(np.float32),
                 labels=out["labels"][keep].cpu().numpy().astype(np.int16),
                 scores=out["scores"][keep].cpu().numpy().astype(np.float32))
        if model == "mask_rcnn":
            r["masks"] = (out["masks"][keep][:, 0] > 0.5).cpu().numpy()
        return r
    return TV.cached(f"{model}_{key}", dict(key=key, shape=img.shape, model=model,
                                            score_min=score_min), run)


def instance_masks(key: str, img: np.ndarray, score_min=0.5) -> dict:
    return detect(key, img, "mask_rcnn", score_min)


def semantic(key: str, img: np.ndarray) -> dict:
    """Per-pixel class scores of DeepLabV3 → label map (H, W) and softmax."""
    def run():
        import torch
        import torch.nn.functional as F
        net, w = _load("deeplab")
        x = w.transforms(resize_size=None)(_tensor(img))[None]
        with torch.no_grad():
            logits = net(x)["out"]
        logits = F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
        prob = logits.softmax(1)[0]
        return dict(labels=prob.argmax(0).cpu().numpy().astype(np.uint8),
                    confidence=(prob.max(0).values.cpu().numpy() * 255).astype(np.uint8))
    return TV.cached(f"deeplab_{key}", dict(key=key, shape=img.shape), run)
