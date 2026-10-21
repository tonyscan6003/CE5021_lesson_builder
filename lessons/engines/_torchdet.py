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


def rpn_proposals(key: str, img: np.ndarray, model="faster_rcnn") -> dict:
    """The region proposal network's output inside Faster R-CNN: the proposals
    passed to the second stage (after the RPN's own NMS), in decreasing order
    of objectness, in the coordinates of `img`."""
    def run():
        import torch
        net, _ = _load(model)
        with torch.no_grad():
            images, _ = net.transform([_tensor(img)])
            feats = net.backbone(images.tensors)
            props, _ = net.rpn(images, feats)
        h, w = images.image_sizes[0]
        s = torch.tensor([img.shape[1] / w, img.shape[0] / h] * 2, device=props[0].device)
        return dict(boxes=(props[0] * s).cpu().numpy().astype(np.float32))
    return TV.cached(f"rpn_{model}_{key}", dict(key=key, shape=img.shape, model=model), run)


def class_activation_map(key: str, img: np.ndarray, side=448) -> dict:
    """CAM (Zhou et al. 2016) from ResNet-50, whose last convolution is followed
    by global average pooling and one dense layer: the class score is the
    weighted sum of the pooled maps, so the same weights applied to the maps
    themselves show where the class evidence lies."""
    def run():
        import torch
        import torch.nn.functional as F
        net, w = _load("resnet50")
        s = side / max(img.shape[:2])
        x = _tensor(img)[None]
        x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            f = net.layer4(net.layer3(net.layer2(net.layer1(
                net.maxpool(net.relu(net.bn1(net.conv1(x))))))))
            logits = net.fc(f.mean((2, 3)))
            c = int(logits.argmax(1))
            cam = torch.einsum("k,khw->hw", net.fc.weight[c], f[0])
        return dict(cls=np.array(c), prob=logits.softmax(1)[0, c].cpu().numpy(),
                    cam=cam.cpu().numpy().astype(np.float32),
                    fmap_shape=np.array(f.shape[1:]))
    return TV.cached(f"cam_{key}", dict(key=key, shape=img.shape, side=side), run)


ZOO = {
    # name: (builder, weights enum, single- or two-stage)
    "Faster R-CNN v2": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights", 2),
    "Faster R-CNN": ("fasterrcnn_resnet50_fpn", "FasterRCNN_ResNet50_FPN_Weights", 2),
    "Faster R-CNN MobileNetV3": ("fasterrcnn_mobilenet_v3_large_fpn",
                                 "FasterRCNN_MobileNet_V3_Large_FPN_Weights", 2),
    "RetinaNet v2": ("retinanet_resnet50_fpn_v2", "RetinaNet_ResNet50_FPN_V2_Weights", 1),
    "RetinaNet": ("retinanet_resnet50_fpn", "RetinaNet_ResNet50_FPN_Weights", 1),
    "FCOS": ("fcos_resnet50_fpn", "FCOS_ResNet50_FPN_Weights", 1),
    "SSD300": ("ssd300_vgg16", "SSD300_VGG16_Weights", 1),
    "SSDlite320": ("ssdlite320_mobilenet_v3_large", "SSDLite320_MobileNet_V3_Large_Weights", 1),
}


def coco_map(name: str) -> float:
    """COCO val2017 box mAP published with torchvision's weights (read from the
    weights' metadata, not typed in)."""
    from torchvision.models import detection
    w = getattr(detection, ZOO[name][1]).DEFAULT
    return float(w.meta["_metrics"]["COCO-val2017"]["box_map"])


def time_detectors(key: str, img: np.ndarray, repeats=3) -> dict:
    """Measured inference time of each detector in ZOO on one image, on the
    device of this build (median of `repeats` runs after a warm-up)."""
    def run():
        import time
        import torch
        from torchvision.models import detection
        names, secs = [], []
        x = [_tensor(img)]
        for name, (fn, wname, _) in ZOO.items():
            w = getattr(detection, wname).DEFAULT
            net = getattr(detection, fn)(weights=w).eval().to(TV.device())
            with torch.no_grad():
                net(x)
                t = []
                for _ in range(repeats):
                    t0 = time.perf_counter(); net(x)
                    if TV.device().type == "cuda":
                        torch.cuda.synchronize()
                    t.append(time.perf_counter() - t0)
            names.append(name); secs.append(float(np.median(t)))
            del net
        return dict(names=np.array(names), seconds=np.array(secs),
                    device=np.array(str(TV.device())))
    return TV.cached(f"timing_{key}", dict(key=key, shape=img.shape, models=list(ZOO),
                                           repeats=repeats, device=str(TV.device())), run)


def raw_detections(key: str, img: np.ndarray, score_min=0.3) -> dict:
    """Faster R-CNN v2 with its own NMS switched off (IoU threshold 1.0), so
    the overlapping boxes that NMS removes can be shown and NMS stepped through."""
    def run():
        import torch
        from torchvision.models import detection
        w = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        net = detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_nms_thresh=1.0,
                                                    box_score_thresh=score_min,
                                                    detections_per_img=500).eval().to(TV.device())
        with torch.no_grad():
            out = net([_tensor(img)])[0]
        return dict(boxes=out["boxes"].cpu().numpy().astype(np.float32),
                    labels=out["labels"].cpu().numpy().astype(np.int16),
                    scores=out["scores"].cpu().numpy().astype(np.float32))
    return TV.cached(f"raw_frcnn_{key}", dict(key=key, shape=img.shape, score_min=score_min), run)


def retinanet_internals(key: str, img: np.ndarray, gt: np.ndarray, cls_name: str,
                        gammas=(0.0, 0.5, 1.0, 2.0, 5.0)) -> dict:
    """RetinaNet's real anchors and classification outputs for one image.

    Anchors are matched to the ground truth `gt` with RetinaNet's rule (IoU >=
    0.5 positive, < 0.4 negative, otherwise ignored). For every anchor the
    sigmoid outputs over all classes are scored against its target (the GT
    class for a positive, all zeros for a negative) with the focal loss for
    each gamma (gamma = 0 is cross-entropy). Returned per anchor: level, IoU,
    label (1 / 0 / -1), p_t of the target class, and the loss per gamma; plus
    the FPN feature-map shapes and channel-mean maps of each level.
    """
    def run():
        import torch
        import torch.nn.functional as F
        from torchvision.models import detection
        from torchvision.ops import box_iou
        w = detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        net, _ = _load("retinanet")
        cats = w.meta["categories"]
        c = cats.index(cls_name)
        with torch.no_grad():
            images, _ = net.transform([_tensor(img)])
            feats = list(net.backbone(images.tensors).values())
            head = net.head(feats)
            anchors = net.anchor_generator(images, feats)[0]
            logits = head["cls_logits"][0]                          # (A, K)
            h, w_ = images.image_sizes[0]
            s = torch.tensor([w_ / img.shape[1], h / img.shape[0]] * 2, device=anchors.device)
            g = torch.as_tensor(gt, dtype=torch.float32, device=anchors.device) * s
            iou = box_iou(anchors, g).max(1).values
            label = torch.full_like(iou, -1)
            label[iou < 0.4] = 0
            label[iou >= 0.5] = 1
            target = torch.zeros_like(logits)
            target[label == 1, c] = 1
            p = logits.sigmoid()
            pt_all = torch.where(target == 1, p, 1 - p)
            ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            losses = {f"loss_g{g_}": ((1 - pt_all) ** g_ * ce).sum(1).cpu().numpy().astype(np.float32)
                      for g_ in gammas}
            pt = torch.where(label == 1, p[:, c], 1 - p.max(1).values)
            sizes = [f.shape[-2] * f.shape[-1] for f in feats]
            per = len(anchors) // sum(sizes)
            level = np.concatenate([np.full(n * per, k) for k, n in enumerate(sizes)])
            maps = {f"map{k}": f[0].norm(dim=0).cpu().numpy().astype(np.float16)
                    for k, f in enumerate(feats)}
        return dict(level=level.astype(np.int8), iou=iou.cpu().numpy().astype(np.float32),
                    label=label.cpu().numpy().astype(np.int8), pt=pt.cpu().numpy().astype(np.float32),
                    anchors=(anchors / s).cpu().numpy().astype(np.float32),
                    shapes=np.array([f.shape[-2:] for f in feats]), gammas=np.array(gammas),
                    **losses, **maps)
    return TV.cached(f"retina_{key}", dict(key=key, shape=img.shape, gt=np.asarray(gt).tolist(),
                                           cls=cls_name, gammas=list(gammas)), run)


def ssd_default_boxes() -> dict:
    """SSD300's default boxes per feature map, from torchvision's own model."""
    def run():
        import torch
        from torchvision.models import detection
        net, _ = _load("ssd300")
        # torchvision's first extra block is conv5 + the modified pool5 + fc6 + fc7;
        # fc6 is the dilated 3 x 3 convolution, hooked here as it is not a prediction map
        fc6 = net.backbone.extra[0][-1][1]
        seen = {}
        hook = fc6.register_forward_hook(lambda m, i, o: seen.update(fc6=o.shape))
        with torch.no_grad():
            x = torch.zeros(1, 3, 300, 300, device=TV.device())
            feats = list(net.backbone(x).values())
            from torchvision.models.detection.image_list import ImageList
            il = ImageList(x, [(300, 300)])
            anchors = net.anchor_generator(il, feats)[0]
        hook.remove()
        shapes = np.array([f.shape[-2:] for f in feats])
        chans = np.array([f.shape[1] for f in feats])
        per_cell = np.array(net.anchor_generator.num_anchors_per_location())
        return dict(shapes=shapes, channels=chans, per_cell=per_cell,
                    anchors=anchors.cpu().numpy().astype(np.float32),
                    fc6_shape=np.array(seen["fc6"][1:]),
                    fc6_conv=np.array([*fc6.kernel_size, *fc6.dilation]))
    return TV.cached("ssd300_default_boxes", dict(model="ssd300", v=3), run)


def fpn_shapes(h=800, w=1216) -> dict:
    """Every feature-map shape (C, H, W) of RetinaNet's ResNet-50 FPN for an
    h x w input, read with forward hooks: the backbone stages C1-C5, the merged
    maps M3-M5 (lateral 1 x 1 conv plus the upsampled level above, the input of
    each 3 x 3 output conv), the levels P3-P7, and the number of input channels
    of the P6 conv, which shows whether P6 is computed from C5 or from P5."""
    def run():
        import torch
        net, _ = _load("retinanet")
        b = net.backbone
        seen, hooks = {}, []
        def keep(name, inp=False):
            return lambda m, i, o: seen.update({name: tuple((i[0] if inp else o).shape[1:])})
        hooks.append(b.body.relu.register_forward_hook(keep("C1")))
        for k in range(1, 5):
            hooks.append(getattr(b.body, f"layer{k}").register_forward_hook(keep(f"C{k + 1}")))
        for k, blk in enumerate(b.fpn.layer_blocks):
            hooks.append(blk.register_forward_hook(keep(f"M{k + 3}", inp=True)))
            hooks.append(blk.register_forward_hook(keep(f"P{k + 3}")))
        hooks.append(b.fpn.extra_blocks.p6.register_forward_hook(keep("P6")))
        hooks.append(b.fpn.extra_blocks.p7.register_forward_hook(keep("P7")))
        with torch.no_grad():
            b(torch.zeros(1, 3, h, w, device=TV.device()))
        for hk in hooks:
            hk.remove()
        out = {k: np.array(v) for k, v in seen.items()}
        out["p6_in"] = np.array(b.fpn.extra_blocks.p6.in_channels)
        out["p6_use_P5"] = np.array(bool(b.fpn.extra_blocks.use_P5))
        return out
    return TV.cached("retinanet_fpn_shapes", dict(model="retinanet", h=h, w=w, v=1), run)


def semantic_scores(key: str, img: np.ndarray, every=2) -> dict:
    """DeepLabV3's full output: per-class softmax scores (21 Pascal VOC
    classes incl. background) at every `every`-th pixel, plus the logits at the
    network's own output resolution (stride 8) for resolution experiments."""
    def run():
        import torch
        import torch.nn.functional as F
        net, w = _load("deeplab")
        x = w.transforms(resize_size=None)(_tensor(img))[None]
        with torch.no_grad():
            feats = net.backbone(x)["out"]
            logits8 = net.classifier(feats)
        logits = F.interpolate(logits8, size=img.shape[:2], mode="bilinear", align_corners=False)
        prob = logits.softmax(1)[0, :, ::every, ::every]
        return dict(prob=prob.cpu().numpy().astype(np.float16),
                    logits8=logits8[0].cpu().numpy().astype(np.float32))
    return TV.cached(f"deeplab_scores_{key}", dict(key=key, shape=img.shape, every=every), run)


def vgg_fully_convolutional(key: str, img: np.ndarray, cls: int) -> dict:
    """VGG-16's classifier made fully convolutional, as in FCN: fc6 (25088 ->
    4096) becomes a 7 x 7 convolution over pool5, fc7 and fc8 become 1 x 1
    convolutions, with exactly the same weights. Applied to an image larger
    than 224 x 224 it gives a map of class scores instead of one vector."""
    def run():
        import torch
        import torch.nn.functional as F
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval().to(TV.device())
        fc6, fc7, fc8 = vgg.classifier[0], vgg.classifier[3], vgg.classifier[6]
        x = TV.to_tensor(img)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        with torch.no_grad():
            p5 = vgg.features((x - mean) / std)
            h = F.relu(F.conv2d(p5, fc6.weight.view(4096, 512, 7, 7), fc6.bias))
            h = F.relu(F.conv2d(h, fc7.weight.view(4096, 4096, 1, 1), fc7.bias))
            s = F.conv2d(h, fc8.weight.view(1000, 4096, 1, 1), fc8.bias)
            prob = s.softmax(1)[0]
            # check: on a 224 x 224 input the score map is 1 x 1 and equals the classifier
            x224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            z = vgg.features((x224 - mean) / std)
            a = vgg.classifier(torch.flatten(vgg.avgpool(z), 1))
            b = F.conv2d(F.relu(F.conv2d(F.relu(F.conv2d(z, fc6.weight.view(4096, 512, 7, 7),
                         fc6.bias)), fc7.weight.view(4096, 4096, 1, 1), fc7.bias)),
                         fc8.weight.view(1000, 4096, 1, 1), fc8.bias)
        return dict(score=prob[cls].cpu().numpy().astype(np.float32),
                    top=prob.argmax(0).cpu().numpy().astype(np.int16),
                    p5_shape=np.array(p5.shape[1:]),
                    check=np.array(float((a[0] - b[0, :, 0, 0]).abs().max())))
    return TV.cached(f"vgg_fcn_{key}", dict(key=key, shape=img.shape, cls=cls), run)


def deeplab_internals(key: str, img: np.ndarray, point: tuple, cls_name: str) -> dict:
    """DeepLabV3 internals for one image: the backbone's 2048-channel output
    (stride 8), the ASPP dilation rates of the model, and the effective
    receptive field of one output pixel: |d score(cls, point) / d input|,
    summed over the colour channels."""
    def run():
        import torch
        net, w = _load("deeplab")
        cats = w.meta["categories"]
        c = cats.index(cls_name)
        x = w.transforms(resize_size=None)(_tensor(img))[None].requires_grad_(True)
        feats = net.backbone(x)["out"]
        logits = net.classifier(feats)
        up = torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear",
                                             align_corners=False)
        up[0, c, point[1], point[0]].backward()
        erf = x.grad[0].abs().sum(0)
        aspp = net.classifier[0]
        rates = [m[0].dilation[0] for m in aspp.convs[1:4]]
        return dict(erf=erf.detach().cpu().numpy().astype(np.float32),
                    feats=feats[0].detach().cpu().numpy().astype(np.float16),
                    rates=np.array(rates), aspp_out=np.array(aspp.project[0].out_channels))
    return TV.cached(f"deeplab_int_{key}", dict(key=key, shape=img.shape, point=list(point),
                                                cls=cls_name), run)


def mask_head_outputs(key: str, img: np.ndarray, score_min=0.5) -> dict:
    """Mask R-CNN's own m x m mask predictions (before they are resized into
    the boxes), captured from the mask head, for the final detections."""
    def run():
        import torch
        net, _ = _load("mask_rcnn")
        store = {}
        h = net.roi_heads.mask_predictor.register_forward_hook(
            lambda m, i, o: store.__setitem__("logits", o))
        with torch.no_grad():
            out = net([_tensor(img)])[0]
        h.remove()
        lg = store["logits"]                                  # (N, classes, 28, 28)
        lab = out["labels"]
        m = lg[torch.arange(len(lab)), lab].sigmoid()
        keep = out["scores"] >= score_min
        return dict(small=m[keep].cpu().numpy().astype(np.float16),
                    boxes=out["boxes"][keep].cpu().numpy().astype(np.float32),
                    labels=lab[keep].cpu().numpy().astype(np.int16),
                    scores=out["scores"][keep].cpu().numpy().astype(np.float32),
                    full=(out["masks"][keep][:, 0] * 255).to(torch.uint8).cpu().numpy())
    return TV.cached(f"mask_head_{key}", dict(key=key, shape=img.shape, score_min=score_min), run)


def cnn_features(key: str, img: np.ndarray) -> dict:
    """ResNet-50's 2048-value feature vector (after global average pooling) for
    an image patch: the 'learned features' R-CNN passes to its SVMs."""
    def run():
        import torch
        net, w = _load("resnet50")
        x = w.transforms()(_tensor(img))[None]
        with torch.no_grad():
            f = net.layer4(net.layer3(net.layer2(net.layer1(net.maxpool(net.relu(
                net.bn1(net.conv1(x))))))))
        return dict(vec=f.mean((2, 3))[0].cpu().numpy().astype(np.float32))
    return TV.cached(f"cnnfeat_{key}", dict(key=key, shape=img.shape), run)


def cam_parts(key: str, img: np.ndarray, k=3, side=448) -> dict:
    """The pieces of a class activation map from ResNet-50: the predicted class,
    the k feature maps that contribute most to its score (weight x pooled value),
    their weights and pooled values, and the CAM itself."""
    def run():
        import torch
        import torch.nn.functional as F
        net, _ = _load("resnet50")
        s = side / max(img.shape[:2])
        x = F.interpolate(_tensor(img)[None], scale_factor=s, mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        with torch.no_grad():
            f = net.layer4(net.layer3(net.layer2(net.layer1(net.maxpool(net.relu(
                net.bn1(net.conv1((x - mean) / std))))))))[0]
            g = f.mean((1, 2))
            logits = net.fc(g[None])[0]
            c = int(logits.argmax())
            wv = net.fc.weight[c]
            contrib = wv * g
            top = torch.argsort(contrib, descending=True)[:k]
            cam = torch.einsum("k,khw->hw", wv, f)
        return dict(cls=np.array(c), channels=top.cpu().numpy(), maps=f[top].cpu().numpy(),
                    weights=wv[top].cpu().numpy(), pooled=g[top].cpu().numpy(),
                    cam=cam.cpu().numpy(), n=np.array(f.shape[0]),
                    share=(contrib[top].sum() / contrib.sum()).cpu().numpy())
    return TV.cached(f"camparts_{key}", dict(key=key, shape=img.shape, k=k, side=side), run)
