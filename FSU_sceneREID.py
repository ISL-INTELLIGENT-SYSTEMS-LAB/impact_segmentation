"""SAM3 + CLIP + LightGlue + SegMASt3R instance matching.
Run: python FSU_sceneREID.py --config sceneREID_config.yaml
Heavy dependencies are imported only for inference; assignment is testable on CPU.
"""
import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
from scipy.optimize import linear_sum_assignment
from huggingface_hub import login

token = os.environ["HF_TOKEN"]
login(token)

def assign_instances(scores, threshold=0.55, margin=0.04, allowed=None):
    """Reject before assignment; bilateral margins and private dummy columns.

    Each real edge must beat every other candidate in both its row and column.
    Scores must be finite in [0,1]. Below-threshold objects are unmatched;
    strong candidates that fail ambiguity/geometry checks are ambiguous.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('Similarity must be a finite 2D matrix')
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError('Similarity must be in [0,1]')
    if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError('Threshold and margin must be in [0,1]')
    n, m = scores.shape
    allowed = np.ones((n, m), dtype=bool) if allowed is None else np.asarray(allowed, bool)
    if allowed.shape != scores.shape:
        raise ValueError('Geometry gate shape differs from similarity')
    eligible = np.zeros((n, m), dtype=bool)
    row_margin = np.zeros((n, m))
    col_margin = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # Evaluate alternatives BEFORE geometry rejection to retain uncertainty.
            r = np.max(np.delete(scores[i], j)) if m > 1 else 0.0
            c = np.max(np.delete(scores[:, j], i)) if n > 1 else 0.0
            row_margin[i, j] = scores[i, j] - r
            col_margin[i, j] = scores[i, j] - c
            eligible[i, j] = (allowed[i, j] and scores[i, j] >= threshold
                              and scores[i, j] > r and scores[i, j] > c
                              and row_margin[i, j] >= margin and col_margin[i, j] >= margin)
    matches = []
    if n and m:
        # Unmatched = zero gain. Each reference has a private dummy target.
        cost = np.full((n, m + n), 1e6)
        cost[:, :m][eligible] = -scores[eligible]
        cost[np.arange(n), m + np.arange(n)] = 0.0
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if j < m and eligible[i, j]:
                matches.append({'reference_index': int(i), 'target_index': int(j),
                                'score': float(scores[i, j]),
                                'row_margin': float(row_margin[i, j]),
                                'column_margin': float(col_margin[i, j])})
    used0 = {x['reference_index'] for x in matches}
    used1 = {x['target_index'] for x in matches}
    def status(index, axis, used):
        if index in used:
            return 'matched'
        candidates = scores[index, :] if axis == 0 else scores[:, index]
        return 'ambiguous' if candidates.size and candidates.max() >= threshold else 'unmatched'
    ref = [status(i, 0, used0) for i in range(n)]
    tgt = [status(j, 1, used1) for j in range(m)]
    return {'matches': matches, 'reference_status': ref, 'target_status': tgt,
            'unmatched_reference': [i for i, s in enumerate(ref) if s == 'unmatched'],
            'unmatched_target': [i for i, s in enumerate(tgt) if s == 'unmatched'],
            'ambiguous_reference': [i for i, s in enumerate(ref) if s == 'ambiguous'],
            'ambiguous_target': [i for i, s in enumerate(tgt) if s == 'ambiguous']}


def crop_from_mask(image, mask, bbox):
    x0, y0 = np.floor(bbox[:2]).astype(int)
    x1, y1 = np.ceil(bbox[2:]).astype(int)
    h, w = image.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    crop = image[y0:y1, x0:x1].copy()
    if not crop.size:
        raise ValueError('Empty detection crop')
    crop[~mask[y0:y1, x0:x1]] = 0
    return crop


def lightglue_geometric_similarity(points0, points1, confidences, min_inliers=8,
                                   ransac_threshold=3.0, sigma=4.0):
    import cv2
    points0 = np.asarray(points0, np.float32).reshape(-1, 2)
    points1 = np.asarray(points1, np.float32).reshape(-1, 2)
    confidences = np.asarray(confidences, float).reshape(-1)
    n = len(points0)
    if n != len(points1) or n != len(confidences):
        raise ValueError('LightGlue points and scores must be aligned')
    if n < max(4, min_inliers):
        return 0.0
    H, inliers = cv2.findHomography(points0, points1, cv2.RANSAC, ransac_threshold)
    if H is None or inliers is None or not np.isfinite(H).all():
        return 0.0
    good = inliers.ravel().astype(bool)
    if good.sum() < min_inliers:
        return 0.0
    projected = cv2.perspectiveTransform(points0[None], H)[0]
    residual = np.linalg.norm(projected - points1, axis=1)
    # Homography reprojection residual, NOT raw inter-crop coordinate distance.
    quality = np.exp(-residual[good] ** 2 / sigma ** 2)
    return float(np.clip(np.sum(quality * confidences[good]) / n, 0, 1))


class AppearanceBranch:
    def __init__(self, config, device):
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.torch = __import__('torch')
        self.device = device
        self.mode = config['CLIP_MODE']
        self.parts = int(config['NUM_PARTS'])
        if self.mode not in ('global', 'parts') or self.parts < 1:
            raise ValueError('CLIP_MODE must be global/parts; NUM_PARTS >= 1')
        self.processor = CLIPImageProcessor.from_pretrained(config['VIT_MODEL_NAME'])
        self.model = CLIPVisionModel.from_pretrained(config['VIT_MODEL_NAME']).eval().to(device)

    def encode(self, crops):
        from PIL import Image
        torch = self.torch
        embeddings = []
        with torch.inference_mode():
            for crop in crops:
                inputs = self.processor(images=Image.fromarray(crop), return_tensors='pt').to(self.device)
                output = self.model(**inputs)
                if self.mode == 'global':
                    emb = output.pooler_output
                else:
                    tokens = output.last_hidden_state[:, 1:]
                    side = int(tokens.shape[1] ** 0.5)
                    if side * side != tokens.shape[1] or self.parts > side:
                        raise ValueError('Invalid CLIP patch grid/part count')
                    grid = tokens.reshape(1, side, side, -1)
                    bands = torch.tensor_split(grid, self.parts, dim=1)
                    emb = torch.cat([b.mean(dim=(1, 2)) for b in bands], dim=1)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                embeddings.append(emb[0].cpu().numpy())
        return np.stack(embeddings)


class LocalBranch:
    def __init__(self, config, device):
        from lightglue import LightGlue, SuperPoint
        self.torch = __import__('torch')
        self.device = device
        self.config = config
        self.extractor = SuperPoint(max_num_keypoints=config['LIGHTGLUE_MAX_KEYPOINTS']).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)

    def encode(self, crops):
        features = []
        with self.torch.inference_mode():
            for crop in crops:
                tensor = self.torch.from_numpy(crop.copy()).float().permute(2, 0, 1).to(self.device) / 255
                # Keep keypoints in crop coordinates for the geometric verification.
                features.append(self.extractor.extract(tensor, resize=None))
        return features

    def score(self, a, b):
        from lightglue.utils import rbd
        with self.torch.inference_mode():
            result = rbd(self.matcher({'image0': a, 'image1': b}))
        fa, fb = rbd(a), rbd(b)
        indices = result['matches0'].detach().cpu().numpy()
        valid = indices >= 0
        p0 = fa['keypoints'].detach().cpu().numpy()[valid]
        p1 = fb['keypoints'].detach().cpu().numpy()[indices[valid]]
        confidence = result['matching_scores0'].detach().cpu().numpy()[valid]
        return lightglue_geometric_similarity(p0, p1, confidence,
            self.config['LIGHTGLUE_MIN_INLIERS'], self.config['LIGHTGLUE_RANSAC_THRESHOLD'],
            self.config['LIGHTGLUE_SIGMA'])


class GeometryBranch:
    """Expose public SegMASt3R forward + pooling + matcher scores for SAM3 masks."""
    def __init__(self, config, device):
        import torch
        repo = os.path.abspath(config['SEGMAST3R_REPO'])
        if not os.path.isfile(os.path.join(repo, 'model_infer.py')):
            raise FileNotFoundError('SEGMAST3R_REPO must point to the official cloned repository')
        sys.path.insert(0, repo)
        # Use the same imports and checkpoint format as the official demo.
        from configs.default import cfg
        from model_infer import MASt3RSegFeatInfer
        from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling
        self.pool = masked_average_pooling
        cfg = cfg.clone()
        cfg.defrost()
        cfg.merge_from_file(os.path.join(repo, config['SEGMAST3R_CONFIG']))
        self.model = MASt3RSegFeatInfer(cfg)
        ckpt = torch.load(config['SEGMAST3R_CHECKPOINT'], map_location='cpu', weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt['state_dict'], strict=False)
        # Do not silently run with random encoder weights.
        critical = [k for k in missing if k in dict(self.model.named_parameters())]
        if critical:
            raise RuntimeError(f'Checkpoint missing trained parameters: {critical[:12]}')
        if unexpected:
            print(f'SegMASt3R: {len(unexpected)} unused checkpoint keys')
        self.model.prepare(device)
        self.device = device
        self.size = int(config['SEGMAST3R_IMAGE_SIZE'])
        if self.size <= 0 or self.size % 16:
            raise ValueError('SEGMAST3R_IMAGE_SIZE must be positive and divisible by 16')
        self.normalization = config['SEGMAST3R_NORMALIZATION']
        if self.normalization not in ('minus_one_one', 'zero_one'):
            raise ValueError('Unknown SegMASt3R normalization')

    def prepare_view(self, image, masks, name):
        import cv2
        import torch
        # Square resize follows the official demo; paired image and mask transforms
        # are identical. Normalization can reproduce either public input path.
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(image).float().permute(2, 0, 1)[None].to(self.device) / 255
        if self.normalization == 'minus_one_one':
            img = img * 2 - 1
        masks = torch.from_numpy(masks.astype(np.float32))[:, None]
        masks = torch.nn.functional.interpolate(masks, size=(self.size, self.size), mode='nearest')[:, 0]
        mask_tensor = masks[None].to(self.device)
        return {'img': img, 'true_shape': torch.tensor([[self.size, self.size]],
                device=self.device, dtype=torch.int64), 'instance': [name]}, mask_tensor

    def score(self, images, masks):
        import torch
        v0, m0 = self.prepare_view(images[0], masks[0], 'view1_0')
        v1, m1 = self.prepare_view(images[1], masks[1], 'view2_0')
        # Tiny masks may disappear during resize. Exclude them from pooling and
        # scatter scores back to original SAM3 indices; they cannot pass the gate.
        keep0 = m0[0].sum(dim=(1, 2)) > 0
        keep1 = m1[0].sum(dim=(1, 2)) > 0
        n, m = len(masks[0]), len(masks[1])
        real = np.zeros((n, m), dtype=float)
        dust0, dust1 = np.ones(n), np.ones(m)
        if keep0.any() and keep1.any():
            with torch.inference_mode():
                desc0, desc1 = self.model(v0, v1)
                a = self.pool(desc0.permute(0, 3, 1, 2), m0[:, keep0])
                b = self.pool(desc1.permute(0, 3, 1, 2), m1[:, keep1])
                transport = self.model.feature_matcher(a, b).exp()[0].cpu().numpy()
            if not np.isfinite(transport).all():
                raise RuntimeError('Nonfinite SegMASt3R transport scores')
            i = np.flatnonzero(keep0.cpu().numpy())
            j = np.flatnonzero(keep1.cpu().numpy())
            real[np.ix_(i, j)] = transport[:-1, :-1]
            dust0[i], dust1[j] = transport[:-1, -1], transport[-1, :-1]
        gate = (real > dust0[:, None]) & (real > dust1[None, :])
        return np.clip(real, 0, 1), gate, dust0, dust1


def get_instances(image, model, processor, config, device):
    import torch
    from PIL import Image
    inputs = processor(images=Image.fromarray(image), text=config['TEXT_PROMPT'], return_tensors='pt').to(device)
    with torch.inference_mode():
        output = model(**inputs)
    result = processor.post_process_instance_segmentation(output,
        threshold=config['SAM_SCORE_THRESHOLD'], mask_threshold=config['SAM_MASK_THRESHOLD'],
        target_sizes=inputs['original_sizes'].tolist())[0]
    masks = result['masks'].detach().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.size == 0:
        masks = np.zeros((0, *image.shape[:2]), dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != image.shape[:2]:
        raise ValueError('SAM3 returned masks with unexpected dimensions')
    masks = masks.astype(bool)
    scores = result['scores'].detach().cpu().numpy()
    boxes = result['boxes'].detach().cpu().numpy().reshape(-1, 4)
    keep = masks.sum(axis=(1, 2)) >= config['MIN_MASK_PIXELS']
    masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
    for i, mask in enumerate(masks):
        y, x = np.where(mask)
        boxes[i] = [x.min(), y.min(), x.max() + 1, y.max() + 1]
    return masks, boxes, scores, np.flatnonzero(keep)


def visualize(images, masks, report, output):
    from PIL import Image, ImageDraw
    panels = []
    pair_ids = {0: {}, 1: {}}
    for identity, match in enumerate(report['matches']):
        pair_ids[0][match['reference_index']] = identity
        pair_ids[1][match['target_index']] = identity
    for side in range(2):
        panel = images[side].copy()
        statuses = report['reference_status' if side == 0 else 'target_status']
        labels = []
        for i, mask in enumerate(masks[side]):
            identity = pair_ids[side].get(i)
            if identity is None:
                color = (255, 160, 0) if statuses[i] == 'ambiguous' else (150, 150, 150)
                label = f'{i}: {statuses[i]}'
            else:
                color = tuple(np.random.default_rng(identity + 41).integers(70, 256, 3).tolist())
                label = f'{i}: pair {identity}'
            panel[mask] = (0.5 * panel[mask] + 0.5 * np.array(color)).astype(np.uint8)
            y, x = np.where(mask)
            labels.append((int(x.min()), int(y.min()), label))
        pil = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil)
        for x, y, label in labels:
            draw.text((x, y), label, fill='white', stroke_width=2, stroke_fill='black')
        panels.append(pil)
    canvas = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)))
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width, 0))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sceneREID_config.yaml')
    args = parser.parse_args()
    import torch
    import yaml
    from PIL import Image, ImageOps
    from transformers import Sam3Model, Sam3Processor
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if any(k in config for k in ('A', 'B', 'C', 'Y')):
        raise ValueError('Use the new named weights; legacy A/B/C/Y settings are unsupported')
    device = torch.device(config['DEVICE'] if config['DEVICE'] != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    weights = np.array([config['CLIP_WEIGHT'], config['LIGHTGLUE_WEIGHT'], config['SEGMAST3R_WEIGHT']], float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, with positive sum')
    if not config['SEGMAST3R_ENABLED'] and weights[2] != 0:
        raise ValueError('Disabled SegMASt3R requires SEGMAST3R_WEIGHT: 0; no silent fallback')
    if config['REQUIRE_SEGMAST3R_SUPPORT'] and not config['SEGMAST3R_ENABLED']:
        raise ValueError('Geometry gate requires SegMASt3R enabled')
    weights /= weights.sum()
    # Standard HF_TOKEN authentication or existing HF login is sufficient.
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    images = []
    for key in ('IMG0_PTH', 'IMG1_PTH'):
        with Image.open(config[key]) as im:
            images.append(np.array(ImageOps.exif_transpose(im).convert('RGB')))
    sam = Sam3Model.from_pretrained(config['SAM_MODEL_ID'], token=token).eval().to(device)
    processor = Sam3Processor.from_pretrained(config['SAM_MODEL_ID'], token=token)
    instances = [get_instances(im, sam, processor, config, device) for im in images]
    masks = [x[0] for x in instances]
    del sam, processor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    n, m = map(len, masks)
    matrices = [np.zeros((n, m), float) for _ in range(3)]
    gate = np.ones((n, m), bool)
    dust0, dust1 = None, None
    if n and m:
        crops = [[crop_from_mask(im, mask, box) for mask, box in zip(item[0], item[1])]
                 for im, item in zip(images, instances)]
        if weights[0] > 0:
            branch = AppearanceBranch(config, device)
            a, b = [branch.encode(c) for c in crops]
            matrices[0] = np.clip(a @ b.T, 0, 1)
            del branch
        if weights[1] > 0:
            branch = LocalBranch(config, device)
            features0, features1 = [branch.encode(c) for c in crops]
            for i in range(n):
                for j in range(m):
                    matrices[1][i, j] = branch.score(features0[i], features1[j])
            del branch, features0, features1
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if config['SEGMAST3R_ENABLED'] and (weights[2] > 0 or config['REQUIRE_SEGMAST3R_SUPPORT']):
            branch = GeometryBranch(config, device)
            matrices[2], geometry_gate, dust0, dust1 = branch.score(images, masks)
            if config['REQUIRE_SEGMAST3R_SUPPORT']:
                gate = geometry_gate
            del branch
    similarity = sum(w * matrix for w, matrix in zip(weights, matrices))
    report = assign_instances(similarity, config['MATCH_THRESHOLD'], config['MATCH_MARGIN'], gate)
    report.update({'images': [config['IMG0_PTH'], config['IMG1_PTH']], 'config': config,
                   'normalized_weights': weights.tolist(), 'similarity': similarity.tolist(),
                   'clip_scores': matrices[0].tolist(), 'lightglue_scores': matrices[1].tolist(),
                   'segmast3r_scores': matrices[2].tolist(), 'geometry_gate': gate.tolist(),
                   'segmast3r_reference_dustbin': None if dust0 is None else dust0.tolist(),
                   'segmast3r_target_dustbin': None if dust1 is None else dust1.tolist(),
                   'reference_detections': [{'index': i, 'sam_index': int(item), 'box': instances[0][1][i].tolist(),
                                             'score': float(instances[0][2][i])} for i, item in enumerate(instances[0][3])],
                   'target_detections': [{'index': i, 'sam_index': int(item), 'box': instances[1][1][i].tolist(),
                                          'score': float(instances[1][2][i])} for i, item in enumerate(instances[1][3])]})
    output = os.path.join(config['PROJECT_NAME'], config['EXPERIMENT_NAME'],
                          datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, 'matches.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    np.savez_compressed(os.path.join(output, 'detections.npz'), reference_masks=masks[0], target_masks=masks[1])
    if config['VISUALIZE_FIG']:
        visualize(images, masks, report, os.path.join(output, 'matches.png'))
    print(json.dumps({k: report[k] for k in ('matches', 'reference_status', 'target_status')}, indent=2))
    print(f'Results: {os.path.abspath(output)}')


if __name__ == '__main__':
    main()
