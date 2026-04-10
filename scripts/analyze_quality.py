#!/usr/bin/env python3
"""Analyze image quality of all training images using classical CV and BRISQUE/NIMA metrics."""
import cv2
import numpy as np
import json
import sys
from pathlib import Path

DATA_DIR = Path('data/training')

BRISQUE_MODEL = str(Path(__file__).resolve().parent.parent / 'models' / 'brisque_model_live.yml')
BRISQUE_RANGE = str(Path(__file__).resolve().parent.parent / 'models' / 'brisque_range_live.yml')

# Try loading BRISQUE (opencv-contrib-python)
try:
    cv2.quality.QualityBRISQUE_compute(np.zeros((10, 10, 3), dtype=np.uint8), BRISQUE_MODEL, BRISQUE_RANGE)
    HAS_BRISQUE = True
except Exception:
    HAS_BRISQUE = False

# Try loading NIMA (lightweight CNN quality scorer)
HAS_NIMA = False
nima_model = None
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    class NIMA(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            base.classifier = nn.Sequential(nn.Dropout(0.75), nn.Linear(1280, 10), nn.Softmax(dim=1))
            self.base = base

        def forward(self, x):
            return self.base(x)

    nima_model = NIMA()
    nima_model.eval()
    nima_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    HAS_NIMA = True
except ImportError:
    pass


def analyze(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return {'file': str(img_path.relative_to(DATA_DIR)), 'error': 'could not read image'}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    contrast = float(gray.std())
    noise = float(cv2.Laplacian(gray, cv2.CV_64F).std())

    issues = []
    if blur < 100:
        issues.append('blurry')
    if brightness < 40:
        issues.append('too dark')
    elif brightness > 220:
        issues.append('overexposed')
    if contrast < 20:
        issues.append('low contrast')
    if w < 320 or h < 240:
        issues.append('low resolution')

    result = {
        'file': str(img_path.relative_to(DATA_DIR)),
        'resolution': f'{w}x{h}',
        'blur_score': round(blur, 1),
        'brightness': round(brightness, 1),
        'contrast': round(contrast, 1),
        'noise': round(noise, 1),
        'issues': issues,
        'sufficient_for_detection': len(issues) == 0
    }

    # BRISQUE score (lower = better, 0-100 typical)
    if HAS_BRISQUE:
        try:
            score = cv2.quality.QualityBRISQUE_compute(img, BRISQUE_MODEL, BRISQUE_RANGE)
            result['brisque'] = round(score[0], 1)
            if score[0] > 60:
                result['issues'].append('poor BRISQUE score')
                result['sufficient_for_detection'] = False
        except Exception:
            pass

    # NIMA score (1-10, higher = better aesthetic/technical quality)
    if HAS_NIMA and nima_model is not None:
        try:
            import torch
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = nima_transform(rgb).unsqueeze(0)
            with torch.no_grad():
                probs = nima_model(tensor).squeeze()
            # Weighted mean: score = sum(i * p_i) for i in 1..10
            score = float((torch.arange(1, 11, dtype=torch.float32) * probs).sum())
            result['nima'] = round(score, 2)
            if score < 3.5:
                result['issues'].append('low NIMA score')
                result['sufficient_for_detection'] = False
        except Exception:
            pass

    return result


if __name__ == '__main__':
    if not DATA_DIR.exists():
        print(f"Directory {DATA_DIR} not found")
        sys.exit(1)

    images = sorted(DATA_DIR.rglob('*'))
    images = [p for p in images if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]

    if not images:
        print(f"No images found in {DATA_DIR}")
        sys.exit(1)

    results = []
    for i, p in enumerate(images, 1):
        print(f"\r  Analyzing {i}/{len(images)}...", end='', flush=True)
        results.append(analyze(p))
    print()

    print(f"\nModels: BRISQUE={'yes' if HAS_BRISQUE else 'no (pip install opencv-contrib-python)'}  NIMA={'yes' if HAS_NIMA else 'no (needs torch/torchvision)'}")

    good = sum(1 for r in results if r.get('sufficient_for_detection'))
    bad = [r for r in results if not r.get('sufficient_for_detection')]

    print(f"\n{'='*60}")
    print(f"Analyzed {len(results)} images — {good} good, {len(bad)} with issues")
    print(f"{'='*60}\n")

    if bad:
        print("Images with quality issues:\n")
        for r in bad:
            issues = ', '.join(r.get('issues', []))
            extra = ''
            if 'brisque' in r:
                extra += f"  brisque={r['brisque']}"
            if 'nima' in r:
                extra += f"  nima={r['nima']}"
            print(f"  ✗ {r['file']}")
            print(f"    [{issues}]  blur={r.get('blur_score','?')}  bright={r.get('brightness','?')}  contrast={r.get('contrast','?')}{extra}\n")
    else:
        print("All images passed quality checks.")

    # Summary statistics
    blur_scores = [r['blur_score'] for r in results if 'blur_score' in r]
    bright_scores = [r['brightness'] for r in results if 'brightness' in r]
    contrast_scores = [r['contrast'] for r in results if 'contrast' in r]
    brisque_scores = [r['brisque'] for r in results if 'brisque' in r]
    nima_scores = [r['nima'] for r in results if 'nima' in r]

    # Count issues by type
    issue_counts = {}
    for r in results:
        for issue in r.get('issues', []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"  Blur:       min={min(blur_scores):.1f}  max={max(blur_scores):.1f}  avg={np.mean(blur_scores):.1f}  median={np.median(blur_scores):.1f}")
    print(f"  Brightness: min={min(bright_scores):.1f}  max={max(bright_scores):.1f}  avg={np.mean(bright_scores):.1f}  median={np.median(bright_scores):.1f}")
    print(f"  Contrast:   min={min(contrast_scores):.1f}  max={max(contrast_scores):.1f}  avg={np.mean(contrast_scores):.1f}  median={np.median(contrast_scores):.1f}")
    if brisque_scores:
        print(f"  BRISQUE:    min={min(brisque_scores):.1f}  max={max(brisque_scores):.1f}  avg={np.mean(brisque_scores):.1f}  median={np.median(brisque_scores):.1f}")
    if nima_scores:
        print(f"  NIMA:       min={min(nima_scores):.2f}  max={max(nima_scores):.2f}  avg={np.mean(nima_scores):.2f}  median={np.median(nima_scores):.2f}")

    if issue_counts:
        print(f"\n  Issues breakdown:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"    {issue}: {count} images ({count*100//len(results)}%)")

    print(f"\nFull results saved to data/training/quality_report.json")
    with open('data/training/quality_report.json', 'w') as f:
        json.dump(results, f, indent=2)
