"""
═══════════════════════════════════════════════════════════════════════════════
  Skin Cancer Detection — Single Image Classifier
  HAM10000 | DINOv2 + SupCon
  ─────────────────────────────────────────────────────────────────────────────
  Evet Fernandes | M.S. Computer Science | CSUDH | Spring 2026

USAGE:
    python predict_single.py --image path/to/your_lesion.jpg
    python predict_single.py --image photo.jpg --checkpoint checkpoints/stage2_best.pt
    python predict_single.py --image photo.jpg --arch resnet152
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Melanoma', 'Nevus', 'BCC', 'AK (Actinic Keratosis)', 'BKL', 'DF', 'VASC']
CLASS_SHORT = ['MEL',      'NV',    'BCC', 'AK',                      'BKL', 'DF', 'VASC']
CLASS_DESC  = {
    'MEL': 'Melanoma — malignant skin cancer, requires urgent referral',
    'NV':  'Melanocytic Nevus — common benign mole',
    'BCC': 'Basal Cell Carcinoma — most common skin cancer, slow-growing',
    'AK':  'Actinic Keratosis — pre-cancerous rough skin patch',
    'BKL': 'Benign Keratosis-like lesion — seborrheic keratosis / lentigo',
    'DF':  'Dermatofibroma — benign fibrous nodule',
    'VASC':'Vascular lesion — angioma or hemorrhage',
}
CLASS_COLORS = ['#E74C3C','#3498DB','#F39C12','#E67E22','#9B59B6','#1ABC9C','#2ECC71']
RISK_LEVEL   = {
    'MEL': ('HIGH',    '#E74C3C'),
    'BCC': ('MEDIUM',  '#F39C12'),
    'AK':  ('MEDIUM',  '#F39C12'),
    'NV':  ('LOW',     '#2ECC71'),
    'BKL': ('LOW',     '#2ECC71'),
    'DF':  ('LOW',     '#2ECC71'),
    'VASC':('LOW',     '#2ECC71'),
}

PALETTE = {
    'bg':     '#0A0A0F',
    'panel':  '#13131A',
    'border': '#252535',
    'accent': '#00E5FF',
    'text':   '#E8E8EE',
    'sub':    '#777788',
}


# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=7, embed_dim=128, unfreeze_last_n=4):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 …")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
        self.feat_dim = self.backbone.embed_dim

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, embed_dim),
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self._mode = 'classify'

    def set_mode(self, m): self._mode = m

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        if return_features:
            return feats
        if self._mode == 'supcon':
            return F.normalize(self.proj_head(feats), dim=1)
        return self.class_head(feats)

    def get_attention_maps(self, x):
        return None  # newer DINOv2 removed get_last_selfattention


class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V2)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))

    def forward(self, x, return_features=False):
        return self.model(x)

    def set_mode(self, m): pass
    def get_attention_maps(self, x): return None


def build_model(arch, ckpt_path, device):
    if arch == 'dinov2':
        model = DINOv2Classifier()
    else:
        model = ResNetBaseline()

    print(f"  Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.set_mode('classify')
    model.eval()
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMS & TTA
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size=518):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tta_augs = [
        [],
        [T.RandomHorizontalFlip(p=1.0)],
        [T.RandomVerticalFlip(p=1.0)],
        [T.RandomRotation(90)],
        [T.RandomRotation(180)],
        [T.RandomRotation(270)],
        [T.ColorJitter(brightness=0.15)],
        [T.ColorJitter(contrast=0.15)],
        [T.RandomResizedCrop(img_size, scale=(0.88, 1.0))],
    ]
    base = [T.Resize((img_size, img_size)), T.ToTensor(), norm]
    return [T.Compose(aug + base) for aug in tta_augs]


@torch.no_grad()
def predict(model, pil_img, tta_transforms, device):
    """Run TTA on a single PIL image, return averaged probs (7,)."""
    all_probs = []
    for tf in tta_transforms:
        x = tf(pil_img).unsqueeze(0).to(device)
        p = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
        all_probs.append(p)

    vp = np.stack(all_probs)                          # (n_views, 7)
    H  = -(vp * np.log(vp + 1e-8)).sum(-1)            # entropy per view
    w  = 1.0 / (H + 1e-8)
    w  = w / w.sum()
    return (w[:, None] * vp).sum(0)                   # weighted avg


@torch.no_grad()
def get_attention(model, pil_img, device, img_size=518):
    """Extract DINO attention map for visualization."""
    if not hasattr(model, 'get_attention_maps'):
        return None
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x    = norm(T.ToTensor()(T.Resize((img_size, img_size))(pil_img))).unsqueeze(0).to(device)
    attn = model.get_attention_maps(x)               # (1, heads, N+1, N+1)
    if attn is None:
        return None
    n_patch = img_size // 14
    amap = attn[0, :, 0, 1:].mean(0).reshape(n_patch, n_patch).cpu().numpy()
    amap = np.array(Image.fromarray(amap).resize((img_size, img_size), Image.BILINEAR))
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    return amap


# ──────────────────────────────────────────────────────────────────────────────
# RESULT FIGURE
# ──────────────────────────────────────────────────────────────────────────────

def render_result(pil_img, probs, attn_map, out_path):
    matplotlib.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor':   PALETTE['panel'],
        'axes.edgecolor':   PALETTE['border'],
        'text.color':       PALETTE['text'],
        'xtick.color':      PALETTE['sub'],
        'ytick.color':      PALETTE['sub'],
        'font.family':      'monospace',
    })

    pred_idx  = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    pred_short= CLASS_SHORT[pred_idx]
    confidence= probs[pred_idx]
    risk_lbl, risk_color = RISK_LEVEL[pred_short]
    referral  = confidence < 0.70

    has_attn  = attn_map is not None
    n_img_cols= 3 if has_attn else 2
    fig       = plt.figure(figsize=(18, 9), facecolor=PALETTE['bg'])
    gs        = gridspec.GridSpec(2, n_img_cols + 1,
                                  figure=fig,
                                  width_ratios=[1]*n_img_cols + [1.6],
                                  hspace=0.35, wspace=0.28)

    # ── Header ──────────────────────────────────────────────────────────
    header_ax = fig.add_axes([0, 0.93, 1, 0.07])
    header_ax.axis('off')
    header_ax.set_facecolor(PALETTE['bg'])
    header_ax.text(0.01, 0.5,
                   'SKIN LESION CLASSIFIER  ·  DINOv2 + SupCon  ·  HAM10000',
                   fontsize=10, color=PALETTE['sub'], va='center', fontfamily='monospace')
    header_ax.text(0.99, 0.5,
                   'Evet Fernandes | CSUDH M.S. CS | Spring 2026',
                   fontsize=9, color=PALETTE['sub'], va='center', ha='right',
                   fontfamily='monospace')

    # ── Original image ───────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(pil_img)
    ax_img.set_title('Input Image', color=PALETTE['sub'], fontsize=9, pad=4)
    ax_img.axis('off')

    # ── Attention / overlay ──────────────────────────────────────────────
    if has_attn:
        img_arr = np.array(pil_img.resize((518, 518)))
        overlay_col = cm.inferno(attn_map)[:, :, :3]
        blend       = np.clip(0.45 * img_arr / 255.0 + 0.55 * overlay_col, 0, 1)

        ax_attn = fig.add_subplot(gs[0, 1])
        ax_attn.imshow(attn_map, cmap='inferno')
        ax_attn.set_title('DINO Attention', color=PALETTE['sub'], fontsize=9, pad=4)
        ax_attn.axis('off')

        ax_blend = fig.add_subplot(gs[0, 2])
        ax_blend.imshow(blend)
        ax_blend.set_title('Attention Overlay', color=PALETTE['sub'], fontsize=9, pad=4)
        ax_blend.axis('off')
    else:
        ax_attn = fig.add_subplot(gs[0, 1])
        ax_attn.axis('off')
        ax_attn.text(0.5, 0.5, 'Attention N/A\n(non-DINOv2)',
                     ha='center', va='center', color=PALETTE['sub'], fontsize=9)

    # ── Probability bar chart ────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :n_img_cols])
    sorted_idx = np.argsort(probs)
    ys    = range(len(CLASS_NAMES))
    bars  = ax_bar.barh(list(ys), probs[sorted_idx],
                        color=[CLASS_COLORS[i] for i in sorted_idx],
                        height=0.6, alpha=0.88)

    ax_bar.set_yticks(list(ys))
    ax_bar.set_yticklabels([CLASS_SHORT[i] for i in sorted_idx], fontsize=9)
    ax_bar.set_xlim(0, 1.08)
    ax_bar.set_xlabel('Probability', color=PALETTE['sub'], fontsize=8)
    ax_bar.set_title('Class Probabilities  (all 7 HAM10000 classes)',
                     color=PALETTE['sub'], fontsize=9, pad=4)
    ax_bar.grid(axis='x', alpha=0.2)
    ax_bar.axvline(0.70, color='yellow', lw=1.2, linestyle='--', alpha=0.7)
    ax_bar.text(0.71, -0.5, 'referral\nthreshold', color='yellow',
                fontsize=7, va='bottom', alpha=0.8)

    for bar, idx in zip(bars, sorted_idx):
        w = bar.get_width()
        ax_bar.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{w:.3f}', va='center', fontsize=8,
                    color='white' if w == probs[pred_idx] else PALETTE['sub'],
                    fontweight='bold' if idx == pred_idx else 'normal')

    # ── Result panel ─────────────────────────────────────────────────────
    ax_res = fig.add_subplot(gs[:, -1])
    ax_res.axis('off')
    ax_res.set_facecolor(PALETTE['panel'])

    y = 0.97
    def txt(s, size=10, color=PALETTE['text'], bold=False, dy=0.06):
        nonlocal y
        ax_res.text(0.08, y, s, fontsize=size,
                    color=color, va='top', fontfamily='monospace',
                    fontweight='bold' if bold else 'normal',
                    transform=ax_res.transAxes)
        y -= dy

    txt('─' * 28, size=8, color=PALETTE['border'], dy=0.04)
    txt('PREDICTION', size=9, color=PALETTE['sub'], dy=0.05)
    txt(pred_name, size=14, color=CLASS_COLORS[pred_idx], bold=True, dy=0.09)
    txt(f'({pred_short})', size=9, color=PALETTE['sub'], dy=0.06)
    txt(f'Confidence:  {confidence:.1%}', size=10, dy=0.06)

    # Confidence bar
    bar_y = y + 0.01
    ax_res.add_patch(plt.Rectangle((0.08, bar_y - 0.015), 0.84, 0.022,
                                   color=PALETTE['border'],
                                   transform=ax_res.transAxes, clip_on=False))
    ax_res.add_patch(plt.Rectangle((0.08, bar_y - 0.015), 0.84 * confidence, 0.022,
                                   color=CLASS_COLORS[pred_idx], alpha=0.85,
                                   transform=ax_res.transAxes, clip_on=False))
    y -= 0.06

    txt('─' * 28, size=8, color=PALETTE['border'], dy=0.04)
    txt('RISK LEVEL', size=9, color=PALETTE['sub'], dy=0.05)
    txt(f'▶  {risk_lbl}', size=13, color=risk_color, bold=True, dy=0.08)
    txt('─' * 28, size=8, color=PALETTE['border'], dy=0.04)
    txt('DESCRIPTION', size=9, color=PALETTE['sub'], dy=0.05)

    # Word-wrap description
    desc = CLASS_DESC[pred_short]
    words = desc.split()
    line, lines = '', []
    for w in words:
        if len(line) + len(w) + 1 > 26:
            lines.append(line); line = w
        else:
            line = (line + ' ' + w).strip()
    if line: lines.append(line)
    for l in lines:
        txt(l, size=8, color=PALETTE['text'], dy=0.045)

    txt('─' * 28, size=8, color=PALETTE['border'], dy=0.04)

    if referral:
        txt('⚠  LOW CONFIDENCE', size=9, color='#FFD700', bold=True, dy=0.055)
        txt('Consider specialist', size=8, color='#FFD700', dy=0.045)
        txt('referral.', size=8, color='#FFD700', dy=0.05)
    else:
        txt('✓  CONFIDENT', size=9, color='#2ECC71', bold=True, dy=0.055)
        txt(f'  (≥ 70% threshold)', size=7, color=PALETTE['sub'], dy=0.05)

    txt('─' * 28, size=8, color=PALETTE['border'], dy=0.04)
    txt('⚠ NOT FOR CLINICAL USE', size=7, color=PALETTE['sub'], dy=0.045)
    txt('Research purposes only.', size=7, color=PALETTE['sub'], dy=0.04)

    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Result saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def find_checkpoint(arch, override):
    if override:
        return override
    candidates = [
        'checkpoints/stage2_best.pt',
        'checkpoints/baseline_best.pt',
        'stage2_best.pt',
        'stage1_best.pt',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description='Single-image skin lesion classifier')
    parser.add_argument('--image',      required=True, help='Path to input image (.jpg/.png)')
    parser.add_argument('--checkpoint', default=None,  help='Path to .pt checkpoint')
    parser.add_argument('--arch',       default='dinov2', choices=['dinov2', 'resnet152'])
    parser.add_argument('--out',        default=None,  help='Output figure path (default: result_<image>.png)')
    parser.add_argument('--no_tta',     action='store_true', help='Skip TTA (faster but less accurate)')
    args = parser.parse_args()

    # ── Validate input ────────────────────────────────────────────────────
    if not os.path.exists(args.image):
        print(f"\n  [ERROR] Image not found: {args.image}")
        sys.exit(1)

    ckpt = find_checkpoint(args.arch, args.checkpoint)
    if not ckpt:
        print("\n  [ERROR] No checkpoint found. Pass --checkpoint path/to/stage2_best.pt")
        sys.exit(1)

    device = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')

    img_size = 518 if args.arch == 'dinov2' else 224

    print("═"*55)
    print("  Skin Lesion Classifier  —  Single Image Mode")
    print(f"  Image : {args.image}")
    print(f"  Model : {args.arch.upper()}  |  Device: {device}")
    print("═"*55)

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model(args.arch, ckpt, device)

    # ── Load image ────────────────────────────────────────────────────────
    pil_img = Image.open(args.image).convert('RGB')
    print(f"  Image size: {pil_img.size[0]}×{pil_img.size[1]}")

    # ── Predict ───────────────────────────────────────────────────────────
    if args.no_tta:
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        x    = norm(T.ToTensor()(T.Resize((img_size, img_size))(pil_img)))
        with torch.no_grad():
            probs = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu().numpy()
    else:
        print(f"  Running TTA inference (9 augmented views) …")
        ttas  = get_transforms(img_size)
        probs = predict(model, pil_img, ttas, device)

    # ── Attention map ─────────────────────────────────────────────────────
    attn_map = None
    if args.arch == 'dinov2':
        print("  Extracting DINO attention map …")
        attn_map = get_attention(model, pil_img, device, img_size)

    # ── Print result ──────────────────────────────────────────────────────
    pred_idx   = int(np.argmax(probs))
    pred_short = CLASS_SHORT[pred_idx]
    confidence = probs[pred_idx]
    risk_lbl, _ = RISK_LEVEL[pred_short]

    print("\n" + "═"*55)
    print(f"  PREDICTION : {CLASS_NAMES[pred_idx]}  ({pred_short})")
    print(f"  CONFIDENCE : {confidence:.1%}")
    print(f"  RISK LEVEL : {risk_lbl}")
    print(f"  REFERRAL   : {'YES — low confidence' if confidence < 0.70 else 'No'}")
    print()
    print("  All class probabilities:")
    for i in np.argsort(probs)[::-1]:
        bar = '█' * int(probs[i] * 30)
        print(f"    {CLASS_SHORT[i]:<5} {probs[i]:.4f}  {bar}")
    print("═"*55)
    print("  ⚠  Research use only — not for clinical diagnosis")
    print("═"*55)

    # ── Save figure ───────────────────────────────────────────────────────
    stem    = os.path.splitext(os.path.basename(args.image))[0]
    out_path= args.out or f'result_{stem}.png'
    render_result(pil_img, probs, attn_map, out_path)


if __name__ == '__main__':
    main()
