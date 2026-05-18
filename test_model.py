"""
═══════════════════════════════════════════════════════════════════════════════
  Skin Cancer Detection — Test & Results Display
  HAM10000 | DINOv2 + SupCon
  ─────────────────────────────────────────────────────────────────────────────
  Evet Fernandes | M.S. Computer Science | CSUDH | Spring 2026

USAGE:
    python test_model.py                        # auto-finds checkpoints & data
    python test_model.py --checkpoint stage2_best.pt
    python test_model.py --data_dir /path/to/HAM10000_images
    python test_model.py --arch resnet152       # if you trained a baseline

OUTPUT:
    results/  ←  all figures + CSV summary saved here
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T

from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, f1_score,
    cohen_kappa_score, confusion_matrix, roc_curve, auc,
    classification_report
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit paths here if needed)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'ham_csv':        'HAM10000_metadata.csv',   # or 'data/HAM10000_metadata.csv'
    'ham_img_dir':    'HAM10000_images',          # merged folder of part_1 + part_2
    'checkpoint':     None,                       # auto-detected below
    'arch':           'dinov2',
    'num_classes':    7,
    'img_size':       518,
    'embed_dim':      128,
    'unfreeze_last_n':4,
    'test_split':     0.15,
    'val_split':      0.15,
    'tta_views':      10,
    'referral_thr':   0.70,
    'batch_size':     8,
    'num_workers':    0,
    'seed':           42,
    'results_dir':    'results',
    'device':         ('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available()
                       else 'cpu'),
}

HAM10000_CLASSES = {
    'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3,
    'bkl': 4, 'df': 5, 'vasc': 6,
}
CLASS_NAMES  = ['Melanoma', 'Nevus', 'BCC', 'AK', 'BKL', 'DF', 'VASC']
CLASS_COLORS = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#1ABC9C','#E67E22']

PALETTE = {
    'bg':      '#0D0D0D',
    'panel':   '#161616',
    'border':  '#2A2A2A',
    'accent':  '#00E5FF',
    'accent2': '#FF4081',
    'text':    '#E8E8E8',
    'sub':     '#888888',
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (copied from main.py so this script is self-contained)
# ──────────────────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes, embed_dim=128, unfreeze_last_n=4):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 …")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
        self.feat_dim = self.backbone.embed_dim

        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            for blk in list(self.backbone.blocks)[-unfreeze_last_n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

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

    def set_mode(self, mode):
        self._mode = mode

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        if return_features:
            return feats
        if self._mode == 'supcon':
            return F.normalize(self.proj_head(feats), dim=1)
        return self.class_head(feats)

    def get_attention_maps(self, x):
        return self.backbone.get_last_selfattention(x)


class ResNetBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V2)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x); x = self.model.layer2(x)
            x = self.model.layer3(x); x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)

    def set_mode(self, m): pass


def build_model(cfg):
    arch = cfg['arch'].lower()
    nc   = cfg['num_classes']
    if arch == 'dinov2':
        return DINOv2Classifier(nc, cfg['embed_dim'], cfg['unfreeze_last_n'])
    elif arch == 'resnet152':
        return ResNetBaseline(nc)
    raise ValueError(f"Unknown arch: {arch}")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class HAM10000Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir, image_ids, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['image_id'].isin(image_ids)].reset_index(drop=True)
        self.img_dir   = img_dir
        self.ids       = df['image_id'].tolist()
        self.labels    = [HAM10000_CLASSES[v] for v in df['dx'].tolist()]
        self.transform = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.ids[idx] + '.jpg')
        img  = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_test_loader(cfg):
    sz  = cfg['img_size']
    tf  = T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df  = pd.read_csv(cfg['ham_csv'])
    ids = df['image_id'].tolist()
    lbl = [HAM10000_CLASSES[v] for v in df['dx'].tolist()]

    _, tmp_ids, _, tmp_lbl = train_test_split(
        ids, lbl, test_size=cfg['val_split'] + cfg['test_split'],
        stratify=lbl, random_state=cfg['seed'])

    val_frac = cfg['val_split'] / (cfg['val_split'] + cfg['test_split'])
    val_ids, test_ids = train_test_split(
        tmp_ids, test_size=1 - val_frac, stratify=tmp_lbl,
        random_state=cfg['seed'])

    test_ds  = HAM10000Dataset(cfg['ham_csv'], cfg['ham_img_dir'], test_ids, tf)
    val_ds   = HAM10000Dataset(cfg['ham_csv'], cfg['ham_img_dir'], val_ids,  tf)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=cfg['num_workers'])
    val_loader  = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=cfg['num_workers'])

    print(f"  Test set : {len(test_ds):,} images")
    print(f"  Val  set : {len(val_ds):,} images")
    return test_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# TTA EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def get_tta_transforms(cfg):
    sz   = cfg['img_size']
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = [T.Resize((sz, sz)), T.ToTensor(), norm]
    augs = [
        [],
        [T.RandomHorizontalFlip(p=1.0)],
        [T.RandomVerticalFlip(p=1.0)],
        [T.RandomRotation(90)],
        [T.RandomRotation(180)],
        [T.RandomRotation(270)],
        [T.ColorJitter(brightness=0.2)],
        [T.ColorJitter(contrast=0.2)],
        [T.GaussianBlur(kernel_size=max(3, int(0.05 * sz) | 1))],
        [T.RandomResizedCrop(sz, scale=(0.85, 1.0))],
    ][:cfg['tta_views']]
    return [T.Compose(a + base) for a in augs]


@torch.no_grad()
def run_inference(model, loader, cfg, use_tta=False):
    """Returns y_prob (N, C), y_true (N,), image_ids list."""
    device     = 'cpu'   # safer for MPS
    model      = model.to(device).eval()
    all_probs, all_true = [], []

    if not use_tta:
        for imgs, labels in tqdm(loader, desc="  Running inference"):
            logits = model(imgs.to(device))
            probs  = torch.softmax(logits, dim=1).numpy()
            all_probs.append(probs)
            all_true.extend(labels.numpy())
    else:
        tta_tfs = get_tta_transforms(cfg)
        ds      = loader.dataset
        bs      = 16
        for i in tqdm(range(0, len(ds), bs), desc="  TTA inference"):
            batch_imgs = []
            batch_lbls = []
            for j in range(i, min(i + bs, len(ds))):
                path = os.path.join(ds.img_dir, ds.ids[j] + '.jpg')
                pil  = Image.open(path).convert('RGB')
                batch_imgs.append(pil)
                batch_lbls.append(ds.labels[j])

            view_probs = []
            for tf in tta_tfs:
                batch_t = torch.stack([tf(img) for img in batch_imgs]).to(device)
                p       = torch.softmax(model(batch_t), dim=1).numpy()
                view_probs.append(p)
            vp = np.stack(view_probs)                     # (n_views, B, C)
            H  = -(vp * np.log(vp + 1e-8)).sum(-1)       # entropy weighting
            w  = 1.0 / (H + 1e-8)
            w  = w / w.sum(0, keepdims=True)
            all_probs.append(np.einsum('vb,vbc->bc', w, vp))
            all_true.extend(batch_lbls)

    model = model.to(cfg['device'])
    y_prob = np.vstack(all_probs)
    y_true = np.array(all_true)
    return y_prob, y_true


def compute_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(axis=1)
    n_cls  = y_prob.shape[1]
    m      = {}

    m['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    m['f1_macro']          = f1_score(y_true, y_pred, average='macro', zero_division=0)
    m['cohens_kappa']      = cohen_kappa_score(y_true, y_pred)

    try:
        m['auc_roc_macro'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        m['auc_roc_macro'] = float('nan')

    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, v in enumerate(f1s):
        m[f'f1_{CLASS_NAMES[i]}'] = v

    # ECE
    conf  = y_prob.max(axis=1)
    pred  = y_prob.argmax(axis=1)
    corr  = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, 16)
    ece   = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum():
            ece += mask.sum() * abs(corr[mask].mean() - conf[mask].mean())
    m['ece'] = ece / len(y_true)

    # Referral rate
    referral_mask = conf < 0.70
    m['referral_rate'] = referral_mask.mean()

    return m


# ──────────────────────────────────────────────────────────────────────────────
# PUBLICATION-QUALITY FIGURES
# ──────────────────────────────────────────────────────────────────────────────

def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  PALETTE['bg'],
        'axes.facecolor':    PALETTE['panel'],
        'axes.edgecolor':    PALETTE['border'],
        'axes.labelcolor':   PALETTE['text'],
        'xtick.color':       PALETTE['sub'],
        'ytick.color':       PALETTE['sub'],
        'text.color':        PALETTE['text'],
        'grid.color':        PALETTE['border'],
        'grid.alpha':        0.5,
        'font.family':       'monospace',
        'axes.titleweight':  'bold',
        'axes.titlesize':    11,
        'axes.labelsize':    9,
    })


def fig1_summary_dashboard(metrics_std, metrics_tta, out_dir):
    """Full-page summary dashboard with key metrics side by side."""
    apply_dark_style()
    fig = plt.figure(figsize=(16, 7), facecolor=PALETTE['bg'])
    fig.suptitle(
        'Skin Cancer Detection — Test Results  |  HAM10000  |  DINOv2 + SupCon',
        fontsize=14, color=PALETTE['accent'], fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    key_metrics = [
        ('Balanced Accuracy', 'balanced_accuracy'),
        ('AUC-ROC (Macro)',   'auc_roc_macro'),
        ('F1 Macro',          'f1_macro'),
        ("Cohen's κ",         'cohens_kappa'),
        ('ECE  (↓ better)',   'ece'),
        ('Referral Rate',     'referral_rate'),
    ]

    for col_idx, (label, mdict) in enumerate([('Standard', metrics_std), ('+ TTA (10 views)', metrics_tta)]):
        ax = fig.add_subplot(gs[col_idx])
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(key_metrics) - 0.5)
        ax.axis('off')
        ax.set_title(label, color=PALETTE['accent2'] if col_idx else PALETTE['accent'],
                     fontsize=12, pad=10)

        for i, (name, key) in enumerate(reversed(key_metrics)):
            val = mdict.get(key, float('nan'))
            y   = i

            # Background bar
            ax.barh(y, 0.95, left=0.025, height=0.65,
                    color=PALETTE['border'], zorder=1)

            # Value bar (skip ECE and referral which aren't 0-1 in same direction)
            bar_val = val if key not in ('ece',) else max(0, 0.5 - val)
            bar_val = min(max(bar_val, 0), 1)
            color   = PALETTE['accent2'] if col_idx else PALETTE['accent']
            ax.barh(y, bar_val * 0.95, left=0.025, height=0.65,
                    color=color, alpha=0.85, zorder=2)

            ax.text(0.97, y, f'{val:.4f}', va='center', ha='right',
                    color='white', fontsize=10, fontweight='bold', zorder=3)
            ax.text(0.02, y, name, va='center', ha='left',
                    color=PALETTE['text'], fontsize=9, zorder=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, '01_summary_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ {path}")


def fig2_confusion_matrix(y_true, y_pred, out_dir, tag=''):
    apply_dark_style()
    cm_arr = confusion_matrix(y_true, y_pred)
    cm_n   = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Confusion Matrix', color=PALETTE['accent'], fontsize=13, fontweight='bold')

    for ax, data, title in zip(axes, [cm_arr, cm_n], ['Counts', 'Normalized']):
        fmt  = 'd' if title == 'Counts' else '.2f'
        cmap = sns.color_palette('Blues', as_cmap=True)
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.5, linecolor=PALETTE['bg'],
                    ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(title, color=PALETTE['text'])
        ax.set_xlabel('Predicted', color=PALETTE['sub'])
        ax.set_ylabel('True',      color=PALETTE['sub'])
        ax.tick_params(colors=PALETTE['sub'], labelsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f'02_confusion_matrix{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ {path}")


def fig3_roc_curves(y_true, y_prob, out_dir, tag=''):
    apply_dark_style()
    n_cls = y_prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor=PALETTE['bg'])
    fig.suptitle('ROC Curves per Class  (★ = sensitivity @ 95% specificity)',
                 color=PALETTE['accent'], fontsize=13, fontweight='bold')
    axes = axes.flatten()

    all_fprs, all_tprs, all_aucs = [], [], []

    for i in range(n_cls):
        ax    = axes[i]
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_a = auc(fpr, tpr)
        all_fprs.append(fpr); all_tprs.append(tpr); all_aucs.append(roc_a)

        ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2)
        ax.fill_between(fpr, tpr, alpha=0.12, color=CLASS_COLORS[i])
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)

        # Mark sensitivity @ 95% specificity
        spec95 = np.where(1 - fpr >= 0.95)[0]
        if len(spec95):
            ax.scatter(fpr[spec95[-1]], tpr[spec95[-1]],
                       color='yellow', marker='*', s=120, zorder=5)

        ax.set_title(f'{CLASS_NAMES[i]}  AUC={roc_a:.3f}',
                     color=CLASS_COLORS[i], fontsize=9)
        ax.set_xlabel('FPR', fontsize=8); ax.set_ylabel('TPR', fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)

    # Macro average in last panel
    ax = axes[n_cls]
    for i in range(n_cls):
        ax.plot(all_fprs[i], all_tprs[i], color=CLASS_COLORS[i],
                lw=1, alpha=0.5, label=CLASS_NAMES[i])
    macro_auc = np.mean(all_aucs)
    ax.set_title(f'All classes  macro={macro_auc:.3f}', color=PALETTE['accent'], fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlabel('FPR', fontsize=8); ax.set_ylabel('TPR', fontsize=8)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.2)

    axes[-1].axis('off')   # hide unused subplot

    plt.tight_layout()
    path = os.path.join(out_dir, f'03_roc_curves{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ {path}")


def fig4_per_class_f1(metrics_std, metrics_tta, out_dir):
    apply_dark_style()
    f1_std = [metrics_std.get(f'f1_{n}', 0) for n in CLASS_NAMES]
    f1_tta = [metrics_tta.get(f'f1_{n}', 0) for n in CLASS_NAMES]
    x      = np.arange(len(CLASS_NAMES))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE['bg'])
    bars1   = ax.bar(x - width/2, f1_std, width, label='Standard',
                     color=PALETTE['accent'], alpha=0.85)
    bars2   = ax.bar(x + width/2, f1_tta,  width, label='+ TTA',
                     color=PALETTE['accent2'], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class F1 Score: Standard vs TTA',
                 color=PALETTE['accent'], fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='white', lw=0.5, linestyle='--', alpha=0.4)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7,
                color=PALETTE['accent'])
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7,
                color=PALETTE['accent2'])

    plt.tight_layout()
    path = os.path.join(out_dir, '04_per_class_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ {path}")


def fig5_confidence_distribution(y_prob, y_true, out_dir):
    apply_dark_style()
    y_pred = y_prob.argmax(axis=1)
    conf   = y_prob.max(axis=1)
    corr   = (y_pred == y_true)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=PALETTE['bg'])
    fig.suptitle('Prediction Confidence Analysis',
                 color=PALETTE['accent'], fontsize=13, fontweight='bold')

    # Panel 1: Correct vs incorrect
    ax = axes[0]
    ax.hist(conf[corr],  bins=40, alpha=0.75, color=PALETTE['accent'],
            label=f'Correct ({corr.sum()})', density=True)
    ax.hist(conf[~corr], bins=40, alpha=0.75, color=PALETTE['accent2'],
            label=f'Wrong ({(~corr).sum()})', density=True)
    ax.axvline(0.70, color='yellow', lw=1.5, linestyle='--', label='Referral thr (0.70)')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Density')
    ax.set_title('Correct vs Incorrect', color=PALETTE['text'])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: Per-class confidence violin
    ax = axes[1]
    cls_confs = [conf[y_true == i] for i in range(len(CLASS_NAMES))]
    parts     = ax.violinplot(cls_confs, positions=range(len(CLASS_NAMES)),
                              showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CLASS_COLORS[i]); pc.set_alpha(0.75)
    parts['cmedians'].set_color('white')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=8, rotation=30)
    ax.set_ylabel('Confidence')
    ax.set_title('Per-Class Confidence', color=PALETTE['text'])
    ax.grid(axis='y', alpha=0.2)

    # Panel 3: Calibration (reliability diagram)
    ax     = axes[2]
    edges  = np.linspace(0, 1, 11)
    midpts = (edges[:-1] + edges[1:]) / 2
    accs   = []
    fracs  = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum():
            accs.append(corr[mask].mean())
            fracs.append(midpts[int(lo*10)])
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Perfect calibration')
    ax.plot(midpts[:len(accs)], accs, 'o-', color=PALETTE['accent'],
            lw=2, ms=5, label='Model')
    ax.fill_between(midpts[:len(accs)], midpts[:len(accs)], accs,
                    alpha=0.15, color=PALETTE['accent2'])
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram (Calibration)', color=PALETTE['text'])
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, '05_confidence_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ {path}")


@torch.no_grad()
def fig6_tsne(model, loader, out_dir, cfg, n_batches=40):
    apply_dark_style()
    print("  Extracting features for t-SNE …")
    device = cfg['device']
    model  = model.to(device).eval()
    feats, labels_list = [], []

    for i, (imgs, lbls) in enumerate(loader):
        if i >= n_batches: break
        f = model(imgs.to(device), return_features=True)
        feats.append(f.cpu().numpy())
        labels_list.extend(lbls.numpy())

    feats  = np.vstack(feats)
    labels = np.array(labels_list)

    print("  Running t-SNE (this takes ~1 min) …")
    emb = TSNE(n_components=2, perplexity=35, n_iter=1200,
               random_state=cfg['seed']).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=PALETTE['bg'])
    for i, (name, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1], c=col,
                   label=name, alpha=0.55, s=10, linewidths=0)

    ax.set_title('t-SNE of DINOv2 Embeddings — HAM10000 Test Set',
                 color=PALETTE['accent'], fontsize=12)
    ax.set_xlabel('t-SNE 1', fontsize=9); ax.set_ylabel('t-SNE 2', fontsize=9)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9,
              framealpha=0.2, facecolor=PALETTE['panel'])
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    path = os.path.join(out_dir, '06_tsne_embeddings.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ {path}")


def fig7_attention_maps(model, loader, out_dir, cfg, n_samples=6):
    if not hasattr(model, 'get_attention_maps'):
        print("  [SKIP] Attention maps — model is not DINOv2")
        return

    apply_dark_style()
    device  = cfg['device']
    sz      = cfg['img_size']
    patch   = 14
    n_patch = sz // patch
    norm_tf = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resize  = T.Resize((sz, sz))

    model = model.to(device).eval()
    ds    = loader.dataset
    samples = []

    for idx in range(min(n_samples, len(ds))):
        path = os.path.join(ds.img_dir, ds.ids[idx] + '.jpg')
        pil  = Image.open(path).convert('RGB')
        pil_rs = resize(pil)
        x    = norm_tf(T.ToTensor()(pil_rs)).unsqueeze(0).to(device)

        with torch.no_grad():
            attn = model.get_attention_maps(x)

        attn_map = attn[0, :, 0, 1:].mean(0).reshape(n_patch, n_patch).cpu().numpy()
        attn_map = np.array(Image.fromarray(attn_map).resize((sz, sz), Image.BILINEAR))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        samples.append((np.array(pil_rs), attn_map, ds.labels[idx]))

    n   = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4.2 * n), facecolor=PALETTE['bg'])
    fig.suptitle('DINOv2 Self-Attention Maps — Lesion Localization (free!)',
                 color=PALETTE['accent'], fontsize=13, fontweight='bold')
    if n == 1: axes = [axes]

    for i, (img, attn, lbl) in enumerate(samples):
        overlay = cm.inferno(attn)[:, :, :3]
        blend   = 0.45 * img / 255.0 + 0.55 * overlay
        axes[i][0].imshow(img)
        axes[i][0].set_title(f'{CLASS_NAMES[lbl]} — original',
                             color=CLASS_COLORS[lbl], fontsize=9)
        axes[i][1].imshow(attn, cmap='inferno')
        axes[i][1].set_title('Attention heatmap', color=PALETTE['sub'], fontsize=9)
        axes[i][2].imshow(np.clip(blend, 0, 1))
        axes[i][2].set_title('Overlay', color=PALETTE['sub'], fontsize=9)
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, '07_attention_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ {path}")


def save_metrics_csv(metrics_std, metrics_tta, out_dir):
    rows = []
    all_keys = sorted(set(list(metrics_std.keys()) + list(metrics_tta.keys())))
    for k in all_keys:
        rows.append({
            'metric':   k,
            'standard': metrics_std.get(k, float('nan')),
            'tta':      metrics_tta.get(k, float('nan')),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'test_metrics.csv')
    df.to_csv(path, index=False)
    print(f"  ✓ {path}")
    return df


def print_full_report(y_true, y_pred, metrics_std, metrics_tta):
    print("\n" + "═"*65)
    print("  FULL TEST RESULTS")
    print("═"*65)
    print(f"\n  {'Metric':<35} {'Standard':>10} {'+ TTA':>10}")
    print("  " + "─"*55)
    rows = [
        ('Balanced Accuracy',       'balanced_accuracy'),
        ('AUC-ROC (Macro OvR)',     'auc_roc_macro'),
        ('F1 Macro',                'f1_macro'),
        ("Cohen's κ",               'cohens_kappa'),
        ('ECE  (↓ better)',         'ece'),
        ('Referral Rate (< 0.70)',  'referral_rate'),
    ]
    for name, key in rows:
        v_std = metrics_std.get(key, float('nan'))
        v_tta = metrics_tta.get(key, float('nan'))
        print(f"  {name:<35} {v_std:>10.4f} {v_tta:>10.4f}")
    print()
    print("  Per-class F1:")
    for n in CLASS_NAMES:
        v_std = metrics_std.get(f'f1_{n}', float('nan'))
        v_tta = metrics_tta.get(f'f1_{n}', float('nan'))
        print(f"    {n:<10} {v_std:>10.4f} {v_tta:>10.4f}")
    print()
    print("  Classification Report (Standard):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))
    print("═"*65)


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────

def find_checkpoint(arch, override=None):
    if override:
        assert os.path.exists(override), f"Checkpoint not found: {override}"
        return override

    candidates = [
        'checkpoints/stage2_best.pt',
        'checkpoints/baseline_best.pt',
        'stage2_best.pt',
        'stage1_best.pt',
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"  Auto-detected checkpoint: {c}")
            return c
    return None


def find_csv_and_imgdir():
    """Try several common layouts for HAM10000."""
    csv_candidates = [
        'HAM10000_metadata.csv',
        'data/HAM10000_metadata.csv',
        'HAM10000_metadata',
    ]
    img_candidates = [
        'HAM10000_images',
        'data/HAM10000_images',
        'HAM10000_images_part_1',   # if not yet merged
    ]
    csv = next((c for c in csv_candidates if os.path.exists(c)), None)
    img = next((c for c in img_candidates if os.path.isdir(c)), None)
    return csv, img


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='HAM10000 Model Tester')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--data_dir',   default=None, help='Path to HAM10000 images folder')
    parser.add_argument('--csv',        default=None, help='Path to HAM10000_metadata.csv')
    parser.add_argument('--arch',       default='dinov2',
                        choices=['dinov2', 'resnet152'])
    parser.add_argument('--no_tta',     action='store_true', help='Skip TTA evaluation')
    parser.add_argument('--no_tsne',    action='store_true', help='Skip t-SNE (slow)')
    parser.add_argument('--no_attn',    action='store_true', help='Skip attention maps')
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg['arch'] = args.arch
    if args.arch != 'dinov2':
        cfg['img_size'] = 224

    os.makedirs(cfg['results_dir'], exist_ok=True)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])

    print("═"*65)
    print("  HAM10000 Skin Cancer — Model Testing")
    print("  Evet Fernandes | CSUDH M.S. CS | Spring 2026")
    print(f"  Device: {cfg['device']}  |  Arch: {cfg['arch'].upper()}")
    print("═"*65)

    # ── Paths ─────────────────────────────────────────────────────────────
    auto_csv, auto_img = find_csv_and_imgdir()
    cfg['ham_csv']     = args.csv      or auto_csv
    cfg['ham_img_dir'] = args.data_dir or auto_img

    if not cfg['ham_csv'] or not os.path.exists(cfg['ham_csv']):
        print("\n  [ERROR] Cannot find HAM10000_metadata.csv")
        print("  Run: python test_model.py --csv /path/to/HAM10000_metadata.csv")
        sys.exit(1)

    if not cfg['ham_img_dir'] or not os.path.isdir(cfg['ham_img_dir']):
        print("\n  [ERROR] Cannot find HAM10000 images folder")
        print("  Make sure part_1 and part_2 images are merged into one folder,")
        print("  then run: python test_model.py --data_dir /path/to/folder")
        sys.exit(1)

    ckpt_path = find_checkpoint(cfg['arch'], args.checkpoint)
    if not ckpt_path:
        print("\n  [ERROR] No checkpoint found. Train the model first, or pass --checkpoint")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n  Loading model from: {ckpt_path}")
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.set_mode('classify')
    model.eval()
    total  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total/1e6:.1f}M")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n  Setting up test/val split …")
    test_loader, val_loader = get_test_loader(cfg)

    # ── Standard inference ────────────────────────────────────────────────
    print("\n  [1/2] Standard inference (no TTA) …")
    y_prob_std, y_true = run_inference(model, test_loader, cfg, use_tta=False)
    metrics_std        = compute_metrics(y_true, y_prob_std)
    y_pred_std         = y_prob_std.argmax(axis=1)

    # ── TTA inference ─────────────────────────────────────────────────────
    if not args.no_tta:
        print(f"\n  [2/2] TTA inference ({cfg['tta_views']} views) …")
        y_prob_tta, _  = run_inference(model, test_loader, cfg, use_tta=True)
        metrics_tta    = compute_metrics(y_true, y_prob_tta)
        y_pred_tta     = y_prob_tta.argmax(axis=1)
    else:
        y_prob_tta  = y_prob_std
        metrics_tta = metrics_std
        y_pred_tta  = y_pred_std

    # ── Print report ──────────────────────────────────────────────────────
    print_full_report(y_true, y_pred_std, metrics_std, metrics_tta)

    # ── Save CSV ──────────────────────────────────────────────────────────
    save_metrics_csv(metrics_std, metrics_tta, cfg['results_dir'])

    # ── Generate figures ──────────────────────────────────────────────────
    print("\n  Generating figures …")
    fig1_summary_dashboard(metrics_std, metrics_tta, cfg['results_dir'])
    fig2_confusion_matrix(y_true, y_pred_std, cfg['results_dir'], tag='_standard')
    fig2_confusion_matrix(y_true, y_pred_tta, cfg['results_dir'], tag='_tta')
    fig3_roc_curves(y_true, y_prob_std, cfg['results_dir'], tag='_standard')
    if not args.no_tta:
        fig3_roc_curves(y_true, y_prob_tta, cfg['results_dir'], tag='_tta')
    fig4_per_class_f1(metrics_std, metrics_tta, cfg['results_dir'])
    fig5_confidence_distribution(y_prob_std, y_true, cfg['results_dir'])

    if not args.no_tsne:
        fig6_tsne(model, test_loader, cfg['results_dir'], cfg)

    if not args.no_attn:
        fig7_attention_maps(model, test_loader, cfg['results_dir'], cfg)

    print("\n" + "═"*65)
    print(f"  All results saved to ./{cfg['results_dir']}/")
    print("  Files generated:")
    for f in sorted(os.listdir(cfg['results_dir'])):
        print(f"    {f}")
    print("═"*65)
    print("  Done ✓")


if __name__ == '__main__':
    main()
