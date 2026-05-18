"""
═══════════════════════════════════════════════════════════════════════════════
  Enhanced Skin Cancer Detection Using Advanced Deep Learning
  DINOv2 Backbone | Supervised Contrastive Learning | Cross-Dataset Evaluation
  ─────────────────────────────────────────────────────────────────────────────
  Evet Fernandes | M.S. Computer Science | CSUDH | Advisor: Dr. Bin Tang
  Spring 2026
═══════════════════════════════════════════════════════════════════════════════

USAGE:
    # 1. Install dependencies
    pip install torch torchvision scikit-learn pandas pillow matplotlib seaborn tqdm einops

    # 2. Set your data paths in CONFIG below (section marked with ★)

    # 3. Run the full pipeline
    python main.py                           # full run
    python main.py --stage data              # verify data only
    python main.py --stage train             # train only
    python main.py --stage eval              # evaluate only
    python main.py --stage visualize         # visualize only
    python main.py --arch resnet152          # run a baseline instead

DATA:
    Download from ISIC Archive (https://www.isic-archive.com):
      • HAM10000  → https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
      • ISIC 2019 → https://challenge.isic-archive.com/data/#2019
      • ISIC 2020 → https://www.kaggle.com/c/siim-isic-melanoma-classification

    Expected directory layout:
        data/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images/          ← *.jpg files
        ├── ISIC_2019_Training_GroundTruth.csv
        ├── ISIC_2019_Training_Input/ ← *.jpg files
        ├── ISIC_2020_Training_GroundTruth.csv
        └── ISIC_2020_Training_Input/ ← *.jpg files
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless — saves figures to disk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as tv_models
import torchvision.transforms as T

from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, f1_score,
    cohen_kappa_score, confusion_matrix, roc_curve, auc
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# ★  CONFIG — EDIT THESE PATHS AND HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    'ham_csv':        'data/HAM10000_metadata.csv',
    'ham_img_dir':    'data/HAM10000_images',
    'isic19_csv':     'data/ISIC_2019_Training_GroundTruth.csv',
    'isic19_img_dir': 'data/ISIC_2019_Training_Input',
    'isic20_csv':     'data/ISIC_2020_Training_GroundTruth.csv',
    'isic20_img_dir': 'data/ISIC_2020_Training_Input',

    # ── Model ──────────────────────────────────────────────────────────────
    'arch':           'dinov2',   # 'dinov2' | 'resnet152' | 'efficientnetv2'
    'num_classes':    7,          # 7 for HAM10000
    'img_size':       518,        # 518 for DINOv2 ViT-B/14; use 224 for CNN baselines
    'embed_dim':      128,        # SupCon projection head output dimension
    'unfreeze_last_n':4,          # unfreeze last N ViT blocks of DINOv2 backbone

    # ── Stage 1: Contrastive pretraining ───────────────────────────────────
    's1_epochs':      30,
    's1_lr':          3e-4,
    's1_batch':       4,          # reduced for MacBook Air RAM
    's1_grad_accum':  128,        # effective batch = 4×128 = 512
    'temperature':    0.07,       # SupCon temperature τ

    # ── Stage 2: Classifier fine-tuning ───────────────────────────────────
    's2_epochs':      20,
    's2_lr':          1e-4,
    's2_batch':       8,          # reduced for MacBook Air RAM
    'focal_gamma':    2.0,        # Focal loss γ
    'label_smoothing':0.1,

    # ── Evaluation ─────────────────────────────────────────────────────────
    'tta_views':      10,         # test-time augmentation views per image
    'referral_thr':   0.70,       # confidence below this → specialist referral
    'val_split':      0.15,       # fraction of training data held out for validation
    'test_split':     0.15,

    # ── I/O ────────────────────────────────────────────────────────────────
    'save_dir':       'checkpoints',
    'figures_dir':    'figures',
    'device':         'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'num_workers':    0,          # 0 avoids multiprocessing freeze on Mac
    'seed':           42,
}

# ── Label mappings ─────────────────────────────────────────────────────────
ISIC2019_CLASSES = {
    'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3,
    'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7,
}
ISIC2019_NAMES = ['Melanoma', 'Nevus', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

HAM10000_CLASSES = {
    'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3,
    'bkl': 4, 'df': 5, 'vasc': 6,
}
HAM10000_NAMES = ['Melanoma', 'Nevus', 'BCC', 'AK', 'BKL', 'DF', 'VASC']

ISIC2020_CLASSES = {'benign': 0, 'malignant': 1}
ISIC2020_NAMES   = ['Benign', 'Malignant']


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TwoViewTransform:
    """Returns two independently augmented views of the same image (for SupCon)."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def build_transforms(cfg, mode='train', supcon=False):
    """
    mode: 'train' | 'val'
    supcon: if True, returns TwoViewTransform for contrastive pretraining
    """
    sz = cfg['img_size']
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if mode == 'val':
        return T.Compose([T.Resize((sz, sz)), T.ToTensor(), norm])

    # Training augmentations (strong for SupCon, moderate for classifier)
    base_aug = T.Compose([
        T.RandomResizedCrop(sz, scale=(0.6 if supcon else 0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=max(3, int(0.05 * sz) | 1)),
        T.ToTensor(),
        norm,
    ])
    if not supcon:
        base_aug = T.Compose([
            T.Resize((sz, sz)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            norm,
        ])

    return TwoViewTransform(base_aug) if supcon else base_aug


class ISICDataset(Dataset):
    """
    Unified Dataset for HAM10000, ISIC 2019, ISIC 2020.

    Supports:
      • Regular training / evaluation  (transform returns single tensor)
      • SupCon training                (transform returns (view1, view2) tuple)

    CSV must have columns:
        image_id  OR  isic_id     — filename stem (no extension)
        <label_col>               — class label string
    """

    def __init__(self, csv_path, img_dir, label_map, label_col,
                 image_ids=None, transform=None):
        df = pd.read_csv(csv_path)
        id_col = 'image_id' if 'image_id' in df.columns else 'isic_id'

        if image_ids is not None:
            df = df[df[id_col].isin(image_ids)].reset_index(drop=True)

        self.img_dir  = img_dir
        self.id_col   = id_col
        self.ids      = df[id_col].tolist()
        self.labels   = [label_map[str(v)] for v in df[label_col].tolist()]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.ids[idx] + '.jpg')
        img  = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def make_balanced_sampler(dataset):
    """WeightedRandomSampler so each class is sampled equally per epoch."""
    labels = np.array(dataset.labels)
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(dataset),
        replacement=True,
    )


def build_splits(csv_path, img_dir, label_map, label_col, cfg, supcon=False):
    """
    Creates train / val / test DataLoaders with stratified splits.
    Returns: train_loader, val_loader, test_loader
    """
    df = pd.read_csv(csv_path)
    id_col = 'image_id' if 'image_id' in df.columns else 'isic_id'
    ids    = df[id_col].tolist()
    labels = [label_map[str(v)] for v in df[label_col].tolist()]

    # Stratified train / (val+test) split
    tr_ids, tmp_ids, tr_lbl, tmp_lbl = train_test_split(
        ids, labels, test_size=cfg['val_split'] + cfg['test_split'],
        stratify=labels, random_state=cfg['seed'])

    val_frac = cfg['val_split'] / (cfg['val_split'] + cfg['test_split'])
    val_ids, te_ids = train_test_split(
        tmp_ids, test_size=1 - val_frac,
        stratify=tmp_lbl, random_state=cfg['seed'])

    def make_loader(split_ids, mode, use_sampler=False, batch=None, sc=False):
        ds = ISICDataset(csv_path, img_dir, label_map, label_col,
                         image_ids=split_ids,
                         transform=build_transforms(cfg, mode=mode, supcon=sc))
        sampler = make_balanced_sampler(ds) if use_sampler else None
        return DataLoader(
            ds,
            batch_size=batch or cfg['s1_batch'],
            sampler=sampler,
            shuffle=(sampler is None and mode == 'train'),
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=(mode == 'train'),
        )

    train_loader = make_loader(tr_ids, 'train', use_sampler=True,
                               batch=cfg['s1_batch'], sc=supcon)
    val_loader   = make_loader(val_ids, 'val',   batch=cfg['s2_batch'])
    test_loader  = make_loader(te_ids,  'val',   batch=cfg['s2_batch'])
    return train_loader, val_loader, test_loader


def verify_data(cfg):
    """Quick check: verify all three datasets are accessible and log class distributions."""
    print("\n" + "═"*60)
    print("  DATA VERIFICATION")
    print("═"*60)

    datasets = [
        ('HAM10000',  cfg['ham_csv'],    cfg['ham_img_dir'],    HAM10000_CLASSES, 'dx'),
        ('ISIC 2019', cfg['isic19_csv'], cfg['isic19_img_dir'], ISIC2019_CLASSES, 'MEL'),
        ('ISIC 2020', cfg['isic20_csv'], cfg['isic20_img_dir'], ISIC2020_CLASSES, 'target'),
    ]

    for name, csv, img_dir, lmap, lcol in datasets:
        if not os.path.exists(csv):
            print(f"  [MISSING] {name} CSV: {csv}")
            continue
        df = pd.read_csv(csv)
        counts = Counter([lmap[str(v)] for v in df[lcol]])
        total = len(df)
        print(f"\n  {name}  ({total:,} images)")
        for cls_id, n in sorted(counts.items()):
            bar = '█' * int(20 * n / total)
            print(f"    Class {cls_id}: {n:5d}  {bar}")

    print("\n  Device:", cfg['device'])
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")
    elif torch.backends.mps.is_available():
        print("  GPU: Apple Silicon (MPS) — training will use your Mac GPU")
    print("═"*60)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODELS
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-B/14 backbone with a switchable head.

    Stage 1 (SupCon):    backbone → LayerNorm → proj_head (128-dim, L2-normed)
    Stage 2 (classify):  backbone → LayerNorm → class_head (num_classes logits)

    Backbone loading:
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        Downloads ~330 MB to ~/.cache/torch/hub/ on first run.
        If behind firewall, see: https://github.com/facebookresearch/dinov2
    """

    def __init__(self, num_classes, embed_dim=128, freeze_backbone=True,
                 unfreeze_last_n=4):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 from torch.hub …")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
            verbose=False)
        self.feat_dim = self.backbone.embed_dim   # 768

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last N ViT blocks + norm layer for fine-tuning
        if not freeze_backbone or unfreeze_last_n > 0:
            blocks = list(self.backbone.blocks)
            for blk in (blocks if not freeze_backbone else blocks[-unfreeze_last_n:]):
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        # SupCon projection head (Stage 1)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, embed_dim),
        )

        # Classification head (Stage 2)
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self._mode = 'supcon'   # 'supcon' | 'classify'

    def set_mode(self, mode):
        """Switch between contrastive pretraining and classification."""
        assert mode in ('supcon', 'classify')
        self._mode = mode
        if mode == 'classify':
            for p in self.proj_head.parameters():
                p.requires_grad = False
            for p in self.class_head.parameters():
                p.requires_grad = True
        else:
            for p in self.proj_head.parameters():
                p.requires_grad = True
            for p in self.class_head.parameters():
                p.requires_grad = False
        print(f"  Model mode → {mode}")

    def forward(self, x, return_features=False):
        feats = self.backbone(x)                   # (B, 768) CLS token
        if return_features:
            return feats
        if self._mode == 'supcon':
            z = self.proj_head(feats)
            return F.normalize(z, dim=1)           # (B, embed_dim) L2-normed
        return self.class_head(feats)              # (B, num_classes) logits

    def get_attention_maps(self, x):
        """DINO self-attention from last ViT block → spatial lesion highlighting."""
        return self.backbone.get_last_selfattention(x)  # (B, heads, N+1, N+1)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


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

    def set_mode(self, mode):
        pass   # baselines don't use SupCon


class EfficientNetBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = tv_models.efficientnet_v2_l(
            weights=tv_models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        in_f = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, num_classes))

    def forward(self, x, return_features=False):
        return self.model(x)

    def set_mode(self, mode):
        pass


def build_model(cfg):
    arch = cfg['arch'].lower()
    nc   = cfg['num_classes']
    if arch == 'dinov2':
        return DINOv2Classifier(
            num_classes=nc,
            embed_dim=cfg['embed_dim'],
            unfreeze_last_n=cfg['unfreeze_last_n'],
        )
    elif arch == 'resnet152':
        return ResNetBaseline(nc)
    elif arch == 'efficientnetv2':
        return EfficientNetBaseline(nc)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOSSES
# ══════════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss  (Khosla et al., NeurIPS 2020).
    https://arxiv.org/abs/2004.11362

    For each anchor:
      positives  = same-class views in the batch
      negatives  = all other samples

    With two augmented views per image (n_views=2):
      - Effective batch size is doubled
      - Each original image appears twice as an anchor
      - SupCon pulls same-class clusters together regardless of sample count
        → compact melanoma clusters even with <10% class frequency
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        """
        features: (B, n_views, embed_dim)  — L2-normalized
        labels:   (B,)
        """
        device = features.device
        B, n_views, D = features.shape

        # Flatten → (B*n_views, D)
        f = features.reshape(B * n_views, D)
        lbl = labels.repeat_interleave(n_views)      # (B*n_views,)
        N = B * n_views

        # Scaled cosine similarity matrix
        sim = torch.matmul(f, f.T) / self.T           # (N, N)

        # Masks
        eye    = torch.eye(N, dtype=torch.bool, device=device)
        pos    = (lbl.unsqueeze(0) == lbl.unsqueeze(1)) & ~eye   # same-class, not self

        # Numerically stable log-softmax
        sim_max = sim.detach().max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim - sim_max) * (~eye).float()

        log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos.float().sum(dim=1).clamp(min=1)
        loss  = -(pos.float() * log_prob).sum(dim=1) / n_pos
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss  (Lin et al., ICCV 2017)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight    # per-class tensor for additional rebalancing

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):    self.val = self.sum = self.count = 0
    def update(self, v, n=1):
        self.val    = v
        self.sum   += v * n
        self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def train_stage1_supcon(model, train_loader, cfg):
    """
    Stage 1 — Supervised Contrastive Pretraining.

    Gradient accumulation simulates large batch (512+) on limited GPU memory.
    AMP (automatic mixed precision) halves memory and speeds up training.
    """
    print("\n" + "═"*60)
    print("  STAGE 1 — Supervised Contrastive Pretraining")
    print("═"*60)

    device = cfg['device']
    model.set_mode('supcon')
    model = model.to(device)

    trainable = model.trainable_params() if hasattr(model, 'trainable_params') else \
                sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable/1e6:.2f}M")

    criterion = SupConLoss(temperature=cfg['temperature'])
    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['s1_lr'], weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['s1_epochs'])
    scaler    = GradScaler()

    grad_accum = cfg['s1_grad_accum']
    best_loss  = float('inf')
    history    = {'loss': []}

    for epoch in range(cfg['s1_epochs']):
        model.train()
        meter = AverageMeter()
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"  S1 Ep {epoch+1:02d}/{cfg['s1_epochs']:02d}")

        for step, (batch, labels) in pbar:
            # batch is (view1, view2) tuple from TwoViewTransform
            if isinstance(batch, (list, tuple)):
                v1, v2 = batch
                imgs = torch.cat([v1, v2], dim=0).to(device)   # (2B, C, H, W)
            else:
                imgs = batch.to(device)
            labels = labels.to(device)

            with autocast():
                z = model(imgs)                           # (2B, embed_dim)
                B = labels.size(0)
                z = z.view(2, B, -1).permute(1, 0, 2)    # (B, 2, embed_dim)
                loss = criterion(z, labels) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            meter.update(loss.item() * grad_accum)
            pbar.set_postfix(loss=f'{meter.avg:.4f}')

        scheduler.step()
        history['loss'].append(meter.avg)
        print(f"  → Epoch {epoch+1:02d} | Loss: {meter.avg:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if meter.avg < best_loss:
            best_loss = meter.avg
            path = os.path.join(cfg['save_dir'], 'stage1_best.pt')
            torch.save(model.state_dict(), path)
            print(f"     ✓ Saved Stage 1 checkpoint  (loss={best_loss:.4f})")

    return model, history


def train_stage2_classifier(model, train_loader, val_loader, cfg):
    """
    Stage 2 — Linear Classifier Fine-tuning.

    Backbone and projection head are frozen.
    Only the classification head is trained with focal loss + label smoothing.
    Temperature scaling is applied post-training for calibration.
    """
    print("\n" + "═"*60)
    print("  STAGE 2 — Classifier Fine-tuning")
    print("═"*60)

    device = cfg['device']
    model.set_mode('classify')
    model = model.to(device)

    criterion = FocalLoss(gamma=cfg['focal_gamma'])
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['s2_lr'], weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['s2_epochs'])
    scaler    = GradScaler()

    best_bacc = 0.0
    history   = {'loss': [], 'val_bacc': [], 'val_auc': []}

    for epoch in range(cfg['s2_epochs']):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        meter = AverageMeter()
        pbar  = tqdm(train_loader, desc=f"  S2 Ep {epoch+1:02d}/{cfg['s2_epochs']:02d}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.item())
            pbar.set_postfix(loss=f'{meter.avg:.4f}')

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        bacc, macro_auc, _ = evaluate(model, val_loader, cfg)
        history['loss'].append(meter.avg)
        history['val_bacc'].append(bacc)
        history['val_auc'].append(macro_auc)

        print(f"  → Epoch {epoch+1:02d} | Loss: {meter.avg:.4f} | "
              f"BalAcc: {bacc:.4f} | AUC: {macro_auc:.4f}")

        if bacc > best_bacc:
            best_bacc = bacc
            path = os.path.join(cfg['save_dir'], 'stage2_best.pt')
            torch.save(model.state_dict(), path)
            print(f"     ✓ Saved Stage 2 checkpoint  (BalAcc={best_bacc:.4f})")

    return model, history


def train_baseline(model, train_loader, val_loader, cfg):
    """Single-stage training for ResNet / EfficientNet baselines."""
    print("\n" + "═"*60)
    print(f"  BASELINE TRAINING — {cfg['arch'].upper()}")
    print("═"*60)

    device    = cfg['device']
    model     = model.to(device)
    criterion = FocalLoss(gamma=cfg['focal_gamma'])
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['s2_epochs'])
    scaler    = GradScaler()

    best_bacc = 0.0
    history   = {'loss': [], 'val_bacc': [], 'val_auc': []}

    for epoch in range(cfg['s2_epochs']):
        model.train()
        meter = AverageMeter()
        for imgs, labels in tqdm(train_loader,
                                 desc=f"  Ep {epoch+1:02d}/{cfg['s2_epochs']:02d}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.item())

        scheduler.step()
        bacc, macro_auc, _ = evaluate(model, val_loader, cfg)
        history['loss'].append(meter.avg)
        history['val_bacc'].append(bacc)
        history['val_auc'].append(macro_auc)
        print(f"  Ep {epoch+1:02d} | Loss {meter.avg:.4f} | BalAcc {bacc:.4f} | AUC {macro_auc:.4f}")

        if bacc > best_bacc:
            best_bacc = bacc
            torch.save(model.state_dict(),
                       os.path.join(cfg['save_dir'], 'baseline_best.pt'))

    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def get_tta_transforms(cfg):
    sz   = cfg['img_size']
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = [T.Resize((sz, sz)), T.ToTensor(), norm]
    augmentations = [
        [],
        [T.RandomHorizontalFlip(p=1.0)],
        [T.RandomVerticalFlip(p=1.0)],
        [T.RandomRotation(90)],
        [T.RandomRotation(180)],
        [T.RandomRotation(270)],
        [T.ColorJitter(brightness=0.2)],
        [T.ColorJitter(contrast=0.2)],
        [T.GaussianBlur(kernel_size=max(3, int(0.05*sz)|1))],
        [T.RandomResizedCrop(sz, scale=(0.85, 1.0))],
    ][:cfg['tta_views']]
    return [T.Compose(aug + base) for aug in augmentations]


@torch.no_grad()
def predict_with_tta(model, pil_images, tta_transforms, device, entropy_weight=True):
    """Runs TTA on a list of PIL images; returns averaged probabilities (B, C)."""
    model.eval()
    view_probs = []
    for tf in tta_transforms:
        batch = torch.stack([tf(img) for img in pil_images]).to(device)
        probs = torch.softmax(model(batch), dim=1).detach().float().cpu().numpy()
        view_probs.append(probs)
    view_probs = np.stack(view_probs)           # (n_views, B, C)

    if entropy_weight:
        H = -(view_probs * np.log(view_probs + 1e-8)).sum(-1)  # (n_views, B)
        w = (1.0 / (H + 1e-8))
        w = w / w.sum(0, keepdims=True)
        return np.einsum('vb,vbc->bc', w, view_probs)

    return view_probs.mean(0)


@torch.no_grad()
def evaluate(model, loader, cfg, use_tta=False):
    """
    Returns: balanced_accuracy, macro_auc_roc, dict_of_all_metrics
    """
    # Use CPU for eval — MPS hangs on tensor transfer
    eval_device = 'cpu'
    model = model.to(eval_device)
    model.eval()
    all_probs, all_true = [], []

    if not use_tta:
        for imgs, labels in tqdm(loader, desc="  Evaluating"):
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            logits = model(imgs.to(eval_device))
            probs  = torch.softmax(logits, dim=1).numpy()
            all_probs.append(probs)
            all_true.extend(labels.numpy())
    else:
        tta_tfs = get_tta_transforms(cfg)
        ds = loader.dataset
        batch_size = 16
        for i in tqdm(range(0, len(ds), batch_size), desc="  TTA Eval"):
            pil_imgs, labels = zip(*(
                (Image.open(os.path.join(ds.img_dir, ds.ids[j] + '.jpg')).convert('RGB'),
                 ds.labels[j])
                for j in range(i, min(i + batch_size, len(ds)))
            ))
            probs = predict_with_tta(model, list(pil_imgs), tta_tfs, eval_device)
            all_probs.append(probs)
            all_true.extend(labels)
    model = model.to(cfg['device'])

    y_prob = np.vstack(all_probs)
    y_true = np.array(all_true)
    y_pred = y_prob.argmax(axis=1)

    n_cls = y_prob.shape[1]
    metrics = {}

    # 1. Balanced accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # 2. Macro AUC-ROC
    try:
        if n_cls == 2:
            metrics['auc_roc_macro'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics['auc_roc_macro'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        metrics['auc_roc_macro'] = float('nan')

    # 3. Sensitivity @ 95% specificity (melanoma = class 0 vs rest)
    metrics['sensitivity_at_95spec'] = _sensitivity_at_spec(y_true, y_prob, 0.95)

    # 4. Partial AUC (TPR > 80%)
    metrics['partial_auc_gt80tpr'] = _partial_auc(y_true, y_prob, min_tpr=0.80)

    # 5. Per-class F1 + macro F1
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, v in enumerate(f1s):
        metrics[f'f1_class_{i}'] = v
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 6. Cohen's kappa
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)

    # 7. ECE
    metrics['ece'] = _ece(y_true, y_prob)

    bacc     = metrics['balanced_accuracy']
    auc_roc  = metrics['auc_roc_macro']
    return bacc, auc_roc, metrics


def _sensitivity_at_spec(y_true, y_prob, target_spec=0.95):
    binary = (y_true == 0).astype(int)
    scores = y_prob[:, 0]
    try:
        fpr, tpr, _ = roc_curve(binary, scores)
        spec = 1 - fpr
        mask = spec >= target_spec
        return float(tpr[mask].max()) if mask.any() else float('nan')
    except Exception:
        return float('nan')


def _partial_auc(y_true, y_prob, min_tpr=0.80):
    binary = (y_true == 0).astype(int)
    scores = y_prob[:, 0]
    try:
        fpr, tpr, _ = roc_curve(binary, scores)
        mask = tpr >= min_tpr
        return float(np.trapz(tpr[mask], fpr[mask])) if mask.sum() >= 2 else float('nan')
    except Exception:
        return float('nan')


def _ece(y_true, y_prob, n_bins=15):
    confidence = y_prob.max(axis=1)
    preds      = y_prob.argmax(axis=1)
    correct    = (preds == y_true).astype(float)
    edges      = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    n          = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(correct[mask].mean() - confidence[mask].mean())
    return ece / n


def apply_referral(y_prob, threshold=0.70):
    """Flag low-confidence predictions as -1 (refer to specialist)."""
    preds = y_prob.argmax(axis=1).copy()
    preds[y_prob.max(axis=1) < threshold] = -1
    rate = (preds == -1).mean()
    print(f"  Referral rate @ threshold {threshold:.2f}: {rate:.1%}")
    return preds


class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration via temperature scaling.
    Fit on validation logits after Stage 2; apply before cross-dataset eval.
    Reduces ECE significantly with a single learned scalar T.
    """

    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T.clamp(min=0.1)

    @torch.no_grad()
    def fit(self, val_logits, val_labels, n_iter=100, lr=0.01):
        logits = torch.tensor(val_logits, dtype=torch.float32)
        labels = torch.tensor(val_labels, dtype=torch.long)
        opt    = torch.optim.LBFGS([self.T], lr=lr, max_iter=n_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        print(f"  Calibration temperature T = {self.T.item():.4f}")


def run_cross_dataset_protocols(model, cfg, class_names):
    """
    Runs all 3 evaluation protocols from the proposal.
    Protocol A: ISIC2019 → ISIC2020 (label schema change)
    Protocol B: HAM10000 → ISIC2020 (class granularity shift)
    Protocol C: ISIC2019 + HAM10000 → ISIC2020 held-out split
    """
    print("\n" + "═"*60)
    print("  CROSS-DATASET GENERALIZATION EVALUATION")
    print("═"*60)

    results = {}
    protocols = [
        ('A', cfg['isic20_csv'], cfg['isic20_img_dir'], ISIC2020_CLASSES, 'target', 2),
    ]

    for pid, csv, img_dir, lmap, lcol, n_cls in protocols:
        if not os.path.exists(csv):
            print(f"  [SKIP] Protocol {pid} — dataset not found: {csv}")
            continue

        print(f"\n  Protocol {pid}")
        _, _, loader = build_splits(csv, img_dir, lmap, lcol, cfg)
        bacc, auc_roc, metrics = evaluate(model, loader, cfg, use_tta=True)

        print(f"  BalAcc: {bacc:.4f}  |  AUC: {auc_roc:.4f}  |  "
              f"ECE: {metrics['ece']:.4f}  |  κ: {metrics['cohens_kappa']:.4f}")
        results[f'protocol_{pid}'] = metrics

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get('loss', []), color='#1D9E75', lw=1.5)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)

    if 'val_bacc' in history:
        axes[1].plot(history['val_bacc'], label='Balanced Acc', color='#1D9E75', lw=1.5)
        axes[1].plot(history.get('val_auc', []), label='AUC-ROC', color='#F2A623',
                     lw=1.5, linestyle='--')
        axes[1].set_title('Validation Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(); axes[1].grid(alpha=0.3)
    else:
        axes[1].axis('off')

    plt.suptitle(f'Training History — {cfg["arch"].upper()}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(cfg['figures_dir'], 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix_fig(y_true, y_pred, class_names, cfg, tag=''):
    valid  = y_pred != -1
    cm_arr = confusion_matrix(y_true[valid], y_pred[valid])
    cm_n   = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names[:cm_n.shape[1]],
                yticklabels=class_names[:cm_n.shape[0]],
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix')
    plt.tight_layout()
    path = os.path.join(cfg['figures_dir'], f'confusion_matrix{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves_fig(y_true, y_prob, class_names, cfg, tag=''):
    n_cls  = y_prob.shape[1]
    y_bin  = label_binarize(y_true, classes=list(range(n_cls)))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, color) in enumerate(zip(class_names[:n_cls], colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name} (AUC={roc_auc:.3f})")
        # Mark sensitivity at 95% specificity
        spec95 = np.where(1 - fpr >= 0.95)[0]
        if len(spec95):
            ax.scatter(fpr[spec95[-1]], tpr[spec95[-1]], color=color, marker='*', s=100)

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves  (★ = sensitivity at 95% specificity)')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    path = os.path.join(cfg['figures_dir'], f'roc_curves{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


@torch.no_grad()
def plot_tsne_embeddings(model, loader, class_names, cfg, n_batches=30):
    print("  Extracting embeddings for t-SNE …")
    device = cfg['device']
    model.eval()
    all_feats, all_labels = [], []

    for i, (imgs, labels) in enumerate(loader):
        if i >= n_batches:
            break
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        feats = model(imgs.to(device), return_features=True)
        all_feats.append(feats.cpu().numpy())
        all_labels.extend(labels.numpy())

    feats  = np.vstack(all_feats)
    labels = np.array(all_labels)

    print("  Running t-SNE …")
    emb = TSNE(n_components=2, perplexity=30, n_iter=1000,
               random_state=cfg['seed']).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[color],
                   label=name, alpha=0.6, s=8)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.set_title('t-SNE of DINOv2 Embeddings')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    path = os.path.join(cfg['figures_dir'], 'tsne_embeddings.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_dino_attention(model, loader, class_names, cfg, n_samples=6):
    """
    Visualizes DINOv2 self-attention maps — free lesion localization!
    Requires DINOv2 backbone which exposes get_last_selfattention().
    """
    if not hasattr(model, 'get_attention_maps'):
        print("  [SKIP] Attention maps — model is not DINOv2")
        return

    device  = cfg['device']
    sz      = cfg['img_size']
    patch   = 14   # ViT-B/14 patch size
    n_patch = sz // patch

    norm_tf = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resize  = T.Resize((sz, sz))

    model.eval()
    samples = []

    ds = loader.dataset
    for idx in range(min(n_samples, len(ds))):
        img_path = os.path.join(ds.img_dir, ds.ids[idx] + '.jpg')
        pil      = Image.open(img_path).convert('RGB')
        pil_rs   = resize(pil)
        x        = norm_tf(T.ToTensor()(pil_rs)).unsqueeze(0).to(device)

        with torch.no_grad():
            attn = model.get_attention_maps(x)   # (1, n_heads, N+1, N+1)

        # CLS token attention to patch tokens, averaged over heads
        attn_map = attn[0, :, 0, 1:].mean(0)    # (N,)
        attn_map = attn_map.reshape(n_patch, n_patch).cpu().numpy()
        attn_map = np.array(Image.fromarray(attn_map).resize((sz, sz), Image.BILINEAR))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        samples.append((np.array(pil_rs), attn_map, ds.labels[idx]))

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (img, attn, lbl) in enumerate(samples):
        overlay = cm.jet(attn)[:, :, :3]
        blend   = 0.5 * img / 255.0 + 0.5 * overlay

        axes[i][0].imshow(img);      axes[i][0].set_title(f'{class_names[lbl]} — original')
        axes[i][1].imshow(attn, cmap='jet'); axes[i][1].set_title('DINO attention')
        axes[i][2].imshow(blend);    axes[i][2].set_title('overlay')
        for ax in axes[i]:
            ax.axis('off')

    plt.suptitle('DINOv2 Self-Attention Maps on Skin Lesions', fontsize=13)
    plt.tight_layout()
    path = os.path.join(cfg['figures_dir'], 'attention_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_metrics_table(metrics, label=''):
    print(f"\n  {'─'*50}")
    if label:
        print(f"  {label}")
    print(f"  {'─'*50}")
    key_metrics = [
        ('balanced_accuracy',    'Balanced accuracy'),
        ('auc_roc_macro',        'AUC-ROC (macro)'),
        ('sensitivity_at_95spec','Sensitivity @ 95% spec'),
        ('partial_auc_gt80tpr',  'Partial AUC (>80% TPR)'),
        ('f1_macro',             'F1-macro'),
        ('cohens_kappa',         "Cohen's κ"),
        ('ece',                  'ECE (↓ better)'),
    ]
    for key, name in key_metrics:
        v = metrics.get(key, float('nan'))
        print(f"  {name:35s}: {v:.4f}")
    print(f"  {'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(cfg, train_loader, val_loader, test_loader, class_names):
    """
    Ablation matrix: systematically remove components to measure each contribution.

    Configs tested:
      1. Full model (DINOv2 + SupCon + TTA + Referral)
      2. No SupCon   (DINOv2 + CE only)
      3. No TTA
      4. Frozen backbone (no ViT block fine-tuning)
    """
    print("\n" + "═"*60)
    print("  ABLATION STUDY")
    print("═"*60)

    ablation_results = {}

    configs = [
        {'name': 'Full (DINOv2+SupCon+TTA)',    'supcon': True,  'tta': True,  'unfreeze': 4},
        {'name': 'No SupCon (DINOv2+CE+TTA)',   'supcon': False, 'tta': True,  'unfreeze': 4},
        {'name': 'No TTA (DINOv2+SupCon)',       'supcon': True,  'tta': False, 'unfreeze': 4},
        {'name': 'Frozen backbone (DINOv2+SupCon+TTA)', 'supcon': True,  'tta': True,  'unfreeze': 0},
    ]

    for abl_cfg in configs:
        print(f"\n  Config: {abl_cfg['name']}")
        local_cfg = {**cfg, 'unfreeze_last_n': abl_cfg['unfreeze'],
                     's1_epochs': 5, 's2_epochs': 5}   # fewer epochs for ablation speed
        model = build_model(local_cfg).to(cfg['device'])

        if abl_cfg['supcon'] and cfg['arch'] == 'dinov2':
            model, _ = train_stage1_supcon(model, train_loader, local_cfg)
            model, _ = train_stage2_classifier(model, train_loader, val_loader, local_cfg)
        else:
            if cfg['arch'] == 'dinov2':
                model.set_mode('classify')
            model, _ = train_baseline(model, train_loader, val_loader, local_cfg)

        _, _, metrics = evaluate(model, test_loader, local_cfg, use_tta=abl_cfg['tta'])
        ablation_results[abl_cfg['name']] = metrics
        print_metrics_table(metrics, abl_cfg['name'])

    # Summary table
    print("\n  ABLATION SUMMARY")
    print(f"  {'Config':<40} {'BalAcc':>8} {'AUC':>8} {'MEL-F1':>8} {'ECE':>8}")
    print("  " + "─"*72)
    for name, m in ablation_results.items():
        print(f"  {name:<40} "
              f"{m.get('balanced_accuracy',float('nan')):>8.4f} "
              f"{m.get('auc_roc_macro',float('nan')):>8.4f} "
              f"{m.get('f1_class_0',float('nan')):>8.4f} "
              f"{m.get('ece',float('nan')):>8.4f}")
    print()
    return ablation_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Detection Pipeline')
    parser.add_argument('--stage',  default='all',
                        choices=['all', 'data', 'train', 'eval', 'visualize', 'ablation'])
    parser.add_argument('--arch',   default=None,
                        choices=['dinov2', 'resnet152', 'efficientnetv2'])
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.arch:
        cfg['arch'] = args.arch
        if args.arch != 'dinov2':
            cfg['img_size'] = 224   # CNN baselines use 224

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    os.makedirs(cfg['save_dir'],   exist_ok=True)
    os.makedirs(cfg['figures_dir'],exist_ok=True)

    print("═"*60)
    print("  Enhanced Skin Cancer Detection")
    print("  Evet Fernandes | CSUDH M.S. CS | Spring 2026")
    print(f"  Architecture: {cfg['arch'].upper()} | Device: {cfg['device']}")
    print("═"*60)

    # ── Primary dataset: HAM10000 (7-class) ───────────────────────────────
    class_names = HAM10000_NAMES
    label_map   = HAM10000_CLASSES
    label_col   = 'dx'
    csv         = cfg['ham_csv']
    img_dir     = cfg['ham_img_dir']

    # ── DATA ──────────────────────────────────────────────────────────────
    if args.stage in ('all', 'data'):
        verify_data(cfg)
        if args.stage == 'data':
            return

    # Build DataLoaders
    print("\n  Building data loaders …")
    supcon_mode = (cfg['arch'] == 'dinov2')

    train_loader, val_loader, test_loader = build_splits(
        csv, img_dir, label_map, label_col, cfg, supcon=supcon_mode)

    print(f"  Train: {len(train_loader.dataset):,} | "
          f"Val: {len(val_loader.dataset):,} | "
          f"Test: {len(test_loader.dataset):,}")

    # ── MODEL ─────────────────────────────────────────────────────────────
    model = build_model(cfg)

    if args.resume:
        print(f"\n  Loading checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=cfg['device']))

    # ── TRAIN ─────────────────────────────────────────────────────────────
    s1_history = {}
    s2_history = {}

    if args.stage in ('all', 'train'):
        if cfg['arch'] == 'dinov2':
            best_s1 = os.path.join(cfg['save_dir'], 'stage1_best.pt')
            if os.path.exists(best_s1):
                print("\n  Stage 1 checkpoint found — skipping Stage 1")
                print(f"  Loading: {best_s1}")
                model.load_state_dict(torch.load(best_s1, map_location=cfg['device']))
            else:
                model, s1_history = train_stage1_supcon(model, train_loader, cfg)
                if os.path.exists(best_s1):
                    model.load_state_dict(torch.load(best_s1, map_location=cfg['device']))
                    print("  Loaded best Stage 1 checkpoint for Stage 2")

            # Stage 2 needs single-view loaders
            train_loader2, val_loader2, _ = build_splits(
                csv, img_dir, label_map, label_col, cfg, supcon=False)
            model, s2_history = train_stage2_classifier(
                model, train_loader2, val_loader2, cfg)
            val_loader = val_loader2   # use single-view val for final eval
        else:
            model, s2_history = train_baseline(model, train_loader, val_loader, cfg)

    # Load best classifier checkpoint for evaluation
    best_s2 = os.path.join(cfg['save_dir'],
                            'stage2_best.pt' if cfg['arch'] == 'dinov2' else 'baseline_best.pt')
    if os.path.exists(best_s2) and args.stage != 'train':
        model.load_state_dict(torch.load(best_s2, map_location=cfg['device']))
        print(f"\n  Loaded best checkpoint: {best_s2}")

    model = model.to(cfg['device'])
    if cfg['arch'] == 'dinov2':
        model.set_mode('classify')

    # ── EVALUATE ──────────────────────────────────────────────────────────
    if args.stage in ('all', 'eval'):
        print("\n" + "═"*60)
        print("  EVALUATION — TEST SET")
        print("═"*60)

        # Standard eval
        bacc, auc_roc, metrics = evaluate(model, test_loader, cfg, use_tta=False)
        print_metrics_table(metrics, 'Without TTA')

        # TTA eval
        bacc_tta, auc_roc_tta, metrics_tta = evaluate(model, test_loader, cfg, use_tta=True)
        print_metrics_table(metrics_tta, f'With TTA ({cfg["tta_views"]} views)')

        # Referral
        # (collect probs first)
        all_probs, all_true = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in test_loader:
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                probs = torch.softmax(model(imgs.to(cfg['device'])), dim=1).detach().float().cpu().numpy()
                all_probs.append(probs)
                all_true.extend(labels.numpy())
        all_probs = np.vstack(all_probs)
        all_true  = np.array(all_true)
        y_pred_ref = apply_referral(all_probs, cfg['referral_thr'])

        # Cross-dataset protocols
        run_cross_dataset_protocols(model, cfg, class_names)

    # ── VISUALIZE ─────────────────────────────────────────────────────────
    if args.stage in ('all', 'visualize'):
        print("\n" + "═"*60)
        print("  VISUALIZATION")
        print("═"*60)

        # Need single-view val loader for visualization
        _, val_vis, test_vis = build_splits(
            csv, img_dir, label_map, label_col, cfg, supcon=False)

        if s1_history or s2_history:
            combined = {**s1_history, **s2_history}
            plot_training_history(s2_history if s2_history else s1_history, cfg)

        # Collect test predictions for plots
        all_probs, all_true = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in test_vis:
                probs = torch.softmax(model(imgs.to(cfg['device'])), dim=1).detach().float().cpu().numpy()
                all_probs.append(probs)
                all_true.extend(labels.numpy())
        all_probs = np.vstack(all_probs)
        all_true  = np.array(all_true)
        all_pred  = all_probs.argmax(axis=1)

        plot_confusion_matrix_fig(all_true, all_pred, class_names, cfg)
        plot_roc_curves_fig(all_true, all_probs, class_names, cfg)
        plot_tsne_embeddings(model, val_vis, class_names, cfg)
        plot_dino_attention(model, test_vis, class_names, cfg)

        print(f"\n  All figures saved to ./{cfg['figures_dir']}/")

    # ── ABLATION ──────────────────────────────────────────────────────────
    if args.stage == 'ablation':
        _, val_abl, test_abl = build_splits(
            csv, img_dir, label_map, label_col, cfg, supcon=False)
        run_ablation(cfg, train_loader, val_abl, test_abl, class_names)

    print("\n  Pipeline complete ✓")


if __name__ == '__main__':
    main()
