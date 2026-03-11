"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PART 2 — Fine-tuning the Pruned MobileNetV4-Small                  ║
║         COCO 2017  ·  AMP  ·  Frozen backbone  ·  CosineAnnealingLR        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Kaggle dashboard inputs  (set via the variables below or pass as env vars):
    PRUNED_MODEL_PATH  — path to pruned_model.pth  (from part1_prune.py)
    COCO_ROOT          — directory containing train2017/, val2017/,
                         annotations/instances_train2017.json, etc.

Recommended Kaggle P100 settings:
    batch_size = 128   (fits comfortably in 16 GB VRAM with AMP)
    num_workers= 4

Outputs written to  ./finetune_outputs/
    ├── finetuned_model.pth          ← full model object
    ├── finetune_history.json
    ├── viz_training_curves.png
    └── viz_epoch_summary_table.png
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import os, json, time, copy, warnings
warnings.filterwarnings("ignore")

# ─── third-party ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from tqdm import tqdm

# ─── Kaggle dashboard inputs (override these or set env vars) ─────────────────
PRUNED_MODEL_PATH = os.environ.get(
    "PRUNED_MODEL_PATH", "./pruning_outputs/pruned_model.pth")
COCO_ROOT         = os.environ.get(
    "COCO_ROOT", "./coco")

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = dict(
    num_classes     = 80,
    img_size        = 224,
    # --- training schedule ---
    total_epochs    = 20,
    frozen_epochs   = 5,       # freeze backbone, train head only
    # --- optimiser ---
    lr_head         = 3e-4,    # LR while backbone frozen
    lr_full         = 5e-5,    # LR after unfreezing
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    # --- data ---
    batch_size      = 128,     # fits P100 16 GB with AMP + pin_memory
    num_workers     = 4,
    subset_size     = None,    # set int for smoke-test, e.g. 4096
    # --- early stopping ---
    patience        = 5,
    min_delta       = 5e-4,
    # --- misc ---
    output_dir      = "./finetune_outputs",
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    seed            = 42,
)

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
DEVICE = CFG["device"]

# ── Colour palette ────────────────────────────────────────────────────────────
C_TRAIN  = "#2E4057"
C_VAL    = "#E84855"
C_FROZEN = "#A8DADC"
C_FULL   = "#F4A261"
BG = "#FAFAFA"; GRID = "#E5E5E5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": BG, "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.labelsize": 11, "axes.titlesize": 13,
    "xtick.labelsize": 9,  "ytick.labelsize": 9,
    "legend.framealpha": 0.9, "legend.fontsize": 10,
})

print(f"[INFO] Device         : {DEVICE}")
print(f"[INFO] Pruned model   : {PRUNED_MODEL_PATH}")
print(f"[INFO] COCO root      : {COCO_ROOT}")


# ════════════════════════════════════════════════════════════════════════════
#  1. DATA
# ════════════════════════════════════════════════════════════════════════════

class COCOClassificationDataset(torch.utils.data.Dataset):
    """
    Single-label classification wrapper over COCO detection.
    Each image is assigned the label of its most-frequent category.
    """
    def __init__(self, img_dir, ann_file, transform=None):
        from pycocotools.coco import COCO
        self.coco      = COCO(ann_file)
        self.img_dir   = img_dir
        self.transform = transform
        self.ids       = list(sorted(self.coco.imgs.keys()))
        cats            = self.coco.loadCats(self.coco.getCatIds())
        self.cat2idx   = {c["id"]: i for i, c in enumerate(cats)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from PIL import Image
        img_id = self.ids[idx]
        info   = self.coco.imgs[img_id]
        img    = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        if anns:
            counts  = {}
            for a in anns:
                counts[a["category_id"]] = counts.get(a["category_id"], 0) + 1
            label = self.cat2idx[max(counts, key=counts.get)]
        else:
            label = 0
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n=2000, nc=80, sz=224):
        self.n, self.nc, self.sz = n, nc, sz
    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.randn(3, self.sz, self.sz), torch.randint(0, self.nc, (1,)).item()


def get_transforms(train=True, img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.08),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.143)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_loaders(cfg):
    train_ann = os.path.join(cfg["coco_root"] if "coco_root" in cfg
                             else COCO_ROOT,
                             "annotations", "instances_train2017.json")
    train_img = os.path.join(COCO_ROOT, "train2017")
    val_ann   = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
    val_img   = os.path.join(COCO_ROOT, "val2017")
    real = os.path.isfile(train_ann) and os.path.isdir(train_img)

    if real:
        print("[INFO] COCO 2017 found — loading real data.")
        train_ds = COCOClassificationDataset(
            train_img, train_ann, transform=get_transforms(True,  cfg["img_size"]))
        val_ds   = COCOClassificationDataset(
            val_img,   val_ann,   transform=get_transforms(False, cfg["img_size"]))
    else:
        print("[WARN] COCO not found — synthetic demo data.")
        train_ds = SyntheticDataset(4000, cfg["num_classes"], cfg["img_size"])
        val_ds   = SyntheticDataset(800,  cfg["num_classes"], cfg["img_size"])

    if cfg["subset_size"]:
        n_tr = min(cfg["subset_size"], len(train_ds))
        n_vl = min(cfg["subset_size"]//4, len(val_ds))
        train_ds = Subset(train_ds, list(range(n_tr)))
        val_ds   = Subset(val_ds,   list(range(n_vl)))

    # ── Persistent workers + prefetch for P100 throughput ─────────────────
    loader_kw = dict(
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        pin_memory  = True,
        persistent_workers = (cfg["num_workers"] > 0),
        prefetch_factor    = 2 if cfg["num_workers"] > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    print(f"[INFO] Train: {len(train_ds)} imgs | "
          f"Val: {len(val_ds)} imgs | "
          f"Batch: {cfg['batch_size']}")
    return train_loader, val_loader


# ════════════════════════════════════════════════════════════════════════════
#  2. LOAD PRUNED MODEL
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[1/5] Loading pruned model from {PRUNED_MODEL_PATH} …")
# weights_only=False is required when loading a full model object (not just
# a state_dict), because the file contains arbitrary Python classes (timm arch).
# This is safe as long as the .pth was produced by part1_prune.py on this machine.
import timm  # must be importable so pickle can reconstruct the timm class
torch.serialization.add_safe_globals([])   # no-op; kept for clarity
model = torch.load(PRUNED_MODEL_PATH, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"      Parameters: {n_params/1e6:.3f} M")


# ════════════════════════════════════════════════════════════════════════════
#  3. BACKBONE FREEZE / UNFREEZE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def freeze_backbone(model):
    """Freeze everything except the final classifier head."""
    for name, param in model.named_parameters():
        if "head" in name or "classifier" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Freeze] Trainable params: {trainable/1e6:.3f} M (head only)")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Unfreeze] Trainable params: {trainable/1e6:.3f} M (full model)")


# ════════════════════════════════════════════════════════════════════════════
#  4. EARLY STOPPING
# ════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=5, min_delta=5e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0

    def __call__(self, val_acc):
        if self.best is None or val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
            return False
        self.counter += 1
        print(f"  [EarlyStopping] {self.counter}/{self.patience} "
              f"(best={self.best:.4f})")
        return self.counter >= self.patience


# ════════════════════════════════════════════════════════════════════════════
#  5. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc="  Train", leave=False, ncols=95,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), \
                       labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (out.argmax(1) == labels).sum().item()
        total      += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), \
                       labels.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)
        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (out.argmax(1) == labels).sum().item()
        total      += bs
    return total_loss / total, correct / total


def fine_tune(model, train_loader, val_loader, cfg):
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["label_smoothing"]).to(DEVICE)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    stopper   = EarlyStopping(patience=cfg["patience"],
                               min_delta=cfg["min_delta"])

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [],
               "lr":         [], "phase":    []}
    best_state, best_acc = None, 0.0

    # ── Phase 1: frozen backbone ──────────────────────────────────────────
    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["frozen_epochs"], eta_min=cfg["lr_head"]/20)

    print(f"\n{'─'*64}")
    print(f"  Phase 1 — Frozen backbone  "
          f"(epochs 1–{cfg['frozen_epochs']})")
    print(f"{'─'*64}")

    for epoch in range(1, cfg["frozen_epochs"] + 1):
        t0 = time.time()
        tr_l, tr_a = train_one_epoch(model, train_loader, criterion,
                                     optimizer, scaler)
        vl_l, vl_a = evaluate(model, val_loader, criterion)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_l); history["val_loss"].append(vl_l)
        history["train_acc"].append(tr_a);  history["val_acc"].append(vl_a)
        history["lr"].append(lr_now);       history["phase"].append("frozen")

        print(f"  Ep {epoch:02d}/{cfg['total_epochs']} [frozen] | "
              f"TrL {tr_l:.4f} TrA {tr_a:.4f} | "
              f"VlL {vl_l:.4f} VlA {vl_a:.4f} | "
              f"LR {lr_now:.2e} | {time.time()-t0:.1f}s")

        if vl_a > best_acc:
            best_acc   = vl_a
            best_state = copy.deepcopy(model.state_dict())

    # ── Phase 2: full fine-tune ───────────────────────────────────────────
    unfreeze_all(model)
    remaining = cfg["total_epochs"] - cfg["frozen_epochs"]
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr_full"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=cfg["lr_full"]/100)

    print(f"\n{'─'*64}")
    print(f"  Phase 2 — Full fine-tune  "
          f"(epochs {cfg['frozen_epochs']+1}–{cfg['total_epochs']})")
    print(f"{'─'*64}")

    for epoch in range(cfg["frozen_epochs"] + 1, cfg["total_epochs"] + 1):
        t0 = time.time()
        tr_l, tr_a = train_one_epoch(model, train_loader, criterion,
                                     optimizer, scaler)
        vl_l, vl_a = evaluate(model, val_loader, criterion)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_l); history["val_loss"].append(vl_l)
        history["train_acc"].append(tr_a);  history["val_acc"].append(vl_a)
        history["lr"].append(lr_now);       history["phase"].append("full")

        print(f"  Ep {epoch:02d}/{cfg['total_epochs']} [full]   | "
              f"TrL {tr_l:.4f} TrA {tr_a:.4f} | "
              f"VlL {vl_l:.4f} VlA {vl_a:.4f} | "
              f"LR {lr_now:.2e} | {time.time()-t0:.1f}s")

        if vl_a > best_acc:
            best_acc   = vl_a
            best_state = copy.deepcopy(model.state_dict())

        if stopper(vl_a):
            print(f"  ⚡ Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    print(f"\n  ✔ Best val accuracy: {best_acc:.4f}")
    return model, history, best_acc


# ════════════════════════════════════════════════════════════════════════════
#  6. RUN
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[2/5] Building data loaders …")
train_loader, val_loader = build_loaders(CFG)

print(f"\n[3/5] Fine-tuning …")
model, history, best_acc = fine_tune(model, train_loader, val_loader, CFG)

print(f"\n[4/5] Saving fine-tuned model …")
ft_path = os.path.join(CFG["output_dir"], "finetuned_model.pth")
torch.save(model.cpu(), ft_path)
print(f"  ✔ Model saved → {ft_path}")

hist_path = os.path.join(CFG["output_dir"], "finetune_history.json")
with open(hist_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"  ✔ History  → {hist_path}")


# ════════════════════════════════════════════════════════════════════════════
#  7. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[5/5] Generating visualisations …")

epochs_done   = list(range(1, len(history["train_loss"]) + 1))
frozen_end    = CFG["frozen_epochs"]
phase_boundary= frozen_end + 0.5   # line between phases


def _shade_phases(ax, total):
    """Shade phase-1 (frozen) region in the background."""
    ax.axvspan(0.5, frozen_end + 0.5, color=C_FROZEN, alpha=0.18,
               label="Frozen backbone")
    ax.axvspan(frozen_end + 0.5, total + 0.5, color=C_FULL, alpha=0.12,
               label="Full fine-tune")
    ax.axvline(phase_boundary, color="#888", ls=":", lw=1.2)


# ── 7-A  Training curves (loss + accuracy + LR) ───────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Loss
ax = axes[0]
_shade_phases(ax, len(epochs_done))
ax.plot(epochs_done, history["train_loss"], "o-", color=C_TRAIN,
        lw=2.2, ms=5, label="Train loss")
ax.plot(epochs_done, history["val_loss"],   "s--", color=C_VAL,
        lw=2.2, ms=5, label="Val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Loss", fontweight="bold")
ax.legend(loc="upper right")
ax.set_xlim(0.5, len(epochs_done) + 0.5)

# Accuracy
ax = axes[1]
_shade_phases(ax, len(epochs_done))
ax.plot(epochs_done, history["train_acc"], "o-",  color=C_TRAIN,
        lw=2.2, ms=5, label="Train acc")
ax.plot(epochs_done, history["val_acc"],   "s--", color=C_VAL,
        lw=2.2, ms=5, label="Val acc")
ax.set_xlabel("Epoch"); ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Accuracy", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(0.5, len(epochs_done) + 0.5)

# LR
ax = axes[2]
_shade_phases(ax, len(epochs_done))
ax.plot(epochs_done, history["lr"], "o-", color="#6A5ACD",
        lw=2.2, ms=5, label="LR")
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule", fontweight="bold")
ax.set_yscale("log")
ax.legend()
ax.set_xlim(0.5, len(epochs_done) + 0.5)

fig.suptitle("Fine-tuning Progress — Pruned MobileNetV4-Small on COCO",
             fontsize=15, fontweight="bold", y=1.02)

# Add legend for phase shading to last subplot
handles_extra = [
    mpatches.Patch(color=C_FROZEN, alpha=0.5, label="Frozen backbone"),
    mpatches.Patch(color=C_FULL,   alpha=0.5, label="Full fine-tune"),
]
axes[2].legend(handles=handles_extra + axes[2].get_legend_handles_labels()[0],
               fontsize=8)

plt.tight_layout()
curves_path = os.path.join(CFG["output_dir"], "viz_training_curves.png")
plt.savefig(curves_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"  ✔ Training curves → {curves_path}")


# ── 7-B  Epoch summary table ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, max(4, len(epochs_done)*0.38 + 1.5)))
ax.axis("off")

col_labels = ["Epoch", "Phase", "Train Loss", "Val Loss",
              "Train Acc", "Val Acc", "LR"]
rows = []
for i, ep in enumerate(epochs_done):
    ph = history["phase"][i].capitalize()
    rows.append([
        str(ep), ph,
        f"{history['train_loss'][i]:.4f}",
        f"{history['val_loss'][i]:.4f}",
        f"{history['train_acc'][i]:.4f}",
        f"{history['val_acc'][i]:.4f}",
        f"{history['lr'][i]:.2e}",
    ])

row_colors = []
for i in range(len(rows)):
    ph = history["phase"][i]
    base = "#EBF5FB" if ph == "frozen" else "#FEF9E7"
    # highlight best val acc row
    if history["val_acc"][i] == max(history["val_acc"]):
        base = "#D5F5E3"
    row_colors.append([base] * len(col_labels))

tbl = ax.table(
    cellText   = rows,
    colLabels  = col_labels,
    cellLoc    = "center",
    loc        = "center",
    cellColours= row_colors,
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.5)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor(C_TRAIN)
    tbl[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Fine-tuning Epoch Summary  (green = best val acc)",
             fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
table_path = os.path.join(CFG["output_dir"], "viz_epoch_summary_table.png")
plt.savefig(table_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"  ✔ Epoch summary table → {table_path}")


# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "╔" + "═"*60 + "╗")
print("║   PART 2 COMPLETE — Fine-tuning                          ║")
print("╠" + "═"*60 + "╣")
rows_s = [
    f"  Total epochs run  : {len(epochs_done)}",
    f"  Best val accuracy : {best_acc:.4f}",
    f"  Finetuned model   : {ft_path}",
]
for r in rows_s:
    print(f"║  {r:<58}║")
print("╚" + "═"*60 + "╝\n")
print("  ➜  Run part3_evaluate_export.py next.")
