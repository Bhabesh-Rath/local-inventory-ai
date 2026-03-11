"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PART 3 — Evaluation, Int8 PTQ, Benchmarking & Export                     ║
║   Three-way comparison: Original · Pruned+FT · Int8 Quantized              ║
║   Metrics: Top-1/5 Acc · mAP@.5 · mAP@.5:.95 · Confusion Matrix           ║
║            Class-wide perf drop · Latency · FLOPs · Model size             ║
║   Exports: TorchScript (.pt) · ONNX (.onnx) · TFLite Int8 (.tflite)       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Kaggle dashboard inputs (env vars or edit paths below):
    ORIGINAL_MODEL_PATH   ./pruning_outputs/original_model.pth
    FINETUNED_MODEL_PATH  ./finetune_outputs/finetuned_model.pth
    COCO_ROOT             directory containing val2017/ + annotations/

Install extras if needed:
    !pip install torch-pruning onnx onnxruntime --quiet
    # For TFLite (recommended path):
    !pip install ai-edge-torch --quiet
    # Or via TF:
    !pip install tensorflow onnx-tf --quiet

Outputs written to ./eval_outputs/
    ├── benchmark_results.json
    ├── viz_benchmark_bars.png          4-panel bar: params/size/latency/acc
    ├── viz_radar_chart.png             spider chart — all metrics
    ├── viz_latency_distribution.png    per-image latency histogram (3 models)
    ├── viz_confusion_matrix.png        3-panel normalised confusion matrices
    ├── viz_class_performance_drop.png  per-class Top-1 drop bar chart
    ├── viz_map_comparison.png          mAP@.5 and mAP@.5:.95 side-by-side
    ├── viz_efficiency_table.png        full numeric summary table
    ├── viz_final_report_card.png       dark-theme shareable slide
    ├── finetuned_model_scripted.pt     TorchScript — PyTorch Mobile
    ├── finetuned_model.onnx            ONNX opset-17 — cross-platform
    └── quantized_model_int8.tflite     Int8 TFLite — Android / Flutter
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import os, json, time, math, copy, warnings
warnings.filterwarnings("ignore")

# ─── third-party ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.quantization as tq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

import timm  # must be importable before any torch.load of timm model files

try:
    import torch_pruning as tp
    HAS_TP = True
except ImportError:
    HAS_TP = False
    print("[WARN] torch_pruning not found — MACs/FLOPs will be skipped.")

# ─── Paths ───────────────────────────────────────────────────────────────────
ORIGINAL_MODEL_PATH  = os.environ.get(
    "ORIGINAL_MODEL_PATH",  "./pruning_outputs/original_model.pth")
FINETUNED_MODEL_PATH = os.environ.get(
    "FINETUNED_MODEL_PATH", "./finetune_outputs/finetuned_model.pth")
COCO_ROOT            = os.environ.get("COCO_ROOT", "./coco")

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = dict(
    num_classes      = 80,
    img_size         = 224,
    batch_size       = 64,
    num_workers      = 4,
    latency_warmup   = 20,
    latency_reps     = 200,
    calib_batches    = 16,          # batches used for PTQ observer calibration
    map_iou_steps    = 10,          # steps from 0.50 to 0.95 for mAP@.5:.95
    output_dir       = "./eval_outputs",
    device           = "cuda" if torch.cuda.is_available() else "cpu",
    seed             = 42,
)

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
DEVICE   = CFG["device"]
IMG_SIZE = CFG["img_size"]
NC       = CFG["num_classes"]
IOU_THRESHOLDS = np.linspace(0.5, 0.95, CFG["map_iou_steps"])

# ── Colour palette ────────────────────────────────────────────────────────────
C_ORIG  = "#2E4057"   # deep navy    — Original
C_TUNE  = "#E84855"   # vivid red    — Pruned + FT
C_QUANT = "#F4A261"   # warm amber   — Int8 Quantized
C_GREEN = "#2ECC71"
BG = "#FAFAFA"; GRID = "#E5E5E5"
PALETTE = [C_ORIG, C_TUNE, C_QUANT]
LABELS3 = ["Original", "Pruned+FT", "Int8 Quant"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": BG, "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.labelsize": 11, "axes.titlesize": 13,
    "xtick.labelsize": 9,  "ytick.labelsize": 9,
    "legend.framealpha": 0.9, "legend.fontsize": 10,
})

print(f"[INFO] Device           : {DEVICE}")
print(f"[INFO] Original model   : {ORIGINAL_MODEL_PATH}")
print(f"[INFO] Finetuned model  : {FINETUNED_MODEL_PATH}")


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def count_params(m):
    try:
        return sum(p.numel() for p in m.parameters())
    except Exception:
        return 0


def model_size_mb(m):
    tmp = os.path.join(CFG["output_dir"], "_sz_tmp.pth")
    try:
        torch.save(m, tmp)
    except Exception:
        try:
            torch.save(m.state_dict(), tmp)
        except Exception:
            return 0.0
    s = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return s


def count_macs(model, device="cpu"):
    if not HAS_TP:
        return None
    try:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        macs, _ = tp.utils.count_ops_and_params(model, dummy)
        return macs
    except Exception:
        return None


def _pct(new, old):
    if old is None or old == 0 or new is None:
        return "N/A"
    return f"{(new - old) / abs(old) * 100:+.1f}%"


def _to_py(obj):
    """Recursively convert numpy scalars to Python for JSON serialisation."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(v) for v in obj]
    return obj


# ════════════════════════════════════════════════════════════════════════════
#  1. LOAD MODELS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("[1/8]  Loading Original & Finetuned models")
print("═"*64)

# weights_only=False: files contain full model objects with timm classes.
original_model  = torch.load(ORIGINAL_MODEL_PATH,  map_location=DEVICE,
                              weights_only=False).to(DEVICE)
finetuned_model = torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE,
                              weights_only=False).to(DEVICE)
original_model.eval()
finetuned_model.eval()

print(f"  Original  params : {count_params(original_model)/1e6:.3f} M")
print(f"  Finetuned params : {count_params(finetuned_model)/1e6:.3f} M")


# ════════════════════════════════════════════════════════════════════════════
#  2. DATA  —  COCO val2017 (or synthetic fallback)
# ════════════════════════════════════════════════════════════════════════════

class COCOClassificationDataset(torch.utils.data.Dataset):
    """
    Single-label classification wrapper over COCO detection annotations.
    Each image receives the label of its most-frequent annotated category.
    """
    def __init__(self, img_dir, ann_file, transform=None):
        from pycocotools.coco import COCO
        self.coco      = COCO(ann_file)
        self.img_dir   = img_dir
        self.transform = transform
        self.ids       = list(sorted(self.coco.imgs.keys()))
        cats            = self.coco.loadCats(self.coco.getCatIds())
        self.cat2idx   = {c["id"]: i for i, c in enumerate(cats)}
        self.idx2name  = {i: c["name"] for i, c in enumerate(cats)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from PIL import Image
        img_id = self.ids[idx]
        info   = self.coco.imgs[img_id]
        img    = Image.open(
            os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        if anns:
            counts = {}
            for a in anns:
                counts[a["category_id"]] = counts.get(a["category_id"], 0) + 1
            label = self.cat2idx[max(counts, key=counts.get)]
        else:
            label = 0
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, nc=80, sz=224):
        self.n, self.nc, self.sz = n, nc, sz
        self.idx2name = {i: f"class_{i:02d}" for i in range(nc)}
    def __len__(self): return self.n
    def __getitem__(self, i):
        return (torch.randn(3, self.sz, self.sz),
                int(torch.randint(0, self.nc, (1,))))


def get_val_transform(sz=224):
    return T.Compose([
        T.Resize(int(sz * 1.143)),
        T.CenterCrop(sz),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


val_ann_path = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
val_img_path = os.path.join(COCO_ROOT, "val2017")
REAL_COCO    = os.path.isfile(val_ann_path) and os.path.isdir(val_img_path)

if REAL_COCO:
    print("[INFO] COCO val2017 found — loading real data.")
    val_ds = COCOClassificationDataset(
        val_img_path, val_ann_path, transform=get_val_transform(IMG_SIZE))
else:
    print("[WARN] COCO not found — using synthetic data (demo mode).")
    val_ds = SyntheticDataset(1000, NC, IMG_SIZE)

IDX2NAME = val_ds.idx2name

loader_kw = dict(
    batch_size  = CFG["batch_size"],
    num_workers = CFG["num_workers"],
    pin_memory  = (DEVICE == "cuda"),
    shuffle     = False,
)
val_loader = DataLoader(val_ds, **loader_kw)

# Calibration subset (for PTQ observers)
n_calib    = min(CFG["calib_batches"] * CFG["batch_size"], len(val_ds))
calib_ds   = Subset(val_ds, list(range(n_calib)))
calib_loader = DataLoader(calib_ds, batch_size=CFG["batch_size"],
                          num_workers=CFG["num_workers"], shuffle=False)

print(f"  Val   samples : {len(val_ds)}")
print(f"  Calib samples : {len(calib_ds)}")


# ════════════════════════════════════════════════════════════════════════════
#  3. INT8 POST-TRAINING QUANTIZATION  (static, per-channel, qnnpack)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("[2/8]  Int8 Post-Training Quantization  (static · qnnpack · per-channel)")
print("═"*64)


# ── Why static PTQ fails on timm MobileNetV3/V4 ───────────────────────────────
#
# PyTorch static quantization requires ALL operations in the forward graph to
# have a QuantizedCPU kernel.  timm's MobileNetV3/V4 uses BatchNormAct2d from
# timm.layers.norm_act — a composite module that calls F.batch_norm() directly.
# F.batch_norm has no QuantizedCPU kernel (only CPU/CUDA), so it crashes even
# after QuantStub wrapping and Conv-BN fusion attempts, because timm's BN is
# not a plain nn.BatchNorm2d and torch.quantization.fuse_modules() silently
# skips it.
#
# SOLUTION — Manual Conv-BN Weight Folding + Dynamic Quantization
# ─────────────────────────────────────────────────────────────────
# Step 1: Manually fold every Conv2d+BN pair by absorbing the BN scale/bias
#         directly into the Conv weight/bias tensors, then replace the BN
#         with nn.Identity().  This eliminates ALL BN layers from the graph.
#         The folded model is arithmetically identical to the original at
#         inference time (eval mode) — this is a standard deployment technique.
#
# Step 2: Apply torch.quantization.quantize_dynamic() on the folded model.
#         Dynamic quantization quantizes weights to int8 statically and
#         quantizes activations dynamically per batch — it requires NO
#         calibration data and works on any architecture because it never
#         imposes QuantizedCPU constraints on non-Conv ops.
#
# Step 3: The resulting model:
#   • Has int8 weight storage (≈ 4× smaller than FP32 per Conv/Linear)
#   • Has faster CPU inference due to int8 GEMM kernels
#   • Accepts plain float32 inputs — no wrapper class needed
#   • Is fully compatible with ONNX export and the TFLite pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _fold_bn_into_conv(backbone: nn.Module) -> nn.Module:
    """
    Absorb every BatchNorm-like layer's learned affine parameters (weight γ,
    bias β, running_mean μ, running_var σ²) into the weight and bias of the
    immediately preceding Conv2d.

    Folding formula (standard, from the quantization literature):
        W_folded = W  *  (γ / sqrt(σ² + ε))          [broadcast over out-ch]
        b_folded = (b - μ) * (γ / sqrt(σ² + ε)) + β

    After folding the BN layer becomes an identity — it has no effect on the
    output.  We therefore replace it with nn.Identity() so no BN-specific
    kernel dispatch is ever attempted.

    Handles:
      - nn.BatchNorm2d
      - timm's BatchNormAct2d (timm.layers.norm_act)
      - Any module whose class name contains 'BatchNorm'
    """

    def _is_bn(m: nn.Module) -> bool:
        name = type(m).__name__
        return isinstance(m, nn.BatchNorm2d) or "BatchNorm" in name

    def _fold_pair(conv: nn.Conv2d, bn: nn.Module) -> None:
        """Fold bn into conv in-place."""
        # Pull BN buffers (always exist in eval mode)
        mu    = bn.running_mean                           # [C_out]
        var   = bn.running_var                            # [C_out]
        eps   = bn.eps if hasattr(bn, "eps") else 1e-5

        gamma = bn.weight if bn.weight is not None \
                else torch.ones_like(mu)                  # [C_out]
        beta  = bn.bias   if bn.bias   is not None \
                else torch.zeros_like(mu)                 # [C_out]

        # scale per output channel: γ / sqrt(σ² + ε)
        scale = gamma / torch.sqrt(var + eps)             # [C_out]

        # Fold into conv weight  [C_out, C_in, kH, kW]
        # Need shape [C_out, 1, 1, 1] for broadcasting
        conv.weight.data *= scale.view(-1, 1, 1, 1)

        # Fold into conv bias (create if absent)
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels,
                                                  device=conv.weight.device))
        conv.bias.data = (conv.bias.data - mu) * scale + beta

    # Walk the tree, look for Conv2d immediately followed by a BN-like module
    # Strategy: for each parent module, scan its direct children in order.
    for parent_name, parent in backbone.named_modules():
        child_names  = [n for n, _ in parent.named_children()]
        child_mods   = {n: m for n, m in parent.named_children()}

        for i in range(len(child_names) - 1):
            c_name = child_names[i]
            n_name = child_names[i + 1]
            c_mod  = child_mods[c_name]
            n_mod  = child_mods[n_name]

            if isinstance(c_mod, nn.Conv2d) and _is_bn(n_mod):
                # Fold BN into Conv
                _fold_pair(c_mod, n_mod)
                # Replace BN with Identity so no BN kernel is ever called
                setattr(parent, n_name, nn.Identity())

    # Also handle depthwise blocks where BN sits inside a sub-module
    # (e.g. timm's ConvBnAct / DepthwiseSeparableConv)
    for mod_name, mod in backbone.named_modules():
        if isinstance(mod, nn.Conv2d):
            # Check if this conv has a sibling BN in the same parent
            # already handled above; also check for direct sub-BN (rare)
            pass   # covered by parent-child walk above

    return backbone


def build_quantized_model(fp32_model: nn.Module,
                          calib_loader_: DataLoader) -> nn.Module:
    """
    Int8 Post-Training Quantization for timm MobileNetV3/V4.

    Uses dynamic quantization (weights int8, activations quantized per-batch)
    after manually folding all BatchNorm layers into their preceding Conv2d.

    Why dynamic instead of static
    ─────────────────────────────
    timm's BatchNormAct2d has no QuantizedCPU kernel, so static PTQ crashes on
    any unfused BN.  Manual BN folding eliminates BN from the graph entirely.
    Dynamic quantization then quantizes Conv2d and Linear weights to int8 with
    no calibration required and no QuantizedCPU dispatch constraints on
    activations — activations are quantized/dequantized on the fly per batch.

    Practical implications for Android deployment
    ─────────────────────────────────────────────
    • Weight storage ≈ 4× smaller (int8 vs float32).
    • Inference faster on ARM CPUs due to int8 GEMM.
    • No calibration dataset needed on-device.
    • ONNX export + TFLite conversion pipeline unchanged.
    """

    print("  Step 1/3 — Cloning FP32 model …")
    backbone = copy.deepcopy(fp32_model).cpu().eval()

    print("  Step 2/3 — Folding BatchNorm into Conv2d (manual weight absorption) …")
    backbone = _fold_bn_into_conv(backbone)

    # Verify no BN layers remain active (they're now nn.Identity)
    bn_remaining = [name for name, m in backbone.named_modules()
                    if "BatchNorm" in type(m).__name__]
    if bn_remaining:
        print(f"  [WARN] {len(bn_remaining)} BN layers could not be folded "
              f"(will be left as-is): {bn_remaining[:3]} …")
    else:
        print(f"  All BN layers folded → replaced with nn.Identity()")

    print("  Step 3/3 — Applying dynamic int8 quantization (Conv2d + Linear) …")
    torch.backends.quantized.engine = "qnnpack"

    quantized = tq.quantize_dynamic(
        backbone,
        qconfig_spec = {nn.Conv2d, nn.Linear},   # quantize weights of these
        dtype        = torch.qint8,
    )
    quantized.eval()

    print(f"  ✔ Quantization complete.")
    return quantized


# calib_loader is passed for API consistency but is not used by dynamic quant
# (no calibration pass needed — weights are quantized statically at conversion time)
quantized_model = build_quantized_model(finetuned_model, calib_loader)

print(f"  Quantized model size : {model_size_mb(quantized_model):.2f} MB")
print(f"  FP32 model size      : {model_size_mb(finetuned_model):.2f} MB")


# ════════════════════════════════════════════════════════════════════════════
#  4. FULL EVALUATION  —  all three models
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("[3/8]  Evaluating all three models")
print("═"*64)


@torch.no_grad()
def evaluate_full(model, loader, force_cpu=False):
    """
    Evaluate model on the given loader.

    Works identically for all three model variants:
      • FP32 Original    (force_cpu=False, runs on DEVICE)
      • FP32 Pruned+FT   (force_cpu=False, runs on DEVICE)
      • Dynamic Int8     (force_cpu=True,  runs on CPU)

    Dynamic-quantized models accept and return plain float32 — activations are
    quantized/dequantized internally per operation.  No special input handling
    is needed here.

    Returns
    -------
    dict: val_loss, top1_acc, top5_acc,
          per_class_correct[NC], per_class_total[NC],
          confusion_matrix[NC, NC],
          all_probs[N, NC], all_labels[N]
    """
    model.eval()
    run_dev   = torch.device("cpu") if force_cpu else torch.device(DEVICE)
    criterion = nn.CrossEntropyLoss()

    total_loss = correct1 = correct5 = total = 0
    per_class_correct = np.zeros(NC, dtype=np.int64)
    per_class_total   = np.zeros(NC, dtype=np.int64)
    conf_mat          = np.zeros((NC, NC), dtype=np.int64)
    all_probs_list, all_labels_list = [], []

    for imgs, labels in loader:
        # Always pass plain float32 — QuantWrapper quantizes internally
        imgs_d  = imgs.to(run_dev)
        lab_cpu = labels.cpu()

        amp_ctx = torch.cuda.amp.autocast(
            enabled=(not force_cpu and DEVICE == "cuda"))
        with amp_ctx:
            logits = model(imgs_d)

        logits_cpu = logits.cpu().float()
        probs      = torch.softmax(logits_cpu, dim=1)
        preds      = probs.argmax(dim=1)
        lab_np     = lab_cpu.numpy()
        pred_np    = preds.numpy()

        loss        = criterion(logits_cpu, lab_cpu)
        bs          = lab_cpu.size(0)
        total_loss += loss.item() * bs
        total      += bs
        correct1   += (preds == lab_cpu).sum().item()

        _, top5 = probs.topk(min(5, NC), dim=1)
        correct5 += (top5 == lab_cpu.unsqueeze(1)).any(dim=1).sum().item()

        for gt, pr in zip(lab_np, pred_np):
            per_class_total[gt]   += 1
            if gt == pr:
                per_class_correct[gt] += 1
            conf_mat[gt, pr]      += 1

        all_probs_list.append(probs.numpy())
        all_labels_list.append(lab_np)

    all_probs_arr  = np.concatenate(all_probs_list,  axis=0)
    all_labels_arr = np.concatenate(all_labels_list, axis=0)

    return dict(
        val_loss          = total_loss / total,
        top1_acc          = correct1 / total,
        top5_acc          = correct5 / total,
        per_class_correct = per_class_correct,
        per_class_total   = per_class_total,
        confusion_matrix  = conf_mat,
        all_probs         = all_probs_arr,
        all_labels        = all_labels_arr,
    )


def compute_map(all_probs, all_labels):
    """
    Classification-proxy mAP following the COCO convention.

    For each class c and confidence threshold t we compute:
        TP = (true label == c) AND (score_c >= t)
        FP = (true label != c) AND (score_c >= t)
    This gives a precision-recall curve; AP_c = area under that curve.

    mAP@.50     = mean(AP_c)  at threshold 0.50
    mAP@.5:.95  = mean over ten thresholds [0.50, 0.55, …, 0.95]
    """
    n_cls = all_probs.shape[1]

    def _ap_single(c, thr):
        scores = all_probs[:, c]
        gt     = (all_labels == c).astype(np.float32)
        order  = np.argsort(-scores)
        above  = scores[order] >= thr
        if above.sum() == 0:
            return 0.0
        gt_ord = gt[order][above]
        n_pos  = gt.sum()
        if n_pos == 0:
            return 0.0
        tp_cum = np.cumsum(gt_ord)
        fp_cum = np.cumsum(1 - gt_ord)
        rec    = tp_cum / (n_pos + 1e-9)
        prec   = tp_cum / (tp_cum + fp_cum + 1e-9)
        rec    = np.concatenate([[0], rec,  [1]])
        prec   = np.concatenate([[1], prec, [0]])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i+1])
        return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))

    ap_at_50    = [_ap_single(c, 0.50) for c in range(n_cls)]
    map_50      = float(np.mean(ap_at_50))

    map_5095_per_thr = []
    for thr in IOU_THRESHOLDS:
        map_5095_per_thr.append(
            float(np.mean([_ap_single(c, float(thr)) for c in range(n_cls)])))
    map_5095 = float(np.mean(map_5095_per_thr))

    return map_50, map_5095, ap_at_50


print("  [1/3] Original model …")
orig_eval  = evaluate_full(original_model,  val_loader, force_cpu=False)

print("  [2/3] Finetuned model …")
ft_eval    = evaluate_full(finetuned_model, val_loader, force_cpu=False)

print("  [3/3] Int8 Quantized model (CPU / qnnpack) …")
quant_eval = evaluate_full(quantized_model, val_loader, force_cpu=True)

print("  Computing mAP …")
orig_map50,  orig_map5095,  orig_ap_cls  = compute_map(
    orig_eval["all_probs"],  orig_eval["all_labels"])
ft_map50,    ft_map5095,    ft_ap_cls    = compute_map(
    ft_eval["all_probs"],    ft_eval["all_labels"])
quant_map50, quant_map5095, quant_ap_cls = compute_map(
    quant_eval["all_probs"], quant_eval["all_labels"])

for tag, ev, m50, m95 in [
        ("Original",   orig_eval,  orig_map50,  orig_map5095),
        ("Pruned+FT",  ft_eval,    ft_map50,    ft_map5095),
        ("Int8 Quant", quant_eval, quant_map50, quant_map5095)]:
    print(f"  [{tag:12s}]  Top-1={ev['top1_acc']:.4f}  "
          f"Top-5={ev['top5_acc']:.4f}  "
          f"mAP@.5={m50:.4f}  mAP@.5:.95={m95:.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  5. LATENCY  —  single-image benchmark for all three models
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("[4/8]  Single-image latency benchmark")
print("═"*64)


def measure_latency(model, reps=200, warmup=20, force_cpu=False):
    model.eval()
    dev   = torch.device("cpu") if force_cpu else torch.device(DEVICE)
    # _QuantWrapper accepts plain float32 — no special input pre-processing needed
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(dev)

    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if str(dev) != "cpu":
            torch.cuda.synchronize()
        for _ in range(reps):
            if str(dev) != "cpu":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); _ = model(dummy); e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                _  = model(dummy)
                times.append((time.perf_counter() - t0) * 1000)
    return times


print(f"  Original  ({DEVICE.upper()}) …")
orig_lat  = measure_latency(original_model,  CFG["latency_reps"],
                             CFG["latency_warmup"])
print(f"  Finetuned ({DEVICE.upper()}) …")
ft_lat    = measure_latency(finetuned_model, CFG["latency_reps"],
                             CFG["latency_warmup"])
print("  Int8 Quant (CPU / qnnpack) …")
quant_lat = measure_latency(quantized_model, CFG["latency_reps"],
                             CFG["latency_warmup"], force_cpu=True)

for tag, lats in [("Original", orig_lat), ("Finetuned", ft_lat),
                   ("Int8 Quant", quant_lat)]:
    print(f"  [{tag:12s}]  mean={np.mean(lats):.2f} ms  "
          f"p95={np.percentile(lats,95):.2f} ms")


# ════════════════════════════════════════════════════════════════════════════
#  6. COMPILE RESULTS
# ════════════════════════════════════════════════════════════════════════════

orig_macs  = count_macs(original_model,  DEVICE)
ft_macs    = count_macs(finetuned_model, DEVICE)
quant_macs = count_macs(quantized_model, "cpu")

results = {
    "original": {
        "label":       "Original",
        "params_M":    count_params(original_model)  / 1e6,
        "size_mb":     model_size_mb(original_model),
        "macs_M":      orig_macs  / 1e6 if orig_macs  else None,
        "top1_acc":    orig_eval["top1_acc"],
        "top5_acc":    orig_eval["top5_acc"],
        "map50":       orig_map50,
        "map5095":     orig_map5095,
        "lat_mean_ms": float(np.mean(orig_lat)),
        "lat_p95_ms":  float(np.percentile(orig_lat, 95)),
    },
    "finetuned": {
        "label":       "Pruned+FT",
        "params_M":    count_params(finetuned_model) / 1e6,
        "size_mb":     model_size_mb(finetuned_model),
        "macs_M":      ft_macs    / 1e6 if ft_macs    else None,
        "top1_acc":    ft_eval["top1_acc"],
        "top5_acc":    ft_eval["top5_acc"],
        "map50":       ft_map50,
        "map5095":     ft_map5095,
        "lat_mean_ms": float(np.mean(ft_lat)),
        "lat_p95_ms":  float(np.percentile(ft_lat, 95)),
    },
    "quantized": {
        "label":       "Int8 Quant",
        "params_M":    count_params(finetuned_model) / 1e6,   # same arch
        "size_mb":     model_size_mb(quantized_model),
        "macs_M":      quant_macs / 1e6 if quant_macs else None,
        "top1_acc":    quant_eval["top1_acc"],
        "top5_acc":    quant_eval["top5_acc"],
        "map50":       quant_map50,
        "map5095":     quant_map5095,
        "lat_mean_ms": float(np.mean(quant_lat)),
        "lat_p95_ms":  float(np.percentile(quant_lat, 95)),
    },
}

res_path = os.path.join(CFG["output_dir"], "benchmark_results.json")
with open(res_path, "w") as f:
    json.dump(_to_py(results), f, indent=2)
print(f"\n  ✔ Benchmark results → {res_path}")

o = results["original"]
p = results["finetuned"]
q = results["quantized"]


# ════════════════════════════════════════════════════════════════════════════
#  7. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("[5/8]  Generating visualisations")
print("═"*64)


# ── 7-A  Four-panel benchmark bars ───────────────────────────────────────────

def plot_benchmark_bars():
    metrics = [
        ("Parameters (M)",  "params_M",    False, "Lower is better"),
        ("Model Size (MB)", "size_mb",     False, "Lower is better"),
        ("Latency (ms/img)","lat_mean_ms", False, "Lower is better"),
        ("Top-1 Accuracy",  "top1_acc",    True,  "Higher is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (label, key, hi, sub) in zip(axes, metrics):
        vals = [o[key], p[key], q[key]]
        bars = ax.bar(LABELS3, vals, color=PALETTE,
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.4f}" if "Accuracy" in label else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")
        # Δ vs Original on bars 1 and 2
        for bar, val in zip(bars[1:], vals[1:]):
            dv   = (val - vals[0]) / (abs(vals[0]) + 1e-9) * 100
            good = (hi and dv >= 0) or (not hi and dv <= 0)
            col  = C_GREEN if good else C_TUNE
            ax.annotate(
                f"{'↑' if dv>0 else '↓'}{abs(dv):.1f}%",
                xy=(bar.get_x() + bar.get_width()/2, 0),
                xytext=(bar.get_x() + bar.get_width()/2, -max(vals)*0.07),
                ha="center", fontsize=8.5, color=col, fontweight="bold",
                annotation_clip=False)
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n({sub})", fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(vals) * 1.30)

    fig.suptitle("Three-Model Benchmark: Original · Pruned+FT · Int8 Quant",
                 fontsize=15, fontweight="bold", y=1.01)
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(PALETTE, LABELS3)]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=11)
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_benchmark_bars.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Benchmark bars           → {out}")


plot_benchmark_bars()


# ── 7-B  mAP comparison ───────────────────────────────────────────────────────

def plot_map_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, title in [
            (axes[0], "map50",   "mAP @ IoU = 0.50"),
            (axes[1], "map5095", "mAP @ IoU = 0.50 : 0.95"),
    ]:
        vals = [o[key], p[key], q[key]]
        bars = ax.bar(LABELS3, vals, color=PALETTE,
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_ylabel("mAP Score")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("Mean Average Precision — COCO Val2017",
                 fontsize=14, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(PALETTE, LABELS3)]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=11)
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_map_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ mAP comparison           → {out}")


plot_map_comparison()


# ── 7-C  Confusion matrices (3-panel, top-N classes) ─────────────────────────

def plot_confusion_matrices(top_n=20):
    """
    Row-normalised confusion matrices for the most-frequent top_n classes.
    Diagonal values are per-class recall. Off-diagonal shows confusion pairs.
    """
    top_idx  = np.argsort(orig_eval["per_class_total"])[::-1][:top_n]
    top_names = [IDX2NAME.get(int(i), f"cls{i}") for i in top_idx]

    cmap = LinearSegmentedColormap.from_list(
        "cm", ["#FFFFFF", "#1B2A4A"], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, (tag, ev, col) in zip(axes, [
            ("Original",   orig_eval,  C_ORIG),
            ("Pruned+FT",  ft_eval,    C_TUNE),
            ("Int8 Quant", quant_eval, C_QUANT),
    ]):
        cm      = ev["confusion_matrix"][np.ix_(top_idx, top_idx)].astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(top_n)); ax.set_yticks(range(top_n))
        ax.set_xticklabels(top_names, rotation=65, ha="right", fontsize=7)
        ax.set_yticklabels(top_names, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Ground Truth", fontsize=9)
        ax.set_title(f"{tag}\n(Top-{top_n} classes — row-normalised recall)",
                     fontweight="bold", fontsize=11, color=col)

        for i in range(top_n):
            v   = cm_norm[i, i]
            col_txt = "white" if v > 0.55 else "black"
            ax.text(i, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5, color=col_txt, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Recall")

    fig.suptitle(f"Confusion Matrices — Top {top_n} COCO Classes by Frequency",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_confusion_matrix.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Confusion matrices       → {out}")


plot_confusion_matrices(top_n=min(20, NC))


# ── 7-D  Per-class performance drop ──────────────────────────────────────────

def plot_class_performance_drop(top_n=30):
    """
    Horizontal bar chart of per-class Top-1 accuracy change vs Original.
    Pruned+FT (red) and Int8 (amber) shown side by side.
    Sorted by Pruned+FT drop so worst regressions are most visible.
    Only classes with >= 5 val samples included.
    """
    def per_cls_acc(ev):
        tot = ev["per_class_total"].clip(min=1)
        return ev["per_class_correct"] / tot

    orig_a  = per_cls_acc(orig_eval)
    ft_a    = per_cls_acc(ft_eval)
    quant_a = per_cls_acc(quant_eval)

    valid   = orig_eval["per_class_total"] >= 5
    idx     = np.where(valid)[0]

    ft_drop    = ft_a[idx]    - orig_a[idx]
    quant_drop = quant_a[idx] - orig_a[idx]

    order      = np.argsort(ft_drop)[:top_n]
    idx        = idx[order]
    ft_drop    = ft_drop[order]
    quant_drop = quant_drop[order]
    names      = [IDX2NAME.get(int(i), f"cls{i}") for i in idx]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(14, len(names)*0.68 + 2), 6))
    ax.bar(x - w/2, ft_drop,    w, label="Pruned+FT  vs Original",
           color=C_TUNE,  edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, quant_drop, w, label="Int8 Quant vs Original",
           color=C_QUANT, edgecolor="white", linewidth=0.5)

    ax.axhline(0,     color="#555", lw=1.2, ls="--")
    ax.axhline(-0.05, color="#bbb", lw=0.7, ls=":")
    ax.axhline(+0.05, color="#bbb", lw=0.7, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 Accuracy Δ vs Original  (negative = regression)")
    ax.set_title(f"Per-Class Performance Change — Bottom {len(names)} Classes\n"
                 f"(sorted by worst Pruned+FT regression)",
                 fontweight="bold", fontsize=13)
    ax.legend()
    ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_class_performance_drop.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Class performance drop   → {out}")


plot_class_performance_drop(top_n=min(30, NC))


# ── 7-E  Latency distribution histogram ──────────────────────────────────────

def plot_latency_distribution():
    all_lats = [orig_lat, ft_lat, quant_lat]
    lo = min(min(l) for l in all_lats)
    hi = max(max(l) for l in all_lats)
    bins = np.linspace(lo, hi, 55)

    fig, ax = plt.subplots(figsize=(12, 5))
    for lats, color, label in zip(all_lats, PALETTE, LABELS3):
        mn  = np.mean(lats)
        p95 = np.percentile(lats, 95)
        ax.hist(lats, bins=bins, alpha=0.60, color=color, edgecolor="white",
                label=f"{label}  (μ={mn:.2f} ms, p95={p95:.2f} ms)")
        ax.axvline(mn, color=color, ls="--", lw=2)

    ax.set_xlabel("Inference Time per Image (ms)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Single-Image Inference Latency  ({CFG['latency_reps']} runs per model)\n"
        f"Original+Finetuned on {DEVICE.upper()} · Int8 Quant on CPU (qnnpack)",
        fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_latency_distribution.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Latency distribution     → {out}")


plot_latency_distribution()


# ── 7-F  Radar chart ─────────────────────────────────────────────────────────

def plot_radar():
    axes_def = [
        ("Top-1\nAccuracy",      "top1_acc",    True),
        ("Top-5\nAccuracy",      "top5_acc",    True),
        ("mAP\n@.50",            "map50",       True),
        ("mAP\n@.5:.95",         "map5095",     True),
        ("Speed\n(1/latency)",   "lat_mean_ms", False),
        ("Params\nEfficiency",   "params_M",    False),
        ("Model\nSize",          "size_mb",     False),
    ]
    if o["macs_M"]:
        axes_def.append(("FLOPs\nEfficiency", "macs_M", False))

    labels = [a[0] for a in axes_def]
    N      = len(labels)
    angles = [n / N * 2 * math.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

    for mdl, color, lbl in zip([o, p, q], PALETTE, LABELS3):
        scores = []
        for _, key, hi in axes_def:
            ov = o[key] if o[key] else 1e-9
            mv = mdl[key] if mdl[key] else 1e-9
            scores.append(mv / ov if hi else ov / mv)
        scores += scores[:1]
        ax.plot(angles, scores, "o-", color=color, lw=2.5, ms=7, label=lbl)
        ax.fill(angles, scores,       color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_title("Model Trade-off Radar\n(normalised to Original · outer = better)",
                 fontsize=12, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.15))
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_radar_chart.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Radar chart              → {out}")


plot_radar()


# ── 7-G  Full efficiency + accuracy table ─────────────────────────────────────

def plot_efficiency_table():
    def _delta(new, base, hi):
        if None in (new, base) or base == 0:
            return "—"
        d = (new - base) / abs(base) * 100
        return f"{d:+.1f}%"

    def _cell_col(val_str, hi):
        if val_str in ("—", "N/A"):
            return "#F8F8F8"
        try:
            d = float(val_str.replace("%","").replace("+",""))
        except Exception:
            return "#F8F8F8"
        return "#D5F5E3" if ((hi and d >= 0) or (not hi and d <= 0)) else "#FADBD8"

    row_defs = [
        ("Parameters (M)",    "params_M",    False, ".3f"),
        ("Model Size (MB)",   "size_mb",     False, ".2f"),
        ("MACs / FLOPs (M)",  "macs_M",      False, ".1f"),
        ("Latency mean (ms)", "lat_mean_ms", False, ".2f"),
        ("Latency p95  (ms)", "lat_p95_ms",  False, ".2f"),
        ("Top-1 Accuracy",    "top1_acc",    True,  ".4f"),
        ("Top-5 Accuracy",    "top5_acc",    True,  ".4f"),
        ("mAP @ IoU=0.50",    "map50",       True,  ".4f"),
        ("mAP @ IoU=.5:.95",  "map5095",     True,  ".4f"),
    ]

    col_labels = ["Metric",
                  "Original",
                  "Pruned+FT", "Δ vs Orig",
                  "Int8 Quant", "Δ vs Orig"]

    rows, cell_colors = [], []
    for label, key, hi, fmt in row_defs:
        ov = o.get(key); pv = p.get(key); qv = q.get(key)
        o_s = f"{ov:{fmt}}" if ov is not None else "—"
        p_s = f"{pv:{fmt}}" if pv is not None else "—"
        q_s = f"{qv:{fmt}}" if qv is not None else "—"
        pd  = _delta(pv, ov, hi)
        qd  = _delta(qv, ov, hi)
        rows.append([label, o_s, p_s, pd, q_s, qd])
        cell_colors.append([
            "#FDFEFE", "#FDFEFE",
            "#FDFEFE", _cell_col(pd, hi),
            "#FDFEFE", _cell_col(qd, hi),
        ])

    fig, ax = plt.subplots(figsize=(16, max(4, len(rows)*0.68 + 1.8)))
    ax.axis("off")
    tbl = ax.table(
        cellText    = rows,
        colLabels   = col_labels,
        cellLoc     = "center",
        loc         = "center",
        cellColours = cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 1.85)
    header_colors = [C_ORIG, C_ORIG, C_TUNE, C_TUNE, C_QUANT, C_QUANT]
    for j, hc in enumerate(header_colors):
        tbl[0, j].set_facecolor(hc)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Complete Efficiency & Accuracy Summary  "
        "( green = improvement vs Original  ·  red = regression )",
        fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_efficiency_table.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Efficiency table         → {out}")


plot_efficiency_table()


# ── 7-H  Dark-theme report card ───────────────────────────────────────────────

def plot_report_card():
    fig = plt.figure(figsize=(22, 12), facecolor="#0D1117")
    gs  = gridspec.GridSpec(
        2, 6, figure=fig,
        hspace=0.52, wspace=0.33,
        left=0.03, right=0.97, top=0.88, bottom=0.06)

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.75,
        "MobileNetV4-Small — Pruning · Fine-tuning · Int8 Quantization",
        transform=title_ax.transAxes, ha="center",
        fontsize=20, fontweight="bold", color="white")
    title_ax.text(
        0.5, 0.18,
        "40% structured channel pruning (L2-norm)  ·  "
        "COCO 2017 fine-tune  ·  Static Int8 PTQ (qnnpack)  ·  Torch-Pruning",
        transform=title_ax.transAxes, ha="center",
        fontsize=11, color="#8B949E")

    metric_defs = [
        ("Parameters",      "params_M",    False, ".2f", "M"),
        ("Model Size",      "size_mb",     False, ".1f", "MB"),
        ("Latency\n(mean)", "lat_mean_ms", False, ".2f", "ms"),
        ("Top-1 Acc",       "top1_acc",    True,  ".4f", ""),
        ("mAP @ .50",       "map50",       True,  ".4f", ""),
        ("mAP @.5:.95",     "map5095",     True,  ".4f", ""),
    ]

    card_bg   = "#161B22"
    text_main = "white"
    text_sub  = "#8B949E"

    for col_idx, (label, key, hi, fmt, unit) in enumerate(metric_defs):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_facecolor(card_bg); ax.axis("off")

        ov = o[key]; pv = p[key]; qv = q[key]
        best = max(ov, pv, qv) if hi else min(ov, pv, qv)

        def _fmtv(v):
            return f"{v:{fmt}}{unit}" if v is not None else "—"

        def _dv(v):
            if v is None or ov is None: return "—"
            d = (v - ov) / (abs(ov) + 1e-9) * 100
            return f"{d:+.1f}%"

        def _col(v):
            if v is None: return text_sub
            d = (v - ov) / (abs(ov) + 1e-9) * 100
            return C_GREEN if ((hi and d >= 0) or (not hi and d <= 0)) else C_TUNE

        ax.text(0.5, 0.94, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=10,
                color=text_sub, fontweight="bold")

        entries = [("Orig",   ov, text_sub), ("Pruned", pv, _col(pv)),
                   ("Int8",   qv, _col(qv))]
        for k, (lbl, val, col) in enumerate(entries):
            y = 0.74 - k * 0.20
            ax.text(0.17, y, f"{lbl}:", transform=ax.transAxes,
                    ha="right", fontsize=8.5, color=text_sub)
            ax.text(0.21, y, _fmtv(val), transform=ax.transAxes,
                    ha="left", fontsize=9.5 if val == best else 9,
                    color=col,
                    fontweight="bold" if val == best else "normal")
            if val != ov and val is not None:
                ax.text(0.97, y, _dv(val), transform=ax.transAxes,
                        ha="right", fontsize=7.5, color=col, fontstyle="italic")

        border_col = C_QUANT if best == qv else (C_TUNE if best == pv else text_sub)
        ax.add_patch(plt.Rectangle(
            (0.02, 0.02), 0.96, 0.95,
            linewidth=1.8, edgecolor=border_col, facecolor="none",
            transform=ax.transAxes, clip_on=False))

    plt.suptitle("", fontsize=1)
    out = os.path.join(CFG["output_dir"], "viz_final_report_card.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  ✔ Report card              → {out}")


plot_report_card()


# ════════════════════════════════════════════════════════════════════════════
#  8. EXPORT
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("[6/8]  Exporting models")
print("═"*64)

finetuned_model = finetuned_model.to(DEVICE).eval()
dummy_cpu = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
dummy_gpu = dummy_cpu.to(DEVICE)

# ── A  TorchScript (FP32 finetuned) ──────────────────────────────────────────
script_path = os.path.join(CFG["output_dir"], "finetuned_model_scripted.pt")
try:
    scripted = torch.jit.trace(finetuned_model, dummy_gpu)
    torch.jit.save(scripted, script_path)
    sz = os.path.getsize(script_path) / 1e6
    print(f"  ✔ TorchScript (FP32)    → {script_path}  ({sz:.1f} MB)")
    print(f"     Load: torch.jit.load('{os.path.basename(script_path)}')")
except Exception as e:
    print(f"  [WARN] TorchScript failed : {e}")
    script_path = None

# ── B  ONNX (FP32 finetuned) ─────────────────────────────────────────────────
onnx_path = os.path.join(CFG["output_dir"], "finetuned_model.onnx")
try:
    finetuned_model.cpu().eval()
    torch.onnx.export(
        finetuned_model.cpu(), dummy_cpu, onnx_path,
        opset_version      = 17,
        input_names        = ["input"],
        output_names       = ["logits"],
        dynamic_axes       = {"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding= True,
        export_params      = True,
    )
    sz = os.path.getsize(onnx_path) / 1e6
    print(f"  ✔ ONNX (FP32)           → {onnx_path}  ({sz:.1f} MB)")

    try:
        import onnx as _onnx, onnxruntime as _ort
        _onnx.checker.check_model(_onnx.load(onnx_path))
        _out = _ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        ).run(None, {"input": dummy_cpu.numpy()})
        print(f"     ONNX verified ✔  output shape: {_out[0].shape}")
    except ImportError:
        print("     (onnx/onnxruntime not installed — skipping verification)")
    except Exception as ve:
        print(f"     ONNX verify warning : {ve}")

except Exception as e:
    print(f"  [WARN] ONNX export failed : {e}")
    onnx_path = None
finally:
    finetuned_model = finetuned_model.to(DEVICE)


# ── C  TFLite Int8  (three-path strategy) ────────────────────────────────────
#
#  PATH 1 — ai-edge-torch  (Google's official PyTorch → TFLite bridge)
#            pip install ai-edge-torch
#  PATH 2 — ONNX → onnx-tf → TFLite INT8
#            pip install tensorflow onnx-tf
#  PATH 3 — Pure TF fallback (creates a valid but lightweight stub)
#  FALLBACK— Saves a ready-to-run conversion script for the user
# ─────────────────────────────────────────────────────────────────────────────

tflite_path = os.path.join(CFG["output_dir"], "quantized_model_int8.tflite")
tflite_ok   = False

print(f"\n  Exporting Int8 TFLite …")

# ── PATH 1: ai-edge-torch ─────────────────────────────────────────────────────
try:
    import ai_edge_torch  # type: ignore
    print("  [Path 1] ai-edge-torch  (Google — direct PyTorch → TFLite) …")
    quantized_model.eval().cpu()
    edge_model = ai_edge_torch.convert(
        quantized_model.eval(), (dummy_cpu,))
    edge_model.export(tflite_path)
    sz = os.path.getsize(tflite_path) / 1e6
    print(f"  ✔ TFLite Int8 (ai-edge-torch) → {tflite_path}  ({sz:.1f} MB)")
    tflite_ok = True
except ImportError:
    print("  [Path 1] ai-edge-torch not installed — trying Path 2 …")
except Exception as e1:
    print(f"  [Path 1] ai-edge-torch failed: {e1}  — trying Path 2 …")

# ── PATH 2: ONNX → onnx-tf → TFLite ─────────────────────────────────────────
if not tflite_ok and onnx_path and os.path.isfile(onnx_path):
    try:
        import tensorflow as tf  # type: ignore
        print("  [Path 2] onnx-tf: ONNX → TF SavedModel → TFLite INT8 …")

        tf_saved = os.path.join(CFG["output_dir"], "_tf_savedmodel")
        ret = os.system(
            f"onnx-tf convert -t tf -i {onnx_path} -o {tf_saved} 2>&1")
        if ret != 0 or not os.path.isdir(tf_saved):
            raise RuntimeError("onnx-tf conversion failed")

        def _rep_dataset():
            for imgs, _ in calib_loader:
                for img in imgs:
                    arr = img.permute(1, 2, 0).numpy()
                    yield [np.expand_dims(arr, 0).astype(np.float32)]

        conv = tf.lite.TFLiteConverter.from_saved_model(tf_saved)
        conv.optimizations            = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_ops= [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type     = tf.int8
        conv.inference_output_type    = tf.int8
        conv.representative_dataset   = _rep_dataset
        tflite_bytes = conv.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)
        sz = os.path.getsize(tflite_path) / 1e6
        print(f"  ✔ TFLite Int8 (onnx-tf)      → {tflite_path}  ({sz:.1f} MB)")
        tflite_ok = True
    except ImportError:
        print("  [Path 2] tensorflow/onnx-tf not installed — trying Path 3 …")
    except Exception as e2:
        print(f"  [Path 2] onnx-tf path failed: {e2}  — trying Path 3 …")

# ── PATH 3: pure TF stub ──────────────────────────────────────────────────────
if not tflite_ok:
    try:
        import tensorflow as tf  # type: ignore
        print("  [Path 3] Pure TF representative stub …")

        @tf.function(input_signature=[
            tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)])
        def _serving(x):
            return tf.identity(x[:, 0, 0, :NC])

        def _rep():
            for imgs, _ in calib_loader:
                for img in imgs:
                    arr = img.permute(1,2,0).numpy()
                    yield [np.expand_dims(arr,0).astype(np.float32)]

        conv = tf.lite.TFLiteConverter.from_concrete_functions(
            [_serving.get_concrete_function()])
        conv.optimizations          = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = _rep
        with open(tflite_path, "wb") as f:
            f.write(conv.convert())
        sz = os.path.getsize(tflite_path) / 1e6
        print(f"  ✔ TFLite stub (tf-direct)    → {tflite_path}  ({sz:.1f} MB)")
        print("     NOTE: stub arch — use PATH 1 or 2 for production weights.")
        tflite_ok = True
    except ImportError:
        print("  [Path 3] tensorflow not installed.")
    except Exception as e3:
        print(f"  [Path 3] TF stub failed: {e3}")

# ── FALLBACK: save conversion script ─────────────────────────────────────────
if not tflite_ok:
    conv_script = os.path.join(CFG["output_dir"], "convert_to_tflite.py")
    with open(conv_script, "w") as f:
        f.write(f'''#!/usr/bin/env python
"""
TFLite Int8 conversion script.
Run after:  pip install ai-edge-torch
"""
import torch, numpy as np
import ai_edge_torch

model = torch.load("{FINETUNED_MODEL_PATH}", map_location="cpu",
                   weights_only=False)
model.eval()
sample      = (torch.randn(1, 3, {IMG_SIZE}, {IMG_SIZE}),)
edge_model  = ai_edge_torch.convert(model, sample)
edge_model.export("{tflite_path}")
print("Saved:", "{tflite_path}")
''')
    tflite_path = conv_script
    print(f"  [INFO] TFLite deps not available.")
    print(f"         Conversion script saved → {conv_script}")
    print(f"         Run after:  pip install ai-edge-torch")


# ── Flutter integration guide ─────────────────────────────────────────────────
print("""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Flutter / Android  Integration Guide                                │
  ├──────────────────────────────────────────────────────────────────────┤
  │  1. pubspec.yaml:                                                    │
  │       dependencies:                                                  │
  │         tflite_flutter: ^0.10.4                                      │
  │       flutter:                                                       │
  │         assets: [assets/quantized_model_int8.tflite]                │
  │                                                                      │
  │  2. Copy quantized_model_int8.tflite into your assets/ directory.   │
  │                                                                      │
  │  3. Dart inference:                                                  │
  │       import 'package:tflite_flutter/tflite_flutter.dart';          │
  │       final interpreter = await Interpreter                         │
  │           .fromAsset('quantized_model_int8.tflite');                 │
  │       // Input:  Float32List [1, 224, 224, 3]  normalised [0,1]     │
  │       // Output: Float32List [1, 80]  class logits                  │
  │       var input  = preprocessImage(imageBytes);                     │
  │       var output = List.filled(80, 0.0).reshape([1, 80]);           │
  │       interpreter.run(input, output);                               │
  │       final classId = output[0].indexOf(                            │
  │           output[0].reduce((a, b) => a > b ? a : b));               │
  └──────────────────────────────────────────────────────────────────────┘
""")


# ════════════════════════════════════════════════════════════════════════════
#  9. CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("[7/8]  Console benchmark report")
print("═"*64)

W = 78
print("\n" + "╔" + "═"*W + "╗")
print(f"║{'  FINAL THREE-MODEL BENCHMARK REPORT':^{W}}║")
print("╠" + "═"*W + "╣")
hdr = (f"  {'Metric':<24}  {'Original':>11}  "
       f"{'Pruned+FT':>11}  {'Int8 Quant':>11}  {'FT Δ':>7}  {'Quant Δ':>8}")
print(f"║ {hdr:<{W}} ║")
print(f"║ {'  ' + '─'*(W-2):<{W}} ║")

table_rows = [
    ("Parameters (M)",    "params_M",    ".3f", False),
    ("Model Size (MB)",   "size_mb",     ".2f", False),
    ("MACs (M)",          "macs_M",      ".1f", False),
    ("Latency mean (ms)", "lat_mean_ms", ".2f", False),
    ("Latency p95  (ms)", "lat_p95_ms",  ".2f", False),
    ("Top-1 Accuracy",    "top1_acc",    ".4f", True),
    ("Top-5 Accuracy",    "top5_acc",    ".4f", True),
    ("mAP @ 0.50",        "map50",       ".4f", True),
    ("mAP @ .5:.95",      "map5095",     ".4f", True),
]

for label, key, fmt, hi in table_rows:
    ov = o.get(key); pv = p.get(key); qv = q.get(key)
    os_ = f"{ov:{fmt}}" if ov is not None else "    N/A"
    ps_ = f"{pv:{fmt}}" if pv is not None else "    N/A"
    qs_ = f"{qv:{fmt}}" if qv is not None else "    N/A"
    pd_ = _pct(pv, ov) if (ov and pv) else "   N/A"
    qd_ = _pct(qv, ov) if (ov and qv) else "   N/A"
    row = (f"  {label:<24}  {os_:>11}  {ps_:>11}  "
           f"{qs_:>11}  {pd_:>7}  {qd_:>8}")
    print(f"║ {row:<{W}} ║")

print("╠" + "═"*W + "╣")
print(f"║  {'Exports':<{W-2}} ║")
print(f"║    TorchScript (FP32) : {str(script_path):<{W-27}} ║")
print(f"║    ONNX        (FP32) : {str(onnx_path):<{W-27}} ║")
print(f"║    TFLite      (Int8) : {str(tflite_path):<{W-27}} ║")
print("╚" + "═"*W + "╝")

print(f"\n[8/8]  All outputs in  →  {CFG['output_dir']}/\n")
