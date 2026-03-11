"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PART 1 — MobileNetV4-Small  ·  Structured Pruning                  ║
║         Torch-Pruning  ·  L2-norm  ·  40% channel reduction                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Install (Kaggle notebook cell):
    !pip install torch-pruning timm --quiet

Outputs written to  ./pruning_outputs/
    ├── pruned_model.pth                  ← full model object (arch preserved)
    ├── original_model.pth
    ├── prune_stats.json
    ├── viz_dependency_graph.txt          ← text dep-graph (printable)
    ├── viz_channel_distribution.png
    ├── viz_pruning_ratio_per_layer.png
    └── viz_weight_magnitude_heatmap.png
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import os, json, copy, warnings, textwrap
warnings.filterwarnings("ignore")

# ─── third-party ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import timm
import torch_pruning as tp

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = dict(
    model_name      = "mobilenetv4_conv_small",
    num_classes     = 80,           # COCO
    pretrained      = True,
    prune_ratio     = 0.40,
    iterative_steps = 5,
    img_size        = 224,
    output_dir      = "./pruning_outputs",
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    seed            = 42,
)

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
DEVICE = CFG["device"]
print(f"[INFO] Device : {DEVICE}")

# ── Colour palette (publication-grade) ───────────────────────────────────────
C_ORIG   = "#2E4057"   # deep navy
C_PRUNED = "#E84855"   # vivid red
C_ACCENT = "#F4A261"   # warm amber
BG       = "#FAFAFA"
GRID     = "#E5E5E5"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    BG,
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.framealpha": 0.9,
    "legend.fontsize":   10,
})


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def model_size_mb(m):
    tmp = os.path.join(CFG["output_dir"], "_sz.pth")
    torch.save(m.state_dict(), tmp)
    s = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return s


def _safe_layer_label(name: str) -> str:
    """Return a short human-readable label even for top-level modules."""
    parts = [p for p in name.split(".") if p]
    if len(parts) == 0:
        return "root"
    if len(parts) == 1:
        return parts[0]
    # prefer last two non-numeric parts
    clean = [p for p in parts if not p.isdigit()]
    if len(clean) >= 2:
        return f"{clean[-2]}.{clean[-1]}"
    return parts[-1]


def collect_conv_info(model):
    """Return list of (label, name, module) for every Conv2d in model."""
    entries = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            label = _safe_layer_label(name)
            entries.append((label, name, mod))
    return entries


# ════════════════════════════════════════════════════════════════════════════
#  1. BUILD MODEL
# ════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Building model …")
model = timm.create_model(
    CFG["model_name"],
    pretrained = CFG["pretrained"],
    num_classes= CFG["num_classes"],
).to(DEVICE)

original_snap = copy.deepcopy(model).cpu()

orig_params = count_params(model)
orig_size   = model_size_mb(model)
print(f"      Parameters : {orig_params/1e6:.3f} M")
print(f"      Disk size  : {orig_size:.2f} MB")


# ════════════════════════════════════════════════════════════════════════════
#  2. DEPENDENCY GRAPH  (printed + saved as text)
# ════════════════════════════════════════════════════════════════════════════

print("\n[2/5] Building dependency graph …")

example_input = torch.randn(1, 3, CFG["img_size"], CFG["img_size"]).to(DEVICE)

# Identify layers to ignore (head + first conv stem)
ignored = []
for m in model.modules():
    if isinstance(m, nn.Linear):
        ignored.append(m)
first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
ignored.append(first_conv)

DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=example_input)

dep_lines = ["MobileNetV4-Small — Dependency Graph",
             "=" * 60]
for i, group in enumerate(DG.get_all_groups(ignored_layers=ignored,
                                             root_module_types=[nn.Conv2d, nn.Linear])):
    dep_lines.append(f"\nGroup {i:03d}")
    dep_lines.append("-" * 40)
    for dep, idxs in group:
        src  = dep.source.name
        tgt  = dep.target.name
        tp_  = dep.__class__.__name__
        dep_lines.append(f"  {src:40s}  ──{tp_:─>20s}──▶  {tgt}")

dep_text = "\n".join(dep_lines)
dep_path = os.path.join(CFG["output_dir"], "viz_dependency_graph.txt")
with open(dep_path, "w") as f:
    f.write(dep_text)

# Also print to stdout (truncated for readability)
print(dep_text[:4000])
if len(dep_text) > 4000:
    print(f"  … (truncated — full graph saved to {dep_path})")
print(f"\n  ✔ Full dependency graph → {dep_path}")


# ════════════════════════════════════════════════════════════════════════════
#  3. PRUNE
# ════════════════════════════════════════════════════════════════════════════

print("\n[3/5] Pruning …")

importance = tp.importance.MagnitudeImportance(p=2)

pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs  = example_input,
    importance      = importance,
    pruning_ratio   = CFG["prune_ratio"],
    iterative_steps = CFG["iterative_steps"],
    ignored_layers  = ignored,
    round_to        = 8,
)

base_macs, base_params_tp = tp.utils.count_ops_and_params(model, example_input)
step_log = []

for step in range(CFG["iterative_steps"]):
    pruner.step()
    macs, params = tp.utils.count_ops_and_params(model, example_input)
    step_log.append({"step": step+1, "params_M": params/1e6, "macs_M": macs/1e6})
    print(f"  Step {step+1}/{CFG['iterative_steps']} | "
          f"Params {params/1e6:.3f}M | MACs {macs/1e6:.1f}M")

final_macs, final_params_tp = tp.utils.count_ops_and_params(model, example_input)
pruned_params = count_params(model)
pruned_size   = model_size_mb(model)

prune_stats = {
    "original_params":    orig_params,
    "pruned_params":      pruned_params,
    "original_params_M":  orig_params / 1e6,
    "pruned_params_M":    pruned_params / 1e6,
    "param_reduction_pct": 100*(orig_params - pruned_params)/orig_params,
    "original_macs_M":    base_macs / 1e6,
    "pruned_macs_M":      final_macs / 1e6,
    "mac_reduction_pct":  100*(base_macs - final_macs)/base_macs,
    "original_size_mb":   orig_size,
    "pruned_size_mb":     pruned_size,
    "step_log":           step_log,
}

stats_path = os.path.join(CFG["output_dir"], "prune_stats.json")
with open(stats_path, "w") as f:
    json.dump(prune_stats, f, indent=2)

print(f"\n  ✔ Pruning summary:")
print(f"     Params : {orig_params/1e6:.3f}M  →  {pruned_params/1e6:.3f}M  "
      f"({prune_stats['param_reduction_pct']:.1f}% removed)")
print(f"     MACs   : {base_macs/1e6:.1f}M  →  {final_macs/1e6:.1f}M  "
      f"({prune_stats['mac_reduction_pct']:.1f}% removed)")
print(f"     Size   : {orig_size:.2f} MB  →  {pruned_size:.2f} MB")
print(f"     Stats  → {stats_path}")


# ════════════════════════════════════════════════════════════════════════════
#  4. SAVE MODELS
# ════════════════════════════════════════════════════════════════════════════

print("\n[4/5] Saving models …")

pruned_path   = os.path.join(CFG["output_dir"], "pruned_model.pth")
original_path = os.path.join(CFG["output_dir"], "original_model.pth")

# Save FULL model objects (not just state_dict) so architecture is portable
torch.save(model.cpu(),           pruned_path)
torch.save(original_snap.cpu(),   original_path)

print(f"  ✔ Pruned model   → {pruned_path}")
print(f"  ✔ Original model → {original_path}")

# Move model back to device for any downstream use in this script
model = model.to(DEVICE)
original_snap = original_snap.to(DEVICE)


# ════════════════════════════════════════════════════════════════════════════
#  5. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n[5/5] Generating visualisations …")


# ── Helper: build matched orig / pruned conv lists by full name ──────────────

def match_convs(orig_model, pruned_model):
    """
    Returns three parallel lists: (labels, orig_ch, pruned_ch)
    Only includes layers present in BOTH models.
    """
    orig_map   = {name: mod for name, mod in orig_model.named_modules()
                  if isinstance(mod, nn.Conv2d)}
    labels, o_ch, p_ch = [], [], []
    for name, mod in pruned_model.named_modules():
        if isinstance(mod, nn.Conv2d) and name in orig_map:
            labels.append(_safe_layer_label(name))
            o_ch.append(orig_map[name].out_channels)
            p_ch.append(mod.out_channels)
    return labels, o_ch, p_ch


# ── 5-A  Channel distribution bar chart ──────────────────────────────────────

def plot_channel_distribution(orig, pruned):
    labels, o_ch, p_ch = match_convs(orig, pruned)
    if not labels:
        print("  [WARN] No matched conv layers — skipping channel distribution plot.")
        return

    x  = np.arange(len(labels))
    w  = 0.38
    fig, ax = plt.subplots(figsize=(max(14, len(labels)*0.55 + 2), 5.5))

    b1 = ax.bar(x - w/2, o_ch, w, label="Original",
                color=C_ORIG,   edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w/2, p_ch, w, label="Pruned (40%)",
                color=C_PRUNED, edgecolor="white", linewidth=0.6)

    # Annotate reduction % on top of pruned bars (every 3rd to avoid clutter)
    for i, (o, p) in enumerate(zip(o_ch, p_ch)):
        if i % 3 == 0 and o > 0:
            pct = (o - p) / o * 100
            ax.text(x[i] + w/2, p + max(o_ch)*0.01,
                    f"−{pct:.0f}%", ha="center", va="bottom",
                    fontsize=7, color=C_PRUNED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Output Channels", fontsize=11)
    ax.set_title("Channel Count per Conv Layer — Original vs Pruned",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(handles=[
        mpatches.Patch(color=C_ORIG,   label=f"Original  ({orig_params/1e6:.2f}M params)"),
        mpatches.Patch(color=C_PRUNED, label=f"Pruned    ({pruned_params/1e6:.2f}M params)"),
    ])
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_channel_distribution.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Channel distribution  → {out}")


# ── 5-B  Per-layer pruning ratio (horizontal bar) ────────────────────────────

def plot_pruning_ratio(orig, pruned):
    labels, o_ch, p_ch = match_convs(orig, pruned)
    if not labels:
        return

    ratios = [100*(o - p)/o if o > 0 else 0.0 for o, p in zip(o_ch, p_ch)]

    fig, ax = plt.subplots(figsize=(8, max(5, len(labels)*0.32 + 1)))
    colors  = [C_PRUNED if r > 0 else "#AAAAAA" for r in ratios]
    y_pos   = np.arange(len(labels))

    bars = ax.barh(y_pos, ratios, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.72)
    ax.axvline(CFG["prune_ratio"]*100, color=C_ACCENT, ls="--",
               lw=2, label=f"Target {CFG['prune_ratio']*100:.0f}%", zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Channels Pruned (%)")
    ax.set_title("Per-Layer Channel Reduction (%)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 110)
    ax.legend()
    ax.invert_yaxis()

    for bar, r in zip(bars, ratios):
        if r > 0:
            ax.text(r + 1, bar.get_y() + bar.get_height()/2,
                    f"{r:.1f}%", va="center", fontsize=7.5, color="#333")

    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_pruning_ratio_per_layer.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Per-layer pruning ratio → {out}")


# ── 5-C  Weight magnitude heatmap ────────────────────────────────────────────

def plot_weight_magnitude_heatmap(orig, pruned, n_layers=20):
    def get_norms(m):
        norms = []
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                w = mod.weight.data.cpu().float()
                norms.append(w.view(w.size(0), -1).norm(p=2, dim=1).numpy())
        return norms[:n_layers]

    o_norms = get_norms(orig)
    p_norms = get_norms(pruned)
    if not o_norms or not p_norms:
        return

    max_ch = max(max(len(n) for n in o_norms), max(len(n) for n in p_norms))

    def pad_mat(norms_list):
        mat = np.full((len(norms_list), max_ch), np.nan)
        for i, n in enumerate(norms_list):
            mat[i, :len(n)] = n
        return mat

    cmap = LinearSegmentedColormap.from_list(
        "mag", ["#0d1b2a", "#1b6ca8", "#f4a261", "#e84855"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mat, title in zip(
            axes,
            [pad_mat(o_norms), pad_mat(p_norms)],
            ["Original — Filter L2 Norms", "Pruned — Filter L2 Norms"]):
        vmax = np.nanpercentile(pad_mat(o_norms), 99)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Layer Index (0 = first conv)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                     label="L2 Norm")

    fig.suptitle(f"Filter Weight Magnitude Heatmap (first {n_layers} Conv layers)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_weight_magnitude_heatmap.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Weight magnitude heatmap → {out}")


# ── 5-D  Iterative pruning steps (params & MACs) ─────────────────────────────

def plot_iterative_steps(step_log, base_params_M, base_macs_M):
    steps  = [0]  + [s["step"]     for s in step_log]
    params = [base_params_M] + [s["params_M"] for s in step_log]
    macs   = [base_macs_M]   + [s["macs_M"]   for s in step_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(steps, params, "o-", color=C_ORIG,   lw=2.5, ms=7, label="Parameters")
    ax1.fill_between(steps, params, alpha=0.12, color=C_ORIG)
    ax1.set_xlabel("Pruning Step"); ax1.set_ylabel("Parameters (M)")
    ax1.set_title("Parameters vs Pruning Step", fontweight="bold")
    ax1.set_xticks(steps)

    ax2.plot(steps, macs,   "o-", color=C_PRUNED, lw=2.5, ms=7, label="MACs")
    ax2.fill_between(steps, macs, alpha=0.12, color=C_PRUNED)
    ax2.set_xlabel("Pruning Step"); ax2.set_ylabel("MACs (M)")
    ax2.set_title("Compute (MACs) vs Pruning Step", fontweight="bold")
    ax2.set_xticks(steps)

    plt.suptitle("Iterative Structured Pruning Progress",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "viz_iterative_pruning_steps.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Iterative steps chart    → {out}")


# ── Run all plots ─────────────────────────────────────────────────────────────

import matplotlib.patches as mpatches  # ensure imported after rcParams set

plot_channel_distribution(original_snap, model)
plot_pruning_ratio(original_snap, model)
plot_weight_magnitude_heatmap(original_snap, model)
plot_iterative_steps(step_log, base_params_tp/1e6, base_macs/1e6)

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "╔" + "═"*60 + "╗")
print("║   PART 1 COMPLETE — Pruning & Visualisation              ║")
print("╠" + "═"*60 + "╣")
rows = [
    f"  Original params  : {orig_params/1e6:.3f} M",
    f"  Pruned   params  : {pruned_params/1e6:.3f} M  "
    f"({prune_stats['param_reduction_pct']:.1f}% ↓)",
    f"  Original MACs    : {base_macs/1e6:.1f} M",
    f"  Pruned   MACs    : {final_macs/1e6:.1f} M  "
    f"({prune_stats['mac_reduction_pct']:.1f}% ↓)",
    f"  Model saved to   : {pruned_path}",
    f"  Stats  saved to  : {stats_path}",
]
for r in rows:
    print(f"║  {r:<58}║")
print("╚" + "═"*60 + "╝\n")
print("  ➜  Run part2_finetune.py next, passing the pruned_model.pth path.")
