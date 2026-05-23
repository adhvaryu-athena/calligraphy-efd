"""Stage 5 — Visualization.

Produces five publication-ready PNG figures at 300 DPI:

    Fig 1: PCA scatter plot of EFD feature vectors (PC1 vs PC2), coloured by class.
    Fig 2: Harmonic reconstruction grid (4 classes x 5 harmonic orders).
    Fig 3: Macro-F1 vs harmonic order, with bootstrap CI error bars.
    Fig 4: Random forest feature importances aggregated by harmonic number,
           with MDI and permutation importance plotted side-by-side when available.
    Diag : Confusion matrix heatmap (diagnostic, not in PRD figure list).

All figures use a consistent colour-blind-safe palette.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyefd import (
    reconstruct_contour, calculate_dc_coefficients, elliptic_fourier_descriptors,
)

CLASS_ORDER = ("serif", "sans-serif", "calligraphic", "display")
CLASS_COLORS = {"serif": "#4477AA", "sans-serif": "#EE6677",
                "calligraphic": "#228833", "display": "#CCBB44"}
CLASS_MARKERS = {"serif": "o", "sans-serif": "s", "calligraphic": "D", "display": "^"}


def _save(fig, path, dpi=300):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_pca_scatter(feats, order=20, glyphs_to_plot=("a", "g"),
                    output_path="outputs/figures/fig1_pca_scatter.png"):
    bundle = feats[order]
    X, y, glyphs, fonts = bundle["X"], bundle["y"], bundle["glyphs"], bundle["fonts"]
    mask = np.isin(glyphs, list(glyphs_to_plot))
    Xs, ys, gs, fs = X[mask], y[mask], glyphs[mask], fonts[mask]
    Xn = StandardScaler().fit_transform(Xs)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xn)
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in CLASS_ORDER:
        for glyph in glyphs_to_plot:
            sel = (ys == cls) & (gs == glyph)
            if not sel.any():
                continue
            ax.scatter(Z[sel, 0], Z[sel, 1], c=CLASS_COLORS[cls],
                       marker=CLASS_MARKERS[cls], s=90, alpha=0.8,
                       edgecolors="white", linewidths=0.8,
                       label=f"{cls} ({glyph})")
            # Only annotate font names on small corpora (<50 fonts); otherwise too noisy.
            if len(np.unique(fs)) < 50:
                for j in np.where(sel)[0]:
                    ax.annotate(fs[j], (Z[j, 0], Z[j, 1]), fontsize=7,
                                alpha=0.6, xytext=(4, 4),
                                textcoords="offset points")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(f"EFD feature vectors in PC1 x PC2 (N={order})", fontsize=12)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    _save(fig, output_path)
    return output_path


def fig_harmonic_reconstruction(outlines, label_map, glyph="a",
                                orders=(1, 3, 5, 10, 20),
                                output_path="outputs/figures/fig2_harmonic_reconstruction.png"):
    class_to_font = {}
    for font, lbl in label_map.items():
        if (lbl in CLASS_ORDER and lbl not in class_to_font
                and font in outlines and glyph in outlines[font]):
            class_to_font[lbl] = font
    classes = [c for c in CLASS_ORDER if c in class_to_font]
    n_rows, n_cols = len(classes), len(orders)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]
    for r, cls in enumerate(classes):
        font = class_to_font[cls]
        contour = outlines[font][glyph]
        locus = calculate_dc_coefficients(contour)
        for c, order in enumerate(orders):
            ax = axes[r, c]
            x_orig = np.append(contour[:, 0], contour[0, 0])
            y_orig = np.append(contour[:, 1], contour[0, 1])
            ax.plot(x_orig, y_orig, color="gray", lw=1.0, alpha=0.55)
            coeffs = elliptic_fourier_descriptors(contour, order=order, normalize=False)
            recon = reconstruct_contour(coeffs, locus=locus, num_points=400)
            ax.plot(np.append(recon[:, 0], recon[0, 0]),
                    np.append(recon[:, 1], recon[0, 1]),
                    color=CLASS_COLORS[cls], lw=1.8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"N = {order}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{cls}\n({font})", fontsize=9, rotation=0,
                              labelpad=48, ha="right", va="center")
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
    fig.suptitle(f"Harmonic reconstruction of glyph '{glyph}' across style classes",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, output_path)
    return output_path


def fig_accuracy_vs_harmonics(results,
                              output_path="outputs/figures/fig3_accuracy_vs_harmonics.png"):
    sens = results["sensitivity"]
    orders = sorted(int(k) for k in sens.keys())
    f1s = np.array([sens[str(o)]["macro_f1"] for o in orders])
    lo = np.array([sens[str(o)]["ci_low_95"] for o in orders])
    hi = np.array([sens[str(o)]["ci_high_95"] for o in orders])
    err = np.vstack([f1s - lo, hi - f1s])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(orders, f1s, yerr=err, fmt="o-", color="#332288",
                ecolor="#999999", capsize=4, lw=1.8, ms=8,
                label="Random forest (group CV)")
    n_classes = len(results["classes"])
    chance = 1.0 / n_classes
    ax.axhline(chance, color="red", ls="--", lw=1.0, alpha=0.6,
               label=f"Chance ({n_classes} classes)")
    ax.set_xlabel("Harmonic order N", fontsize=11)
    ax.set_ylabel("Macro-F1 (95% bootstrap CI)", fontsize=11)
    ax.set_title("Classification accuracy vs harmonic order", fontsize=12)
    ax.set_xticks(orders); ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    _save(fig, output_path)
    return output_path


def _aggregate_importance_by_harmonic(imp, order):
    """Sum per-coefficient importance into per-harmonic totals.

    Layout after normalize=True drops a_1, b_1, c_1: harmonic 1 has 1 feature (d_1),
    harmonics 2..N have 4 features each.
    """
    imp = np.asarray(imp)
    assert len(imp) == 4 * order - 3, f"importance length {len(imp)} != {4*order - 3}"
    h_imp = np.zeros(order)
    pos = 0
    h_imp[0] = imp[pos]; pos += 1
    for n in range(2, order + 1):
        h_imp[n - 1] = imp[pos:pos + 4].sum(); pos += 4
    return h_imp


def fig_feature_importance(results, order=20,
                            output_path="outputs/figures/fig4_feature_importance.png"):
    """Side-by-side MDI and permutation importance per harmonic.

    Falls back to MDI-only when permutation importance is missing from results.
    """
    rf = results["main"]["random_forest"]["group"]
    mdi = rf.get("feature_importances")
    perm = rf.get("feature_importances_perm")
    if mdi is None:
        raise RuntimeError("Random forest MDI importances missing from results.")

    mdi_h = _aggregate_importance_by_harmonic(mdi, order)
    harmonics = np.arange(1, order + 1)
    has_perm = perm is not None

    if has_perm:
        perm_h = _aggregate_importance_by_harmonic(perm, order)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
        ax1.bar(harmonics, mdi_h, color="#117733", edgecolor="black", lw=0.4)
        ax1.set_title("MDI (mean decrease in impurity)", fontsize=11)
        ax1.set_xlabel("Harmonic number n", fontsize=11)
        ax1.set_ylabel("RF importance (summed over a_n,b_n,c_n,d_n)", fontsize=11)
        ax1.set_xticks(harmonics)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax2.bar(harmonics, perm_h, color="#882255", edgecolor="black", lw=0.4)
        ax2.set_title("Permutation importance (Strobl 2007)", fontsize=11)
        ax2.set_xlabel("Harmonic number n", fontsize=11)
        ax2.set_ylabel("Permutation importance", fontsize=11)
        ax2.set_xticks(harmonics)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
        for ax, h_imp in ((ax1, mdi_h), (ax2, perm_h)):
            for i in np.argsort(h_imp)[-3:][::-1]:
                ax.text(harmonics[i], h_imp[i] + max(h_imp) * 0.01,
                        f"n={harmonics[i]}", ha="center", fontsize=9, fontweight="bold")
        fig.suptitle(f"Importance of each harmonic (N={order}, RF under group CV)",
                     fontsize=12, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(harmonics, mdi_h, color="#117733", edgecolor="black", lw=0.4)
        ax.set_xlabel("Harmonic number n", fontsize=11)
        ax.set_ylabel("RF importance (MDI, summed over a_n,b_n,c_n,d_n)", fontsize=11)
        ax.set_title(f"Importance of each harmonic (N={order}, RF under group CV)",
                     fontsize=12)
        ax.set_xticks(harmonics)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        for i in np.argsort(mdi_h)[-3:][::-1]:
            ax.text(harmonics[i], mdi_h[i] + max(mdi_h) * 0.01,
                    f"n={harmonics[i]}", ha="center", fontsize=9, fontweight="bold")
    _save(fig, output_path)
    return output_path


def fig_confusion_matrix(results, classifier="random_forest", cv_type="group",
                          output_path="outputs/figures/diag_confusion_matrix.png"):
    block = results["main"][classifier][cv_type]
    classes = block["classes"]
    cm = np.array(block["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(f"Confusion matrix ({classifier}, {cv_type} CV)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    _save(fig, output_path)
    return output_path


def run(feats, outlines, label_map, results, output_dir="outputs/figures",
        primary_order=20, verbose=True):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if verbose: print("Drawing Fig 1: PCA scatter")
    paths["fig1"] = str(fig_pca_scatter(feats, order=primary_order,
                                         output_path=output_dir / "fig1_pca_scatter.png"))
    if verbose: print("Drawing Fig 2: harmonic reconstruction grid")
    paths["fig2"] = str(fig_harmonic_reconstruction(outlines, label_map, glyph="a",
                                                     output_path=output_dir / "fig2_harmonic_reconstruction.png"))
    if verbose: print("Drawing Fig 3: accuracy vs harmonics")
    paths["fig3"] = str(fig_accuracy_vs_harmonics(results,
                                                   output_path=output_dir / "fig3_accuracy_vs_harmonics.png"))
    if verbose: print("Drawing Fig 4: feature importance (MDI + permutation)")
    paths["fig4"] = str(fig_feature_importance(results, order=primary_order,
                                                output_path=output_dir / "fig4_feature_importance.png"))
    if verbose: print("Drawing diagnostic: confusion matrix")
    paths["diag"] = str(fig_confusion_matrix(results,
                                              output_path=output_dir / "diag_confusion_matrix.png"))
    return paths
