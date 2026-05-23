"""Phase 6 — Streamlit font similarity explorer.

Run with:
    streamlit run src/web_tool.py

Features:
1. UPLOAD A FONT: .ttf/.otf -> EFD features -> 10 nearest neighbors
2. BROWSE THE CORPUS: UMAP scatter (Google labels or discovered clusters)
3. HARMONIC RECONSTRUCTION: glyph 'a' at N=5,10,15,20
"""

import logging
import os
import pickle
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — fail gracefully
# ---------------------------------------------------------------------------

try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    print("streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False
    st.error("numpy not installed.")

try:
    import pandas as pd
    _HAS_PD = True
except ImportError:
    _HAS_PD = False
    st.error("pandas not installed.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    st.error("matplotlib not installed.")

try:
    import pyefd
    _HAS_PYEFD = True
except ImportError:
    _HAS_PYEFD = False

try:
    from fontTools.ttLib import TTFont
    _HAS_FONTTOOLS = True
except ImportError:
    _HAS_FONTTOOLS = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
CANONICAL_DIR = OUTPUTS / "embeddings" / "canonical"
SIMILARITY_DIR = OUTPUTS / "similarity"
CLUSTERING_RESULTS = OUTPUTS / "clustering_results.json"
FONTS_CSV = OUTPUTS / "fonts.csv"
OUTLINES_PKL = OUTPUTS / "outlines.pkl"


# ---------------------------------------------------------------------------
# Data loading (all cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_fonts_df():
    """Load fonts metadata CSV."""
    if not FONTS_CSV.exists():
        return None
    try:
        return pd.read_csv(FONTS_CSV)
    except Exception as exc:
        _logger.warning("Failed to load fonts.csv: %s", exc)
        return None


@st.cache_data
def load_umap_embedding():
    """Load canonical UMAP embedding (Config A, mean).

    Looks for *_A_mean_umap_*.npz in the canonical embeddings directory.
    Returns (embedding, font_names, labels) or (None, None, None).
    """
    if not CANONICAL_DIR.exists():
        return None, None, None

    candidates = sorted(CANONICAL_DIR.glob("A_mean_umap_*.npz"))
    if not candidates:
        # Fall back to any umap file
        candidates = sorted(CANONICAL_DIR.glob("*_umap_*.npz"))
    if not candidates:
        return None, None, None

    try:
        data = np.load(candidates[0], allow_pickle=True)
        embedding = data["embedding"]
        font_names = data["font_names"].tolist() if "font_names" in data else []
        labels = data["labels"].tolist() if "labels" in data else []
        return embedding, font_names, labels
    except Exception as exc:
        _logger.warning("Failed to load UMAP embedding: %s", exc)
        return None, None, None


@st.cache_data
def load_clustering_results():
    """Load clustering_results.json. Returns dict or None."""
    if not CLUSTERING_RESULTS.exists():
        return None
    try:
        import json
        with open(CLUSTERING_RESULTS) as f:
            return json.load(f)
    except Exception as exc:
        _logger.warning("Failed to load clustering_results.json: %s", exc)
        return None


@st.cache_data
def load_similarity_matrix():
    """Load cosine pairwise similarity for Config A / mean.

    Returns (D, font_names) or (None, None).
    """
    if not SIMILARITY_DIR.exists():
        return None, None

    # Prefer cosine A_mean
    candidates = sorted(SIMILARITY_DIR.glob("pairwise_A_mean_cosine.npz"))
    if not candidates:
        candidates = sorted(SIMILARITY_DIR.glob("pairwise_A_mean_*.npz"))
    if not candidates:
        candidates = sorted(SIMILARITY_DIR.glob("pairwise_*.npz"))
    if not candidates:
        return None, None

    try:
        data = np.load(candidates[0], allow_pickle=True)
        D = data["D"]
        font_names = data["font_names"].tolist() if "font_names" in data else []
        return D, font_names
    except Exception as exc:
        _logger.warning("Failed to load similarity matrix: %s", exc)
        return None, None


@st.cache_data
def load_font_features_canonical():
    """Load Config A / mean font-level feature matrix.

    Returns (X, font_names) or (None, None).
    """
    npz_path = OUTPUTS / "font_features_A_mean.npz"
    if not npz_path.exists():
        return None, None
    try:
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(float)
        font_names = data["font_names"].tolist() if "font_names" in data else []
        return X, font_names
    except Exception as exc:
        _logger.warning("Failed to load canonical features: %s", exc)
        return None, None


@st.cache_data
def load_outlines():
    """Load outlines.pkl. Returns dict or None."""
    if not OUTLINES_PKL.exists():
        return None
    try:
        with open(OUTLINES_PKL, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        _logger.warning("Failed to load outlines.pkl: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Feature extraction for uploaded font
# ---------------------------------------------------------------------------

def extract_features_from_upload(font_bytes: bytes):
    """Extract Config A / mean EFD features from uploaded font bytes.

    Returns (feature_vector, error_message).
    """
    if not _HAS_FONTTOOLS:
        return None, "fontTools not installed."
    if not _HAS_PYEFD:
        return None, "pyefd not installed."

    try:
        from src.counters import extract_glyph_features
        from src.aggregate import aggregate_to_font_level
    except ImportError as exc:
        return None, f"Cannot import project modules: {exc}"

    import tempfile

    GLYPHS = "abcdefghijklmnopqrstuvwxyz"
    N_HARMONICS = 20

    with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
        tmp.write(font_bytes)
        tmp_path = tmp.name

    try:
        font = TTFont(tmp_path)
        cmap = font.getBestCmap()
        glyph_set = font.getGlyphSet()

        glyph_vecs = []
        for char in GLYPHS:
            glyph_name = cmap.get(ord(char)) if cmap else None
            if glyph_name is None:
                continue
            try:
                glyph = glyph_set[glyph_name]
            except KeyError:
                continue
            result = extract_glyph_features(glyph, font, config="A", n_harmonics=N_HARMONICS)
            if result is not None:
                glyph_vecs.append(result["features"])

        if not glyph_vecs:
            return None, "No usable glyphs found in uploaded font."

        font_vec = aggregate_to_font_level(glyph_vecs, mode="mean")
        return font_vec, None

    except Exception as exc:
        return None, f"Feature extraction failed: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def find_nearest_neighbors(query_vec: "np.ndarray", X: "np.ndarray",
                            font_names: list, k: int = 10):
    """Find k nearest neighbors in feature matrix X using cosine distance.

    Returns list of (font_name, cosine_distance) tuples.
    """
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_distances

    if X.shape[0] == 0:
        return []

    # Pad/trim query to match X width
    qv = query_vec.copy()
    if qv.shape[0] < X.shape[1]:
        padded = np.zeros(X.shape[1])
        padded[: qv.shape[0]] = qv
        qv = padded
    elif qv.shape[0] > X.shape[1]:
        qv = qv[: X.shape[1]]

    qv = qv.reshape(1, -1)
    dists = cosine_distances(qv, X)[0]
    top_k_idx = np.argsort(dists)[:k]
    return [(font_names[i], float(dists[i])) for i in top_k_idx]


# ---------------------------------------------------------------------------
# Get best-k cluster labels from clustering_results.json
# ---------------------------------------------------------------------------

def get_best_cluster_labels(clustering_results, font_names_order):
    """Extract best-k kmeans labels for Config A / mean variant.

    Returns (labels_list, best_k) aligned to font_names_order, or (None, None).
    """
    if clustering_results is None:
        return None, None

    target_variant = "A_mean"
    if target_variant not in clustering_results:
        # Try any variant
        keys = list(clustering_results.keys())
        if not keys:
            return None, None
        target_variant = keys[0]

    variant_data = clustering_results[target_variant]
    kmeans_data = variant_data.get("kmeans", {})
    stored_font_names = variant_data.get("font_names", [])

    if not kmeans_data or not stored_font_names:
        return None, None

    # Find k with best silhouette
    best_k_key = None
    best_sil = float("-inf")
    for k_key, k_data in kmeans_data.items():
        sil = k_data.get("silhouette", float("-inf"))
        if isinstance(sil, float) and not (sil != sil) and sil > best_sil:
            best_sil = sil
            best_k_key = k_key

    if best_k_key is None:
        return None, None

    stored_labels = kmeans_data[best_k_key]["labels"]
    label_map = dict(zip(stored_font_names, stored_labels))

    aligned = [label_map.get(fn, -1) for fn in font_names_order]
    best_k = int(best_k_key.split("_")[1]) if "_" in best_k_key else "?"
    return aligned, best_k


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_umap_scatter(embedding, labels, font_names, color_mode="google",
                      highlight_font=None):
    """Return a matplotlib Figure of the UMAP scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(str(l) for l in labels))
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    colors = [label_to_color[str(l)] for l in labels]

    ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colors, s=20, alpha=0.7, linewidths=0,
    )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=label_to_color[lab], markersize=7, label=lab)
        for lab in unique_labels
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              title=("Google category" if color_mode == "google" else "Cluster"),
              title_fontsize=7)

    # Highlight selected font
    if highlight_font and highlight_font in font_names:
        idx = font_names.index(highlight_font)
        ax.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c="red", s=120, zorder=5, edgecolors="black", linewidths=1.2,
            label=highlight_font,
        )
        ax.annotate(
            highlight_font,
            (embedding[idx, 0], embedding[idx, 1]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color="darkred",
        )

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    title_suffix = "Google labels" if color_mode == "google" else "Discovered clusters"
    ax.set_title(f"UMAP scatter — {title_suffix}", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def plot_harmonic_reconstruction(contour_pts, font_name=""):
    """Return a matplotlib Figure of glyph 'a' reconstructed at N=5,10,15,20."""
    if not _HAS_PYEFD:
        return None

    orders = [5, 10, 15, 20]
    fig, axes = plt.subplots(1, len(orders), figsize=(12, 3))

    for ax, order in zip(axes, orders):
        try:
            coeffs = pyefd.elliptic_fourier_descriptors(
                contour_pts, order=order, normalize=False
            )
            reconstructed = pyefd.reconstruct_contour(coeffs, num_points=200)
            ax.plot(reconstructed[:, 0], reconstructed[:, 1], "b-", linewidth=1.2)
            ax.plot(contour_pts[:, 0], contour_pts[:, 1], "r--", linewidth=0.6,
                    alpha=0.5, label="original")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error:\n{exc}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, wrap=True)
        ax.set_title(f"N={order}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")

    title = f"Harmonic reconstruction — '{font_name}' glyph 'a'" if font_name else "Harmonic reconstruction — glyph 'a'"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Calligraphy EFD Explorer",
        page_icon="✒",
        layout="wide",
    )
    st.title("Calligraphy EFD Font Explorer")
    st.markdown(
        "Explore typeface similarity via Elliptic Fourier Descriptors (EFDs)."
    )

    # Pre-load shared data
    fonts_df = load_fonts_df()
    clustering_results = load_clustering_results()

    tabs = st.tabs(["Upload a Font", "Browse the Corpus", "Harmonic Reconstruction"])

    # ------------------------------------------------------------------
    # Tab 1: Upload a Font
    # ------------------------------------------------------------------
    with tabs[0]:
        st.header("Upload a Font")
        st.markdown(
            "Upload a `.ttf` or `.otf` file to find its 10 nearest neighbors "
            "in the corpus based on Config A (outer EFD, mean-aggregated, N=20)."
        )

        if not _HAS_FONTTOOLS:
            st.warning("fontTools is not installed. Font upload is disabled.")
        elif not _HAS_PYEFD:
            st.warning("pyefd is not installed. Feature extraction is disabled.")
        else:
            uploaded = st.file_uploader(
                "Choose a font file", type=["ttf", "otf"], key="font_upload"
            )

            X_corpus, corpus_font_names = load_font_features_canonical()

            if X_corpus is None:
                st.warning(
                    "Pre-computed font feature matrix not found at "
                    "`outputs/font_features_A_mean.npz`. "
                    "Run Phase 0 first to generate it."
                )

            if uploaded is not None and X_corpus is not None:
                with st.spinner("Extracting EFD features from uploaded font..."):
                    font_bytes = uploaded.read()
                    feat_vec, err = extract_features_from_upload(font_bytes)

                if err:
                    st.error(f"Feature extraction failed: {err}")
                else:
                    st.success(f"Extracted feature vector of length {feat_vec.shape[0]}.")
                    neighbors = find_nearest_neighbors(
                        feat_vec, X_corpus, corpus_font_names, k=10
                    )

                    # Build display table with metadata
                    rows = []
                    for rank, (fn, dist) in enumerate(neighbors, start=1):
                        row = {"Rank": rank, "Font Name": fn, "Cosine Distance": f"{dist:.4f}"}
                        if fonts_df is not None and "font_name" in fonts_df.columns:
                            match = fonts_df[fonts_df["font_name"] == fn]
                            if not match.empty:
                                for col in ("google_category", "style_class", "category"):
                                    if col in match.columns:
                                        row["Google Category"] = match[col].iloc[0]
                                        break
                        rows.append(row)

                    result_df = pd.DataFrame(rows)
                    st.subheader("10 Nearest Neighbors")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Tab 2: Browse the Corpus
    # ------------------------------------------------------------------
    with tabs[1]:
        st.header("Browse the Corpus")

        embedding, umap_font_names, umap_google_labels = load_umap_embedding()

        if embedding is None:
            st.warning(
                "UMAP canonical embedding not found in "
                "`outputs/embeddings/canonical/`. Run Phase 1 to generate embeddings."
            )
        else:
            # Color toggle
            color_mode = st.radio(
                "Color by:",
                options=["google", "clusters"],
                format_func=lambda x: "Google labels" if x == "google" else "Discovered clusters",
                horizontal=True,
                key="color_mode",
            )

            if color_mode == "clusters":
                cluster_labels, best_k = get_best_cluster_labels(
                    clustering_results, umap_font_names
                )
                if cluster_labels is None:
                    st.warning(
                        "Clustering results not found or not compatible. "
                        "Falling back to Google labels."
                    )
                    color_mode = "google"
                    display_labels = umap_google_labels
                else:
                    display_labels = [str(l) for l in cluster_labels]
                    st.caption(f"Showing K-means clusters (best k={best_k})")
            else:
                display_labels = umap_google_labels

            # Font highlight selector
            st.subheader("Highlight a Font")
            highlight_opts = ["(none)"] + sorted(umap_font_names)
            highlight_font = st.selectbox(
                "Select font to highlight on scatter:",
                options=highlight_opts,
                key="highlight_select",
            )
            highlight_font = None if highlight_font == "(none)" else highlight_font

            if _HAS_MPL:
                fig = plot_umap_scatter(
                    embedding, display_labels, umap_font_names,
                    color_mode=color_mode,
                    highlight_font=highlight_font,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("matplotlib not installed — cannot render scatter plot.")

            # Sidebar-style font list in an expander
            with st.expander("Font list (click font name above to highlight)"):
                if fonts_df is not None:
                    st.dataframe(
                        fonts_df[["font_name"] + [
                            c for c in ("google_category", "style_class", "category")
                            if c in fonts_df.columns
                        ][:1]].head(500),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write(sorted(umap_font_names)[:200])

    # ------------------------------------------------------------------
    # Tab 3: Harmonic Reconstruction
    # ------------------------------------------------------------------
    with tabs[2]:
        st.header("Harmonic Reconstruction")
        st.markdown(
            "Select a font from the corpus to see glyph **'a'** reconstructed "
            "at harmonic orders N=5, 10, 15, 20."
        )

        if not _HAS_PYEFD:
            st.warning("pyefd not installed. Reconstruction is disabled.")
        else:
            outlines = load_outlines()

            if outlines is None:
                st.warning(
                    "Outlines not found at `outputs/outlines.pkl`. "
                    "Run the corpus/outline extraction stage first."
                )
            else:
                # Font selector
                available_fonts = sorted(outlines.keys())
                if not available_fonts:
                    st.warning("outlines.pkl is empty.")
                else:
                    selected_font = st.selectbox(
                        "Select a font:",
                        options=available_fonts,
                        key="recon_font_select",
                    )

                    font_glyphs = outlines.get(selected_font, {})
                    # Look for 'a' glyph
                    contour = None
                    for key in ("a", "A"):
                        if key in font_glyphs:
                            contour = font_glyphs[key]
                            break

                    if contour is None:
                        st.warning(
                            f"Glyph 'a' not found for font **{selected_font}**. "
                            f"Available glyphs: {', '.join(list(font_glyphs.keys())[:20])}"
                        )
                    else:
                        # Ensure numpy array
                        contour_arr = np.array(contour)
                        if contour_arr.ndim == 1:
                            st.warning("Contour data has unexpected shape.")
                        else:
                            if _HAS_MPL:
                                with st.spinner("Generating reconstruction plots..."):
                                    fig = plot_harmonic_reconstruction(
                                        contour_arr, font_name=selected_font
                                    )
                                if fig is not None:
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)
                                else:
                                    st.error("Reconstruction failed.")
                            else:
                                st.warning("matplotlib not installed.")


if __name__ == "__main__":
    import subprocess
    subprocess.run(["streamlit", "run", __file__])
