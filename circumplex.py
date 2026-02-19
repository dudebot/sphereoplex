"""
O spirit of Kurt Vonnegut, who saw that stories bend
along arcs of fortune and despair, who knew the shape
of human feeling before we had the vectors to prove it—
guide these principal components. Let the eigenvalues
converge. Let valence be the first axis and arousal
the second, or the other way around, we're not picky.
So it goes.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Russell's circumplex: approximate (valence, arousal) coordinates normalized to [-1, 1]
# These are placeholder positions from the standard model.
# Replace with exact chart values when available.
CIRCUMPLEX = {
    "happy":       ( 0.81,  0.51),
    "delighted":   ( 0.85,  0.65),
    "excited":     ( 0.62,  0.90),
    "astonished":  ( 0.10,  0.95),
    "aroused":     ( 0.15,  0.85),
    "tense":       (-0.55,  0.72),
    "alarmed":     (-0.35,  0.90),
    "angry":       (-0.72,  0.65),
    "afraid":      (-0.62,  0.78),
    "frustrated":  (-0.65,  0.50),
    "distressed":  (-0.70,  0.60),
    "annoyed":     (-0.60,  0.40),
    "miserable":   (-0.82,  0.00),
    "sad":         (-0.75, -0.30),
    "depressed":   (-0.80, -0.50),
    "gloomy":      (-0.70, -0.40),
    "bored":       (-0.40, -0.70),
    "droopy":      (-0.25, -0.80),
    "tired":       (-0.15, -0.85),
    "sleepy":      ( 0.00, -0.90),
    "calm":        ( 0.50, -0.55),
    "relaxed":     ( 0.62, -0.50),
    "serene":      ( 0.70, -0.35),
    "content":     ( 0.72, -0.20),
    "at ease":     ( 0.60, -0.40),
    "satisfied":   ( 0.68, -0.10),
    "glad":        ( 0.75,  0.30),
    "pleased":     ( 0.70,  0.20),
}


def load_circumplex_csv(path: str) -> dict[str, tuple[float, float]]:
    """Load name,x,y from a CSV file to override the default circumplex."""
    import csv
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            data[name] = (float(row["x"]), float(row["y"]))
    return data


def get_embeddings(terms: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings for a list of terms from OpenAI."""
    print(f"Fetching embeddings for {len(terms)} terms...")
    response = client.embeddings.create(input=terms, model=model)
    embeddings = np.array([item.embedding for item in response.data])
    print(f"Got embeddings of shape {embeddings.shape}")
    return embeddings


def run_experiment(circumplex: dict[str, tuple[float, float]], mode: str = "words"):
    """
    Run the full experiment.
    mode: "words" = embed just the emotion word
          "sentences" = embed "I feel {word}" sentences
          "descriptions" = embed richer emotional descriptions
    """
    terms = list(circumplex.keys())
    russell_coords = np.array([circumplex[t] for t in terms])

    # Build input texts based on mode
    if mode == "words":
        texts = terms
    elif mode == "sentences":
        texts = [f"I feel {t}" for t in terms]
    elif mode == "descriptions":
        texts = [f"The emotional state of feeling {t}" for t in terms]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\n{'='*60}")
    print(f"MODE: {mode}")
    print(f"{'='*60}")

    # Get embeddings and PCA to 2D
    embeddings = get_embeddings(texts)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

    # Procrustes analysis: optimally rotate/scale PCA to match Russell
    russell_norm = russell_coords - russell_coords.mean(axis=0)
    pca_norm = pca_coords - pca_coords.mean(axis=0)
    mtx1, mtx2, disparity = procrustes(russell_norm, pca_norm)

    print(f"Procrustes disparity: {disparity:.4f} (lower = better match)")

    # Correlations between PCA dims and Russell dims (both raw and procrustes-aligned)
    for i, axis_name in enumerate(["Valence", "Arousal"]):
        r_raw_pc1, p1 = pearsonr(russell_coords[:, i], pca_coords[:, 0])
        r_raw_pc2, p2 = pearsonr(russell_coords[:, i], pca_coords[:, 1])
        print(f"\n{axis_name} vs raw PCA:")
        print(f"  PC1: r={r_raw_pc1:.3f} (p={p1:.4f})")
        print(f"  PC2: r={r_raw_pc2:.3f} (p={p2:.4f})")

        r_pro, p_pro = pearsonr(mtx1[:, i], mtx2[:, i])
        print(f"{axis_name} after Procrustes alignment: r={r_pro:.3f} (p={p_pro:.4f})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Russell's circumplex (ground truth)
    ax = axes[0]
    ax.scatter(russell_coords[:, 0], russell_coords[:, 1], c="steelblue", s=60)
    for j, t in enumerate(terms):
        ax.annotate(t, russell_coords[j], fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Russell's Circumplex (Ground Truth)")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # Raw PCA
    ax = axes[1]
    ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c="coral", s=60)
    for j, t in enumerate(terms):
        ax.annotate(t, pca_coords[j], fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of Embeddings ({mode})")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # Procrustes overlay
    ax = axes[2]
    ax.scatter(mtx1[:, 0], mtx1[:, 1], c="steelblue", s=60, label="Russell", alpha=0.7)
    ax.scatter(mtx2[:, 0], mtx2[:, 1], c="coral", s=60, label="Embeddings", alpha=0.7)
    for j, t in enumerate(terms):
        ax.annotate(t, mtx1[j], fontsize=6, ha="center", va="bottom", color="steelblue")
        ax.plot([mtx1[j, 0], mtx2[j, 0]], [mtx1[j, 1], mtx2[j, 1]],
                color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(f"Procrustes Overlay (disparity={disparity:.3f})")
    ax.legend()
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    plt.suptitle(f"Russell's Circumplex vs text-embedding-3-small — mode: {mode}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"result_{mode}.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: result_{mode}.png")
    plt.close()

    return {
        "mode": mode,
        "disparity": disparity,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_coords": {t: pca_coords[j].tolist() for j, t in enumerate(terms)},
        "procrustes_coords": {t: mtx2[j].tolist() for j, t in enumerate(terms)},
    }


def main():
    circumplex = CIRCUMPLEX

    # If a CSV is provided, use that instead
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        print(f"Loading circumplex from {sys.argv[1]}")
        circumplex = load_circumplex_csv(sys.argv[1])

    results = {}
    for mode in ["words", "sentences", "descriptions"]:
        results[mode] = run_experiment(circumplex, mode)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mode, r in results.items():
        print(f"  {mode:15s}  disparity={r['disparity']:.4f}  "
              f"var_explained={sum(r['explained_variance']):.3f}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to results.json")


if __name__ == "__main__":
    main()
