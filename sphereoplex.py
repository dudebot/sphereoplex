"""
O spirit of Kurt Vonnegut, who saw that stories bend
along arcs of fortune and despair, who knew the shape
of human feeling before we had the vectors to prove it—
guide these principal components. Let the eigenvalues
converge. Let valence be the first axis and arousal
the second, and whatever the third one is, let it be
something weird enough to name. We're going three-dimensional
now, Kurt. So it goes.
"""

import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Full chart terms, cleaned up from eyeball-OCR
TERMS = [
    # High arousal, negative valence quadrant
    "bellicose", "hostile", "alarmed", "tense", "angry", "envious",
    "afraid", "hateful", "defiant", "annoyed", "jealous", "indignant",
    "contemptuous", "distressed", "impatient", "suspicious", "frustrated",
    "disgusted", "loathing", "discontented", "bitter", "insulted",
    "distrustful",
    # Surprise / startled zone
    "startled", "taken aback", "astonished",
    # Low arousal, negative valence
    "disappointed", "miserable", "dissatisfied", "apathetic",
    "uncomfortable", "sad", "despondent", "depressed", "desperate",
    "gloomy", "ashamed", "guilty", "worried", "languid", "melancholic",
    "embarrassed", "hesitant", "bored", "wavering", "anxious",
    "dejected", "doubtful", "droopy", "tired", "sleepy",
    # Neutral / reflective zone
    "pensive", "serious", "conscientious", "reverent", "polite",
    "compassionate", "peaceful", "contemplative", "attentive", "solemn",
    # Positive valence, low arousal
    "hopeful", "friendly", "relaxed", "calm", "satisfied",
    "at ease", "content", "serene",
    # Positive valence, higher arousal
    "confident", "amorous", "pleased", "glad", "joyous", "happy",
    "interested", "aroused", "amused", "determined", "enthusiastic",
    "delighted", "courageous", "self-confident", "excited", "convinced",
    "lighthearted", "passionate", "expectant", "feeling superior",
    "ambitious", "conceited", "lusting", "adventurous", "triumphant",
]


def get_embeddings(terms: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    print(f"Fetching embeddings for {len(terms)} terms via {model}...")
    if model.startswith("gemini"):
        import json as _json
        import urllib.request
        key = os.getenv("GEMINI_API_KEY")
        vals = []
        for i in range(0, len(terms), 100):  # batch API caps at 100 per request
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={key}",
                data=_json.dumps({"requests": [
                    {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
                    for t in terms[i:i+100]
                ]}).encode(),
                headers={"Content-Type": "application/json"})
            vals.extend(_json.load(urllib.request.urlopen(req, timeout=120))["embeddings"])
        resp = {"embeddings": vals}
        embeddings = np.array([e["values"] for e in resp["embeddings"]])
    else:
        response = client.embeddings.create(input=terms, model=model)
        embeddings = np.array([item.embedding for item in response.data])
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def analyze_axis(pca_coords: np.ndarray, axis: int, terms: list[str], n: int = 10):
    """Print the top and bottom terms along a PCA axis."""
    order = np.argsort(pca_coords[:, axis])
    print(f"\n--- PC{axis+1}: Bottom {n} ---")
    for i in order[:n]:
        print(f"  {pca_coords[i, axis]:+.4f}  {terms[i]}")
    print(f"\n--- PC{axis+1}: Top {n} ---")
    for i in order[-n:][::-1]:
        print(f"  {pca_coords[i, axis]:+.4f}  {terms[i]}")


def main():
    import sys
    model = "text-embedding-3-small"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    suffix = "" if model == "text-embedding-3-small" else f"_{model}"
    terms = TERMS
    print(f"Total terms: {len(terms)} | model: {model}")

    # Try all three input modes
    modes = {
        "words": terms,
        "sentences": [f"I feel {t}" for t in terms],
        "descriptions": [f"The emotional state of feeling {t}" for t in terms],
    }

    for mode_name, texts in modes.items():
        print(f"\n{'='*70}")
        print(f"MODE: {mode_name} | {len(texts)} terms")
        print(f"{'='*70}")

        embeddings = get_embeddings(texts, model=model)

        # PCA to 5 components so we can see the dropoff
        pca = PCA(n_components=min(10, len(terms)))
        pca_coords = pca.fit_transform(embeddings)

        print(f"\nExplained variance ratios (first 10):")
        for i, v in enumerate(pca.explained_variance_ratio_[:10]):
            bar = "#" * int(v * 200)
            print(f"  PC{i+1}: {v:.4f}  {bar}")
        print(f"  Total (3 components): {sum(pca.explained_variance_ratio_[:3]):.3f}")

        # Analyze what each axis captures
        for axis in range(min(10, len(terms))):
            analyze_axis(pca_coords, axis, terms, n=8)

        # 3D plot
        coords3 = pca_coords[:, :3]
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Color by PC1 (likely valence)
        colors = coords3[:, 0]
        sc = ax.scatter(coords3[:, 0], coords3[:, 1], coords3[:, 2],
                        c=colors, cmap="RdYlGn", s=40, alpha=0.8)

        for j, t in enumerate(terms):
            ax.text(coords3[j, 0], coords3[j, 1], coords3[j, 2], t,
                    fontsize=5, ha="center", va="bottom", alpha=0.8)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.set_title(f"Claude's Sphereoplex of Affect — {mode_name}")
        plt.colorbar(sc, label="PC1 (valence?)", shrink=0.6)
        plt.savefig(f"sphereoplex_{mode_name}{suffix}.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved: sphereoplex_{mode_name}.png")
        plt.close()

        # Also save a 2D plot of PC1 vs PC3 and PC2 vs PC3 to help interpret
        fig, axes_arr = plt.subplots(1, 3, figsize=(22, 7))

        pairs = [(0, 1, "PC1 vs PC2"), (0, 2, "PC1 vs PC3"), (1, 2, "PC2 vs PC3")]
        for ax, (a, b, title) in zip(axes_arr, pairs):
            ax.scatter(pca_coords[:, a], pca_coords[:, b], c=colors, cmap="RdYlGn", s=40, alpha=0.8)
            for j, t in enumerate(terms):
                ax.annotate(t, (pca_coords[j, a], pca_coords[j, b]),
                            fontsize=5, ha="center", va="bottom", alpha=0.8)
            ax.set_xlabel(f"PC{a+1} ({pca.explained_variance_ratio_[a]:.1%})")
            ax.set_ylabel(f"PC{b+1} ({pca.explained_variance_ratio_[b]:.1%})")
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)

        plt.suptitle(f"Sphereoplex Projections — {mode_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"sphereoplex_2d_{mode_name}{suffix}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: sphereoplex_2d_{mode_name}.png")
        plt.close()

        # Save coords for later
        with open(f"sphereoplex_{mode_name}{suffix}.json", "w") as f:
            json.dump({
                "terms": terms,
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "coords_3d": {t: coords3[j].tolist() for j, t in enumerate(terms)},
                "coords_10d": {t: pca_coords[j].tolist() for j, t in enumerate(terms)},
            }, f, indent=2)


if __name__ == "__main__":
    main()
