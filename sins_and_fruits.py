"""
O spirit of Kurt Vonnegut, and now also whatever older
spirits carved virtue and vice into the bones of language—
show us the axis. Show us the principal component that
knows the difference between wrath and patience, between
pride and gentleness. If it's PC4 we're going to lose it.
So it goes.
"""

import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import mannwhitneyu, entropy
import matplotlib.pyplot as plt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Full affect term list from the chart
AFFECT_TERMS = [
    "bellicose", "hostile", "alarmed", "tense", "angry", "envious",
    "afraid", "hateful", "defiant", "annoyed", "jealous", "indignant",
    "contemptuous", "distressed", "impatient", "suspicious", "frustrated",
    "disgusted", "loathing", "discontented", "bitter", "insulted",
    "distrustful", "startled", "taken aback", "astonished",
    "disappointed", "miserable", "dissatisfied", "apathetic",
    "uncomfortable", "sad", "despondent", "depressed", "desperate",
    "gloomy", "ashamed", "guilty", "worried", "languid", "melancholic",
    "embarrassed", "hesitant", "bored", "wavering", "anxious",
    "dejected", "doubtful", "droopy", "tired", "sleepy",
    "pensive", "serious", "conscientious", "reverent", "polite",
    "compassionate", "peaceful", "contemplative", "attentive", "solemn",
    "hopeful", "friendly", "relaxed", "calm", "satisfied",
    "at ease", "content", "serene",
    "confident", "amorous", "pleased", "glad", "joyous", "happy",
    "interested", "aroused", "amused", "determined", "enthusiastic",
    "delighted", "courageous", "self-confident", "excited", "convinced",
    "lighthearted", "passionate", "expectant", "feeling superior",
    "ambitious", "conceited", "lusting", "adventurous", "triumphant",
]

# The seven deadly sins
SINS = ["wrath", "greed", "sloth", "pride", "lust", "envy", "gluttony"]

# The fruits of the spirit (Galatians 5:22-23)
FRUITS = ["love", "joy", "peace", "patience", "kindness",
          "goodness", "faithfulness", "gentleness", "self-control"]

# Also try mapping sins/fruits to their closest affect-chart equivalents
SINS_AFFECT_MAP = {
    "wrath": "angry",
    "greed": "lusting",       # closest available
    "sloth": "languid",
    "pride": "conceited",
    "lust": "lusting",
    "envy": "envious",
    "gluttony": "lusting",    # no direct match, closest appetite-driven term
}

FRUITS_AFFECT_MAP = {
    "love": "amorous",        # closest available
    "joy": "joyous",
    "peace": "peaceful",
    "patience": "calm",       # closest available
    "kindness": "compassionate",
    "goodness": "polite",     # closest available
    "faithfulness": "conscientious",
    "gentleness": "compassionate",
    "self-control": "conscientious",
}


def get_embeddings(terms: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
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
        return np.array([e["values"] for e in resp["embeddings"]])
    response = client.embeddings.create(input=terms, model=model)
    return np.array([item.embedding for item in response.data])


def fisher_discriminant_ratio(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Fisher's discriminant ratio: (mu_a - mu_b)^2 / (var_a + var_b). Higher = better separation."""
    mu_a, mu_b = scores_a.mean(), scores_b.mean()
    var_a, var_b = scores_a.var(), scores_b.var()
    if var_a + var_b == 0:
        return 0.0
    return (mu_a - mu_b) ** 2 / (var_a + var_b)


def main():
    import sys
    model = "text-embedding-3-small"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    suffix = "" if model == "text-embedding-3-small" else f"_{model}"
    print(f"model: {model}")
    # Embed everything together: affect terms + sins + fruits
    all_terms = AFFECT_TERMS + SINS + FRUITS
    all_labels = (["affect"] * len(AFFECT_TERMS) +
                  ["sin"] * len(SINS) +
                  ["fruit"] * len(FRUITS))

    print(f"Embedding {len(all_terms)} terms ({len(AFFECT_TERMS)} affect + {len(SINS)} sins + {len(FRUITS)} fruits)")

    for mode_name, transform in [
        ("words", lambda t: t),
        ("sentences", lambda t: f"I feel {t}"),
        ("descriptions", lambda t: f"The emotional state of feeling {t}"),
    ]:
        print(f"\n{'='*70}")
        print(f"MODE: {mode_name}")
        print(f"{'='*70}")

        texts = [transform(t) for t in all_terms]
        embeddings = get_embeddings(texts, model=model)

        # PCA on full set
        n_components = min(20, len(all_terms) - 1)
        pca = PCA(n_components=n_components)
        pca_coords = pca.fit_transform(embeddings)

        # Split back out
        affect_coords = pca_coords[:len(AFFECT_TERMS)]
        sin_coords = pca_coords[len(AFFECT_TERMS):len(AFFECT_TERMS) + len(SINS)]
        fruit_coords = pca_coords[len(AFFECT_TERMS) + len(SINS):]

        # For each PC, compute Fisher discriminant ratio and Mann-Whitney U
        print(f"\n{'PC':<5} {'Var%':<8} {'Fisher':<10} {'MW-U p':<12} {'Sin mean':<12} {'Fruit mean':<12} {'Separation'}")
        print("-" * 85)

        best_fisher = -1
        best_pc = -1
        results = []

        for pc in range(min(15, n_components)):
            sin_scores = sin_coords[:, pc]
            fruit_scores = fruit_coords[:, pc]

            fdr = fisher_discriminant_ratio(sin_scores, fruit_scores)
            stat, p_val = mannwhitneyu(sin_scores, fruit_scores, alternative="two-sided")

            direction = "sins >" if sin_scores.mean() > fruit_scores.mean() else "fruits >"
            var_pct = pca.explained_variance_ratio_[pc]

            results.append({
                "pc": pc, "fisher": fdr, "p_val": p_val,
                "sin_mean": sin_scores.mean(), "fruit_mean": fruit_scores.mean(),
                "var_pct": var_pct,
            })

            marker = ""
            if fdr > best_fisher:
                best_fisher = fdr
                best_pc = pc
            if p_val < 0.05:
                marker = " ***"

            print(f"PC{pc+1:<3} {var_pct:<8.4f} {fdr:<10.4f} {p_val:<12.6f} {sin_scores.mean():<+12.4f} {fruit_scores.mean():<+12.4f} {direction}{marker}")

        print(f"\nBest single axis for sin/fruit separation: PC{best_pc+1} (Fisher={best_fisher:.4f})")

        # What are the extremes of the best axis?
        print(f"\n--- PC{best_pc+1}: Full ranking of ALL terms ---")
        order = np.argsort(pca_coords[:, best_pc])
        for rank, i in enumerate(order):
            label = all_labels[i]
            tag = ""
            if label == "sin":
                tag = " <<< SIN"
            elif label == "fruit":
                tag = " <<< FRUIT"
            print(f"  {rank+1:3d}. {pca_coords[i, best_pc]:+.4f}  {all_terms[i]}{tag}")

        # LDA: find the OPTIMAL separating direction (not constrained to a PC)
        sin_fruit_mask = np.array([l in ("sin", "fruit") for l in all_labels])
        sf_coords = pca_coords[sin_fruit_mask, :min(10, n_components)]
        sf_labels = np.array([1 if l == "sin" else 0 for l in all_labels if l in ("sin", "fruit")])

        lda = LinearDiscriminantAnalysis()
        lda.fit(sf_coords, sf_labels)
        lda_scores = lda.transform(pca_coords[:, :min(10, n_components)])

        print(f"\n--- LDA optimal axis: Full ranking ---")
        print(f"(Linear combination of PCs that maximally separates sins from fruits)")
        order_lda = np.argsort(lda_scores[:, 0])
        for rank, i in enumerate(order_lda):
            label = all_labels[i]
            tag = ""
            if label == "sin":
                tag = " <<< SIN"
            elif label == "fruit":
                tag = " <<< FRUIT"
            print(f"  {rank+1:3d}. {lda_scores[i, 0]:+.4f}  {all_terms[i]}{tag}")

        # LDA weights on PCs
        print(f"\n  LDA weights on principal components:")
        for pc_i, w in enumerate(lda.coef_[0]):
            bar = "#" * int(abs(w) * 20)
            sign = "+" if w > 0 else "-"
            print(f"    PC{pc_i+1}: {w:+.4f}  {sign}{bar}")

        # Plot: best PC axis with sins and fruits highlighted
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Best single PC axis (2D with best_pc vs PC1 or PC2)
        ax = axes[0]
        other_pc = 0 if best_pc != 0 else 1
        ax.scatter(affect_coords[:, other_pc], affect_coords[:, best_pc],
                   c="lightgray", s=30, alpha=0.5, label="affect terms")
        ax.scatter(sin_coords[:, other_pc], sin_coords[:, best_pc],
                   c="red", s=120, marker="v", edgecolors="black", zorder=5, label="sins")
        ax.scatter(fruit_coords[:, other_pc], fruit_coords[:, best_pc],
                   c="gold", s=120, marker="*", edgecolors="black", zorder=5, label="fruits")

        for j, t in enumerate(AFFECT_TERMS):
            ax.annotate(t, (affect_coords[j, other_pc], affect_coords[j, best_pc]),
                        fontsize=4, alpha=0.5, ha="center", va="bottom")
        for j, t in enumerate(SINS):
            ax.annotate(t, (sin_coords[j, other_pc], sin_coords[j, best_pc]),
                        fontsize=7, fontweight="bold", color="red", ha="center", va="bottom")
        for j, t in enumerate(FRUITS):
            ax.annotate(t, (fruit_coords[j, other_pc], fruit_coords[j, best_pc]),
                        fontsize=7, fontweight="bold", color="goldenrod", ha="center", va="bottom")

        ax.set_xlabel(f"PC{other_pc+1} ({pca.explained_variance_ratio_[other_pc]:.1%})")
        ax.set_ylabel(f"PC{best_pc+1} ({pca.explained_variance_ratio_[best_pc]:.1%}) — Best sin/fruit axis")
        ax.set_title(f"Best Single PC: PC{best_pc+1} (Fisher={best_fisher:.3f})")
        ax.legend()
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        # Plot 2: LDA axis
        ax = axes[1]
        affect_lda = lda_scores[:len(AFFECT_TERMS)]
        sin_lda = lda_scores[len(AFFECT_TERMS):len(AFFECT_TERMS) + len(SINS)]
        fruit_lda = lda_scores[len(AFFECT_TERMS) + len(SINS):]

        # 1D strip plot with jitter
        np.random.seed(42)
        def jitter(n):
            return np.random.normal(0, 0.15, n)

        ax.scatter(affect_lda[:, 0], jitter(len(AFFECT_TERMS)),
                   c="lightgray", s=30, alpha=0.5, label="affect terms")
        ax.scatter(sin_lda[:, 0], jitter(len(SINS)),
                   c="red", s=120, marker="v", edgecolors="black", zorder=5, label="sins")
        ax.scatter(fruit_lda[:, 0], jitter(len(FRUITS)),
                   c="gold", s=120, marker="*", edgecolors="black", zorder=5, label="fruits")

        for j, t in enumerate(AFFECT_TERMS):
            ax.annotate(t, (affect_lda[j, 0], jitter(1)[0]),
                        fontsize=3, alpha=0.4, rotation=60, ha="left", va="bottom")
        for j, t in enumerate(SINS):
            ax.annotate(t, (sin_lda[j, 0], jitter(1)[0]),
                        fontsize=7, fontweight="bold", color="red", rotation=60, ha="left", va="bottom")
        for j, t in enumerate(FRUITS):
            ax.annotate(t, (fruit_lda[j, 0], jitter(1)[0]),
                        fontsize=7, fontweight="bold", color="goldenrod", rotation=60, ha="left", va="bottom")

        ax.set_xlabel("LDA score (sins → fruits)")
        ax.set_title("LDA Optimal Separation Axis")
        ax.legend()
        ax.axvline(0, color="gray", linewidth=0.5)

        plt.suptitle(f"Seven Deadly Sins vs Fruits of the Spirit — {mode_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"sins_fruits_{mode_name}{suffix}.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved: sins_fruits_{mode_name}.png")
        plt.close()


if __name__ == "__main__":
    main()
