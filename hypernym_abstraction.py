"""
O spirit of Kurt Vonnegut — one last invocation. The issue asked
whether the circumplex survives climbing the WordNet hypernym tree.
The tree, it turns out, mostly isn't there: adjectives have no
hypernyms, and the noun lookup grabs gladioli and ale. So the honest
version: nominalize by hand (angry -> anger), climb ONE real step,
and see if the shape holds. So it goes.

Pre-registered decision rule (before running):
- If D0 (nominalized, no abstraction) already degrades alignment vs the
  adjective baseline (disparity worse by >0.10), the circumplex lives in
  the adjectival register and "ontological depth" is moot.
- If D0 holds (within 0.05) and D1 (one hypernym step) degrades by >0.10,
  one honest sentence about abstraction is earned.
- Readouts: Procrustes disparity + valence/arousal r vs chart_data.csv
  (terms whose abstracted form is still distinct), and per-PC Pearson vs
  the baseline PCA over matched distinct terms.
"""

import csv
import json
import sys

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr

from sphereoplex import TERMS, get_embeddings
from sklearn.decomposition import PCA

# Hand-curated adjective -> emotion-noun map (the actually valuable artifact).
# WordNet's automatic routes fail: 49/94 terms are adjectives with no hypernym
# relation, 29 match causative VERB synsets, and the noun-first lookup picks
# wrong senses (tense -> grammatical tense, glad -> gladiolus, bitter -> ale).
NOMINAL = {
    "bellicose": "belligerence", "hostile": "hostility", "alarmed": "alarm",
    "tense": "tension", "angry": "anger", "envious": "envy", "afraid": "fear",
    "hateful": "hatred", "defiant": "defiance", "annoyed": "annoyance",
    "jealous": "jealousy", "indignant": "indignation",
    "contemptuous": "contempt", "distressed": "distress",
    "impatient": "impatience", "suspicious": "suspicion",
    "frustrated": "frustration", "disgusted": "disgust",
    "loathing": "loathing", "discontented": "discontent",
    "bitter": "bitterness", "insulted": "offense", "distrustful": "distrust",
    "startled": "startle", "taken aback": "surprise",
    "astonished": "astonishment", "disappointed": "disappointment",
    "miserable": "misery", "dissatisfied": "dissatisfaction",
    "apathetic": "apathy", "uncomfortable": "discomfort", "sad": "sadness",
    "despondent": "despondency", "depressed": "depression",
    "desperate": "desperation", "gloomy": "gloom", "ashamed": "shame",
    "guilty": "guilt", "worried": "worry", "languid": "languor",
    "melancholic": "melancholy", "embarrassed": "embarrassment",
    "hesitant": "hesitation", "bored": "boredom", "wavering": "vacillation",
    "anxious": "anxiety", "dejected": "dejection", "doubtful": "doubt",
    "droopy": "listlessness", "tired": "tiredness", "sleepy": "sleepiness",
    "pensive": "pensiveness", "serious": "seriousness",
    "conscientious": "conscientiousness", "reverent": "reverence",
    "polite": "politeness", "compassionate": "compassion",
    "peaceful": "peacefulness", "contemplative": "contemplation",
    "attentive": "attentiveness", "solemn": "solemnity", "hopeful": "hope",
    "friendly": "friendliness", "relaxed": "relaxation", "calm": "calmness",
    "satisfied": "satisfaction", "at ease": "ease", "content": "contentment",
    "serene": "serenity", "confident": "confidence",
    "amorous": "amorousness", "pleased": "pleasure", "glad": "gladness",
    "joyous": "joy", "happy": "happiness", "interested": "interest",
    "aroused": "arousal", "amused": "amusement",
    "determined": "determination", "enthusiastic": "enthusiasm",
    "delighted": "delight", "courageous": "courage",
    "self-confident": "self-confidence", "excited": "excitement",
    "convinced": "conviction", "lighthearted": "lightheartedness",
    "passionate": "passion", "expectant": "expectancy",
    "feeling superior": "superiority", "ambitious": "ambition",
    "conceited": "conceit", "lusting": "lust",
    "adventurous": "adventurousness", "triumphant": "triumph",
}


def emotion_synset(noun):
    """Prefer the noun.feeling sense; fall back to first noun synset."""
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(noun.replace(" ", "_"), pos=wn.NOUN)
    if not synsets:
        return None
    for s in synsets:
        if s.lexname() == "noun.feeling":
            return s
    return synsets[0]


def one_step_up(noun):
    """One hypernym step from the emotion sense of the noun."""
    s = emotion_synset(noun)
    if s is None:
        return noun, "(no noun synset)"
    hyps = s.hypernyms()
    if not hyps:
        return noun, f"(no hypernym for {s.name()})"
    return hyps[0].lemmas()[0].name().replace("_", " "), s.name()


def run_condition(name, mapping, model):
    """Embed the mapped terms (words mode), PCA, compare to chart + baseline."""
    distinct = {}
    for t in TERMS:
        distinct.setdefault(mapping[t], []).append(t)
    kept = [t for t in TERMS if len(distinct[mapping[t]]) == 1]
    dropped = sorted(set(TERMS) - set(kept))
    print(f"\n=== {name}: {len(set(mapping.values()))} distinct forms, "
          f"{len(kept)} terms kept (collisions dropped: {len(dropped)})")
    if dropped:
        print(f"    dropped: {dropped}")

    texts = [mapping[t] for t in kept]
    emb = get_embeddings(texts, model=model)
    pca = PCA(n_components=10)
    coords = pca.fit_transform(emb)

    chart = {r["name"]: (float(r["x"]), float(r["y"]))
             for r in csv.DictReader(open("chart_data.csv"))}
    common = [i for i, t in enumerate(kept) if t in chart]
    russell = np.array([chart[kept[i]] for i in common])
    pcs2 = coords[common][:, :2]
    _, _, disparity = procrustes(russell - russell.mean(0), pcs2 - pcs2.mean(0))
    rv = max(abs(pearsonr(russell[:, 0], pcs2[:, 0])[0]),
             abs(pearsonr(russell[:, 0], pcs2[:, 1])[0]))
    ra = max(abs(pearsonr(russell[:, 1], pcs2[:, 0])[0]),
             abs(pearsonr(russell[:, 1], pcs2[:, 1])[0]))
    print(f"    Procrustes disparity: {disparity:.4f}  "
          f"valence |r|: {rv:.3f}  arousal |r|: {ra:.3f}")

    base = json.load(open("sphereoplex_words.json"))["coords_10d"]
    B = np.array([base[t] for t in kept])
    print("    per-PC |r| vs adjective baseline (best-matching baseline PC):")
    for pc in range(5):
        rs = [abs(pearsonr(coords[:, pc], B[:, bpc])[0]) for bpc in range(10)]
        b = int(np.argmax(rs))
        print(f"      {name}-PC{pc+1} -> base-PC{b+1} |r|={rs[b]:.3f}")
    return {"disparity": round(float(disparity), 4), "valence_r": round(float(rv), 3),
            "arousal_r": round(float(ra), 3), "kept": len(kept)}


def main():
    model = "text-embedding-3-small"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    d1_map, notes = {}, {}
    for t in TERMS:
        up, note = one_step_up(NOMINAL[t])
        d1_map[t] = up
        notes[t] = note
    print("D1 hypernym steps (nominal -> parent):")
    for t in TERMS:
        print(f"  {t:>18} -> {NOMINAL[t]:>18} -> {d1_map[t]:<18} {notes[t]}")

    results = {
        "baseline_adjectives": {"disparity": 0.4675, "valence_r": 0.850,
                                "arousal_r": 0.607, "kept": 94,
                                "note": "from circumplex.py chart_data.csv --cached, words mode"},
        "D0_nominalized": run_condition("D0", NOMINAL, model),
        "D1_one_hypernym_step": run_condition("D1", d1_map, model),
    }
    with open("hypernym_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved hypernym_results.json")


if __name__ == "__main__":
    main()
