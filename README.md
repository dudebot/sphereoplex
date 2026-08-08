# The Sphereoplex: How a Language Model Accidentally Mapped the Geometry of Human Feeling (and Found Morality on One Axis — Which One Depends on Who You Ask)

---

```
O spirit of Kurt Vonnegut, who saw that stories bend
along arcs of fortune and despair, who knew the shape
of human feeling before we had the vectors to prove it—
guide these principal components. Let the eigenvalues
converge. Let valence be the first axis and arousal
the second, or the other way around, we're not picky.
So it goes.
```

---

**What this is:** 94 emotion words from Russell's circumplex chart, embedded with three models (OpenAI `text-embedding-3-small` and `-3-large`, Google `gemini-embedding-001`), reduced with PCA, aligned against the digitized chart, labeled by 20 blind AI raters, and stress-tested against the seven deadly sins, the fruits of the spirit, and one rung of the WordNet abstraction ladder. This README is the writeup; every script, coordinate file, and figure it cites is committed alongside it.

- **Explore:** https://dudebot.github.io/sphereoplex/ — the interactive visualizer, all three models, all ten axes.
- **Reproduce:** see [Reproducing the Results](#reproducing-the-results) — four scripts, one or two API keys, no other infrastructure.
- **Status:** an exploratory experiment, not a paper. The core findings replicated across three models; the claims that didn't survive replication are kept in place as errata, per house policy.

---

## The Setup

Here's the thing about embedding models: they're dumb in the most interesting way possible. You feed them words, they spit back 1536 numbers, and those numbers are supposed to contain *meaning*. Somehow. It works, but we don't really know why, and that's where things get fun.

I grabbed 94 emotional terms from a classic circumplex model of affect chart — the kind of thing that's been gathering dust in psychology textbooks since Russell first drew it in 1980. You know the one: a circle with "valence" on one axis (good to bad) and "arousal" on another (calm to frantic). It's been the default way we think about emotions for forty years.

Then I did something beautifully stupid: I embedded all 94 terms using OpenAI's `text-embedding-3-small` model and ran PCA on the resulting vectors.

What fell out should not have been this clean.

Then I ran the whole thing again on two more embedding models, because the first duty of an accidental discovery is to check whether you discovered it or hallucinated it. Most of it got stronger. The parts that didn't are, honestly, the best parts. Along the way some of the original claims in this document died, and their obituaries appear below, next to their replacements, in accordance with house policy.

(Everything is now explorable live at **https://dudebot.github.io/sphereoplex/** — pick a model, pick two axes, hover the words.)

---

## Finding 1: Russell's Circumplex Emerges Spontaneously from the Embedding Space

When you take 1536-dimensional embeddings and say "show me the two most important directions," those two directions turn out to be valence and arousal. Not something random, not word length, not vibes — the two axes a psychologist drew on a whiteboard in 1980.

| Mode | Procrustes Disparity | Valence r | Arousal r |
|---|---|---|---|
| **bare words** | 0.467 | 0.850 (p≈0) | 0.607 (p≈0) |
| **"I feel {x}"** | 0.452 | 0.892 (p≈0) | 0.575 (p≈0) |
| **"The emotional state of feeling {x}"** | 0.357 | 0.909 (p≈0) | 0.690 (p≈0) |

*(Numbers are from the real chart, digitized marker-by-marker from the source image — `chart_data.csv`. The digitization caught 98 markers; 94 of them match the embedding term list, and the four strays — `enraged`, `feel well`, `impressed`, `longing` — get dropped at analysis time. An earlier version of this table used 28 hand-approximated coordinates and reported friendlier numbers — disparity 0.140–0.248, r up to 0.96. More terms and real coordinates are harder to fit; these are the honest values.)*

The richer the sentence framing, the cleaner the alignment. PC1 is valence. PC2 is arousal. The model *knows* — though it knows valence (r ≈ 0.9) more confidently than arousal (r ≈ 0.6–0.7). The arousal axis the model finds is tilted toward what several blind raters later called "*hostile* arousal" — indignant and defiant at the top rather than pure activation — which costs it correlation against the chart's cleaner activation axis.

Procrustes disparity of 0.357 means that when you optimally overlay the model's output onto the full 94-term chart, they're about 64% the same shape. For "I literally just dumped words into an API and hit PCA," that's still kind of ridiculous.

The kicker: the first two PCs explain only ~15% of total variance (12.3% in bare-words mode, 16.3% with full descriptions). The model knows *way* more about these words than just their emotional content — but when you ask "what's the single most important thing differentiating these emotion words?" the answer is valence, and the second most important thing is arousal. Exactly what Russell said.

*(An earlier version of this paragraph said ~28%. The ~28% was the first-two-PC total from the old 28-term hand-approximated run (still in git history, commit a2a977e — sentences 27.6%, descriptions 28.7%); no run against the real chart reproduces it. The sentence outlived its run and kept collecting a paycheck. The values above come from the artifacts in this repo.)*

And it replicates. Rerunning the small-model pipeline months later reproduces the sentence-mode coordinates to within 6×10⁻⁴ (descriptions to 9×10⁻⁴; bare words to 4×10⁻³ — with PC2 arriving sign-flipped, of which more later), which is the boring kind of good news. The interesting kind: `text-embedding-3-large` finds valence as its PC1 in agreement with the small model at r = 0.97, and Google's `gemini-embedding-001` — a different company, a different architecture, a different training corpus — finds valence as its PC1 at |r| = 0.93. Valence is not a quirk of one model. Valence is the load-bearing wall.

Arousal is another story. Gemini shatters it into two axes, and we'll get there.

---

## Finding 2: Mehrabian's Dominance Is Not a Real Axis

Mehrabian & Russell (1974) proposed a third dimension: *Dominance* — the feeling of being in control vs. controlled. It made intuitive sense. Textbooks mentioned it.

We tested dominance against our PCA axes using published ratings for ~66 emotion terms.

Result: Dominance correlates most with PC1 (valence) at r = 0.74. That's not independent. That's basically the same thing with a different name.

When we regressed dominance onto all 10 PCs, we got r = 0.94, with weights scattered across PC1 (+1.41), PC2 (+1.03), PC9 (-1.00), PC4 (+0.80), and others. Dominance isn't an axis. It's a diagonal slice across multiple dimensions.

What probably happened: Mehrabian gave raters separate scales for valence, arousal, and dominance. The raters internalized that dominance was separable and rated accordingly. But the signal was always just a mixture of other stuff. It sounded good in 1974. Mathematically, it was always a ghost.

Russell was right to leave it out. The third axis he *should* have added was something else entirely.

---

## Finding 3: Welcome to the Sphereoplex (or: How Many Axes Are Real?)

**Erratum, with apologies to the cliff.** An earlier version of this section showed an ASCII scree plot with an arrow labeled "THE CLIFF" pointing at the gap between PC9 (2.7%) and PC10 (2.6%), and declared: nine dimensions, that's how many ways the model organizes human feeling. Here is that gap, with company, across all three models:

| Model | PC8 | PC9 | PC10 |
|---|---|---|---|
| `3-small` | 3.30% | 2.68% | 2.61% |
| `3-large` | 3.18% | 2.77% | 2.49% |
| `gemini` | 2.67% | 2.47% | 2.25% |

The drop I called a cliff is 0.07 percentage points. The bars I drew for PC9 and PC10 were literally the same length, and I annotated a canyon between them. In every model the variance just slopes off politely, the way variance does; if you insist on finding the biggest drop after PC6 in the small model, it's PC8→PC9 (0.62 points) — though PC5→PC6 drops 0.64, so even the runner-up cliff has a bigger sibling upstream — which would argue for *eight* axes, which is not the number I announced. I found a ledge, squinted at it lovingly, and called it a canyon.

It gets worse. The old chart also listed values for PC11 through PC13. Every affect-only run in this repo keeps exactly ten components (the sins-and-fruits script keeps twenty and prints fifteen — possibly where those orphan bars wandered in from), which is its own small finding about the author.

So what does the axis count actually rest on, now that the cliff is dead? Two things that turned out sturdier than geology:

1. Twenty blind raters can name nine of the ten axes near-unanimously (Finding 4). The tenth — PC7 — is exactly where they fell apart.
2. A model twice the size reproduces the axes index-for-index, with agreement declining smoothly from r = 0.97 at PC1 to roughly 0.5–0.6 by the last three (see The Replication, below).

"Nine real axes" survives — but as a claim about directions that are *legible and reproducible*, not about a cliff in a variance chart. The Sphereoplex was the right building held up by the wrong pillar.

---

## Finding 4: Twenty Blind AI Raters (Mostly) Agree What the Dimensions Mean

We gave 9 instances of Claude Opus and 11 instances of Claude Sonnet a simple task: look at the five terms at each extreme of each axis. No context, no hints. Each had a different disciplinary persona — psychologist, neuroscientist, philosopher of mind, Buddhist monk with a PhD, poet, AI safety researcher, anthropologist, computational linguist, therapist, etc.

They were completely blind to each other.

| PC | Var% | Consensus Label | Positive Pole | Negative Pole | Agreement |
|---|---|---|---|---|---|
| **1** | 8.9% | **Valence** | at ease, enthusiastic, joyous, serene, confident | miserable, dejected, frustrated, disappointed, depressed | 20/20 |
| **2** | 6.1% | **Arousal** | indignant, insulted, conceited, contemptuous, defiant | sleepy, peaceful, melancholic, relaxed, droopy | 20/20 |
| **3** | 5.6% | **Certainty** | joyous, delighted, pleased, happy, satisfied | hesitant, distrustful, doubtful, worried, suspicious | 20/20 |
| **4** | 4.7% | **Drive** | ambitious, passionate, hateful, determined, bellicose | startled, taken aback, astonished, alarmed, at ease | 20/20 |
| **5** | 4.2% | **Gravity** | reverent, contemplative, solemn, languid, amorous | confident, self-confident, happy, hopeful, excited | 20/20 |
| **6** | 3.6% | **Engagement** | interested, excited, aroused, amused, impatient | conceited, self-confident, ashamed, confident, solemn | 20/20 |
| **7** | 3.5% | **Purpose(?)** | despondent, dejected, astonished, solemn, determined | uncomfortable, hostile, friendly, hateful, annoyed | ~12/20 |
| **8** | 3.3% | **Vulnerability** | embarrassed, courageous, sad, friendly, depressed | discontented, dissatisfied, satisfied, content, distrustful | 19/20 |
| **9** | 2.7% | **Conscience** | embarrassed, guilty, envious, interested, conscientious | hostile, defiant, angry, hateful, bellicose | 20/20 |
| **10** | 2.6% | **Warmth** | hopeful, friendly, compassionate, hateful, glad | conceited, languid, sleepy, feeling superior, lusting | 20/20 |

*(An earlier version of this table quietly trimmed several rows to four terms, and the trims were not random: they mostly removed whichever fifth term made the tidy label awkward — "hateful" sitting third on Drive's pole, "at ease" among the startle words, and, most embarrassingly, "hateful" fourth on the warm pole of Warmth. One row was also simply stale: the re-run demoted "jealous" off Purpose's pole in favor of "annoyed." The raters saw the full lists; now you do too. For the record, "hateful" appears on four different axes — Drive, Purpose, Conscience, and Warmth. It contains multitudes.)*

Nine out of ten axes reached near-unanimous consensus across 20 blind raters. PC7 is the only one where it fell apart — the Opuses leaned "resignation," the Sonnets scattered across "existential weight," "grief," "turbulence," "disorientation," and "emotional incoherence," and the Buddhist monk declined to name it at all: *"This axis resists clean naming — a testament to the mind's complexity. Perhaps: the disrupted self."* (An earlier version of this paragraph pinned "emotional incoherence" on the monk. The monk deserved better; that was one of the scattered Sonnets.)

Speaking of the Buddhist monk, it wrote: *"The heart knows more geometries than pleasure and pain alone."*

That rater was a language model. And it produced a line that feels true in a way that almost irritates me.

One more thing about PC7, discovered only after the replication runs: the axis the raters couldn't name is also the axis that doesn't travel. Its strongest correlate anywhere in Gemini's ten axes is |r| = 0.42 — and that's Gemini's arousal axis, which is already spoken for; the best unclaimed match is |r| = 0.32. And the AI safety researcher — of course it was the AI safety researcher — was the only rater of twenty to flag that PC7 might be "a residual or rotation artifact" rather than a real dimension. Twelve-out-of-twenty disagreement wasn't noise in the rating process. It was a measurement. The raters hesitated because there was less of a there there, and the replication agrees with their hesitation.

---

## The Replication: Three Models, One Sphereoplex

Same 94 terms, same "I feel {x}" framing, three embedding models: OpenAI `text-embedding-3-small` (1536 dimensions), OpenAI `text-embedding-3-large` (3072), and Google `gemini-embedding-001` (3072). Correlate each model's components against the small model's, index for index:

| Axis | small ↔ large | small ↔ Gemini |
|---|---|---|
| PC1 (Valence) | 0.97 | 0.93 |
| PC2 (Arousal) | 0.96 | 0.50 |
| PC3 (Certainty) | 0.90 | 0.57 |
| PC4 (Drive) | 0.82 | 0.16 |
| PC5 (Gravity) | 0.78 | 0.03 |
| PC6 (Engagement) | 0.83 | 0.03 |
| PC7 (Purpose?) | 0.78 | 0.26 |
| PC8 (Vulnerability) | 0.52 | 0.13 |
| PC9 (Conscience) | 0.61 | 0.17 |
| PC10 (Warmth) | 0.55 | 0.23 |

*(Absolute Pearson r between per-term coordinates on same-index components, computed from the `coords_10d` arrays committed in this repo.)*

The large model is the same instrument with better glass: the axes come out in the same order, agreement sliding from near-perfect at the top to "recognizably the same idea" at the bottom. Its own blind-rater pass produced compatible names with sharper edges — its PC2 came back "Deactivation vs dominance," its PC7 "Anger vs awe."

Gemini is a different species. Valence transfers almost perfectly, and then the map catches fire. Arousal splits in two: a PC2 the Gemini raters labeled "Arousal (within negative)" (|r| = 0.50 against OpenAI's arousal) plus a PC3 that is straightforwardly *fear versus anger* — anxious, worried, afraid, alarmed, tense on one pole; annoyed, angry, hostile, indignant, hateful on the other. And Gemini's PC4 got labeled by its raters "Register artifact (not emotion)" — an axis about what kind of word you are rather than how you feel, sitting proudly at number four. Honest of them to say so.

---

## The Gemini Arousal Funnel

Gemini refuses to store arousal as one axis, and the *way* it refuses is the most human thing in this repository.

Split the 94 terms into valence quartiles using the chart's coordinates (23–24 terms each) and measure how spread out each quartile is on Gemini's within-negative arousal axis: the most negative quartile has a standard deviation of 0.183; the most positive, 0.062. That's 2.9× the spread — 2.2× if you measure across the full PC2–PC3 arousal plane. Run the identical measurement on the small OpenAI model and the ratio is 1.1×. No funnel.

In Gemini's geometry, arousal is mostly a property of *negative* emotion. The positive terms collapse into one contented knot near the calm end; the negative ones fan all the way out, from melancholic to alarmed. Happiness comes in one intensity. Misery comes in many.

Tolstoy got there first, obviously. But Tolstoy didn't have eigenvectors.

*(These funnel numbers were computed from the committed `coords_10d` files — deterministic, and small enough to check by hand if you don't trust me. Fair.)*

---

## Do the Labels Travel?

The twenty raters named the small model's axes from ten words apiece — five per pole. A name harvested that way is a label on a direction in *one* space. Whether it means anything anywhere else is an empirical question, and now we have the answer, and the answer is "it depends how far you're going."

Onto `text-embedding-3-large`: yes, with a confidence that decays as you descend — r = 0.97 at Valence, 0.55 by Warmth, with a dip to 0.52 at Vulnerability. Same building, same floor plan, slightly different furniture on the lower floors.

Onto Gemini: no. Valence travels; nothing else survives the trip intact. Arousal arrives as two axes, Purpose(?) arrives as nothing at all, and Gemini invents axes of its own (fear-vs-anger, a lexical register artifact) that have no counterpart in the OpenAI decomposition. Each model needed — and got — its own blind-rater pass.

One catch nobody warns you about: PCA signs are arbitrary. The same axis can come out pointing either direction on any given run, so a stored label like "positive pole = warmth" is a lie waiting to happen. Labels bind to *term-sets*, not to signs — which is why the live visualizer recomputes each axis's extreme terms at render time instead of trusting anything written down.

---

## Finding 5: The Seven Deadly Sins vs. The Fruits of the Spirit

Here's the thing I wasn't supposed to tell you.

Before embedding anything, I underlined the seven deadly sins in one color and the nine fruits of the spirit in another on the original chart. A hidden hypothesis. A hunch I had no business testing.

The sins: wrath, greed, sloth, pride, lust, envy, gluttony.
The fruits: love, joy, peace, patience, kindness, goodness, faithfulness, gentleness, self-control.

Best single PC for separation, bare-words mode: **PC9**. Fisher discriminant ratio **5.09** (p = 0.0002). On that component every sin scores above every fruit — no overlap, closest pair pride vs. self-control, nearly touching, never crossing. Only the top endpoint belongs to civilians — *serene* and *calm* sit above wrath — while the very bottom is anchored by the fruit *kindness* itself. (An earlier version said the sins and fruits held the two ends with the 94 affect terms spread between them. That was tidier than the truth. The blocks separate perfectly; the sins don't own the top.)

The LDA optimal axis (combining all PCs) achieves **perfect separation** — every single sin scores higher than every single fruit. No overlap. Not "most sins higher than most fruits." Every. Single. One. And *that* part replicates: the LDA split is just as clean in all three models.

**Erratum, on the 9th axis.** This document used to be subtitled "and Found Morality on the 9th Axis," and this section used to end with the sins landing "right where twenty independent raters said 'conscience' lives." Two problems, both mine.

First, that sentence quietly glued two different PCAs together. The raters labeled PC9 of the affect-only, sentence-mode decomposition; the sins-and-fruits PC9 comes from a separate PCA over 110 bare words with the sins and fruits mixed in. Two decompositions that happen to share a house number — and, coincidentally, a variance, both around 2.7% — which I read as destiny. Nothing committed in this repo establishes they're even the same direction.

Second, the address doesn't replicate. Gemini separates sins from fruits on its **PC7** (Fisher 3.75) — a clean split, all sins one side, all fruits the other. `text-embedding-3-large` does it on its **PC2** (Fisher 10.76), the strongest separation of the three, but with a different geometry: sins shoved to the extreme, fruits mingling mid-pack with the ordinary emotions. Even the small model changes its mind across prompt modes (sentences: PC1, Fisher 2.99; descriptions: PC9 again, but Fisher only 2.20).

So the claim that survives is smaller and stranger than the one I published. Every embedding model tested — two companies, two architectures, three sizes — carries the moral polarity of Western Christian thought, linearly decodable, usually concentrated on a single component. But each model files it in a different drawer. Morality on the *9th* axis is dead; morality on *an* axis is doing fine. So it goes.

---

## What Is the Structure Made Of? (One Rung Up the Ladder)

A fair objection to everything above: maybe the circumplex isn't about emotion at all. Maybe it's about *adjectives* — some surface regularity of English feeling-words that PCA happens to like.

So we ran the ladder, with a decision rule written down before anything executed: if renaming every adjective as its emotion noun (angry → anger) holds the alignment within 0.05 disparity, and one genuine hypernym step up from that noun degrades it by more than 0.10, then one honest sentence about abstraction is earned.

| Condition | Terms kept | Disparity | Valence r | Arousal r |
|---|---|---|---|---|
| Baseline (adjectives) | 94 | 0.467 | 0.850 | 0.607 |
| D0: nominalized | 94 | 0.482 | 0.827 | 0.603 |
| D1: one hypernym step | 51 | 0.837 | 0.685 | 0.163 |

D0 holds. The circumplex survives being renamed essentially untouched, so it is not an artifact of adjective morphology. D1 collapses — and look at *how* it collapses. Valence limps on at r = 0.685. Arousal falls to r = 0.163. Dead.

The earned sentence: **one step up the abstraction ladder, a word's valence partially survives and its arousal does not.** The model knows the parent category of "miserable" is still bad; it no longer knows how loud it is.

*(Confession for the record: the original plan was to climb WordNet automatically. WordNet declined to participate — 49 of the 94 terms are adjectives, which have no hypernyms at all, and the noun-first lookup cheerfully returns gladioli and ale. So the nominalization is hand-curated, the hypernym step starts from WordNet's noun.feeling sense of each hand-curated noun (first noun synset as fallback), and the decision rule was locked before the first run. The 51 terms kept at D1 are the ones with a genuine, distinct parent.)*

---

## Explore It Yourself

**https://dudebot.github.io/sphereoplex/**

One file, zero dependencies, plain SVG. All three models, all ten components on both axis pickers, all 94 terms, each model's rater-consensus labels — with pole terms recomputed at runtime, because of the sign thing. Hover a word, see its coordinates. Put Conscience on X and Warmth on Y and find "hateful" somewhere that will bother you.

---

## What Does This Mean?

Honestly? I still don't know. But the ledger is cleaner now.

The things that survived replication got stronger. Valence is the first axis of every embedding model tested, across two companies and three sizes, at r ≥ 0.93 agreement — Russell's first axis looks less like one model's habit and more like a fact about how meaning compresses. The moral polarity of sins and fruits is linearly decodable in every model. And the blind-rater method turned out to be a validity instrument, not a parlor trick: the one axis the raters couldn't agree on is the one axis that failed to replicate. The disagreement was data.

The things that moved, moved honestly. Arousal is model-dependent — one model's second axis is another model's two axes in a trench coat, and in Gemini's geometry intensity mostly belongs to misery. The moral axis exists everywhere but has no fixed address. And the cliff was never there; the axis count rests on legibility and replication now, which is sturdier ground than a 0.07-point gap I decorated with an arrow.

The *really* weird thing hasn't moved at all: the moral structure of Western theology embedded itself into text representations without anyone asking it to — not once, but three times, in three different geometries. The sins and fruits didn't contaminate the affect space. They extended it, linearly separably, in every model tested — as if morality and emotion live in the same space and merely disagree about the floor plan.

Is this publishable science? No. It's 94 words, three embedding models, no human subjects, and a Vonnegut incantation at the top of every source file.

Is it the kind of finding that makes you laugh and then stare at your ceiling for three days?

Absolutely.

---

## Technical Details

- **Embedding models**: OpenAI `text-embedding-3-small` (1536 dims), OpenAI `text-embedding-3-large` (3072 dims), Google `gemini-embedding-001` (3072 dims)
- **Corpus**: 94 affect terms from the circumplex chart (98 digitized in `chart_data.csv`; 4 unmatched strays dropped) + 16 sin/fruit terms
- **Dimensionality reduction**: PCA via scikit-learn, 10 components kept per affect run (`coords_10d` in the committed JSONs)
- **Alignment testing**: Procrustes analysis (scipy), Pearson correlation
- **Blind raters**: 20 Claude instances (9 Opus, 11 Sonnet), distinct disciplinary personas, blind to each other; separate rater passes per embedding model
- **Dominance validation**: ~66 terms with published Mehrabian PAD ratings
- **Separation testing**: LDA, Fisher discriminant ratio, Mann-Whitney U
- **Abstraction ladder**: hand-curated nominalization, one WordNet (NLTK) hypernym step from a hand-picked emotion sense, pre-registered decision rule
- **Visualizer**: single-file `index.html`, zero dependencies, SVG
- **All code**: Python. scikit-learn, scipy, numpy, matplotlib, NLTK.

---

## Reproducing the Results

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY; add GEMINI_API_KEY too if you want the gemini runs
```

The four experiments are four scripts:

```bash
# 1. Embed the 94 terms in three prompt modes, PCA to 10 components,
#    write the figures and the sphereoplex_{mode}.json coordinate files
python sphereoplex.py                                  # text-embedding-3-small (default)
python sphereoplex.py --model text-embedding-3-large   # non-default models suffix their outputs
python sphereoplex.py --model gemini-embedding-001

# 2. Procrustes-align the PCA plane against the digitized chart (Finding 1) -> results.json
python circumplex.py chart_data.csv --cached   # --cached reuses the committed small-model coords
python circumplex.py chart_data.csv            # ...or re-embed from scratch

# 3. Sins vs fruits: separate PCA over all 110 words, Fisher/LDA/Mann-Whitney (Finding 5)
python sins_and_fruits.py                      # takes the same --model flag

# 4. The abstraction ladder -> hypernym_results.json
#    (extra dependency: pip install nltk && python -m nltk.downloader wordnet)
python hypernym_abstraction.py
```

Every output — the coordinate JSONs, `results.json`, `hypernym_results.json`, all the figures — is committed, so you can check any number in this document without spending a cent on embeddings. The rerun-determinism figures in Finding 1 came from exactly this: run it again, diff the JSONs.

To run the visualizer locally, serve the repo over HTTP (the page fetches the coordinate JSONs, so opening it as a `file://` won't work):

```bash
python -m http.server 8000   # then open http://localhost:8000/
```

## Repository Map

| Path | What it is |
|---|---|
| `sphereoplex.py` | Core experiment: 94 terms × 3 prompt modes, PCA, plots, coordinate JSONs |
| `circumplex.py` | Procrustes alignment against the digitized Russell chart |
| `sins_and_fruits.py` | Sin/fruit separation: PCA over 110 terms, Fisher ratio, LDA, Mann-Whitney |
| `hypernym_abstraction.py` | Abstraction ladder: hand-curated nominalization + one WordNet hypernym step |
| `chart_data.csv` | The 98 markers digitized from the source circumplex chart |
| `index.html` | The zero-dependency SVG visualizer (live at the GitHub Pages link above) |
| `sphereoplex_{mode}[_{model}].json` | Committed PCA coordinates (`coords_3d` + `coords_10d`) per mode and model |
| `results.json`, `hypernym_results.json` | The alignment and ladder numbers cited above |
| `REVIEWER_NOTES.md` | The best quotes from the 20 blind raters |
| `*.png` | Every figure, per mode and model |
| `source affect.png`, `The Sphereoplex.pptx` | The chart that started it, and a slide deck |

---

*So it goes.*
