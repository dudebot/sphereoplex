# The Sphereoplex: How a Language Model Accidentally Mapped the Geometry of Human Feeling (and Found Morality on the 9th Axis)

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

## The Setup

Here's the thing about embedding models: they're dumb in the most interesting way possible. You feed them words, they spit back 1536 numbers, and those numbers are supposed to contain *meaning*. Somehow. It works, but we don't really know why, and that's where things get fun.

I grabbed ~94 emotional terms from a classic circumplex model of affect chart — the kind of thing that's been gathering dust in psychology textbooks since Russell first drew it in 1980. You know the one: a circle with "valence" on one axis (good to bad) and "arousal" on another (calm to frantic). It's been the default way we think about emotions for forty years.

Then I did something beautifully stupid: I embedded all 94 terms using OpenAI's `text-embedding-3-small` model and ran PCA on the resulting vectors.

What fell out should not have been this clean.

---

## Finding 1: Russell's Circumplex Emerges Spontaneously from the Embedding Space

When you take 1536-dimensional embeddings and say "show me the two most important directions," those two directions turn out to be valence and arousal. Not something random, not word length, not vibes — the two axes a psychologist drew on a whiteboard in 1980.

| Mode | Procrustes Disparity | Valence r | Arousal r |
|---|---|---|---|
| **bare words** | 0.248 | 0.939 (p≈0) | 0.789 (p≈0) |
| **"I feel {x}"** | 0.209 | 0.961 (p≈0) | 0.809 (p≈0) |
| **"The emotional state of feeling {x}"** | 0.140 | 0.956 (p≈0) | 0.902 (p≈0) |

The richer the sentence framing, the cleaner the alignment. PC1 is valence. PC2 is arousal. The model *knows*.

Procrustes disparity of 0.140 means that when you optimally overlay the model's output onto Russell's original chart, they're about 86% the same shape. For "I literally just dumped words into an API and hit PCA," that's kind of ridiculous.

The kicker: the first two PCs explain only ~28% of total variance. The model knows *way* more about these words than just their emotional content — but when you ask "what's the single most important thing differentiating these emotion words?" the answer is valence, and the second most important thing is arousal. Exactly what Russell said.

---

## Finding 2: Mehrabian's Dominance Is Not a Real Axis

Mehrabian & Russell (1974) proposed a third dimension: *Dominance* — the feeling of being in control vs. controlled. It made intuitive sense. Textbooks mentioned it.

We tested dominance against our PCA axes using published ratings for ~66 emotion terms.

Result: Dominance correlates most with PC1 (valence) at r = 0.74. That's not independent. That's basically the same thing with a different name.

When we regressed dominance onto all 10 PCs, we got r = 0.94, with weights scattered across PC1 (+1.41), PC2 (+1.03), PC9 (-1.00), PC4 (+0.80), and others. Dominance isn't an axis. It's a diagonal slice across multiple dimensions.

What probably happened: Mehrabian gave raters separate scales for valence, arousal, and dominance. The raters internalized that dominance was separable and rated accordingly. But the signal was always just a mixture of other stuff. It sounded good in 1974. Mathematically, it was always a ghost.

Russell was right to leave it out. The third axis he *should* have added was something else entirely.

---

## Finding 3: Welcome to the Sphereoplex (Nine Real Axes)

Run a scree plot on the PCA variance:

```
PC1:  8.9%  ████████████████████████████████████████████
PC2:  6.1%  ██████████████████████████████
PC3:  5.6%  ████████████████████████████
PC4:  4.7%  ███████████████████████
PC5:  4.2%  █████████████████████
PC6:  3.6%  ██████████████████
PC7:  3.5%  ██████████████████
PC8:  3.3%  █████████████████
PC9:  2.7%  ██████████████
                ↕ THE CLIFF
PC10: 2.6%  ██████████████
PC11: 2.4%  ████████████
PC12: 2.3%  ████████████
PC13: 2.3%  ████████████
     ...flattens into noise...
```

There's a cliff between PC9 and PC10. Beyond it, variance flatlines. Nine dimensions. That's how many ways the embedding model has learned to organize human feeling.

---

## Finding 4: Twenty Blind AI Raters All Agree What the Dimensions Mean

We gave 9 instances of Claude Opus and 11 instances of Claude Sonnet a simple task: look at the five terms at each extreme of each axis. No context, no hints. Each had a different disciplinary persona — psychologist, neuroscientist, philosopher of mind, Buddhist monk with a PhD, poet, AI safety researcher, anthropologist, computational linguist, therapist, etc.

They were completely blind to each other.

| PC | Var% | Consensus Label | Positive Pole | Negative Pole | Agreement |
|---|---|---|---|---|---|
| **1** | 8.9% | **Valence** | at ease, enthusiastic, joyous, serene, confident | miserable, dejected, frustrated, disappointed, depressed | 20/20 |
| **2** | 6.1% | **Arousal** | indignant, insulted, conceited, contemptuous, defiant | sleepy, peaceful, melancholic, relaxed, droopy | 20/20 |
| **3** | 5.6% | **Certainty** | joyous, delighted, pleased, happy, satisfied | hesitant, distrustful, doubtful, worried, suspicious | 20/20 |
| **4** | 4.7% | **Drive** | ambitious, passionate, determined, bellicose | startled, taken aback, astonished, alarmed | 20/20 |
| **5** | 4.2% | **Gravity** | reverent, contemplative, solemn, languid | confident, self-confident, happy, hopeful, excited | 20/20 |
| **6** | 3.6% | **Engagement** | interested, excited, aroused, amused, impatient | conceited, self-confident, ashamed, solemn | 20/20 |
| **7** | 3.5% | **Purpose(?)** | despondent, dejected, astonished, determined, solemn | uncomfortable, hostile, friendly, hateful, jealous | ~12/20 |
| **8** | 3.3% | **Vulnerability** | embarrassed, sad, courageous, friendly, depressed | discontented, dissatisfied, satisfied, content | 19/20 |
| **9** | 2.7% | **Conscience** | embarrassed, interested, guilty, envious, conscientious | hostile, defiant, angry, hateful, bellicose | 20/20 |
| **10** | 2.6% | **Warmth** | hopeful, compassionate, friendly, glad | conceited, languid, sleepy, feeling superior, lusting | 20/20 |

Nine out of ten axes reached near-unanimous consensus across 20 blind raters. PC7 is the only one where it fell apart — the Opuses called it "resignation," the Sonnets scattered across "existential weight," "grief," "turbulence," and the Buddhist monk straight up called it "emotional incoherence."

Speaking of the Buddhist monk, it wrote: *"The heart knows more geometries than pleasure and pain alone."*

That rater was a language model. And it produced a line that feels true in a way that almost irritates me.

---

## Finding 5: The Seven Deadly Sins vs. The Fruits of the Spirit

Here's the thing I wasn't supposed to tell you.

Before embedding anything, I underlined the seven deadly sins in one color and the nine fruits of the spirit in another on the original chart. A hidden hypothesis. A hunch I had no business testing.

The sins: wrath, greed, sloth, pride, lust, envy, gluttony.
The fruits: love, joy, peace, patience, kindness, goodness, faithfulness, gentleness, self-control.

Best single PC for separation: **PC9** — the one labeled "Conscience" by the blind raters.

Fisher discriminant ratio: **5.09** (p = 0.0002)

On the PC9 ranking, ALL 9 fruits cluster at one end and ALL 7 sins cluster at the other, with the 94 affect terms spread between them.

The LDA optimal axis (combining all PCs) achieves **perfect separation** — every single sin scores higher than every single fruit. No overlap. Not "most sins higher than most fruits." Every. Single. One.

The embedding model — trained on internet text, optimized for semantic similarity, knowing nothing about theology — learned the moral structure of Western Christian thought and placed it on the 9th principal component, right where twenty independent raters said "conscience" lives.

That component explains 2.7% of the variance in how the model thinks about emotional words.

---

## What Does This Mean?

Honestly? I don't know yet.

The obvious thing: Russell's two-dimensional model was always an incomplete projection of something higher-dimensional. The embedding space knows there are at least nine meaningful axes in human emotion.

The weird thing: Independent AI raters can recognize emotional dimensions just from seeing the extreme points. The structure is legible. You don't need all 1536 dimensions to understand what's happening.

The *really* weird thing: The moral structure of Western theology embedded itself into a text representation without anyone asking it to. The sins and fruits didn't contaminate the affect space — they extended it perfectly, as if morality and emotion live in the same geometric space.

Is this publishable science? No. It's 94 words, no human subjects, and a Vonnegut incantation in the source code.

Is it the kind of finding that makes you laugh and then stare at your ceiling for three days?

Absolutely.

---

## Technical Details

- **Embedding model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Corpus**: 94 affect terms from circumplex chart + 16 sin/fruit terms
- **Dimensionality reduction**: PCA via scikit-learn
- **Alignment testing**: Procrustes analysis (scipy), Pearson correlation
- **Blind raters**: 20 Claude instances (9 Opus 4.6, 11 Sonnet 4.6), distinct disciplinary personas
- **Dominance validation**: ~66 terms with published Mehrabian PAD ratings
- **Separation testing**: LDA, Fisher discriminant ratio, Mann-Whitney U
- **All code**: Python. Pandas, scikit-learn, scipy, numpy, matplotlib.

---

*So it goes.*
