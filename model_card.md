# Model Card: Mood Machine

This model card documents the **Mood Machine** project, a short-text mood classifier
built during the AI 110 "Mood Machine" lab. The repository contains two versions:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

This card primarily documents the **rule-based model**, which is the version that was
fully implemented, with a comparison section for the ML extension.

---

## 1. Model Overview

**Model type:**
Rule-based lexicon classifier (primary), with an optional scikit-learn ML model for comparison.

**Intended purpose:**
Classify short, social-media-style text posts into one of four moods: `positive`,
`negative`, `neutral`, or `mixed`. The system is a teaching artifact meant to expose how
simple systems interpret language and where they break, not a production sentiment tool.

**How it works (brief):**
The rule-based model tokenizes a post, then scores it by comparing tokens against curated
positive and negative word lists. Positive words add to the score, negative words subtract.
The final numeric score is mapped to a mood label. Two enhancements go beyond the starter:
simple negation handling (a negation word flips the sentiment of a nearby word) and a
`mixed` label for posts that carry both positive and negative signals that cancel out.

---

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` contains **16 posts total**: the 6 starter posts plus 10 that were added
to cover more realistic language. `TRUE_LABELS` holds one human label per post, kept aligned
one-to-one (`len(SAMPLE_POSTS) == len(TRUE_LABELS)`).

**Labeling process:**
Labels were assigned by reading each post and choosing the closest of the four allowed
labels. For clearly-worded posts this was straightforward. For ambiguous posts, the label
reflects the most likely intended tone rather than the literal words. Several posts were
deliberately chosen to be hard to label so they could act as edge cases in evaluation.

**Posts that were hard to label / could reasonably differ:**
- `"This is fine"` — labeled `neutral`, but reads as sarcastic in many real contexts.
- `"I guess it's fine, whatever"` — labeled `neutral`, though the tone leans resigned/negative.
- `"so tired but also kinda excited??"` — labeled `mixed`; a reader could argue positive.

**Important characteristics of the dataset:**
- Contains slang ("lowkey", "highkey", "no cap", "ngl", "vibing")
- Contains emojis (🙃, 💀, 😂, 🥳)
- Contains sarcasm ("i absolutely love getting stuck in traffic 🙃")
- Contains genuinely mixed emotions
- All posts are short (one line) and written in casual, English, Gen-Z-leaning register

**Possible issues with the dataset:**
- **Small size (16 posts):** far too few to train or evaluate a model reliably.
- **Register skew:** heavily informal English slang; underrepresents formal writing,
  other dialects, and non-English language.
- **Label subjectivity:** several posts have defensible alternative labels.
- **Class balance:** the four labels are not evenly represented.

---

## 3. How the Rule-Based Model Works

**Preprocessing (`preprocess`):**
- Strips leading/trailing whitespace and lowercases the text.
- Removes punctuation but **preserves apostrophes** (so `don't` stays one token) and
  **preserves emoji characters** instead of discarding them.
- Splits on whitespace into tokens.

**Scoring rules (`score_text` / shared `_analyze_tokens` helper):**
- Score starts at 0.
- Each token found in `POSITIVE_WORDS` adds +1; each token in `NEGATIVE_WORDS` subtracts 1.
- **Negation handling:** if a sentiment word is preceded within 2 tokens by a negation word
  ("not", "no", "never", "don't", "can't", etc.), its contribution is flipped. So
  `"I am not happy about this"` scores negative instead of positive.

**Label thresholds (`predict_label`):**
- `score > 0` → `positive`
- `score < 0` → `negative`
- `score == 0` with **both** positive and negative hits → `mixed`
- `score == 0` with **no** sentiment hits → `neutral`

**Strengths of this approach:**
- Predictable and fully explainable — every decision traces to specific words (the `explain`
  method reports exact positive/negative hits and the final score).
- Handles clearly-worded, in-vocabulary posts well.
- Negation handling correctly resolves a common failure ("not happy", "not bad").

**Weaknesses of this approach:**
- **Sarcasm:** a single strong keyword dominates (see Section 5).
- **Out-of-vocabulary slang:** words like "sick", "fire", "vibing" aren't in the lists, so
  they contribute nothing.
- **Emojis:** although preserved through preprocessing, they carry no score because they
  aren't in the word lists, so 💀 or 😂 have no effect on the label.
- **Mixed emotions with out-of-vocab words:** posts like "tired but hopeful" may score
  wrong because only some of the emotional words are in the lists.

---

## 4. How the ML Model Works

> **To complete from your actual run of `python ml_experiments.py`.** The description below
> reflects the intended design per the repo; confirm the specifics against the code and
> replace the bracketed notes with what you observe.

**Features used:**
Bag-of-words representation (scikit-learn `CountVectorizer`) built from the posts in
`SAMPLE_POSTS`.

**Training data:**
Trained on `SAMPLE_POSTS` with `TRUE_LABELS` as targets — the same 16-post dataset the
rule-based model is evaluated on.

**Training behavior:**
`[Fill in: what happened to accuracy/predictions when you added posts or changed labels?]`

**Strengths and weaknesses:**
- *Strengths:* learns associations automatically from labels, so it can pick up on words
  the rule lists don't contain (e.g., slang) if they appear in training.
- *Weaknesses:* with only 16 examples it will overfit heavily, likely memorizing the
  training posts and generalizing poorly. It may also latch onto spurious cues (a single
  distinctive word) rather than meaning.

---

## 5. Evaluation

**How the model was evaluated:**
Both models are evaluated on the labeled posts in `dataset.py` (predicted label vs. true
label, reported as accuracy). Note this is **training-set accuracy**, not a held-out test
set, so it overstates real-world performance — especially for the ML model.

**Rule-based accuracy:**
`[Fill in from python main.py — e.g. "X / 16 correct = XX%"]`

**ML accuracy:**
`[Fill in from python ml_experiments.py]`

**Examples of correct predictions (rule-based):**
- `"I love this class so much"` → `positive`. "love" is a positive word, no negation. Correct.
- `"Today was a terrible day"` → `negative`. "terrible" is a negative word. Correct.
- `"I am not happy about this"` → `negative`. Negation flips "happy". Correct — this one
  would fail without the negation enhancement.

**Examples of incorrect predictions (rule-based):**
- `"i absolutely love getting stuck in traffic 🙃"` → predicted **positive**, true label
  **negative**. The model sees "love" (+1), the emoji and sarcasm carry no weight, so one
  keyword decides the label. Classic single-keyword-dominates failure.
- `"ngl kinda over this week 💀"` → predicted **neutral**, true label **negative**. None of
  "ngl", "over", or 💀 are in the word lists, so the score is 0. The slang and emoji that
  carry all the meaning are invisible to the model.

**How the two models' failures differ:**
`[Fill in after running both — does the ML model get the slang/sarcasm cases the rules miss,
or does it fail on different posts? Note whether its mistakes look "memorized" vs. "reasoned".]`

---

## 6. Limitations

- **Very small dataset (16 posts)** — not enough to evaluate reliably or generalize.
- **Cannot detect sarcasm** — demonstrated by the "stuck in traffic" example above.
- **Vocabulary-bound** — the rule-based model only understands words on its lists; unfamiliar
  slang scores as neutral.
- **Emojis ignored** — preserved in tokens but never scored.
- **Training-accuracy only** — no held-out test set, so reported numbers are optimistic.
- **Register-specific** — tuned to short, casual, English posts; likely misreads formal
  writing and other dialects/languages.

---

## 7. Ethical Considerations

- **Misclassifying distress:** a post expressing real distress could be scored neutral or
  even positive (e.g., sarcasm), which would be harmful if such a system were used to route
  support or flag risk.
- **Dialect and community bias:** the word lists and dataset encode one narrow slice of
  English. Communities using different slang, dialects, or languages would be
  systematically misread.
- **Privacy:** running mood detection over personal messages raises consent and
  surveillance concerns, independent of accuracy.
- **Overconfidence in a simple system:** the numeric score can look authoritative even when
  the underlying reasoning is a single keyword match.

---

## 8. Ideas for Improvement

- Add substantially more labeled data across more registers and languages.
- Use a proper **held-out test set** instead of reporting training accuracy.
- Add emoji-to-sentiment mapping so emojis actually affect the score.
- Expand slang coverage in the word lists (while noting this scales poorly).
- Use TF-IDF instead of raw counts for the ML model.
- Move to a small neural network or a pretrained transformer for context and sarcasm.
- Add confidence/uncertainty output so borderline `mixed`/`neutral` cases are flagged rather
  than forced into a single label.