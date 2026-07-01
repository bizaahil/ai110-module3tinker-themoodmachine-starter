# The Mood Machine

The Mood Machine is a small text classifier that starts with a **rule-based** approach and
can be extended with a **tiny machine learning model**. It guesses whether a short piece of
text sounds **positive**, **negative**, **neutral**, or **mixed** based on the words and
data you give it.

The point of the lab is not to build an accurate sentiment tool — it's to see, hands-on, how
a simple system interprets language, where it breaks, and why data and design choices matter
long before any "AI" is involved.

---

## Repo Structure

```plaintext
├── dataset.py         # Word lists + labeled example posts (expanded during the lab)
├── mood_analyzer.py   # Rule-based classifier: preprocess, score, predict, explain
├── main.py            # Runs the rule-based model + interactive demo, prints an accuracy eval
├── ml_experiments.py  # A tiny ML classifier using scikit-learn (bag-of-words)
├── model_card.md      # Model card documenting behavior, data, limits, and ethics
└── requirements.txt   # Dependencies for the optional ML exploration
```

---

## Getting Started

1. Open this folder in VS Code.
2. Make sure your Python environment is active.
3. Install dependencies:

```bash
    pip install -r requirements.txt
```

4. Run the rule-based model + interactive demo:

```bash
    python main.py
```

   The top of the output shows an automatic evaluation (each post, its predicted label, the
   true label, and an accuracy score), followed by an interactive prompt where you can type
   your own text. Type `quit` or press Enter on an empty line to exit.

   > Note: pressing `Ctrl+C` at the prompt shows a `KeyboardInterrupt` traceback. That is the
   > normal way the interactive loop exits, **not** an error in your code.

5. Try the ML version:

```bash
    python ml_experiments.py
```

   This trains a small scikit-learn classifier on the same labeled posts and lets you compare
   its behavior to the rule-based model.

---

## What This Implementation Does

The rule-based `MoodAnalyzer` in this repo implements:

- **Preprocessing** — lowercasing, whitespace trimming, punctuation stripping while keeping
  apostrophes (so `don't` stays one token) and emoji characters intact.
- **Lexicon scoring** — positive words add to a score, negative words subtract from it.
- **Negation handling** — a negation word ("not", "no", "never", "don't", …) within two
  tokens flips the sentiment of the following word, so `"I am not happy"` reads as negative.
- **A `mixed` label** — when positive and negative signals both appear and cancel to zero,
  the post is labeled `mixed` rather than `neutral`; `neutral` is reserved for posts with no
  sentiment hits at all.
- **Explanations** — `explain()` reports the exact positive/negative hits and final score,
  so every prediction is traceable.

The dataset (`dataset.py`) has been expanded to **16 labeled posts** including slang, emojis,
sarcasm, and genuinely mixed emotions — several chosen specifically to stress-test the model.

---

## What You Will Do

During this lab you will:

- Implement the missing parts of the rule-based `MoodAnalyzer`.
- Add new positive and negative words.
- Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- Stress-test the model, observe incorrect predictions, and reason about *why* they happen.
- Train a tiny ML model and compare its behavior to your rule-based system.
- Complete the model card with your findings on data, behavior, limitations, and improvements.

The goal is to reason about how models behave, how data shapes them, and why even small
design choices matter.

---

## Known Limitations (by design)

This system is intentionally brittle so you can see where rules fail:

- **Sarcasm** is read literally — `"i absolutely love getting stuck in traffic 🙃"` scores
  positive because "love" dominates.
- **Out-of-vocabulary slang** ("sick", "fire", "ngl") contributes nothing to the score.
- **Emojis** are preserved as tokens but carry no sentiment weight.
- **Accuracy** is measured on the training data itself, so the numbers are optimistic.

These are documented more fully in `model_card.md`.

---

## Tips

- Start with preprocessing before touching the scoring rules.
- When debugging, print tokens, scores, and intermediate matches — don't guess.
- Use an AI assistant to brainstorm edge-case posts or to *explain* a failure, but verify
  its reasoning against the actual code and data.
- Examples that mislead or confuse the model teach you the most. Go looking for them.