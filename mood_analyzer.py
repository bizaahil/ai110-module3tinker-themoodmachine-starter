# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the sentiment of a nearby word (simple negation handling).
NEGATION_WORDS = {
    "not", "no", "never",
    "don't", "dont", "doesn't", "doesnt",
    "didn't", "didnt", "can't", "cant",
    "won't", "wont", "isn't", "isnt",
}

# How many tokens back we check for a negation word.
NEGATION_WINDOW = 2


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Improvements over the starter version:
          - Strips punctuation (so "excited??" -> "excited")
          - Keeps apostrophes so contractions like "don't" stay one token
          - Keeps emoji characters intact instead of stripping them
          - Lowercases and splits on whitespace
        """
        cleaned = text.strip().lower()

        # Remove punctuation but keep word characters, apostrophes, whitespace,
        # and common emoji unicode ranges.
        cleaned = re.sub(
            r"[^\w\s'\U0001F300-\U0001FAFF\u2600-\u27BF]",
            " ",
            cleaned,
        )
        tokens = cleaned.split()

        return tokens

    # ---------------------------------------------------------------------
    # Shared scoring helper
    # ---------------------------------------------------------------------

    def _analyze_tokens(self, tokens: List[str]) -> Tuple[int, List[str], List[str]]:
        """
        Core scoring logic shared by score_text, predict_label, and explain.

        Returns:
          (score, positive_hits, negative_hits)

        Handles simple negation: if a sentiment word is preceded within
        NEGATION_WINDOW tokens by a negation word, its contribution flips.
        """
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            window_start = max(0, i - NEGATION_WINDOW)
            is_negated = any(
                t in NEGATION_WORDS for t in tokens[window_start:i]
            )

            if token in self.positive_words:
                if is_negated:
                    score -= 1
                    negative_hits.append(f"not {token}")
                else:
                    score += 1
                    positive_hits.append(token)
            elif token in self.negative_words:
                if is_negated:
                    score += 1
                    positive_hits.append(f"not {token}")
                else:
                    score -= 1
                    negative_hits.append(token)

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score, negative words decrease it.
        Negation (e.g. "not happy") flips the contribution of the word
        that follows it.
        """
        tokens = self.preprocess(text)
        score, _, _ = self._analyze_tokens(tokens)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - score > 0                         -> "positive"
          - score < 0                         -> "negative"
          - score == 0, no sentiment hits      -> "neutral"
          - score == 0, both pos & neg hits    -> "mixed"
        """
        tokens = self.preprocess(text)
        score, positive_hits, negative_hits = self._analyze_tokens(tokens)

        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        elif positive_hits and negative_hits:
            # Score cancelled out, but there were real signals pulling
            # in both directions -> mixed, not neutral.
            return "mixed"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.
        """
        tokens = self.preprocess(text)
        score, positive_hits, negative_hits = self._analyze_tokens(tokens)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )