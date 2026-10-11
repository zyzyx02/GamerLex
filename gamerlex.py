# -*- coding: utf-8 -*-
"""GamerLex helper: use the lexicon together with VADER.

    from gamerlex import GamerLexAnalyzer
    gl = GamerLexAnalyzer("GamerLex_v1.0.csv")           # all four tiers
    gl.polarity_scores("gg, the boss fights slap but the netcode is janky")

    gl = GamerLexAnalyzer("GamerLex_v1.0.csv", tiers=CURATED)     # hand-audited tiers only
    gl = GamerLexAnalyzer("GamerLex_v1.0.csv", tiers=("core",))   # one tier only

Requires: vaderSentiment.
"""
import csv
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


CURATED = ("core", "extended", "vader_override")
ALL_TIERS = CURATED + ("auto",)


class GamerLexAnalyzer:
    def __init__(self, path="GamerLex_v1.0.csv", tiers=ALL_TIERS, use_vader=True):
        self.entries = {}
        multi = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["tier"] not in tiers:
                    continue
                term = row["term"]
                self.entries[term.replace(" ", "_")] = float(row["valence"])
                if " " in term:
                    multi.append(term)
        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon = dict(self.analyzer.lexicon) if use_vader else {}
        self.analyzer.lexicon.update(self.entries)
        self._ws = re.compile(r"\s+")
        if multi:
            parts = sorted(multi, key=len, reverse=True)
            pat = "|".join(r"\s+".join(re.escape(w) for w in t.split()) for t in parts)
            self._rx = re.compile(r"(?<![A-Za-z0-9_])(?:" + pat + r")(?![A-Za-z0-9_])", re.IGNORECASE)
        else:
            self._rx = None

    def join_multiword(self, text):
        """Joins multi-word GamerLex terms with underscores so VADER sees them as one token."""
        if self._rx is None:
            return text
        return self._rx.sub(lambda m: self._ws.sub("_", m.group(0)), text)

    def polarity_scores(self, text):
        return self.analyzer.polarity_scores(self.join_multiword(text))
