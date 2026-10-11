# -*- coding: utf-8 -*-
"""GamerLex helper: use the lexicon together with VADER.

In Python:

    from gamerlex import GamerLexAnalyzer
    gl = GamerLexAnalyzer()                                  # all four tiers
    gl.polarity_scores("gg, the boss fights slap but the netcode is janky")
    gl.matches("gg, the boss fights slap but the netcode is janky")

    gl = GamerLexAnalyzer(tiers=CURATED)                     # hand-audited tiers only
    gl = GamerLexAnalyzer(tiers=("core",))                   # one tier only

From the command line:

    python gamerlex.py "gg, the boss fights slap but the netcode is janky"
    python gamerlex.py --curated "first text" "second text"
    python gamerlex.py < my_texts.txt                        # one text per line

Requires: vaderSentiment.
"""
import csv
import os
import re
import sys

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


CURATED = ("core", "extended", "vader_override")
ALL_TIERS = CURATED + ("auto",)
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GamerLex_v1.0.csv")
_TOKEN = re.compile(r"[a-z0-9'*_-]+")


class GamerLexAnalyzer:
    def __init__(self, path=DEFAULT_PATH, tiers=ALL_TIERS, use_vader=True):
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
        """VADER scores (neg, neu, pos, compound) with the GamerLex entries added."""
        return self.analyzer.polarity_scores(self.join_multiword(text))

    def matches(self, text):
        """GamerLex terms found in the text, in order, as (term, valence) pairs."""
        out = []
        for tok in _TOKEN.findall(self.join_multiword(text).lower()):
            tok = tok.strip("'-*")
            if tok in self.entries:
                out.append((tok.replace("_", " "), self.entries[tok]))
        return out


def _main(argv):
    tiers = ALL_TIERS
    if "--curated" in argv:
        tiers = CURATED
        argv = [a for a in argv if a != "--curated"]
    texts = argv if argv else [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        print(__doc__)
        return
    gl = GamerLexAnalyzer(tiers=tiers)
    vader = SentimentIntensityAnalyzer()
    for t in texts:
        s = gl.polarity_scores(t)
        terms = ", ".join(f"{w} ({v:+.2f})" for w, v in gl.matches(t)) or "none"
        print(t)
        print(f"  VADER compound:            {vader.polarity_scores(t)['compound']:+.4f}")
        print(f"  VADER + GamerLex compound: {s['compound']:+.4f}  (neg {s['neg']:.3f}, neu {s['neu']:.3f}, pos {s['pos']:.3f})")
        print(f"  GamerLex terms:            {terms}")


if __name__ == "__main__":
    _main(sys.argv[1:])
