# -*- coding: utf-8 -*-
"""
GamerLex :: valencia kiosztasa, kiadasi csomag es szamszeru kiertekeles
======================================================================

MIT CSINAL (egy futassal):
    1. A korpuszt veletlenszeruen harom reszre osztja (seed 42):
       70% tanito, 10% validacios, 20% teszt. A benchmark mondatait
       tartalmazo review-k egyik reszbe sem kerulnek be.
    2. A tanito reszen minden lexikonszora kiszamolja, mennyire kotodik az
       ajanlo (voted_up) review-khoz, a review hosszara korrigalva
       (hossz-savonkent retegzett Mantel-Haenszel log-eselyhanyados).
    3. Szures, csak a tanito resz alapjan:
         - a tokenizaloval nem illesztheto bejegyzes kiesik;
         - ami kevesebb mint 5 tanito review-ban fordul elo, kiesik;
         - a VADER-felulirasi reteg szava kiesik (marad a VADER-nel), ha a
           korpusz szerint is egyertelmuen negativ.
    4. A log-eselyhanyadost a VADER skalajara (-4..+4) illeszti olyan
       horgonyszavakon, amelyek a VADER-ben benne vannak, a GamerLexben
       nincsenek. Ket illesztest probal ki (egy meredekseg / elojelenkent
       kulon meredekseg), es a validacios reszen a jobbat valasztja.
    5. A teszt reszen osszeveti: VADER, VADER + GamerLex retegenkent, a
       GamerLex onmagaban, az automatikus reteg kulon, es ket kuralatlan alapvonal
       (a 30 000 leggyakoribb token automatikus valenciaval). Mutatok: ROC AUC, review-hossz szerint
       retegzett AUC, kiegyensulyozott pontossag, lefedettseg; a VADER-hez
       mert kulonbsegre bootstrap konfidenciaintervallum.
       A benchmark mondatokon Spearman-korrelaciot szamol az emberi
       ertekelessel. Ha a bench/gamerlex_bench_GT_vak.xlsx ki van toltve,
       a harom ertekelo medianjaval is.
    6. Automatikus reteg ('auto'): a leggyakoribb tokenek kozul azok a tartalmas
       szavak, amelyek sem a VADER-ben, sem a kuralt GamerLexben nincsenek, es
       erosen kotodnek az ajanlashoz (elore rogzitett kuszobok). Funkcioszavak,
       szamjegyes vagy irasjeles tokenek es tulajdonnevek nem kerulnek bele.
    7. Mufajonkenti osszevetes (ha a korpuszban vannak genre_* oszlopok).
    8. Kiirja a kiadasi csomagot.

MIT OLVAS:
    steam_sppm_clustered_deduped.parquet   (review_df.reviews.review, review_df.reviews.voted_up)
    lexicon_master_v2.csv                  (a gamerslang_lexicon_apply_audit.py kimenete)
    bench/bench_merged.csv                 (nem kotelezo; a mondatszintu kiertekeleshez)
    bench/gamerlex_bench_GT_vak.xlsx       (nem kotelezo; a harmadik ertekelo pontszamai)
    Mindet maga keresi meg (aktualis mappa, a szkript mappaja, output/gamerslang,
    H:/output/gamerslang, a korpuszt a H:/ gyokerben is). Ha valamit nem talal,
    kiirja, hol kereste.

MIT IR (a lexikon mappajanak 'release' almappajaba):
    GamerLex_v1.0.csv           a lexikon: term, valence, tier, category, statisztikak
    GamerLex_v1.0_vader.txt     VADER-formatum, minden reteg (tobbszavas kifejezes alahuzassal)
    GamerLex_v1.0_curated_vader.txt   ugyanez az automatikus reteg nelkul
    GamerLex_v1.0.json          ugyanez JSON-ban
    GamerLex_v1.0_dropped.csv   a szuresen kiesett bejegyzesek az okkal
    GamerLex_v1.0_anchors.csv   a skalaillesztes horgonyszavai
    GamerLex_v1.0_results.json  minden szamszeru eredmeny gepileg olvashatoan
    GamerLex_v1.0_report.txt    ugyanez olvashato formaban
    gamerlex.py                 segedmodul a hasznalathoz (VADER-rel)
    README.md, CITATION.cff     a kozzetetelhez
  es a lexikon mappajanak 'eval' almappajaba (review-szoveget tartalmaz, NEM kozzetetelre):
    GamerLex_eval_test_sample.csv   a tesztminta a valtozatok pontszamaival
    GamerLex_induced_baseline.csv   a kuralatlan alapvonal szavai

HOGYAN FUTTASD (a H:/ gyokerbol):
    python output\\gamerslang\\gamerlex_build.py
    python output\\gamerslang\\gamerlex_build.py --force           (meglevo kiadas felulirasa)
    python output\\gamerslang\\gamerlex_build.py --test-n 300000   (nagyobb tesztminta, lassabb)

Fuggosegek: pandas, numpy, pyarrow, vaderSentiment.
Varhato futasido: nagyjabol 20-25 perc.
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

VERSION = "1.0"
PARQUET = "steam_sppm_clustered_deduped.parquet"
LEXFILE = "lexicon_master_v2.csv"
TEXT_COL = "review_df.reviews.review"
VOTE_COL = "review_df.reviews.voted_up"
TOK = re.compile(r"[a-z0-9'*-]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
LEN_EDGES = [5, 10, 20, 40, 80, 160, 320]
P_TRAIN, P_VAL = 0.70, 0.10                 # a maradek 20% a teszt
MIN_TRAIN_N = 5
OVR_DIFF, OVR_Z = -0.10, -3.0               # felulirasi reteg: ennel negativabb szo marad a VADER-nel
ANCHOR_MIN_N = 500
SHRINK_N0 = 25.0
IND_K = 30_000                              # kuralatlan alapvonal: ennyi leggyakoribb token
IND_SAMPLE = 300_000                        # ennyi review-bol becsuli a gyakorisagot (cimke nelkul)
IND_Z = 3.0                                 # csak a szignifikansan kotodo tokenek kapnak valenciat
# automatikus reteg ('auto'): elore rogzitett szabalyok, nem a tesztadaton hangolva
AUTO_MIN_N = 100                            # legalabb ennyi tanito review
AUTO_MIN_Z = 5.0                            # legalabb ekkora |z|
AUTO_MIN_VAL = 0.5                          # legalabb ekkora |valencia|
CAP_MIN_N = 20                              # tulajdonnev-szures: legalabb ennyi mondatkozi elofordulas
CAP_RATIO = 0.5                             # ... es ezek legalabb ekkora resze nagy kezdobetus
WORD_FORM = re.compile(r"[a-z]+(?:-[a-z]+)*")  # az automatikus retegbe csak betukbol allo (esetleg kotojeles) szo kerulhet
SENT_MARKS = ".!?\n:;-*"
TOKC = re.compile(r"[A-Za-z0-9'*-]+")
# funkcioszavak (a scikit-learn angol stoplistaja; a VADER tagado- es fokozoszavai futas kozben adodnak hozza)
STOPWORDS = set("""
a about above across after afterwards again against all almost alone along already also although always am
among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back
be became because become becomes becoming been before beforehand behind being below beside besides between
beyond bill both bottom but by call can cannot cant co con could couldnt cry de describe detail do done down
due during each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything
everywhere except few fifteen fifty fill find fire first five for former formerly forty found four from front
full further get give go had has hasnt have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred i ie if in inc indeed interest into is it its itself keep last latter
latterly least less ltd made many may me meanwhile might mill mine more moreover most mostly move much must my
myself name namely neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of
off often on once one only onto or other others otherwise our ours ourselves out over own part per perhaps
please put rather re same see seem seemed seeming seems serious several she should show side since sincere six
sixty so some somehow someone something sometime sometimes somewhere still such system take ten than that the
their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin
third this those though three through throughout thru thus to together too top toward towards twelve twenty
two un under until up upon us very via was we well were what whatever when whence whenever where whereafter
whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will
with within without would yet you your yours yourself yourselves
""".split())
VAL_CLIP = 4.0


# ---------------------------------------------------------------- segedfuggvenyek

def find_file(name, cands, what, required=True):
    for c in cands:
        p = os.path.join(c, name)
        if os.path.exists(p):
            return p
    if required:
        print(f"[HIBA] Nem talalom: {name} ({what}). Itt kerestem:")
        for c in cands:
            print("   ", os.path.abspath(os.path.join(c, name)))
        sys.exit(1)
    return None


def pick_col(cols, wanted, suffix):
    if wanted in cols:
        return wanted
    hits = [c for c in cols if c.endswith(suffix)]
    if len(hits) == 1:
        print(f"  [megjegyzes] '{wanted}' helyett ezt az oszlopot hasznalom: {hits[0]}")
        return hits[0]
    print(f"[HIBA] Nincs '{wanted}' oszlop a parquet fajlban. A fajl oszlopai:")
    for c in cols:
        print("   ", c)
    sys.exit(1)


def avg_ranks(x):
    x = np.asarray(x, dtype=float)
    uniq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return (cum - (cnt - 1) / 2.0)[inv]


def auc(scores, y):
    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = avg_ranks(scores)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def strat_auc(scores, y, bins, nb):
    """Hossz-savonkent szamolt AUC, a savok pozitiv-negativ parjainak szamaval sulyozva."""
    num = den = 0.0
    for b in range(nb):
        m = bins == b
        yb = y[m]
        n1 = int(yb.sum())
        n0 = int(m.sum()) - n1
        if n1 == 0 or n0 == 0:
            continue
        w = float(n1) * float(n0)
        num += w * auc(scores[m], yb)
        den += w
    return float(num / den) if den else float("nan")


def spearman(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def kripp_alpha(ratings, metric="interval"):
    """Krippendorff-alfa; ratings: ertekelok x egysegek (np.nan = hianyzik), coincidence-matrixszal."""
    r = np.asarray(ratings, dtype=float)
    vals = np.unique(r[~np.isnan(r)])
    if len(vals) < 2:
        return float("nan")
    k = len(vals)
    pos = {v: i for i, v in enumerate(vals)}
    o = np.zeros((k, k))
    for j in range(r.shape[1]):
        col = r[:, j][~np.isnan(r[:, j])]
        mu = len(col)
        if mu < 2:
            continue
        for a in range(mu):
            for b in range(mu):
                if a != b:
                    o[pos[col[a]], pos[col[b]]] += 1.0 / (mu - 1)
    nc = o.sum(axis=1)
    n = nc.sum()
    if metric == "interval":
        dist = (vals[:, None] - vals[None, :]) ** 2
    else:                                            # ordinalis
        cum = np.cumsum(nc)
        dist = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                lo, hi = min(a, b), max(a, b)
                dist[a, b] = ((cum[hi] - (cum[lo - 1] if lo > 0 else 0)) - (nc[lo] + nc[hi]) / 2.0) ** 2
    d_o = (o * dist).sum() / n
    d_e = (np.outer(nc, nc) * dist).sum() / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def balanced_acc(scores, y):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(scores) >= 0          # 0 (nincs erzelmi szo) = ajanlo, mert az a tobbsegi osztaly
    tpr = pred[y].mean() if y.any() else float("nan")
    tnr = (~pred[~y]).mean() if (~y).any() else float("nan")
    return float((tpr + tnr) / 2.0)


def build_joiner(multi_terms):
    """A tobbszavas kifejezeseket alahuzassal egy tokenne fuzi (kis- es nagybetutol fuggetlenul)."""
    if not multi_terms:
        return lambda t: t
    parts = sorted(multi_terms, key=len, reverse=True)
    pat = "|".join(r"\s+".join(re.escape(w) for w in term.split()) for term in parts)
    rx = re.compile(r"(?<![A-Za-z0-9_])(?:" + pat + r")(?![A-Za-z0-9_])", re.IGNORECASE)
    ws = re.compile(r"\s+")
    return lambda t: rx.sub(lambda m: ws.sub("_", m.group(0)), t)


def make_analyzer(SIA, entries=None, vader=True):
    """entries: {token: valence}; vader=False eseten csak a GamerLex szokincse marad."""
    a = SIA()
    if not vader:
        a.lexicon = {}
    else:
        a.lexicon = dict(a.lexicon)
    if entries:
        a.lexicon.update(entries)
    return a


def score_texts(analyzer, texts):
    return np.array([analyzer.polarity_scores(t)["compound"] for t in texts], dtype=float)


HELPER_PY = '''# -*- coding: utf-8 -*-
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
        self._ws = re.compile(r"\\s+")
        if multi:
            parts = sorted(multi, key=len, reverse=True)
            pat = "|".join(r"\\s+".join(re.escape(w) for w in t.split()) for t in parts)
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
'''


# ---------------------------------------------------------------- fo program

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-n", type=int, default=100_000, help="tesztminta merete (review)")
    ap.add_argument("--val-n", type=int, default=50_000, help="validacios minta merete (review)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=200, help="bootstrap ismetlesek szama az AUC-kulonbseghez")
    ap.add_argument("--boot-bench", type=int, default=1000, help="bootstrap ismetlesek szama a benchmarkon")
    ap.add_argument("--force", action="store_true", help="meglevo kiadas felulirasa")
    args = ap.parse_args()
    t0 = time.time()

    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    lexpath = find_file(LEXFILE, [here, os.path.join(cwd, "output", "gamerslang"), cwd, r"H:\output\gamerslang"],
                        "a megtisztitott lexikon; elobb futtasd: python output\\gamerslang\\gamerslang_lexicon_apply_audit.py")
    lexdir = os.path.dirname(lexpath)
    outdir = os.path.join(lexdir, "release")
    base = f"GamerLex_v{VERSION}"
    f_main = os.path.join(outdir, base + ".csv")
    evaldir = os.path.join(lexdir, "eval")
    f_eval = os.path.join(evaldir, "GamerLex_eval_test_sample.csv")
    existing = [p for p in (f_main, f_eval) if os.path.exists(p)]
    if existing and not args.force:
        print("[MEGALLTAM] Mar letezik kiadas, nem irom felul (--force kell):")
        for p in existing:
            print("   ", p)
        sys.exit(1)
    parquet = find_file(PARQUET, [cwd, os.path.abspath(os.path.join(here, "..", "..")), here, "H:\\"], "korpusz")
    benchpath = find_file("bench_merged.csv", [os.path.join(lexdir, "bench"), lexdir], "benchmark", required=False)
    gtpath = find_file("gamerlex_bench_GT_vak.xlsx", [os.path.join(lexdir, "bench"), lexdir], "harmadik ertekelo", required=False)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[HIBA] Hianyzik a pyarrow csomag. Telepites: pip install pyarrow")
        sys.exit(1)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
    except ImportError:
        print("[HIBA] Hianyzik a vaderSentiment csomag. Telepites: pip install vaderSentiment")
        sys.exit(1)

    print("Lexikon:  ", lexpath)
    print("Korpusz:  ", parquet)
    print("Benchmark:", benchpath if benchpath else "(nincs meg, a mondatszintu kiertekeles kimarad)")

    lex = pd.read_csv(lexpath, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = {"word", "tier", "slang_type"}
    if not need <= set(lex.columns):
        print("[HIBA] Hianyzo oszlop(ok) a lexikonfajlban:", ", ".join(sorted(need - set(lex.columns))))
        sys.exit(1)
    lex["word"] = lex.word.str.strip().str.lower()
    lex = lex[lex.word != ""].drop_duplicates("word").reset_index(drop=True)
    n_input = len(lex)
    vader_lex = SIA().lexicon

    # ---- szokincs: GamerLex-bejegyzesek + VADER-horgonyszavak
    words = list(lex.word)
    single = {}
    multi = defaultdict(list)
    matchable = np.zeros(len(words), dtype=bool)
    for i, w in enumerate(words):
        toks = TOK.findall(w)
        if not toks or "".join(toks) != re.sub(r"\s+", "", w):
            continue
        matchable[i] = True
        if len(toks) == 1:
            single[toks[0]] = i
        else:
            multi[toks[0]].append((tuple(toks), i))
    glset = set(words)
    anchors = [w for w in vader_lex if w not in glset and len(w) >= 2 and TOK.fullmatch(w) and re.search(r"[a-z]", w)]
    for j, w in enumerate(anchors):
        single.setdefault(w, len(words) + j)
    V = len(words) + len(anchors)
    vocab = set(single)
    starters = set(multi)
    print(f"Lexikonbejegyzes: {n_input}  (illesztheto {int(matchable.sum())})  |  VADER-horgonyjelolt: {len(anchors)}")

    # ---- benchmark mondatok (ezeket tartalmazo review-k kimaradnak)
    bench = None
    bench_set = set()
    if benchpath:
        bench = pd.read_csv(benchpath, encoding="utf-8-sig")
        if {"sentence_id", "sentence", "A1_sent", "A2_sent"} <= set(bench.columns):
            bench["sentence"] = bench.sentence.astype(str)
            bench = bench[~bench.sentence.str.startswith("#NAME")].copy()
            bench_set = {s.strip() for s in bench.sentence}
        else:
            print("  [megjegyzes] a bench_merged.csv oszlopai nem a vartak, a mondatszintu kiertekeles kimarad")
            bench = None

    # ---- egy menet a korpuszon
    pf = pq.ParquetFile(parquet)
    cols = pf.schema_arrow.names
    tcol = pick_col(cols, TEXT_COL, ".review")
    vcol = pick_col(cols, VOTE_COL, "voted_up")
    total = pf.metadata.num_rows
    p_val = min(1.0, args.val_n / max(total * P_VAL, 1))
    p_test = min(1.0, args.test_n / max(total * (1 - P_TRAIN - P_VAL), 1))
    print(f"Review a korpuszban: {total:,}")
    print("A parquet oszlopai:", ", ".join(cols))

    # ---- elo-menet: a leggyakoribb tokenek a kuralatlan alapvonalhoz (cimket nem hasznal)
    rng0 = np.random.default_rng(args.seed + 7)
    p0 = min(1.0, IND_SAMPLE / max(total, 1))
    df_cnt = Counter()
    mid_tot = Counter()                       # mondat kozbeni elofordulasok (tulajdonnev-szureshez)
    mid_cap = Counter()                       # ebbol nagy kezdobetus
    for batch in pf.iter_batches(batch_size=50_000, columns=[tcol]):
        texts = batch.column(0).to_pylist()
        u0 = rng0.random(len(texts))
        for k, txt in enumerate(texts):
            if u0[k] < p0 and txt is not None:
                txt = str(txt)
                seen_tok = set()
                prev_end = 0
                first = True
                for mt in TOKC.finditer(txt):
                    w = mt.group(0)
                    lw = w.lower()
                    seen_tok.add(lw)
                    gap = txt[prev_end:mt.start()]
                    at_start = first or any(ch in gap for ch in SENT_MARKS)
                    first = False
                    prev_end = mt.end()
                    if not at_start:
                        mid_tot[lw] += 1
                        if w[0].isupper() and w[1:] == w[1:].lower():
                            mid_cap[lw] += 1
                df_cnt.update(seen_tok)
    ind_tokens = [w for w, c in df_cnt.most_common() if len(w) >= 2 and re.search(r"[a-z]", w)][:IND_K]
    cap_ratio = {w: (mid_cap.get(w, 0) / mid_tot[w] if mid_tot.get(w, 0) >= CAP_MIN_N else float("nan")) for w in ind_tokens}
    del df_cnt, mid_tot, mid_cap
    n_extra = 0
    for w in ind_tokens:
        if w not in single:
            single[w] = V + n_extra
            n_extra += 1
    V += n_extra
    vocab = set(single)
    print(f"Kuralatlan alapvonal szokincse: {len(ind_tokens):,} token ({time.time() - t0:.0f} mp)")

    rng = np.random.default_rng(args.seed)
    NB = len(LEN_EDGES) + 1
    n2l = [0] * (V * NB)                      # szo x hossz-sav, sima listaban (gyorsabb, mint a numpy elemenkent)
    npos2l = [0] * (V * NB)
    Nb = np.zeros(NB, dtype=np.int64)
    NPOSb = np.zeros(NB, dtype=np.int64)
    counts = {"train": 0, "val": 0, "test": 0, "bench_kizarva": 0, "ures": 0}
    pos_counts = {"train": 0, "val": 0, "test": 0}
    val_texts, val_y, test_texts, test_y = [], [], [], []
    gcols = [c for c in cols if c.startswith("genre_")]
    test_genres = []
    next_report = 500_000
    seen = 0

    for batch in pf.iter_batches(batch_size=50_000, columns=[tcol, vcol] + gcols):
        texts = batch.column(0).to_pylist()
        votes = batch.column(1).to_pylist()
        gmat = [batch.column(2 + gi).to_pylist() for gi in range(len(gcols))]
        u = rng.random(len(texts))
        u2 = rng.random(len(texts))
        for k, (txt, vote) in enumerate(zip(texts, votes)):
            seen += 1
            if seen >= next_report:
                print(f"  {seen:,} review  ({time.time() - t0:.0f} mp)")
                next_report += 500_000
            if txt is None or vote is None:
                counts["ures"] += 1
                continue
            txt = str(txt)
            try:
                pos = int(vote) == 1
            except (TypeError, ValueError):
                pos = str(vote).strip().lower() in ("true", "1")
            toks = TOK.findall(txt.lower())
            if not toks:
                counts["ures"] += 1
                continue
            if bench_set and any(s.strip() in bench_set for s in SENT_SPLIT.split(txt.strip())):
                counts["bench_kizarva"] += 1
                continue
            if u[k] >= P_TRAIN:
                if u[k] < P_TRAIN + P_VAL:
                    counts["val"] += 1
                    pos_counts["val"] += pos
                    if u2[k] < p_val:
                        val_texts.append(txt)
                        val_y.append(pos)
                else:
                    counts["test"] += 1
                    pos_counts["test"] += pos
                    if u2[k] < p_test:
                        test_texts.append(txt)
                        test_y.append(pos)
                        if gcols:
                            test_genres.append([1 if (g[k] is not None and float(g[k]) > 0) else 0 for g in gmat])
                continue
            # tanito resz
            counts["train"] += 1
            pos_counts["train"] += pos
            lt = len(toks)
            b = 0
            while b < NB - 1 and lt > LEN_EDGES[b]:
                b += 1
            Nb[b] += 1
            NPOSb[b] += pos
            tset = set(toks)
            hit = {single[t] for t in tset & vocab}
            if tset & starters:
                for j, t in enumerate(toks):
                    if t in starters:
                        for seq, idx in multi[t]:
                            if j + len(seq) <= lt and tuple(toks[j:j + len(seq)]) == seq:
                                hit.add(idx)
            for idx in hit:
                q = idx * NB + b
                n2l[q] += 1
                if pos:
                    npos2l[q] += 1

    if counts["train"] == 0:
        print("[HIBA] A tanito resz ures.")
        sys.exit(1)
    print(f"Korpusz-menet kesz ({time.time() - t0:.0f} mp). Tanito: {counts['train']:,}  validacios: {counts['val']:,}  "
          f"teszt: {counts['test']:,}  benchmark miatt kizarva: {counts['bench_kizarva']:,}  ures: {counts['ures']:,}")

    # ---- statisztikak a tanito reszen
    n2 = np.array(n2l, dtype=np.int64).reshape(V, NB)
    npos2 = np.array(npos2l, dtype=np.int64).reshape(V, NB)
    n = n2.sum(axis=1)
    npos = npos2.sum(axis=1)
    base_b = NPOSb / np.maximum(Nb, 1)
    exp_cnt = (n2 * base_b).sum(axis=1)
    var_cnt = (n2 * base_b * (1 - base_b)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(n > 0, npos / np.maximum(n, 1), np.nan)
        exp = np.where(n > 0, exp_cnt / np.maximum(n, 1), np.nan)
        A = npos2.astype(float)
        B = (n2 - npos2).astype(float)
        C = NPOSb - A
        D = (Nb - NPOSb) - B
        T = np.maximum(Nb, 1).astype(float)
        num = (A * D / T).sum(axis=1) + 0.5
        den = (B * C / T).sum(axis=1) + 0.5
        lor = np.where(n > 0, np.log(num / den), np.nan)
        z = np.where(var_cnt > 0, (npos - exp_cnt) / np.sqrt(np.maximum(var_cnt, 1e-12)), np.nan)
    assoc = rec - exp

    G = len(words)
    lex["matchable"] = matchable
    lex["n_train"] = n[:G]
    lex["rec_share"] = rec[:G]
    lex["expected_rec_share"] = exp[:G]
    lex["assoc"] = assoc[:G]
    lex["log_odds"] = lor[:G]
    lex["z"] = z[:G]
    lex["vader_valence"] = [vader_lex.get(w, np.nan) for w in words]
    lex["multiword"] = lex.word.str.contains(" ").astype(int)

    # ---- szures (csak tanito adatokbol)
    reason = pd.Series("", index=lex.index)
    reason[~lex.matchable] = "not matchable by the reference tokenizer"
    reason[(reason == "") & (lex.n_train < MIN_TRAIN_N)] = f"fewer than {MIN_TRAIN_N} training reviews"
    ovr_neg = (reason == "") & (lex.tier == "vader_override") & (lex.assoc <= OVR_DIFF) & (lex.z <= OVR_Z)
    reason[ovr_neg] = "override candidate with clearly negative corpus association (VADER value retained)"
    dropped = lex[reason != ""].copy()
    dropped["ok"] = reason[reason != ""]
    keep = lex[reason == ""].copy().reset_index(drop=True)
    print(f"Szures utan megmarad: {len(keep)}  |  kiesett: {len(dropped)}  {dropped.ok.value_counts().to_dict()}")

    # ---- skalaillesztes a VADER-re horgonyszavakon
    a_idx = np.array([single[w] for w in anchors if single.get(w, -1) >= G], dtype=int)
    anc = pd.DataFrame({"word": [w for w in anchors if single.get(w, -1) >= G]})
    anc["vader_valence"] = [vader_lex[w] for w in anc.word]
    anc["n_train"] = n[a_idx]
    anc["log_odds"] = lor[a_idx]
    anc["assoc"] = assoc[a_idx]
    anc = anc[(anc.n_train >= ANCHOR_MIN_N) & anc.log_odds.notna()].reset_index(drop=True)
    if len(anc) < 50:
        print(f"[HIBA] Tul keves horgonyszo ({len(anc)}), a skalaillesztes nem megbizhato.")
        sys.exit(1)
    x = anc.log_odds.values
    y = anc.vader_valence.values
    k_single = float((x * y).sum() / (x * x).sum())
    mp, mn = x > 0, x < 0
    k_pos = float((x[mp] * y[mp]).sum() / (x[mp] ** 2).sum()) if mp.sum() >= 30 else float("nan")
    k_neg = float((x[mn] * y[mn]).sum() / (x[mn] ** 2).sum()) if mn.sum() >= 30 else float("nan")
    cal_stats = {
        "n_anchors": int(len(anc)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": spearman(x, y),
        "sign_agreement": float((np.sign(x) == np.sign(y)).mean()),
        "k_single": k_single, "k_pos": k_pos, "k_neg": k_neg,
    }
    print(f"Horgonyszo: {len(anc)}  |  Pearson {cal_stats['pearson']:.3f}  Spearman {cal_stats['spearman']:.3f}  "
          f"elojel-egyezes {cal_stats['sign_agreement']:.3f}")
    print(f"Meredekseg: egy = {k_single:.3f}  |  pozitiv oldal = {k_pos:.3f}  negativ oldal = {k_neg:.3f}")

    def valence_for(kp, kn):
        lo = keep.log_odds.values
        k = np.where(lo >= 0, kp, kn)
        v = k * lo * (keep.n_train.values / (keep.n_train.values + SHRINK_N0))
        return np.clip(v, -VAL_CLIP, VAL_CLIP)

    cals = {"single": (k_single, k_single)}
    if math.isfinite(k_pos) and math.isfinite(k_neg) and k_pos > 0 and k_neg > 0:
        cals["piecewise"] = (k_pos, k_neg)
    if k_single <= 0:
        print("[HIBA] A skalaillesztes meredeksege nem pozitiv; a korpuszbol mert irany nem egyezik a VADER-rel.")
        sys.exit(1)

    multi_terms = [w for w in keep.word if " " in w]
    joiner = build_joiner(multi_terms)

    def entries_for(vals, tiers=None):
        d = {}
        for w, v, t in zip(keep.word, vals, keep.tier):
            if tiers is None or t in tiers:
                d[w.replace(" ", "_")] = float(v)
        return d

    # ---- az illesztes kivalasztasa a validacios reszen
    val_y_arr = np.array(val_y, dtype=bool)
    val_joined = [joiner(t) for t in val_texts]
    print(f"Validacios minta: {len(val_texts):,} review. Illesztes kivalasztasa...")
    val_res = {}
    for name, (kp, kn) in cals.items():
        an = make_analyzer(SIA, entries_for(valence_for(kp, kn)))
        val_res[name] = auc(score_texts(an, val_joined), val_y_arr)
        print(f"    {name:10s} AUC (VADER + GamerLex, validacios): {val_res[name]:.4f}")
    best = max(val_res, key=lambda q: val_res[q])
    kp, kn = cals[best]
    keep["valence"] = np.round(valence_for(kp, kn), 2)
    print(f"Valasztott illesztes: {best}  (k+ = {kp:.3f}, k- = {kn:.3f})")

    # ---- kuralatlan alapvonal: a leggyakoribb tokenek ugyanigy szamolt valenciaval
    ind_idx = np.array([single[w] for w in ind_tokens], dtype=int)
    ind = pd.DataFrame({"word": ind_tokens, "n_train": n[ind_idx], "log_odds": lor[ind_idx], "z": z[ind_idx], "assoc": assoc[ind_idx],
                        "rec_share": rec[ind_idx], "expected_rec_share": exp[ind_idx]})
    ind = ind[(ind.n_train >= MIN_TRAIN_N) & ind.log_odds.notna() & (ind.z.abs() >= IND_Z)].reset_index(drop=True)
    lo_i = ind.log_odds.values
    ind["valence"] = np.round(np.clip(np.where(lo_i >= 0, kp, kn) * lo_i * (ind.n_train.values / (ind.n_train.values + SHRINK_N0)),
                                      -VAL_CLIP, VAL_CLIP), 2)
    ind["in_vader"] = ind.word.isin(set(vader_lex)).astype(int)
    ind["in_gamerlex"] = ind.word.isin(set(keep.word)).astype(int)
    ind_all = dict(zip(ind.word, ind.valence.astype(float)))
    ind_new = {w: v for w, v in ind_all.items() if w not in vader_lex}
    print(f"Kuralatlan alapvonal: {len(ind_all):,} token kap valenciat (|z| >= {IND_Z:g}); ebbol a VADER-ben nincs: {len(ind_new):,}; "
          f"a GamerLexben is szerepel: {int(ind.in_gamerlex.sum()):,}")

    # ---- automatikus reteg: tartalmas szavak elore rogzitett szabalyokkal
    try:
        import vaderSentiment.vaderSentiment as _vs
        func_words = set(STOPWORDS) | set(getattr(_vs, "NEGATE", [])) | set(getattr(_vs, "BOOSTER_DICT", {}))
    except Exception:
        func_words = set(STOPWORDS)
    aux = {"do", "does", "did", "ca", "wo", "is", "are", "was", "were", "could", "would", "should", "have", "has", "had"}

    def token_class(w):
        if w in func_words or "n't" in w or (w.endswith("nt") and w[:-2] in aux):
            return "function"
        if re.search(r"\d", w):
            return "digit"
        if not WORD_FORM.fullmatch(w):
            return "other_form"
        if len(w) <= 2:
            return "short"
        cr = cap_ratio.get(w, float("nan"))
        if cr == cr and cr >= CAP_RATIO and not w.startswith("i'"):
            return "proper_noun"
        return "content"

    ind["token_class"] = ind.word.map(token_class)
    ind["cap_ratio"] = ind.word.map(lambda w: cap_ratio.get(w, float("nan")))
    auto = ind[(ind.token_class == "content") & (ind.in_vader == 0) & (ind.in_gamerlex == 0) & (ind.n_train >= AUTO_MIN_N)
               & (ind.z.abs() >= AUTO_MIN_Z) & (ind.valence.abs() >= AUTO_MIN_VAL)].copy()
    n_curated = len(keep)
    auto_rows = pd.DataFrame({"word": auto.word.values, "tier": "auto", "slang_type": "general", "matchable": True,
                              "n_train": auto.n_train.values, "rec_share": auto.rec_share.values,
                              "expected_rec_share": auto.expected_rec_share.values, "assoc": auto.assoc.values,
                              "log_odds": auto.log_odds.values, "z": auto.z.values, "vader_valence": np.nan,
                              "multiword": 0, "valence": auto.valence.values})
    keep = pd.concat([keep, auto_rows], ignore_index=True)
    class_counts = {k: int(v) for k, v in ind.token_class.value_counts().items()}
    print(f"Automatikus reteg: {len(auto_rows):,} szo (tartalmas, betukbol all, nincs a VADER-ben es a kuralt retegekben, n >= {AUTO_MIN_N}, "
          f"|z| >= {AUTO_MIN_Z:g}, |valencia| >= {AUTO_MIN_VAL}).  Tokenosztalyok az alapvonalban: {class_counts}")

    # ---- kiertekeles a teszt reszen
    test_y_arr = np.array(test_y, dtype=bool)
    test_joined = [joiner(t) for t in test_texts]
    print(f"Tesztminta: {len(test_texts):,} review. Kiertekeles (ez a leglassabb resz)...")
    cur_words = list(keep.word[keep.tier != "auto"])
    keepset_single = {w for w in cur_words if " " not in w}
    autoset = set(keep.word[keep.tier == "auto"])
    keep_multi = defaultdict(list)
    for w in cur_words:
        if " " in w:
            tk = TOK.findall(w)
            keep_multi[tk[0]].append(tuple(tk))

    def tok_info(txt):
        toks = TOK.findall(txt.lower())
        lt = len(toks)
        bb = 0
        while bb < NB - 1 and lt > LEN_EDGES[bb]:
            bb += 1
        found = bool(set(toks) & keepset_single)
        if not found:
            for j, t in enumerate(toks):
                if t in keep_multi and any(tuple(toks[j:j + len(seq)]) == seq for seq in keep_multi[t]):
                    found = True
                    break
        return lt, bb, found, found or bool(set(toks) & autoset)

    info = [tok_info(t) for t in test_texts]
    test_len = np.array([i[0] for i in info], dtype=int)
    test_bin = np.array([i[1] for i in info], dtype=int)
    has = np.array([i[2] for i in info], dtype=bool)          # van benne kuralt GamerLex-szo
    has_any = np.array([i[3] for i in info], dtype=bool)      # van benne barmelyik reteg szava
    vals = keep.valence.values
    CURATED = ("core", "extended", "vader_override")
    ALL_TIERS = CURATED + ("auto",)
    # (nev, bejegyzesek, VADER szokincse is, tobbszavas kifejezesek osszefuzese)
    variants = [
        ("VADER", None, True, False),
        ("VADER + core", entries_for(vals, ("core",)), True, True),
        ("VADER + core + extended", entries_for(vals, ("core", "extended")), True, True),
        ("VADER + GamerLex curated", entries_for(vals, CURATED), True, True),
        ("VADER + auto tier", entries_for(vals, ("auto",)), True, False),
        ("VADER + GamerLex full", entries_for(vals, ALL_TIERS), True, True),
        ("GamerLex full only", entries_for(vals, ALL_TIERS), False, True),
        ("VADER + induced (new words)", ind_new, True, False),
        ("VADER + induced (all frequent words)", ind_all, True, False),
    ]
    full_name = "VADER + GamerLex full"
    cur_name = "VADER + GamerLex curated"
    scores = {}
    test_rows = []
    for name, entries, use_v, joined in variants:
        ts = time.time()
        an = make_analyzer(SIA, entries, vader=use_v)
        sc = score_texts(an, test_joined if joined else test_texts)
        scores[name] = sc
        row = {
            "variant": name,
            "n_added_entries": int(len(entries)) if entries else 0,
            "auc": auc(sc, test_y_arr),
            "auc_length_stratified": strat_auc(sc, test_y_arr, test_bin, NB),
            "auc_with_term": auc(sc[has], test_y_arr[has]),
            "balanced_accuracy": balanced_acc(sc, test_y_arr),
            "coverage": float((sc != 0).mean()),
        }
        test_rows.append(row)
        print(f"    {name:38s} AUC {row['auc']:.4f} | retegzett AUC {row['auc_length_stratified']:.4f} | AUC (van kuralt szo) {row['auc_with_term']:.4f} | "
              f"kiegy. pontossag {row['balanced_accuracy']:.4f} | lefedettseg {row['coverage']:.4f}  ({time.time() - ts:.0f} mp)")

    # bootstrap: kulonbseg a VADER-hez kepest, minden valtozatra
    brng = np.random.default_rng(args.seed + 1)
    m = len(test_y_arr)
    others = [v[0] for v in variants if v[0] != "VADER"]
    b_auc = {v[0]: [] for v in variants}           # minden ujramintavetelezesben minden valtozat erteke
    b_str = {v[0]: [] for v in variants}
    for _ in range(max(args.boot, 0)):
        ii = brng.integers(0, m, m)
        yy = test_y_arr[ii]
        bb = test_bin[ii]
        for nm in b_auc:
            b_auc[nm].append(auc(scores[nm][ii], yy))
            b_str[nm].append(strat_auc(scores[nm][ii], yy, bb, NB))
    b_auc = {k: np.array(v) for k, v in b_auc.items()}
    b_str = {k: np.array(v) for k, v in b_str.items()}
    d_boot = {nm: list(b_auc[nm] - b_auc["VADER"]) for nm in others}
    ds_boot = {nm: list(b_str[nm] - b_str["VADER"]) for nm in others}
    base_row = test_rows[0]
    for row in test_rows[1:]:
        nm = row["variant"]
        row["delta_auc"] = row["auc"] - base_row["auc"]
        row["delta_auc_strat"] = row["auc_length_stratified"] - base_row["auc_length_stratified"]
        row["delta_auc_ci95"] = [float(np.percentile(d_boot[nm], 2.5)), float(np.percentile(d_boot[nm], 97.5))] if d_boot[nm] else [float("nan")] * 2
        row["delta_auc_strat_ci95"] = [float(np.percentile(ds_boot[nm], 2.5)), float(np.percentile(ds_boot[nm], 97.5))] if ds_boot[nm] else [float("nan")] * 2
        print(f"    kulonbseg a VADER-hez: {nm:38s} AUC {row['delta_auc']:+.4f} [{row['delta_auc_ci95'][0]:+.4f}, {row['delta_auc_ci95'][1]:+.4f}] | "
              f"retegzett {row['delta_auc_strat']:+.4f} [{row['delta_auc_strat_ci95'][0]:+.4f}, {row['delta_auc_strat_ci95'][1]:+.4f}]")
    full_row = [r for r in test_rows if r["variant"] == full_name][0]
    cur_row = [r for r in test_rows if r["variant"] == cur_name][0]
    d_auc = full_row["delta_auc"]
    ci = full_row["delta_auc_ci95"]

    # paros osszevetesek a GamerLex es az alapvonalak kozott (ugyanazokon az ujramintavetelezeseken)
    PAIRS = [(full_name, "VADER + induced (new words)"), (full_name, "VADER + induced (all frequent words)"),
             (full_name, cur_name), (full_name, "VADER + auto tier"),
             (cur_name, "VADER + induced (new words)"), (cur_name, "VADER + induced (all frequent words)")]
    PAIRS = [(a, b) for a, b in PAIRS if a in scores and b in scores]
    rowd = {r["variant"]: r for r in test_rows}

    def ci95(x):
        x = np.asarray([v for v in x if v == v])
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan")] * 2

    pair_rows = []
    for a, b in PAIRS:
        pr = {"a": a, "b": b,
              "delta_auc": rowd[a]["auc"] - rowd[b]["auc"], "delta_auc_ci95": ci95(b_auc[a] - b_auc[b]) if args.boot else [float("nan")] * 2,
              "delta_auc_strat": rowd[a]["auc_length_stratified"] - rowd[b]["auc_length_stratified"],
              "delta_auc_strat_ci95": ci95(b_str[a] - b_str[b]) if args.boot else [float("nan")] * 2}
        pair_rows.append(pr)
        print(f"    paros: {a} - {b}: AUC {pr['delta_auc']:+.4f} [{pr['delta_auc_ci95'][0]:+.4f}, {pr['delta_auc_ci95'][1]:+.4f}] | "
              f"retegzett {pr['delta_auc_strat']:+.4f} [{pr['delta_auc_strat_ci95'][0]:+.4f}, {pr['delta_auc_strat_ci95'][1]:+.4f}]")

    # ---- mufajonkent: ajanlasi arany, atlagos pontszam es AUC a VADER-rel, illetve a teljes GamerLexszel
    genre_rows = []
    genre_corr = {}
    genre_note = ""
    if gcols and len(test_genres) == len(test_texts):
        GM = np.array(test_genres, dtype=bool)
        genre_note = ("a review can belong to several genres" if GM.sum(axis=1).max() > 1
                      else "each review carries one genre label; genres with fewer than 200 test reviews are not shown")
        sv, sg, sc_cur = scores["VADER"], scores[full_name], scores[cur_name]
        for gi, gc in enumerate(gcols):
            mg = GM[:, gi]
            if mg.sum() < 200 or test_y_arr[mg].all() or (~test_y_arr[mg]).all():
                continue
            genre_rows.append({"genre": gc.replace("genre_", ""), "n": int(mg.sum()), "rec_share": float(test_y_arr[mg].mean()),
                               "mean_vader": float(sv[mg].mean()), "mean_gamerlex": float(sg[mg].mean()),
                               "auc_vader": auc(sv[mg], test_y_arr[mg]), "auc_gamerlex": auc(sg[mg], test_y_arr[mg]),
                               "auc_strat_vader": strat_auc(sv[mg], test_y_arr[mg], test_bin[mg], NB),
                               "auc_strat_curated": strat_auc(sc_cur[mg], test_y_arr[mg], test_bin[mg], NB),
                               "auc_strat_gamerlex": strat_auc(sg[mg], test_y_arr[mg], test_bin[mg], NB)})
        if len(genre_rows) >= 5:
            rs = [r["rec_share"] for r in genre_rows]
            genre_corr = {"n_genres": len(genre_rows),
                          "spearman_recshare_vs_mean_vader": spearman([r["mean_vader"] for r in genre_rows], rs),
                          "spearman_recshare_vs_mean_gamerlex": spearman([r["mean_gamerlex"] for r in genre_rows], rs)}
        print("Mufajonkent (tesztminta):")
        for r in genre_rows:
            print(f"    {r['genre']:14s} n={r['n']:6d}  ajanlo {r['rec_share']:.3f} | atlag VADER {r['mean_vader']:+.3f}  GamerLex {r['mean_gamerlex']:+.3f} | "
                  f"AUC VADER {r['auc_vader']:.4f}  GamerLex {r['auc_gamerlex']:.4f} | retegzett {r['auc_strat_vader']:.4f} -> kuralt {r['auc_strat_curated']:.4f} -> teljes {r['auc_strat_gamerlex']:.4f}")
        if genre_corr:
            print(f"    Mufajok ajanlasi aranya es atlagos pontszama kozti Spearman: VADER {genre_corr['spearman_recshare_vs_mean_vader']:+.3f}, "
                  f"GamerLex {genre_corr['spearman_recshare_vs_mean_gamerlex']:+.3f}  ({genre_corr['n_genres']} mufaj)")

    # ---- mondatszintu benchmark: ket annotator atlaga, es ha megvan, a harom ertekelo medianja
    bench_rows = []
    bench_sets = []
    bench_pairs = []
    agreement = {}
    gt_status = "not available"
    if bench is not None:
        b2 = bench[bench.A1_sent.notna() & bench.A2_sent.notna()].copy()
        if len(b2) >= 30:
            R2 = np.vstack([b2.A1_sent.astype(float).values, b2.A2_sent.astype(float).values])
            agreement = {"spearman_A1_A2": spearman(R2[0], R2[1]), "alpha_interval_two": kripp_alpha(R2, "interval"),
                         "alpha_ordinal_two": kripp_alpha(R2, "ordinal")}
        if len(b2) >= 30:
            bench_sets.append(("mean of two annotators", b2, ((b2.A1_sent.astype(float) + b2.A2_sent.astype(float)) / 2.0).values))
        if gtpath:
            try:
                gt = pd.read_excel(gtpath, sheet_name="ertekeles")
                gt["GT"] = pd.to_numeric(gt["GT_sentiment"], errors="coerce")
                gt = gt[gt.GT.isin([-2, -1, 0, 1, 2])][["sentence_id", "GT"]].drop_duplicates("sentence_id")
                n_filled = len(gt)
                if n_filled >= 0.9 * len(bench):
                    b3 = b2.merge(gt, on="sentence_id", how="inner")
                    R3 = np.vstack([b3.A1_sent.astype(float).values, b3.A2_sent.astype(float).values, b3.GT.astype(float).values])
                    med = np.median(R3, axis=0)
                    bench_sets.append(("median of three raters", b3, med))
                    names3 = ["A1", "A2", "GT"]
                    agreement["n_sentences"] = int(R3.shape[1])
                    agreement["pairs"] = []
                    for i in range(3):
                        for j in range(i + 1, 3):
                            x, y = R3[i], R3[j]
                            agreement["pairs"].append({"raters": names3[i] + "-" + names3[j], "spearman": spearman(x, y),
                                                       "exact": float((x == y).mean()), "within_one": float((np.abs(x - y) <= 1).mean())})
                    agreement["leave_one_out_spearman"] = {names3[i]: spearman(R3[i], R3[[q for q in range(3) if q != i]].mean(axis=0)) for i in range(3)}
                    agreement["alpha_interval_three"] = kripp_alpha(R3, "interval")
                    agreement["alpha_ordinal_three"] = kripp_alpha(R3, "ordinal")
                    gt_status = f"used ({n_filled} sentences rated)"
                else:
                    gt_status = f"incomplete ({n_filled} of {len(bench)} sentences rated), not used"
            except Exception as e:                      # hibas vagy nyitva levo fajl ne allitsa meg a futast
                gt_status = f"could not be read ({type(e).__name__})"
        print("Harmadik ertekelo (GT):", gt_status)
        if agreement:
            print(f"Ertekelok egyezese: A1-A2 Spearman {agreement['spearman_A1_A2']:.4f}, Krippendorff-alfa (intervallum) ket ertekelo {agreement['alpha_interval_two']:.4f}"
                  + (f", harom ertekelo {agreement['alpha_interval_three']:.4f}" if "alpha_interval_three" in agreement else ""))
            for pr in agreement.get("pairs", []):
                print(f"    {pr['raters']}: Spearman {pr['spearman']:.4f}, pontos egyezes {pr['exact']:.3f}, legfeljebb 1 elteres {pr['within_one']:.3f}")
    for gold_name, bdf, human in bench_sets:
        sents = bdf.sentence.tolist()
        sents_j = [joiner(x) for x in sents]
        bs = {}
        for name, entries, use_v, joined in variants:
            an = make_analyzer(SIA, entries, vader=use_v)
            bs[name] = score_texts(an, sents_j if joined else sents)
        rb = np.random.default_rng(args.seed + 2)
        nb_s = len(human)
        sb = {nm: [] for nm in bs}
        for _ in range(max(args.boot_bench, 0)):
            ii = rb.integers(0, nb_s, nb_s)
            for nm in sb:
                sb[nm].append(spearman(bs[nm][ii], human[ii]))
        sb = {k: np.array(v) for k, v in sb.items()}
        dboot = {nm: list(sb[nm] - sb["VADER"]) for nm in bs if nm != "VADER"}
        nonneutral = human != 0
        print(f"Benchmark ({nb_s} mondat), Spearman: {gold_name}")
        rho0 = spearman(bs["VADER"], human)
        for name in bs:
            rho = spearman(bs[name], human)
            r = {"gold": gold_name, "n": int(nb_s), "variant": name, "spearman_human": rho, "coverage": float((bs[name] != 0).mean()),
                 "sign_accuracy_nonneutral": float((np.sign(bs[name][nonneutral]) == np.sign(human[nonneutral])).mean()) if nonneutral.any() else float("nan"),
                 "n_nonneutral": int(nonneutral.sum())}
            if name != "VADER":
                dd = [x for x in dboot[name] if x == x]
                r["delta"] = rho - rho0
                r["delta_ci95"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))] if dd else [float("nan")] * 2
                print(f"    {name:38s} {rho:.4f}  kulonbseg {r['delta']:+.4f} [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}]  (lefedettseg {r['coverage']:.4f})")
            else:
                print(f"    {name:38s} {rho:.4f}  (lefedettseg {r['coverage']:.4f})")
            bench_rows.append(r)
        for a, b in PAIRS:
            pr = {"gold": gold_name, "a": a, "b": b, "delta": spearman(bs[a], human) - spearman(bs[b], human),
                  "delta_ci95": ci95(sb[a] - sb[b]) if args.boot_bench else [float("nan")] * 2}
            bench_pairs.append(pr)
            print(f"    paros: {a} - {b}: {pr['delta']:+.4f} [{pr['delta_ci95'][0]:+.4f}, {pr['delta_ci95'][1]:+.4f}]")

    # ---- leiro szamok
    kv = keep[keep.vader_valence.notna()]
    flips = int((np.sign(kv.valence) != np.sign(kv.vader_valence)).sum())
    tier_counts = keep.tier.value_counts().to_dict()
    cat_counts = keep.slang_type.value_counts().to_dict()
    desc = {
        "n_entries": int(len(keep)), "n_curated": int(n_curated), "n_auto": int(len(keep) - n_curated),
        "n_input": int(n_input), "n_dropped": int(len(dropped)),
        "dropped_by_reason": {k: int(v) for k, v in dropped.ok.value_counts().items()},
        "by_tier": {k: int(v) for k, v in tier_counts.items()},
        "by_category": {k: int(v) for k, v in cat_counts.items()},
        "n_multiword": int(keep.multiword.sum()),
        "n_also_in_vader": int(len(kv)), "n_sign_differs_from_vader": flips,
        "valence_negative_le_-0.5": int((keep.valence <= -0.5).sum()),
        "valence_neutral": int(((keep.valence > -0.5) & (keep.valence < 0.5)).sum()),
        "valence_positive_ge_0.5": int((keep.valence >= 0.5).sum()),
        "median_n_train": float(keep.n_train.median()),
    }
    stable = keep[(keep.n_train >= 500) & (keep.tier != "auto")]
    top_neg = stable.sort_values("valence").head(15)[["word", "valence", "n_train"]].values.tolist()
    top_pos = stable.sort_values("valence", ascending=False).head(15)[["word", "valence", "n_train"]].values.tolist()
    stable_a = keep[(keep.n_train >= 500) & (keep.tier == "auto")]
    top_neg_auto = stable_a.sort_values("valence").head(15)[["word", "valence", "n_train"]].values.tolist()
    top_pos_auto = stable_a.sort_values("valence", ascending=False).head(15)[["word", "valence", "n_train"]].values.tolist()
    freq_auto = keep[keep.tier == "auto"].sort_values("n_train", ascending=False).head(25)[["word", "valence", "n_train"]].values.tolist()
    auto_summary = {
        "n": int(len(keep) - n_curated),
        "rule": {"token_class": "content", "word_form": WORD_FORM.pattern, "not_in_vader": True, "not_in_curated_tiers": True,
                 "min_train_reviews": AUTO_MIN_N, "min_abs_z": AUTO_MIN_Z, "min_abs_valence": AUTO_MIN_VAL,
                 "candidate_pool": f"{IND_K} most frequent tokens of a label-free sample of {IND_SAMPLE} reviews",
                 "function_words": "scikit-learn English stop list plus VADER negations and boosters, and n't contractions",
                 "proper_noun": {"min_mid_sentence_occurrences": CAP_MIN_N, "min_capitalised_share": CAP_RATIO}},
        "token_classes_in_candidate_pool": class_counts,
        "n_negative": int((keep[keep.tier == "auto"].valence < 0).sum()),
        "n_positive": int((keep[keep.tier == "auto"].valence > 0).sum()),
        "most_frequent": freq_auto, "top_negative": top_neg_auto, "top_positive": top_pos_auto,
    }
    ovr = keep[keep.tier == "vader_override"]
    ovr_summary = {
        "n": int(len(ovr)),
        "mean_vader_valence": float(ovr.vader_valence.mean()) if len(ovr) else float("nan"),
        "mean_gamerlex_valence": float(ovr.valence.mean()) if len(ovr) else float("nan"),
        "share_sign_differs": float((np.sign(ovr.valence) != np.sign(ovr.vader_valence)).mean()) if len(ovr) else float("nan"),
    }

    results = {
        "version": VERSION, "built": datetime.date.today().isoformat(), "seed": args.seed,
        "corpus": {"reviews_total": int(total), "train": counts["train"], "validation": counts["val"], "test": counts["test"],
                   "excluded_benchmark": counts["bench_kizarva"], "empty": counts["ures"],
                   "rec_share_train": pos_counts["train"] / max(counts["train"], 1),
                   "rec_share_test": pos_counts["test"] / max(counts["test"], 1),
                   "length_bins": [{"tokens": (f"{(LEN_EDGES[i - 1] + 1) if i else 1}-{LEN_EDGES[i]}" if i < NB - 1 else f"{LEN_EDGES[-1] + 1}+"),
                                    "reviews": int(Nb[i]), "rec_share": float(base_b[i])} for i in range(NB)]},
        "method": {"split": [P_TRAIN, P_VAL, round(1 - P_TRAIN - P_VAL, 2)], "min_train_reviews": MIN_TRAIN_N,
                   "override_rule": {"assoc_le": OVR_DIFF, "z_le": OVR_Z}, "anchor_min_reviews": ANCHOR_MIN_N,
                   "shrinkage_n0": SHRINK_N0, "valence_clip": VAL_CLIP, "tokenizer": TOK.pattern},
        "lexicon": desc, "override_tier": ovr_summary, "auto_tier": auto_summary,
        "calibration": dict(cal_stats, chosen=best, k_pos_used=kp, k_neg_used=kn, validation_auc=val_res,
                            validation_sample=len(val_texts)),
        "induced_baseline": {"top_k_tokens": IND_K, "frequency_sample": IND_SAMPLE, "min_abs_z": IND_Z,
                             "n_entries": int(len(ind_all)), "n_not_in_vader": int(len(ind_new)),
                             "n_also_in_gamerlex": int(ind.in_gamerlex.sum())},
        "evaluation_reviews": {"test_sample": len(test_texts), "share_with_curated_term": float(has.mean()),
                               "share_with_any_term": float(has_any.mean()),
                               "rows": test_rows, "delta_auc_full_vs_vader": d_auc, "delta_auc_ci95": ci,
                               "bootstrap": args.boot},
        "evaluation_genres": {"rows": genre_rows, "across_genres": genre_corr, "note": genre_note},
        "evaluation_sentences": {"third_rater": gt_status, "rows": bench_rows, "paired": bench_pairs, "rater_agreement": agreement},
        "evaluation_reviews_paired": pair_rows,
        "top_negative": top_neg, "top_positive": top_pos,
    }

    # ---- kiiras
    os.makedirs(outdir, exist_ok=True)
    out = keep.rename(columns={"word": "term", "slang_type": "category", "n_train": "n_reviews"})
    for c in ("rec_share", "expected_rec_share", "assoc", "log_odds"):
        out[c] = out[c].astype(float).round(4)
    out["z"] = out.z.astype(float).round(2)
    out_cols = ["term", "valence", "tier", "category", "multiword", "n_reviews", "rec_share", "expected_rec_share",
                "assoc", "log_odds", "z", "vader_valence"]
    out = out.sort_values("term")[out_cols]
    out.to_csv(f_main, index=False, encoding="utf-8-sig")
    with open(os.path.join(outdir, base + "_vader.txt"), "w", encoding="utf-8", newline="\n") as f:
        for r in out.itertuples():
            f.write(f"{r.term.replace(' ', '_')}\t{r.valence}\n")
    with open(os.path.join(outdir, base + "_curated_vader.txt"), "w", encoding="utf-8", newline="\n") as f:
        for r in out.itertuples():
            if r.tier != "auto":
                f.write(f"{r.term.replace(' ', '_')}\t{r.valence}\n")
    with open(os.path.join(outdir, base + ".json"), "w", encoding="utf-8") as f:
        json.dump({r.term: {"valence": float(r.valence), "tier": r.tier, "category": r.category} for r in out.itertuples()},
                  f, ensure_ascii=False, indent=1)
    dcols = [c for c in ["word", "tier", "slang_type", "n_train", "assoc", "z", "vader_valence", "ok"] if c in dropped.columns]
    dropped[dcols].rename(columns={"ok": "reason"}).to_csv(os.path.join(outdir, base + "_dropped.csv"), index=False, encoding="utf-8-sig")
    anc.round(4).to_csv(os.path.join(outdir, base + "_anchors.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(outdir, base + "_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(outdir, "gamerlex.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(HELPER_PY)
    # belso kiertekelesi fajlok (review-szoveget tartalmaznak, nem a kiadas reszei)
    os.makedirs(evaldir, exist_ok=True)
    ev = pd.DataFrame({"text": test_texts, "recommended": test_y_arr.astype(int), "n_tokens": test_len,
                       "length_bin": test_bin, "has_curated_term": has.astype(int), "has_any_term": has_any.astype(int)})
    if gcols and len(test_genres) == len(test_texts):
        GMo = np.array(test_genres, dtype=int)
        for gi, gc in enumerate(gcols):
            ev[gc] = GMo[:, gi]
    for nm, sc in scores.items():
        ev["score :: " + nm] = np.round(sc, 4)
    ev.to_csv(f_eval, index=False, encoding="utf-8-sig")
    ind.round(4).to_csv(os.path.join(evaldir, "GamerLex_induced_baseline.csv"), index=False, encoding="utf-8-sig")

    c = results["corpus"]
    L = []
    L.append(f"GamerLex v{VERSION} :: build report ({results['built']})")
    L.append("=" * 60)
    L.append(f"Corpus: {c['reviews_total']:,} Steam reviews; train {c['train']:,}, validation {c['validation']:,}, test {c['test']:,}; "
             f"excluded (contain benchmark sentences) {c['excluded_benchmark']:,}; empty {c['empty']:,}.")
    L.append(f"Share of recommending reviews: train {c['rec_share_train']:.4f}, test {c['rec_share_test']:.4f}.")
    L.append("")
    L.append(f"Lexicon: {desc['n_entries']} entries = {desc['n_curated']} curated (candidates {desc['n_input']}, dropped {desc['n_dropped']}) "
             f"+ {desc['n_auto']} in the automatic tier.")
    L.append("  by tier:     " + ", ".join(f"{k} {desc['by_tier'].get(k, 0)}" for k in ALL_TIERS))
    L.append("  by category: " + ", ".join(f"{k} {v}" for k, v in desc["by_category"].items()))
    L.append(f"  median training reviews per entry: {desc['median_n_train']:.0f}")
    L.append(f"  multi-word terms: {desc['n_multiword']}; also in VADER: {desc['n_also_in_vader']}, "
             f"of which sign differs from VADER: {desc['n_sign_differs_from_vader']}.")
    L.append(f"  valence <= -0.5: {desc['valence_negative_le_-0.5']}; between: {desc['valence_neutral']}; >= +0.5: {desc['valence_positive_ge_0.5']}.")
    L.append("  dropped: " + "; ".join(f"{k}: {v}" for k, v in desc["dropped_by_reason"].items()))
    L.append("")
    L.append(f"Scaling to the VADER range: {cal_stats['n_anchors']} anchor words (in VADER, not in GamerLex, >= {ANCHOR_MIN_N} training reviews).")
    L.append(f"  Pearson {cal_stats['pearson']:.3f}, Spearman {cal_stats['spearman']:.3f}, sign agreement {cal_stats['sign_agreement']:.3f}.")
    L.append(f"  slopes: single {k_single:.3f}; positive side {k_pos:.3f}; negative side {k_neg:.3f}.")
    L.append("  validation AUC: " + ", ".join(f"{k} {v:.4f}" for k, v in val_res.items()) + f"  ->  chosen: {best}")
    L.append("")
    L.append(f"Override tier: {ovr_summary['n']} terms; mean VADER valence {ovr_summary['mean_vader_valence']:+.2f}, "
             f"mean GamerLex valence {ovr_summary['mean_gamerlex_valence']:+.2f}; sign differs for {ovr_summary['share_sign_differs']:.1%}.")
    L.append("")
    L.append(f"Automatic tier: {auto_summary['n']} words ({auto_summary['n_negative']} negative, {auto_summary['n_positive']} positive). Rule: content word "
             f"made of letters, not in VADER, not in the curated tiers, >= {AUTO_MIN_N} training reviews, |z| >= {AUTO_MIN_Z:g}, |valence| >= {AUTO_MIN_VAL}.")
    L.append("  token classes among the induced candidates: " + ", ".join(f"{k} {v}" for k, v in class_counts.items()))
    L.append("  most frequent: " + ", ".join(f"{w} ({v:+.2f})" for w, v, _ in freq_auto))
    L.append("")
    L.append(f"Review-level evaluation on {len(test_texts):,} held-out reviews (target: recommended yes/no); "
             f"{has.mean():.1%} contain at least one curated term, {has_any.mean():.1%} a term of any tier.")
    L.append(f"  {'variant':38s} {'entries':>8s} {'AUC':>7s} {'AUCstr':>7s} {'AUC*':>7s} {'bal.acc':>8s} {'coverage':>9s}")
    for r in test_rows:
        L.append(f"  {r['variant']:38s} {r['n_added_entries']:8d} {r['auc']:7.4f} {r['auc_length_stratified']:7.4f} {r['auc_with_term']:7.4f} "
                 f"{r['balanced_accuracy']:8.4f} {r['coverage']:9.4f}")
    L.append("  AUCstr = AUC computed within review-length bands and averaged (weights: positive-negative pairs per band).")
    L.append("  AUC* = only reviews with at least one curated GamerLex term. Coverage = share of reviews with a non-zero score.")
    L.append(f"  Differences from VADER with 95% bootstrap CI ({args.boot} resamples):")
    for r in test_rows[1:]:
        L.append(f"    {r['variant']:38s} AUC {r['delta_auc']:+.4f} [{r['delta_auc_ci95'][0]:+.4f}, {r['delta_auc_ci95'][1]:+.4f}]   "
                 f"AUCstr {r['delta_auc_strat']:+.4f} [{r['delta_auc_strat_ci95'][0]:+.4f}, {r['delta_auc_strat_ci95'][1]:+.4f}]")
    L.append(f"  Paired differences between lexicons (same {args.boot} resamples):")
    for pr in pair_rows:
        L.append(f"    {pr['a']} minus {pr['b']}: AUC {pr['delta_auc']:+.4f} [{pr['delta_auc_ci95'][0]:+.4f}, {pr['delta_auc_ci95'][1]:+.4f}]   "
                 f"AUCstr {pr['delta_auc_strat']:+.4f} [{pr['delta_auc_strat_ci95'][0]:+.4f}, {pr['delta_auc_strat_ci95'][1]:+.4f}]")
    L.append(f"  Induced baseline: the {IND_K:,} most frequent tokens, valence computed the same way, kept if |z| >= {IND_Z:g}: "
             f"{len(ind_all):,} entries ({len(ind_new):,} not in VADER; {int(ind.in_gamerlex.sum()):,} also in GamerLex).")
    L.append("  Note: GamerLex valences were learned from recommendations in the training part; VADER uses no such information.")
    L.append("")
    if genre_rows:
        L.append(f"By genre (test sample; {genre_note}):")
        L.append(f"  {'genre':14s} {'n':>7s} {'rec.':>6s} {'mean VADER':>11s} {'mean full':>10s} {'AUCstr VADER':>13s} {'curated':>8s} {'full':>7s}")
        for r in genre_rows:
            L.append(f"  {r['genre']:14s} {r['n']:7d} {r['rec_share']:6.3f} {r['mean_vader']:+11.3f} {r['mean_gamerlex']:+10.3f} "
                     f"{r['auc_strat_vader']:13.4f} {r['auc_strat_curated']:8.4f} {r['auc_strat_gamerlex']:7.4f}")
        if genre_corr:
            L.append(f"  Spearman across {genre_corr['n_genres']} genres between recommending share and mean score: "
                     f"VADER {genre_corr['spearman_recshare_vs_mean_vader']:+.3f}, GamerLex full {genre_corr['spearman_recshare_vs_mean_gamerlex']:+.3f}.")
        L.append("")
    L.append(f"Third rater file: {gt_status}.")
    if agreement:
        L.append(f"Rater agreement: A1-A2 Spearman {agreement['spearman_A1_A2']:.4f}; Krippendorff's alpha (interval) two raters {agreement['alpha_interval_two']:.4f}, "
                 f"(ordinal) {agreement['alpha_ordinal_two']:.4f}"
                 + (f"; three raters {agreement['alpha_interval_three']:.4f} (interval), {agreement['alpha_ordinal_three']:.4f} (ordinal)." if "alpha_interval_three" in agreement else "."))
        for pr in agreement.get("pairs", []):
            L.append(f"  {pr['raters']}: Spearman {pr['spearman']:.4f}, exact {pr['exact']:.3f}, within one point {pr['within_one']:.3f}")
        if "leave_one_out_spearman" in agreement:
            L.append("  each rater against the mean of the other two (Spearman): " + ", ".join(f"{k} {v:.4f}" for k, v in agreement["leave_one_out_spearman"].items()))
    for gold_name in dict.fromkeys(r["gold"] for r in bench_rows):
        rows_g = [r for r in bench_rows if r["gold"] == gold_name]
        L.append(f"Sentence-level evaluation on {rows_g[0]['n']} benchmark sentences (Spearman with the {gold_name}; "
                 f"difference from VADER with 95% bootstrap CI, {args.boot_bench} resamples):")
        for r in rows_g:
            extra = f"   {r['delta']:+.4f} [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}]" if "delta" in r else ""
            L.append(f"  {r['variant']:38s} {r['spearman_human']:7.4f}   coverage {r['coverage']:.4f}   sign acc. {r['sign_accuracy_nonneutral']:.4f}{extra}")
        L.append(f"  sign acc. = share of the {rows_g[0]['n_nonneutral']} sentences with a non-zero gold rating where the score has the same sign.")
        for pr in [q for q in bench_pairs if q["gold"] == gold_name]:
            L.append(f"  paired: {pr['a']} minus {pr['b']}: {pr['delta']:+.4f} [{pr['delta_ci95'][0]:+.4f}, {pr['delta_ci95'][1]:+.4f}]")
    L.append("")
    L.append("Most negative curated terms (>= 500 training reviews): " + ", ".join(f"{w} ({v:+.2f})" for w, v, _ in top_neg))
    L.append("Most positive curated terms (>= 500 training reviews): " + ", ".join(f"{w} ({v:+.2f})" for w, v, _ in top_pos))
    L.append("Most negative automatic-tier words (>= 500 training reviews): " + ", ".join(f"{w} ({v:+.2f})" for w, v, _ in top_neg_auto))
    L.append("Most positive automatic-tier words (>= 500 training reviews): " + ", ".join(f"{w} ({v:+.2f})" for w, v, _ in top_pos_auto))
    report = "\n".join(L)
    with open(os.path.join(outdir, base + "_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    def md_table(rows, heads, fmt):
        s = "| " + " | ".join(heads) + " |\n|" + "|".join(["---"] * len(heads)) + "|\n"
        for r in rows:
            s += "| " + " | ".join(fmt(r)) + " |\n"
        return s

    def dtxt(r, key):
        return "%+.4f (%+.4f to %+.4f)" % (r[key], r[key + "_ci95"][0], r[key + "_ci95"][1]) if key in r else "reference"

    tab_reviews = md_table(test_rows, ["Lexicon", "Added entries", "AUC", "AUC, length-stratified", "Difference from VADER, length-stratified (95% CI)",
                                       "Balanced accuracy", "Coverage"],
                           lambda r: [r["variant"], "%d" % r["n_added_entries"], "%.4f" % r["auc"], "%.4f" % r["auc_length_stratified"],
                                      dtxt(r, "delta_auc_strat"), "%.4f" % r["balanced_accuracy"], "%.4f" % r["coverage"]])
    tab_bench = ""
    for gold_name in dict.fromkeys(r["gold"] for r in bench_rows):
        rows_g = [r for r in bench_rows if r["gold"] == gold_name]
        tab_bench += "\nGold standard: %s (%d sentences).\n\n" % (gold_name, rows_g[0]["n"])
        tab_bench += md_table(rows_g, ["Lexicon", "Spearman", "Difference from VADER (95% CI)", "Sign accuracy", "Coverage"],
                              lambda r: [r["variant"], "%.4f" % r["spearman_human"], dtxt(r, "delta"), "%.4f" % r["sign_accuracy_nonneutral"], "%.4f" % r["coverage"]])
        bp = [q for q in bench_pairs if q["gold"] == gold_name]
        if bp:
            tab_bench += "\nPaired differences (95% CI):\n\n" + md_table(bp, ["Lexicon A", "Lexicon B", "Spearman A minus B (95% CI)"],
                                                                         lambda q: [q["a"], q["b"], "%+.4f (%+.4f to %+.4f)" % (q["delta"], q["delta_ci95"][0], q["delta_ci95"][1])])
    tier_txt = ", ".join("%d %s" % (desc["by_tier"].get(k, 0), k) for k in ALL_TIERS)
    tab_pairs = md_table(pair_rows, ["Lexicon A", "Lexicon B", "AUC A minus B (95% CI)", "Length-stratified AUC A minus B (95% CI)"],
                         lambda q: [q["a"], q["b"], "%+.4f (%+.4f to %+.4f)" % (q["delta_auc"], q["delta_auc_ci95"][0], q["delta_auc_ci95"][1]),
                                    "%+.4f (%+.4f to %+.4f)" % (q["delta_auc_strat"], q["delta_auc_strat_ci95"][0], q["delta_auc_strat_ci95"][1])]) if pair_rows else ""
    agr_txt = ""
    if agreement:
        agr_txt = ("Agreement between the two annotators: Spearman %.2f, Krippendorff's alpha %.2f (interval scale)." %
                   (agreement["spearman_A1_A2"], agreement["alpha_interval_two"]))
        if "alpha_interval_three" in agreement:
            agr_txt += (" With the third rater (the author, rating blind to annotator labels and lexicon scores) alpha is %.2f; "
                        "pairwise Spearman correlations range from %.2f to %.2f." %
                        (agreement["alpha_interval_three"], min(p["spearman"] for p in agreement["pairs"]), max(p["spearman"] for p in agreement["pairs"])))
    auto_examples = ", ".join("*%s*" % w for w, _, _ in freq_auto[:8])
    share_any = "%.1f%%" % (100 * has_any.mean())
    d_cur = "%+.4f (95%% bootstrap CI %+.4f to %+.4f)" % (cur_row["delta_auc"], cur_row["delta_auc_ci95"][0], cur_row["delta_auc_ci95"][1])
    d_full = "%+.4f (95%% bootstrap CI %+.4f to %+.4f)" % (d_auc, ci[0], ci[1])
    tab_genre = ""
    if genre_rows:
        tab_genre = md_table(genre_rows, ["Genre", "Reviews", "Recommending share", "Mean score, VADER", "Mean score, GamerLex full",
                                          "AUC, length-stratified, VADER", "with curated tiers", "with all tiers"],
                             lambda r: [r["genre"], "%d" % r["n"], "%.3f" % r["rec_share"], "%+.3f" % r["mean_vader"], "%+.3f" % r["mean_gamerlex"],
                                        "%.4f" % r["auc_strat_vader"], "%.4f" % r["auc_strat_curated"], "%.4f" % r["auc_strat_gamerlex"]])
    n_ind_all, n_ind_new = "{:,}".format(len(ind_all)), "{:,}".format(len(ind_new))
    n_ind_overlap = "{:,}".format(len(ind_all) - len(ind_new))
    ind_k_txt = "{:,}".format(IND_K)
    share_has = "%.1f%%" % (100 * has.mean())
    p_test_txt = "%.0f%%" % (100 * (1 - P_TRAIN - P_VAL))
    snippet = 'sia.lexicon.update({t: float(v) for t, v in (line.rstrip("\\n").split("\\t") for line in f)})'
    readme = f"""# GamerLex v{VERSION}

GamerLex is a sentiment lexicon for video game reviews, built from {c['reviews_total']:,} Steam reviews. Its curated part is a hand-audited vocabulary of gaming slang and jargon; an optional automatic tier adds frequent review vocabulary selected by rule. Every entry carries a corpus-derived valence on the VADER scale. The lexicon is designed to be used together with VADER: its entries add vocabulary that VADER does not know and replace VADER values for words whose meaning shifts in game reviews (for example *kill*, *scary*, *brutal*).

Built on {results['built']}. All numbers in this file were produced by `gamerlex_build.py` in the same run.

## Contents

| File | Description |
|---|---|
| `{base}.csv` | The lexicon with statistics (see columns below) |
| `{base}_vader.txt` | Tab-separated `term<TAB>valence`, all tiers, loadable into VADER (multi-word terms joined with `_`) |
| `{base}_curated_vader.txt` | The same without the automatic tier |
| `{base}.json` | Term, valence, tier and category as JSON |
| `gamerlex.py` | Helper class that plugs the lexicon into VADER and handles multi-word terms |
| `{base}_results.json` | Every figure reported here, machine readable |
| `{base}_anchors.csv` | Anchor words used for scaling to the VADER range |
| `{base}_dropped.csv` | Candidate entries removed by the build filters, with the reason |

## Size

{desc['n_entries']} entries: {desc['n_curated']} in three hand-audited tiers and {desc['n_auto']} in the automatic tier ({tier_txt}). {desc['n_multiword']} are multi-word terms.

- **core**: gaming slang and jargon (genres, mechanics, roles, platform and industry terms).
- **extended**: general evaluative vocabulary that is frequent in reviews and missing from VADER, including misspellings.
- **vader_override**: words VADER rates as negative that are descriptive or positive in game reviews (combat vocabulary, horror vocabulary, slang with reversed polarity).
- **auto**: frequent content words that are absent from VADER and from the three tiers above and are clearly associated with the recommendation. They were selected by a fixed rule (step 5 below) and were not reviewed by hand. The most frequent ones are {auto_examples}. Many of these name an aspect of a game or structure an argument, so their valence tells in what kind of review the word tends to occur. That is weaker evidence of sentiment than the valence of an evaluative word. The tier is optional and can be switched off (see Usage).

{desc['n_also_in_vader']} entries also occur in VADER; for {desc['n_sign_differs_from_vader']} of them the GamerLex valence has the opposite sign.

## Columns of `{base}.csv`

| Column | Meaning |
|---|---|
| `term` | Lower-case term; multi-word terms contain spaces |
| `valence` | Sentiment valence on the VADER scale (-4 to +4) |
| `tier` | `core`, `extended`, `vader_override` or `auto` |
| `category` | Coarse label inherited from the seed term (mechanic, reaction, industry, skill, toxic, meta); indicative only. `general` for the automatic tier |
| `multiword` | 1 for multi-word terms |
| `n_reviews` | Number of training reviews that contain the term |
| `rec_share` | Share of those reviews that recommend the game |
| `expected_rec_share` | Share expected from review length alone |
| `assoc` | `rec_share` minus `expected_rec_share` |
| `log_odds` | Length-stratified (Mantel-Haenszel) log odds ratio of recommendation |
| `z` | Standardised difference between observed and expected recommending reviews |
| `vader_valence` | VADER value of the term, if VADER contains it |

## How the valence is computed

1. The corpus is split at random into training ({P_TRAIN:.0%}), validation ({P_VAL:.0%}) and test ({p_test_txt}) parts (seed {args.seed}). Reviews that contain a benchmark sentence are left out.
2. Short reviews are almost always positive and long reviews are more often negative (recommending share {base_b[0]:.3f} for 1-{LEN_EDGES[0]} tokens, {base_b[-1]:.3f} for more than {LEN_EDGES[-1]} tokens). Each term is therefore compared with reviews of similar length: the association is a Mantel-Haenszel log odds ratio stratified by {NB} length bands, estimated on the training part only.
3. The log odds ratio is shrunk towards zero for rare terms (factor n / (n + {SHRINK_N0:.0f})) and mapped to the VADER range with a slope fitted on {cal_stats['n_anchors']} anchor words. Anchors are words that VADER contains, GamerLex does not, and that occur in at least {ANCHOR_MIN_N} training reviews. On these anchors the corpus measure and VADER agree in sign for {cal_stats['sign_agreement']:.1%} of the words (Spearman {cal_stats['spearman']:.2f}). The mapping chosen on the validation part is `{best}` (slope {kp:.3f} for positive and {kn:.3f} for negative log odds).
4. Curated terms with fewer than {MIN_TRAIN_N} training reviews are removed. A `vader_override` candidate is removed, and the VADER value kept, if the corpus also shows a clearly negative association (`assoc` <= {OVR_DIFF} and `z` <= {OVR_Z}).
5. Candidates for the automatic tier are the {ind_k_txt} most frequent tokens in a sample of {IND_SAMPLE:,} reviews (frequency only, no labels). A token enters the tier if it consists of letters (hyphens allowed inside), is not a function word (scikit-learn English stop list, VADER negations and boosters, contractions with *n't*), is not written with a capital initial in at least {CAP_RATIO:.0%} of its mid-sentence occurrences (proper names), is absent from VADER and from the curated tiers, occurs in at least {AUTO_MIN_N} training reviews, and has |`z`| >= {AUTO_MIN_Z:g} and |`valence`| >= {AUTO_MIN_VAL}. These thresholds are constants of the build script and were not tuned on the test part.

## Results

Review-level evaluation on {len(test_texts):,} held-out reviews. The task is to rank reviews by the VADER compound score so that recommending reviews come first (ROC AUC). {share_has} of the test reviews contain at least one curated GamerLex term and {share_any} contain a term of any tier.

{tab_reviews}
Paired differences between lexicons on the same bootstrap resamples:

{tab_pairs}
"GamerLex curated" is the three hand-audited tiers together; "GamerLex full" adds the automatic tier. Review length is strongly related to the recommendation, so the length-stratified AUC (computed within {NB} length bands and averaged) is the fairer comparison. Unstratified AUC difference from VADER alone: {d_cur} with the curated tiers and {d_full} with all tiers ({args.boot} resamples). Coverage is the share of reviews that receive a non-zero score. Balanced accuracy treats a score of zero or above as a recommendation.

The two "induced" rows are uncurated baselines. They take the {ind_k_txt} most frequent tokens of the corpus, compute valence exactly as for GamerLex and keep every token with a clear association (|z| >= {IND_Z:g}): {n_ind_all} tokens, of which {n_ind_new} are not in VADER. The first adds only tokens that VADER does not contain; the second adds all of them and also replaces the VADER values of the {n_ind_overlap} tokens that both contain. Both rows show what a purely automatic lexicon reaches on this task, and they set the level against which the curated vocabulary has to be judged.

GamerLex valences were estimated from recommendations in the training part, so this comparison shows how well they generalise to unseen reviews. VADER was built without such information, which should be kept in mind when reading the difference.
"""
    if tab_genre:
        readme += f"""
Results by genre on the same test sample ({genre_note}):

{tab_genre}"""
        if genre_corr:
            readme += f"""
Across the {genre_corr['n_genres']} genres, the Spearman correlation between the share of recommending reviews and the mean score is {genre_corr['spearman_recshare_vs_mean_vader']:+.3f} for VADER and {genre_corr['spearman_recshare_vs_mean_gamerlex']:+.3f} for VADER with all GamerLex tiers.
"""
    if bench_rows:
        readme += f"""
Sentence-level evaluation on benchmark sentences rated on a five-point scale (Spearman correlation with the human rating; differences from VADER with a bootstrap interval of {args.boot_bench} resamples):
{tab_bench}
Sign accuracy is computed on sentences with a non-zero gold rating. {agr_txt} The benchmark sentences were sampled to contain at least one candidate term and agreement between raters is moderate, so these figures are indicative.
"""
    readme += f"""
In the override tier ({ovr_summary['n']} terms) the mean VADER valence is {ovr_summary['mean_vader_valence']:+.2f} and the mean GamerLex valence is {ovr_summary['mean_gamerlex_valence']:+.2f}; the sign differs for {ovr_summary['share_sign_differs']:.1%} of the terms.

## Usage

```python
from gamerlex import GamerLexAnalyzer

gl = GamerLexAnalyzer("{base}.csv")                      # all four tiers
print(gl.polarity_scores("gg, the boss fights are brutal and the horror is terrifying"))

from gamerlex import CURATED
gl_curated = GamerLexAnalyzer("{base}.csv", tiers=CURATED)   # without the automatic tier
```

Without the helper, single-word entries can be added to VADER directly (use `{base}_curated_vader.txt` to leave out the automatic tier):

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
with open("{base}_vader.txt", encoding="utf-8") as f:
    {snippet}
```

Reference tokenizer used for all counts: lower-casing followed by the regular expression `{TOK.pattern}`.

When reading `{base}.csv` with pandas, pass `keep_default_na=False`: the lexicon contains the term *null*, which pandas would otherwise read as a missing value.

## Limitations

- Valence is derived from review-level recommendations, so it measures the company a term keeps, which is a proxy for sentence-level sentiment.
- The automatic tier was not reviewed by hand. It contains aspect and discourse words whose valence reflects typical use in Steam reviews and may not hold in other text types.
- Negation and sense ambiguity are left to the sentiment analyser. Examples: *not scary* is usually a complaint, and *sick* is negative in *sick of* and positive in *sick game*.
- The candidate terms came from embedding neighbours of slang seeds, a manual audit and a glossary cross-check; coverage of game jargon is broad but not exhaustive.
- The `category` labels are inherited from seed terms and were not assigned one by one.
- The corpus is English-language Steam reviews; values may not transfer to other platforms or languages.

## Citation

[PLACEHOLDER: add the reference of the accompanying paper and the DOI of the archived release once they exist.]

Guzsvinecz, T. ({datetime.date.today().year}). GamerLex: a lexicon of gaming slang and jargon with corpus-derived valence (Version {VERSION}) [Data set].

## Licence

[PLACEHOLDER: licence to be confirmed by the author. CC BY 4.0 is the usual choice for a citable lexical resource.]
"""
    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(readme)
    cff = f"""cff-version: 1.2.0
message: "If you use GamerLex, please cite it as below."
title: "GamerLex: a lexicon of gaming slang and jargon with corpus-derived valence"
type: dataset
version: "{VERSION}"
date-released: "{results['built']}"
authors:
  - family-names: Guzsvinecz
    given-names: Tibor
    affiliation: "University of Pannonia"
keywords:
  - sentiment analysis
  - lexicon
  - video games
  - Steam reviews
  - slang
"""
    with open(os.path.join(outdir, "CITATION.cff"), "w", encoding="utf-8", newline="\n") as f:
        f.write(cff)

    print("\n" + report)
    print("\nKiadasi csomag:", outdir)
    print(f"Futasido: {time.time() - t0:.0f} mp")


if __name__ == "__main__":
    main()
