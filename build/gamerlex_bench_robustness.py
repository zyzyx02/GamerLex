# -*- coding: utf-8 -*-
"""
GamerLex :: mondatszintu robusztussagi ellenorzesek a benchmarkon
=================================================================

MIERT KELL:
    Az A1 es A2 annotator az annotalo lapon latta, hogy a review ajanlja-e a
    jatekot. A GamerLex valenciai szinten az ajanlasbol szarmaznak, igy az
    emberi ertekeles es a lexikon kozotti egyezes egy resze a lathato cimken
    keresztul is letrejohet. Ez a szkript negy ellenorzest vegez:
      1. minden ertekelo (A1, A2, GT) es minden lexikon korrelacioja a cimkevel;
      2. cimken beluli Spearman: kulon az ajanlo es a nem ajanlo review-k
         mondatain, a ket csoport elemszamaval sulyozva (a cimke hatasa igy
         kiesik); paros bootstrap, a cimke szerint retegezve;
      3. a "the"-reszminta: azok a mondatok, amelyek csak a korai jeloltlista
         "the" tokenje miatt kerultek a benchmarkba, tehat gyakorlatilag
         szuretlen mondatok (a kivalasztasi torzitas ellenorzese);
      4. azok a mondatok, amelyekben nincs az audit soran hozzaadott kuralt
         szo (az auditlapon latszott, hany benchmarkmondatban szerepel egy
         szo; ez a kuralt szavak kivalasztasat befolyasolhatta).
    Elotte a teljes mintan is kiszamolja a paros kulonbsegeket.
    Mindegyik ellenorzes ket aranystandarddal fut: a harom ertekelo medianja,
    illetve csak GT (o vakon, cimke nelkul ertekelt).

MIT OLVAS (a gamerlex_build.py kimenetei es a benchmark):
    release/GamerLex_v1.0.csv
    eval/GamerLex_induced_baseline.csv
    bench/bench_merged.csv
    bench/gamerlex_bench_GT_vak.xlsx
    lexicon_audit.xlsx                 (nem kotelezo; a 4. ellenorzeshez)
    A mappat maga keresi: a szkript mappaja, az aktualis mappa
    output/gamerslang almappaja, az aktualis mappa, H:/output/gamerslang.
    Ha nem talalja, kiirja, hol kereste.

MIT IR:
    eval/GamerLex_bench_robustness.txt    olvashato riport
    eval/GamerLex_bench_robustness.json   ugyanez gepileg olvashatoan
    Meglevo kimenetet csak --force eseten ir felul.

HOGYAN FUTTASD (a H:/ gyokerbol):
    python output\\gamerslang\\gamerlex_bench_robustness.py
    python output\\gamerslang\\gamerlex_bench_robustness.py --force

Fuggosegek: pandas, numpy, openpyxl, vaderSentiment, valamint a
gamerlex_build.py ugyanabban a mappaban. Futasido: par perc.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def find_dir():
    cwd = os.getcwd()
    cands = [HERE, os.path.join(cwd, "output", "gamerslang"), cwd, r"H:\output\gamerslang"]
    need = [os.path.join("release", "GamerLex_v1.0.csv"), os.path.join("eval", "GamerLex_induced_baseline.csv"),
            os.path.join("bench", "bench_merged.csv"), os.path.join("bench", "gamerlex_bench_GT_vak.xlsx")]
    for c in cands:
        if all(os.path.exists(os.path.join(c, n)) for n in need):
            return c
    print("[HIBA] Nem talalom a szukseges fajlokat. Itt kerestem:")
    for c in cands:
        for n in need:
            print("   ", os.path.abspath(os.path.join(c, n)))
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap ismetlesek szama")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--force", action="store_true", help="meglevo kimenet felulirasa")
    args = ap.parse_args()

    try:
        import gamerlex_build as gb
    except ImportError:
        print("[HIBA] A gamerlex_build.py-nak ugyanebben a mappaban kell lennie:", HERE)
        sys.exit(1)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA

    base = find_dir()
    out_txt = os.path.join(base, "eval", "GamerLex_bench_robustness.txt")
    out_json = os.path.join(base, "eval", "GamerLex_bench_robustness.json")
    if (os.path.exists(out_txt) or os.path.exists(out_json)) and not args.force:
        print("[MEGALLTAM] Mar letezik kimenet, nem irom felul (--force kell):", out_txt)
        sys.exit(1)
    print("Mappa:", base)

    gt = pd.read_excel(os.path.join(base, "bench", "gamerlex_bench_GT_vak.xlsx"), sheet_name="ertekeles")
    gt["GT"] = pd.to_numeric(gt["GT_sentiment"], errors="coerce")
    gt = gt[gt.GT.isin([-2, -1, 0, 1, 2])][["sentence_id", "GT"]].drop_duplicates("sentence_id")
    bm = pd.read_csv(os.path.join(base, "bench", "bench_merged.csv"), encoding="utf-8-sig")
    bm["sentence"] = bm.sentence.astype(str)
    bm = bm[~bm.sentence.str.startswith("#NAME") & bm.A1_sent.notna() & bm.A2_sent.notna()]
    b = bm.merge(gt, on="sentence_id", how="inner").reset_index(drop=True)
    if len(b) < 100:
        print("[HIBA] Tul keves mondat a harom ertekelovel:", len(b))
        sys.exit(1)
    A1 = b.A1_sent.astype(float).values
    A2 = b.A2_sent.astype(float).values
    G = b.GT.astype(float).values
    med = np.median(np.vstack([A1, A2, G]), axis=0)
    y = b.recommended.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).values
    golds = {"median of three raters": med, "third rater only (blind)": G}

    g = pd.read_csv(os.path.join(base, "release", "GamerLex_v1.0.csv"), keep_default_na=False, na_values=[""], encoding="utf-8-sig")
    ind = pd.read_csv(os.path.join(base, "eval", "GamerLex_induced_baseline.csv"), keep_default_na=False, na_values=[""], encoding="utf-8-sig")
    vl = SIA().lexicon
    sents = b.sentence.tolist()
    joiner = gb.build_joiner([t for t in g.term if " " in t])
    sj = [joiner(s) for s in sents]
    e_full = {t.replace(" ", "_"): float(v) for t, v in zip(g.term, g.valence)}
    e_cur = {t.replace(" ", "_"): float(v) for t, v, ti in zip(g.term, g.valence, g.tier) if ti != "auto"}
    i_all = dict(zip(ind.word.astype(str), ind.valence.astype(float)))
    i_new = {w: v for w, v in i_all.items() if w not in vl}
    lex = {"VADER": (None, sents), "VADER + GamerLex curated": (e_cur, sj), "VADER + GamerLex full": (e_full, sj),
           "VADER + induced (new words)": (i_new, sents), "VADER + induced (all frequent words)": (i_all, sents)}
    sc = {k: gb.score_texts(gb.make_analyzer(SIA, e), t) for k, (e, t) in lex.items()}
    full = "VADER + GamerLex full"
    pairs = [(full, "VADER"), (full, "VADER + induced (new words)"), (full, "VADER + induced (all frequent words)"),
             (full, "VADER + GamerLex curated"), ("VADER + GamerLex curated", "VADER")]

    def ci(x):
        x = np.asarray([v for v in x if v == v])
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan")] * 2

    res = {"n_sentences": int(len(b)), "n_recommended": int(y.sum()), "bootstrap": args.boot}
    L = ["GamerLex :: sentence-level robustness checks", "=" * 60,
         f"Sentences rated by all three raters: {len(b)}; from recommending reviews: {int(y.sum())} ({y.mean():.1%}).", ""]

    # 1. korrelacio a lathato cimkevel
    yl = y.astype(float)
    res["spearman_with_label"] = {"A1": gb.spearman(A1, yl), "A2": gb.spearman(A2, yl), "GT": gb.spearman(G, yl),
                                  "median of three raters": gb.spearman(med, yl)}
    res["spearman_lexicon_with_label"] = {k: gb.spearman(v, yl) for k, v in sc.items()}
    L.append("1. Spearman correlation with the review label (A1 and A2 saw the label while rating, GT did not):")
    L.append("   raters:   " + ", ".join(f"{k} {v:.4f}" for k, v in res["spearman_with_label"].items()))
    L.append("   lexicons: " + ", ".join(f"{k} {v:.4f}" for k, v in res["spearman_lexicon_with_label"].items()))
    L.append("")

    # 1b. teljes minta, mindket aranystandarddal (paros bootstrap)
    rng0 = np.random.default_rng(args.seed + 2)
    res["overall"] = {}
    L.append("1b. Whole benchmark:")
    for gname, gold in golds.items():
        point = {k: gb.spearman(v, gold) for k, v in sc.items()}
        boot = {k: [] for k in sc}
        for _ in range(max(args.boot, 0)):
            ii = rng0.integers(0, len(b), len(b))
            for k, v in sc.items():
                boot[k].append(gb.spearman(v[ii], gold[ii]))
        boot = {k: np.array(v) for k, v in boot.items()}
        prs = [{"a": a, "b": c, "delta": point[a] - point[c], "ci95": ci(boot[a] - boot[c])} for a, c in pairs]
        res["overall"][gname] = {"spearman": point, "paired": prs}
        L.append(f"   gold: {gname}")
        L.append("     " + ", ".join(f"{k} {v:.4f}" for k, v in point.items()))
        for p in prs:
            L.append(f"     {p['a']} minus {p['b']}: {p['delta']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")
    L.append("")

    # 2. cimken beluli korrelacio
    groups = [np.where(y)[0], np.where(~y)[0]]

    def pooled(v, gold, idx_list):
        num = den = 0.0
        for ix in idx_list:
            r = gb.spearman(v[ix], gold[ix])
            if r == r:
                num += r * len(ix)
                den += len(ix)
        return num / den if den else float("nan")

    rng = np.random.default_rng(args.seed)
    res["within_label"] = {}
    L.append("2. Within-label Spearman (computed separately for recommending and non-recommending reviews, weighted by group size):")
    for gname, gold in golds.items():
        point = {k: pooled(v, gold, groups) for k, v in sc.items()}
        boot = {k: [] for k in sc}
        for _ in range(max(args.boot, 0)):
            ib = [rng.choice(ix, len(ix)) for ix in groups]
            for k, v in sc.items():
                boot[k].append(pooled(v, gold, ib))
        boot = {k: np.array(v) for k, v in boot.items()}
        prs = [{"a": a, "b": c, "delta": point[a] - point[c], "ci95": ci(boot[a] - boot[c])} for a, c in pairs]
        res["within_label"][gname] = {"spearman": point, "paired": prs,
                                      "by_label": {lab: {k: gb.spearman(v[ix], gold[ix]) for k, v in sc.items()}
                                                   for lab, ix in (("recommended", groups[0]), ("not recommended", groups[1]))}}
        L.append(f"   gold: {gname}")
        L.append("     " + ", ".join(f"{k} {v:.4f}" for k, v in point.items()))
        for p in prs:
            L.append(f"     {p['a']} minus {p['b']}: {p['delta']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")
    L.append("")

    # 3. "the"-reszminta
    def toks(s):
        return [t.strip() for t in str(s).split(",") if t.strip()]

    the_only = b.slang_found.apply(lambda s: toks(s) == ["the"]).values if "slang_found" in b.columns else np.zeros(len(b), bool)
    ix0 = np.where(the_only)[0]
    res["the_only_subset"] = {"n": int(len(ix0)), "share_recommended": float(y[ix0].mean()) if len(ix0) else float("nan")}
    L.append(f"3. Sentences that entered the benchmark only through the token 'the' (approximately unselected): n = {len(ix0)}, "
             f"recommending share {res['the_only_subset']['share_recommended']:.3f}.")
    if len(ix0) >= 50:
        rng = np.random.default_rng(args.seed + 1)
        for gname, gold in golds.items():
            point = {k: gb.spearman(v[ix0], gold[ix0]) for k, v in sc.items()}
            boot = {k: [] for k in sc}
            for _ in range(max(args.boot, 0)):
                ii = rng.choice(ix0, len(ix0))
                for k, v in sc.items():
                    boot[k].append(gb.spearman(v[ii], gold[ii]))
            boot = {k: np.array(v) for k, v in boot.items()}
            prs = [{"a": a, "b": c, "delta": point[a] - point[c], "ci95": ci(boot[a] - boot[c])} for a, c in pairs]
            res["the_only_subset"][gname] = {"spearman": point, "paired": prs}
            L.append(f"   gold: {gname}")
            L.append("     " + ", ".join(f"{k} {v:.4f}" for k, v in point.items()))
            for p in prs:
                L.append(f"     {p['a']} minus {p['b']}: {p['delta']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")
    L.append("")

    # 4. az audit soran hozzaadott kuralt szavakat nem tartalmazo mondatok
    #    (az auditlapon latszott, hany benchmarkmondatban szerepel egy szo)
    audit_path = os.path.join(base, "lexicon_audit.xlsx")
    if os.path.exists(audit_path):
        import re as _re
        au = pd.read_excel(audit_path, sheet_name="audit", keep_default_na=False)
        new_terms = set(au[au.eredet.astype(str).str.startswith("ÚJ")].word.astype(str).str.strip().str.lower())
        added = set(g[g.tier != "auto"].term) & new_terms
        tokre = _re.compile(r"[a-z0-9'*-]+")
        a_single = {t for t in added if " " not in t}
        a_multi = [tuple(t.split()) for t in added if " " in t]

        def has_added(snt):
            tk = tokre.findall(snt.lower())
            if set(tk) & a_single:
                return True
            return any(tuple(tk[j:j + len(q)]) == q for q in a_multi for j in range(len(tk) - len(q) + 1))

        ixa = np.where(~b.sentence.apply(has_added).values)[0]
        res["without_audit_additions"] = {"n_added_terms": int(len(added)), "n": int(len(ixa))}
        L.append(f"4. Sentences without any curated term added during the audit ({len(added)} terms): n = {len(ixa)}.")
        rng = np.random.default_rng(args.seed + 3)
        for gname, gold in golds.items():
            point = {k: gb.spearman(v[ixa], gold[ixa]) for k, v in sc.items()}
            boot = {k: [] for k in sc}
            for _ in range(max(args.boot, 0)):
                ii = rng.choice(ixa, len(ixa))
                for k, v in sc.items():
                    boot[k].append(gb.spearman(v[ii], gold[ii]))
            boot = {k: np.array(v) for k, v in boot.items()}
            prs = [{"a": a, "b": c, "delta": point[a] - point[c], "ci95": ci(boot[a] - boot[c])} for a, c in pairs]
            res["without_audit_additions"][gname] = {"spearman": point, "paired": prs}
            L.append(f"   gold: {gname}")
            L.append("     " + ", ".join(f"{k} {v:.4f}" for k, v in point.items()))
            for p in prs:
                L.append(f"     {p['a']} minus {p['b']}: {p['delta']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")
        L.append("")
    else:
        L.append("4. lexicon_audit.xlsx not found; check without audit additions skipped.")
        L.append("")
    L.append(f"Intervals: 95% percentile bootstrap, {args.boot} resamples (within-label: resampled separately within each label group).")

    report = "\n".join(L)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(report)
    print("\nKiirva:", out_txt)


if __name__ == "__main__":
    main()
