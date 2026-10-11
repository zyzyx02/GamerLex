# GamerLex v1.0

GamerLex is a sentiment lexicon for video game reviews, built from 4,691,606 Steam reviews. Its curated part is a hand-audited vocabulary of gaming slang and jargon; an optional automatic tier adds frequent review vocabulary selected by rule. Every entry carries a corpus-derived valence on the VADER scale. The lexicon is designed to be used together with VADER: its entries add vocabulary that VADER does not know and replace VADER values for words whose meaning shifts in game reviews (for example *kill*, *scary*, *brutal*).

Built on 2026-10-11. All numbers in this file were produced by `gamerlex_build.py` in the same run.

## Contents

| File | Description |
|---|---|
| `GamerLex_v1.0.csv` | The lexicon with statistics (see columns below) |
| `GamerLex_v1.0_vader.txt` | Tab-separated `term<TAB>valence`, all tiers, loadable into VADER (multi-word terms joined with `_`) |
| `GamerLex_v1.0_curated_vader.txt` | The same without the automatic tier |
| `GamerLex_v1.0.json` | Term, valence, tier and category as JSON |
| `gamerlex.py` | Helper class that plugs the lexicon into VADER and handles multi-word terms |
| `GamerLex_v1.0_results.json` | Every figure reported here, machine readable |
| `GamerLex_v1.0_anchors.csv` | Anchor words used for scaling to the VADER range |
| `GamerLex_v1.0_dropped.csv` | Candidate entries removed by the build filters, with the reason |

## Size

8046 entries: 4223 in three hand-audited tiers and 3823 in the automatic tier (3593 core, 436 extended, 194 vader_override, 3823 auto). 281 are multi-word terms.

- **core**: gaming slang and jargon (genres, mechanics, roles, platform and industry terms).
- **extended**: general evaluative vocabulary that is frequent in reviews and missing from VADER, including misspellings.
- **vader_override**: words VADER rates as negative that are descriptive or positive in game reviews (combat vocabulary, horror vocabulary, slang with reversed polarity).
- **auto**: frequent content words that are absent from VADER and from the three tiers above and are clearly associated with the recommendation. They were selected by a fixed rule (step 5 below) and were not reviewed by hand. The most frequent ones are *just*, *story*, *bit*, *buy*, *different*, *experience*, *music*, *work*. Many of these name an aspect of a game or structure an argument, so their valence tells in what kind of review the word tends to occur. That is weaker evidence of sentiment than the valence of an evaluative word. The tier is optional and can be switched off (see Usage).

292 entries also occur in VADER; for 178 of them the GamerLex valence has the opposite sign.

## Columns of `GamerLex_v1.0.csv`

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

1. The corpus is split at random into training (70%), validation (10%) and test (20%) parts (seed 42). Reviews that contain a benchmark sentence are left out.
2. Short reviews are almost always positive and long reviews are more often negative (recommending share 0.902 for 1-5 tokens, 0.697 for more than 320 tokens). Each term is therefore compared with reviews of similar length: the association is a Mantel-Haenszel log odds ratio stratified by 8 length bands, estimated on the training part only.
3. The log odds ratio is shrunk towards zero for rare terms (factor n / (n + 25)) and mapped to the VADER range with a slope fitted on 1449 anchor words. Anchors are words that VADER contains, GamerLex does not, and that occur in at least 500 training reviews. On these anchors the corpus measure and VADER agree in sign for 72.9% of the words (Spearman 0.54). The mapping chosen on the validation part is `piecewise` (slope 1.275 for positive and 1.320 for negative log odds).
4. Curated terms with fewer than 5 training reviews are removed. A `vader_override` candidate is removed, and the VADER value kept, if the corpus also shows a clearly negative association (`assoc` <= -0.1 and `z` <= -3.0).
5. Candidates for the automatic tier are the 30,000 most frequent tokens in a sample of 300,000 reviews (frequency only, no labels). A token enters the tier if it consists of letters (hyphens allowed inside), is not a function word (scikit-learn English stop list, VADER negations and boosters, contractions with *n't*), is not written with a capital initial in at least 50% of its mid-sentence occurrences (proper names), is absent from VADER and from the curated tiers, occurs in at least 100 training reviews, and has |`z`| >= 5 and |`valence`| >= 0.5. These thresholds are constants of the build script and were not tuned on the test part.

## Results

Review-level evaluation on 99,389 held-out reviews. The task is to rank reviews by the VADER compound score so that recommending reviews come first (ROC AUC). 62.8% of the test reviews contain at least one curated GamerLex term and 81.2% contain a term of any tier.

| Lexicon | Added entries | AUC | AUC, length-stratified | Difference from VADER, length-stratified (95% CI) | Balanced accuracy | Coverage |
|---|---|---|---|---|---|---|
| VADER | 0 | 0.7495 | 0.8245 | reference | 0.6974 | 0.8952 |
| VADER + core | 3593 | 0.7912 | 0.8509 | +0.0264 (+0.0249 to +0.0276) | 0.7438 | 0.9217 |
| VADER + core + extended | 4029 | 0.7961 | 0.8538 | +0.0293 (+0.0278 to +0.0307) | 0.7484 | 0.9228 |
| VADER + GamerLex curated | 4223 | 0.7856 | 0.8674 | +0.0430 (+0.0414 to +0.0446) | 0.7385 | 0.9230 |
| VADER + auto tier | 3823 | 0.8088 | 0.8656 | +0.0412 (+0.0396 to +0.0427) | 0.7520 | 0.9365 |
| VADER + GamerLex full | 8046 | 0.8292 | 0.8930 | +0.0685 (+0.0665 to +0.0706) | 0.7789 | 0.9521 |
| GamerLex full only | 8046 | 0.8003 | 0.7933 | -0.0311 (-0.0361 to -0.0257) | 0.7468 | 0.8085 |
| VADER + induced (new words) | 12414 | 0.8594 | 0.8915 | +0.0671 (+0.0641 to +0.0697) | 0.8069 | 0.9769 |
| VADER + induced (all frequent words) | 14429 | 0.8985 | 0.9137 | +0.0892 (+0.0857 to +0.0925) | 0.8378 | 0.9767 |

Paired differences between lexicons on the same bootstrap resamples:

| Lexicon A | Lexicon B | AUC A minus B (95% CI) | Length-stratified AUC A minus B (95% CI) |
|---|---|---|---|
| VADER + GamerLex full | VADER + induced (new words) | -0.0302 (-0.0328 to -0.0281) | +0.0015 (-0.0000 to +0.0032) |
| VADER + GamerLex full | VADER + induced (all frequent words) | -0.0693 (-0.0728 to -0.0661) | -0.0207 (-0.0227 to -0.0189) |
| VADER + GamerLex full | VADER + GamerLex curated | +0.0436 (+0.0416 to +0.0456) | +0.0256 (+0.0244 to +0.0269) |
| VADER + GamerLex full | VADER + auto tier | +0.0204 (+0.0184 to +0.0220) | +0.0274 (+0.0260 to +0.0287) |
| VADER + GamerLex curated | VADER + induced (new words) | -0.0738 (-0.0770 to -0.0709) | -0.0241 (-0.0260 to -0.0216) |
| VADER + GamerLex curated | VADER + induced (all frequent words) | -0.1129 (-0.1172 to -0.1094) | -0.0463 (-0.0488 to -0.0434) |

"GamerLex curated" is the three hand-audited tiers together; "GamerLex full" adds the automatic tier. Review length is strongly related to the recommendation, so the length-stratified AUC (computed within 8 length bands and averaged) is the fairer comparison. Unstratified AUC difference from VADER alone: +0.0361 (95% bootstrap CI +0.0339 to +0.0382) with the curated tiers and +0.0797 (95% bootstrap CI +0.0765 to +0.0827) with all tiers (200 resamples). Coverage is the share of reviews that receive a non-zero score. Balanced accuracy treats a score of zero or above as a recommendation.

The two "induced" rows are uncurated baselines. They take the 30,000 most frequent tokens of the corpus, compute valence exactly as for GamerLex and keep every token with a clear association (|z| >= 3): 14,429 tokens, of which 12,414 are not in VADER. The first adds only tokens that VADER does not contain; the second adds all of them and also replaces the VADER values of the 2,015 tokens that both contain. Both rows show what a purely automatic lexicon reaches on this task, and they set the level against which the curated vocabulary has to be judged.

GamerLex valences were estimated from recommendations in the training part, so this comparison shows how well they generalise to unseen reviews. VADER was built without such information, which should be kept in mind when reading the difference.

Results by genre on the same test sample (each review carries one genre label; genres with fewer than 200 test reviews are not shown):

| Genre | Reviews | Recommending share | Mean score, VADER | Mean score, GamerLex full | AUC, length-stratified, VADER | with curated tiers | with all tiers |
|---|---|---|---|---|---|---|---|
| Action | 60008 | 0.808 | +0.388 | +0.422 | 0.8207 | 0.8669 | 0.8908 |
| Adventure | 17634 | 0.834 | +0.498 | +0.532 | 0.8275 | 0.8613 | 0.8928 |
| Casual | 7237 | 0.849 | +0.505 | +0.506 | 0.8266 | 0.8615 | 0.8888 |
| Puzzle | 860 | 0.855 | +0.492 | +0.541 | 0.8461 | 0.8783 | 0.8991 |
| RPG | 3027 | 0.786 | +0.447 | +0.470 | 0.8255 | 0.8721 | 0.9013 |
| Racing | 1049 | 0.802 | +0.439 | +0.401 | 0.8628 | 0.8833 | 0.9049 |
| Simulation | 6935 | 0.790 | +0.413 | +0.383 | 0.8291 | 0.8762 | 0.8993 |
| Sports | 250 | 0.828 | +0.470 | +0.435 | 0.8852 | 0.9229 | 0.9432 |
| Strategy | 2191 | 0.772 | +0.425 | +0.448 | 0.8190 | 0.8624 | 0.8956 |

Across the 9 genres, the Spearman correlation between the share of recommending reviews and the mean score is +0.700 for VADER and +0.583 for VADER with all GamerLex tiers.

Sentence-level evaluation on benchmark sentences rated on a five-point scale (Spearman correlation with the human rating; differences from VADER with a bootstrap interval of 1000 resamples):

Gold standard: mean of two annotators (1462 sentences).

| Lexicon | Spearman | Difference from VADER (95% CI) | Sign accuracy | Coverage |
|---|---|---|---|---|
| VADER | 0.5706 | reference | 0.6335 | 0.8454 |
| VADER + core | 0.6149 | +0.0443 (+0.0286 to +0.0604) | 0.7130 | 0.9371 |
| VADER + core + extended | 0.6199 | +0.0494 (+0.0330 to +0.0660) | 0.7202 | 0.9405 |
| VADER + GamerLex curated | 0.6392 | +0.0686 (+0.0456 to +0.0899) | 0.7146 | 0.9405 |
| VADER + auto tier | 0.6152 | +0.0446 (+0.0253 to +0.0632) | 0.7242 | 0.9425 |
| VADER + GamerLex full | 0.6604 | +0.0898 (+0.0624 to +0.1157) | 0.7663 | 0.9795 |
| GamerLex full only | 0.4158 | -0.1548 (-0.2086 to -0.0995) | 0.6248 | 0.9008 |
| VADER + induced (new words) | 0.6257 | +0.0551 (+0.0247 to +0.0849) | 0.7695 | 0.9993 |
| VADER + induced (all frequent words) | 0.6249 | +0.0543 (+0.0122 to +0.0967) | 0.7886 | 0.9993 |

Paired differences (95% CI):

| Lexicon A | Lexicon B | Spearman A minus B (95% CI) |
|---|---|---|
| VADER + GamerLex full | VADER + induced (new words) | +0.0347 (+0.0139 to +0.0576) |
| VADER + GamerLex full | VADER + induced (all frequent words) | +0.0355 (+0.0052 to +0.0650) |
| VADER + GamerLex full | VADER + GamerLex curated | +0.0212 (+0.0057 to +0.0384) |
| VADER + GamerLex full | VADER + auto tier | +0.0452 (+0.0279 to +0.0624) |
| VADER + GamerLex curated | VADER + induced (new words) | +0.0135 (-0.0131 to +0.0404) |
| VADER + GamerLex curated | VADER + induced (all frequent words) | +0.0143 (-0.0208 to +0.0484) |

Gold standard: median of three raters (1462 sentences).

| Lexicon | Spearman | Difference from VADER (95% CI) | Sign accuracy | Coverage |
|---|---|---|---|---|
| VADER | 0.5787 | reference | 0.7256 | 0.8454 |
| VADER + core | 0.6176 | +0.0389 (+0.0231 to +0.0561) | 0.7980 | 0.9371 |
| VADER + core + extended | 0.6241 | +0.0454 (+0.0283 to +0.0632) | 0.8090 | 0.9405 |
| VADER + GamerLex curated | 0.6455 | +0.0668 (+0.0444 to +0.0879) | 0.8035 | 0.9405 |
| VADER + auto tier | 0.6225 | +0.0437 (+0.0250 to +0.0623) | 0.7947 | 0.9425 |
| VADER + GamerLex full | 0.6638 | +0.0851 (+0.0582 to +0.1115) | 0.8364 | 0.9795 |
| GamerLex full only | 0.4073 | -0.1715 (-0.2270 to -0.1172) | 0.6509 | 0.9008 |
| VADER + induced (new words) | 0.6338 | +0.0551 (+0.0238 to +0.0838) | 0.8364 | 0.9993 |
| VADER + induced (all frequent words) | 0.6197 | +0.0410 (-0.0016 to +0.0813) | 0.8430 | 0.9993 |

Paired differences (95% CI):

| Lexicon A | Lexicon B | Spearman A minus B (95% CI) |
|---|---|---|
| VADER + GamerLex full | VADER + induced (new words) | +0.0300 (+0.0087 to +0.0514) |
| VADER + GamerLex full | VADER + induced (all frequent words) | +0.0441 (+0.0156 to +0.0739) |
| VADER + GamerLex full | VADER + GamerLex curated | +0.0183 (+0.0027 to +0.0343) |
| VADER + GamerLex full | VADER + auto tier | +0.0414 (+0.0254 to +0.0581) |
| VADER + GamerLex curated | VADER + induced (new words) | +0.0117 (-0.0145 to +0.0375) |
| VADER + GamerLex curated | VADER + induced (all frequent words) | +0.0258 (-0.0078 to +0.0595) |

Sign accuracy is computed on sentences with a non-zero gold rating. Agreement between the two annotators: Spearman 0.54, Krippendorff's alpha 0.45 (interval scale). With the third rater (the author, rating blind to annotator labels and lexicon scores) alpha is 0.58; pairwise Spearman correlations range from 0.54 to 0.73. The benchmark sentences were sampled to contain at least one candidate term and agreement between raters is moderate, so these figures are indicative.

In the override tier (194 terms) the mean VADER valence is -2.00 and the mean GamerLex valence is +0.37; the sign differs for 72.7% of the terms.

## Usage

```python
from gamerlex import GamerLexAnalyzer

gl = GamerLexAnalyzer("GamerLex_v1.0.csv")                      # all four tiers
print(gl.polarity_scores("gg, the boss fights are brutal and the horror is terrifying"))

from gamerlex import CURATED
gl_curated = GamerLexAnalyzer("GamerLex_v1.0.csv", tiers=CURATED)   # without the automatic tier
```

Without the helper, single-word entries can be added to VADER directly (use `GamerLex_v1.0_curated_vader.txt` to leave out the automatic tier):

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
with open("GamerLex_v1.0_vader.txt", encoding="utf-8") as f:
    sia.lexicon.update({t: float(v) for t, v in (line.rstrip("\n").split("\t") for line in f)})
```

Reference tokenizer used for all counts: lower-casing followed by the regular expression `[a-z0-9'*-]+`.

When reading `GamerLex_v1.0.csv` with pandas, pass `keep_default_na=False`: the lexicon contains the term *null*, which pandas would otherwise read as a missing value.

## Limitations

- Valence is derived from review-level recommendations, so it measures the company a term keeps, which is a proxy for sentence-level sentiment.
- The automatic tier was not reviewed by hand. It contains aspect and discourse words whose valence reflects typical use in Steam reviews and may not hold in other text types.
- Negation and sense ambiguity are left to the sentiment analyser. Examples: *not scary* is usually a complaint, and *sick* is negative in *sick of* and positive in *sick game*.
- The candidate terms came from embedding neighbours of slang seeds, a manual audit and a glossary cross-check; coverage of game jargon is broad but not exhaustive.
- The `category` labels are inherited from seed terms and were not assigned one by one.
- The corpus is English-language Steam reviews; values may not transfer to other platforms or languages.

## Reproducing the build

The scripts in `build/` produce every file and figure of this release in one run:

```
python build/gamerlex_build.py
python build/gamerlex_bench_robustness.py
```

They need the Steam review corpus, the sentence benchmark and the third rater's ratings, which are not redistributed here because they contain review texts. With these inputs and the same package versions, the run reproduces the release (all random steps use fixed seeds). `build/lexicon_master_v2.csv` is the list of 4,344 audited candidate terms that enters the build.

## Citation

If you use GamerLex, please cite the accompanying paper:

[Preprint reference and link placeholder]

The archived release will receive a DOI on Zenodo: [DOI placeholder]

## Licence

The lexicon files are released under CC BY 4.0 and the code under the MIT licence; see `LICENSE-DATA.md` and `LICENSE`.
