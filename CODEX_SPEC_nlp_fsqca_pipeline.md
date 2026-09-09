\# Campus Founders — NLP + fs-QCA Pipeline (Codex build spec, v2)



\*\*Purpose.\*\* Replace manual thematic coding with two reproducible NLP methods, then run a

fuzzy-set QCA (fs-QCA) whose OUTCOME is \*feedback type\* (a belief-endorsing → belief-challenging

ladder), with acceptance embedded inside that ladder.



\*\*Two NLP methods (locked, both drawn from the project's own method list):\*\*

\- Unsupervised: \*\*BERTopic in guided / semi-supervised mode\*\* — emergent themes steered by

&#x20; Zellweger \& Zenger (Z\&Z) seed vocabulary. Covers the list entries "BERTopic",

&#x20; "Guided topic modeling", and "Semi-supervised topic modeling".

\- Supervised: \*\*SetFit\*\* (the list's "Few-shot classification") — few-shot contrastive

&#x20; fine-tuning of a sentence transformer, trained on the existing codebook. Optional

&#x20; LLM-as-second-coder (list Section G) on the hardest dimensions to lift inter-coder agreement.



\*\*Environment:\*\* local machine, Python 3.11, optional CUDA GPU. fs-QCA step in R.



\---



\## 0. The two-source insight (most important design rule)



Each pitch session has TWO speakers that must be separated:



| Source | Produces | Used as |

|--------|----------|---------|

| Founder turns (pitch + their answers) | Problem Articulation, Solution Fit, the 10 Theoretical-Ground dimensions, Innovation, Business Model, Team | \*\*fs-QCA CONDITIONS\*\* |

| Evaluator turns (panel questions + comments) | Feedback type (8 codes) | \*\*fs-QCA OUTCOME\*\* |



So segmentation must tag speaker role, and SetFit trains TWO families of classifiers — one on

founder turns (conditions), one on evaluator turns (feedback type, multi-label).



\---



\## 1. Repository layout



```

cf-nlp-qca/

├── data/

│   ├── raw/transcripts/            # one .txt per startup; diarised "Founder:" / "Panel:" prefixes if possible

│   ├── interim/segments.parquet    # segmented, speaker-tagged, cleaned

│   └── processed/

│       ├── topic\_membership.csv     # BERTopic theme proportions per startup (founder side)

│       ├── conditions\_setfit.csv    # SetFit P(high) per condition per startup (founder side)

│       ├── feedback\_setfit.csv      # SetFit P(type) per feedback code per startup (evaluator side)

│       └── qca\_calibrated.csv       # final fs-QCA input (conditions + feedback outcome\[s])

├── labels/

│   ├── condition\_labels.csv         # startup\_id + binary calibrated conditions (founder side)

│   └── feedback\_labels.csv          # startup\_id + multi-label feedback codes (evaluator side)

├── src/

│   ├── segment.py

│   ├── bertopic\_run.py

│   ├── setfit\_conditions.py         # train + predict founder-side conditions

│   ├── setfit\_feedback.py           # train + predict evaluator-side feedback type (multi-label)

│   ├── aggregate.py                 # segment scores -> startup scores

│   └── calibrate\_for\_qca.py

├── qca/

│   └── fsqca.R

├── requirements.txt

├── environment.md

└── README.md

```



\---



\## 2. Dependencies (`requirements.txt`)



```

bertopic>=0.16

sentence-transformers>=3.0

setfit>=1.1

umap-learn>=0.5

hdbscan>=0.8

scikit-learn>=1.4

pandas>=2.2

pyarrow>=15

nltk>=3.8

torch>=2.2            # CUDA build if GPU present

```



R side: `install.packages(c("QCA","SetMethods"))` (Dusa's QCA package).

GPU check for `environment.md`: `python -c "import torch; print(torch.cuda.is\_available())"`.

SetFit / BERTopic body = `sentence-transformers/all-mpnet-base-v2` (GPU) or `all-MiniLM-L6-v2` (CPU).



\---



\## 3. Segmentation (`segment.py`)



\- Input: `data/raw/transcripts/<startup\_id>.txt`.

\- Detect speaker role from diarisation prefixes (regex `^(Founder|Team|Panel|Mentor|Evaluator|Judge):`).

&#x20; If transcripts are NOT diarised, flag it — feedback-type coding needs evaluator turns isolated;

&#x20; ask the researcher to mark panel turns or supply a separate evaluator-comment file.

\- Split each turn into sentences (nltk `sent\_tokenize`). Columns: `segment\_id`, `startup\_id`,

&#x20; `role` (founder/evaluator), `source` (pitch/qa), `text`.

\- Drop segments < 4 tokens.

\- Output: `data/interim/segments.parquet`.



\---



\## 4. Unsupervised — guided BERTopic (`bertopic\_run.py`)



Run on FOUNDER segments only. Goal: emergent themes that are theory-connected.



Seed the model with Z\&Z-derived vocabulary so topics align to the belief-test-respond cycle:



```python

seed\_topic\_list = \[

&#x20;   \["problem", "pain", "need", "customer", "segment"],            # belief formation

&#x20;   \["assumption", "hypothesis", "mechanism", "causal", "theory"], # belief structure

&#x20;   \["test", "experiment", "pilot", "mvp", "a/b", "validate"],     # belief testing

&#x20;   \["churn", "conversion", "retention", "willingness to pay"],    # evidence / metrics

&#x20;   \["pivot", "revise", "update", "feedback", "iterate"],          # responding

]

```



Small-corpus tuning (defaults dump most segments to the -1 outlier cluster — override):



```python

from bertopic import BERTopic

from sentence\_transformers import SentenceTransformer

from umap import UMAP

from sklearn.cluster import KMeans



embed = SentenceTransformer("all-mpnet-base-v2")

umap\_model = UMAP(n\_neighbors=5, n\_components=5, min\_dist=0.0, metric="cosine", random\_state=42)

cluster\_model = KMeans(n\_clusters=12, random\_state=42)



topic\_model = BERTopic(

&#x20;   embedding\_model=embed, umap\_model=umap\_model, hdbscan\_model=cluster\_model,

&#x20;   seed\_topic\_list=seed\_topic\_list, calculate\_probabilities=True, verbose=True,

)

topics, probs = topic\_model.fit\_transform(founder\_segment\_texts)

```



\- For a stricter codebook tie-in, also run semi-supervised mode: pass `y=labels` (integer codes

&#x20; derived from human Theoretical-Ground tags) to `fit\_transform` so the manifold respects known codes.

\- Validate: c\_v coherence (aim >= 0.5); hand-map each topic to a Z\&Z phase; report the mapping table.

\- Aggregate (`aggregate.py`): per startup, proportion of founder segments in topic k = raw

&#x20; membership in theme k -> fuzzy signal for QCA.



\---



\## 5. Supervised — SetFit (`setfit\_conditions.py`, `setfit\_feedback.py`)



\### 5a. Founder-side conditions

Train ONE binary SetFit model per calibrated condition from `labels/condition\_labels.csv`:



| condition  | positive class (=1)                                            |

|------------|-----------------------------------------------------------------|

| PA\_high    | Validated/Structured, Theory-in-use, Empirically informed       |

| SF\_high    | Strong alignment, High fit validated in context                 |

| TG\_belief  | Belief presence = explicit or structured theory                 |

| TG\_test    | Test targeting OR test design quality present                   |

| TG\_respond | Feedback interpretation OR response-to-evidence present         |



Do NOT train on the raw 0–8 / 0–4 ordinal scales (62 cases / 9 classes is too sparse). Keep the

ordinal scale for human reporting only. Train at startup level on the concatenated founder text.



```python

from setfit import SetFitModel, Trainer, TrainingArguments

model = SetFitModel.from\_pretrained("sentence-transformers/all-mpnet-base-v2")

args = TrainingArguments(batch\_size=16, num\_epochs=4, num\_iterations=20)

trainer = Trainer(model=model, args=args, train\_dataset=train\_ds, eval\_dataset=eval\_ds)

trainer.train()

proba = model.predict\_proba(texts)   # P(class=1) = fuzzy membership

```



Validate per condition: leave-one-out CV, macro-F1, and Cohen's κ vs human codes (the construct-

validity claim for BOTH project goals).



\### 5b. Evaluator-side feedback type (multi-label)

Feedback type is multi-label (a session can contain several codes). Train a multi-label SetFit

(`multi\_target\_strategy="one-vs-rest"`) on evaluator turns, labels from `labels/feedback\_labels.csv`:

`POS\_REINF, CLARIFY\_REQ, DIAG\_PROB, PRESC\_ACT, TEST\_GUIDE, DISCONFIRM, EXT\_SIGNAL, TEAM\_CAP`.



Output per startup = P(each feedback code) and/or the share of evaluator turns in each code.



\### 5c. Optional LLM-as-second-coder

For the subtle Z\&Z dimensions (e.g. "co-produced advice", "test least-probable assumption first"),

run an LLM deductive pass against the codebook definitions, compare to SetFit, and report agreement.

Use only to lift κ on hard codes — not as the primary classifier (prompt-sensitivity, weak calibration).



\---



\## 6. Calibration bridge (`calibrate\_for\_qca.py`)



fs-QCA needs set membership in \[0,1] via the direct method (Ragin): three anchors — full

membership, crossover (0.5), full non-membership.



\- SetFit P(high) / P(type): logistic direct method; set crossover at the decision boundary.

\- BERTopic proportions: anchors at the 0.95 / 0.50 / 0.05 quantiles, or theory-driven.



\### Building the OUTCOME from feedback type (the key change)

Implement BOTH and compare:



1\. \*\*Per-type fuzzy outcomes.\*\* Calibrate each feedback code as its own fuzzy outcome; run one

&#x20;  fs-QCA per code. Answers "which configurations elicit diagnostic / testing / disconfirming feedback?"



2\. \*\*Z\&Z feedback-ladder score (single ordered outcome).\*\* Map the 8 codes onto the belief-cycle

&#x20;  position and build one fuzzy "belief-challenge intensity" outcome:

&#x20;  - Endorse end (low): POS\_REINF

&#x20;  - Diagnostic middle: CLARIFY\_REQ, DIAG\_PROB, EXT\_SIGNAL, TEAM\_CAP

&#x20;  - Belief-revision end (high): PRESC\_ACT, TEST\_GUIDE, DISCONFIRM

&#x20;  Calibrate the per-session weighted position along this ladder to \[0,1]. Acceptance maps onto the

&#x20;  low-challenge (belief-endorsed) end — this is the "acceptance embedded inside feedback" link.



Output `qca\_calibrated.csv`: one row per startup; columns = calibrated conditions + outcome(s).



\---



\## 7. fs-QCA in R (`qca/fsqca.R`)



```r

library(QCA)

d <- read.csv("../data/processed/qca\_calibrated.csv", row.names = "startup\_id")

conds <- c("PA\_high","SF\_high","TG\_belief","TG\_test","TG\_respond","TEAM\_struct","PLATFORM","B2B\_pure")



\# necessity for the ladder outcome

print(superSubset(d, outcome = "FB\_ladder", conditions = conds, relation = "necessity", incl.cut = 0.9))



\# sufficiency

tt  <- truthTable(d, outcome = "FB\_ladder", conditions = conds, incl.cut = 0.8, n.cut = 2, show.cases = TRUE, sort.by = "incl")

print(minimize(tt, details = TRUE, include = "?"))

```



Repeat for each per-type outcome (FB\_TEST\_GUIDE, FB\_DISCONFIRM, FB\_POS\_REINF, ...). Report

necessity (consistency/coverage), truth table, intermediate + parsimonious solutions, and

solution consistency/coverage for each.



\---



\## 8. Validation \& robustness

\- BERTopic: c\_v coherence; topic→Z\&Z-phase mapping table.

\- SetFit: κ vs human per condition AND per feedback code; macro-F1; LOO-CV.

\- Calibration: report 3 anchors per measure; skewness check.

\- fs-QCA: vary incl.cut (0.75/0.80/0.85); report deviant cases; compare ladder vs per-type.



\---



\## 9. Codex prompt sequence (run one stage at a time, inspect, then continue)

1\. Scaffold repo (section 1) + `requirements.txt` (section 2).

2\. `segment.py` (section 3) — confirm speaker-role detection works on a sample transcript.

3\. `bertopic\_run.py` + `aggregate.py` (section 4).

4\. `setfit\_conditions.py` (5a) with LOO-CV and κ.

5\. `setfit\_feedback.py` (5b) multi-label.

6\. `calibrate\_for\_qca.py` (section 6) — both outcome strategies.

7\. `qca/fsqca.R` (section 7) — ladder outcome first, then parametrise over per-type outcomes.

Do not let Codex chain all stages unsupervised — each stage needs a human look at its output.

