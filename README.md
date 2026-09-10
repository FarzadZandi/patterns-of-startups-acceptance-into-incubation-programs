# Patterns of Startup Acceptance into XXX Incubation Programs

This repository contains a privacy-safe research pipeline for studying which combinations of startup characteristics are associated with acceptance into an incubation program and which founder-side characteristics elicit different forms of evaluator feedback. It combines natural-language processing (NLP), supervised text classification, fuzzy-set calibration, and fuzzy-set Qualitative Comparative Analysis (fs-QCA).

`XXX` is used throughout this README as a pseudonym for the organization that supplied the research material. Raw transcripts, startup identities, coding workbooks, trained models, and row-level analytical outputs are intentionally excluded from version control.

## Technical summary

The project treats startup evaluation as a **configurational** problem: acceptance may follow from several different combinations of conditions rather than from one variable acting independently. Founder speech supplies the conditions; evaluator speech supplies feedback outcomes. The implemented pipeline:

1. segments pitch and question-and-answer transcripts by speaker role;
2. discovers founder-side themes with guided BERTopic;
3. estimates five theory-derived founder conditions with SetFit classifiers;
4. calibrates the coded variables and topic proportions into fuzzy-set memberships;
5. models acceptance, rejection, and feedback-challenge outcomes with fs-QCA; and
6. renders an R Markdown report containing necessity, sufficiency, truth-table, solution-path, veto, and robustness analyses.

The code and report templates record an analytical cohort of **70 startups** from three program batches: **27 accepted** and **43 rejected**. This corresponds to an observed acceptance rate of **38.6%**. The mean-weight feedback ladder places **14 of 70 cases (20.0%)** above the 0.5 crossover; the more permissive maximum-weight version places **22 of 70 cases (31.4%)** above it.

These are reproducible sample and calibration checkpoints documented by the implementation—not final causal findings. The privacy-safe repository does not include the confidential inputs or generated QCA solution objects needed to independently verify or publish specific sufficient configurations, consistency scores, or coverage scores. Those results must be regenerated in the authorized local environment.

## Research questions

The pipeline is designed to address four related questions:

1. Which combinations of problem articulation, solution fit, theory-grounded behavior, team structure, platform orientation, and business model are sufficient for program acceptance?
2. Are any individual conditions necessary for acceptance or rejection?
3. Which combinations of founder characteristics are associated with more belief-challenging evaluator feedback?
4. Do text-derived testing themes add explanatory value beyond the human-coded conditions?

The analysis is asymmetric. A pathway associated with acceptance is not assumed to be the logical inverse of a pathway associated with rejection; both outcomes are analyzed separately.

## Analytical design

### Two-source measurement strategy

The central design rule is to keep the two speakers analytically separate:

| Evidence source | Information extracted | Analytical role |
|---|---|---|
| Founder turns | Problem articulation, solution fit, theory-grounded belief formation, testing, response to evidence, team, market, revenue, and business-model signals | fs-QCA conditions |
| Evaluator turns | Positive reinforcement, clarification requests, problem diagnosis, prescribed action, testing guidance, disconfirmation, external signals, and team-capability feedback | Feedback outcomes |
| Administrative decision | Accepted or rejected | Primary outcome |

This separation limits target leakage: evaluator language and the administrative decision are not used as founder-side explanatory conditions.

### Core fuzzy-set conditions

| Set | Meaning | Source |
|---|---|---|
| `PA_fuzzy` | Strength and empirical grounding of problem articulation | Human ordinal coding |
| `SF_fuzzy` | Plausibility and validation of solution fit | Human ordinal coding |
| `TGB_f` | Explicit or structured theoretical beliefs | Human binary coding |
| `TGT_f` | Evidence of targeted testing | Human binary coding |
| `TGR_f` | Evidence of interpreting and responding to feedback | Human binary coding |
| `TEAM_f` | Structured team capability | Human binary coding |
| `PLAT_f` | Non-platform orientation as implemented in the calibration script | Human binary coding, reverse calibrated |
| `B2B_f` | Pure business-to-business orientation | Human binary coding |
| `T_testing_f` | Relative prevalence of pilot, MVP, experiment, and testing discourse | BERTopic-derived robustness condition |

The binary conditions are represented as `0.05` for full non-membership and `0.95` for full membership. Problem articulation and solution fit use theory-informed ordinal mappings. Topic shares use three-anchor direct calibration.

### Outcomes

`OUTCOME_accept` represents the administrative decision, calibrated to `0.95` for accepted cases and `0.05` otherwise.

Evaluator feedback is also summarized as a belief-challenge ladder:

| Feedback code | Interpretation | Challenge weight |
|---|---|---:|
| `POS_REINF` | Positive reinforcement | 0.00 |
| `CLARIFY_REQ` | Clarification request | 0.20 |
| `EXT_SIGNAL` | External signal or contextual information | 0.30 |
| `DIAG_PROB` | Diagnostic problem feedback | 0.50 |
| `TEAM_CAP` | Team-capability feedback | 0.50 |
| `PRESC_ACT` | Prescribed action | 0.70 |
| `TEST_GUIDE` | Guidance toward testing | 0.85 |
| `DISCONFIRM` | Direct disconfirmation | 1.00 |

The pipeline produces mean-weight, maximum-weight, and dominant-code versions of this outcome. Comparing them is important because a session with one strongly challenging comment can score differently depending on whether intensity is averaged across all active feedback types or represented by its most challenging type.

## Pipeline architecture

```text
Private transcripts and coding files
                |
                v
      Speaker-aware segmentation
                |
        +-------+-------+
        |               |
        v               v
 Founder segments   Evaluator labels
        |               |
   +----+----+          |
   |         |          |
   v         v          v
BERTopic   SetFit   Feedback ladder
   |         |          |
   +---------+----------+
             |
             v
      Fuzzy-set calibration
             |
             v
     Necessity and truth tables
             |
             v
  Parsimonious/intermediate solutions
             |
             v
       HTML research report
```

### Stage 1: speaker-aware transcript segmentation

[`src/segment.py`](src/segment.py) reads files from batches 3–5, normalizes batch-specific filenames, detects pitch and Q&A boundaries, assigns founder/evaluator roles, splits turns into sentences, removes segments shorter than five tokens, and writes `data/interim/segments.parquet`.

Speaker detection is heuristic. Transcripts with reliable diarization prefixes are preferable. Ambiguous or combined transcripts require manual review because speaker misclassification can move text from a condition source to an outcome source.

### Stage 2: guided BERTopic

[`src/bertopic_run.py`](src/bertopic_run.py) embeds founder segments with `sentence-transformers/all-MiniLM-L6-v2`, reduces the embeddings with UMAP, and fits 12 KMeans clusters inside BERTopic. Seed vocabulary steers the representation toward problem framing, beliefs, testing, metrics, iteration, teams, and business models.

KMeans was selected because density-based clustering assigned too much of the small corpus to an outlier topic. This gives every segment a topic membership, but it also forces noisy discourse into one of the 12 clusters. Four clusters are explicitly marked as noise during validation, and researcher review of topic words and representative quotations remains essential.

[`src/bertopic_validation.py`](src/bertopic_validation.py) supports:

- c-v topic coherence;
- one-sided Mann–Whitney comparisons between accepted and non-accepted cases;
- human-readable labels for all 12 topics;
- representative quotation extraction for usable topics; and
- export of decision comparisons and representative passages.

### Stage 3: few-shot SetFit classification

[`src/setfit_conditions.py`](src/setfit_conditions.py) fits one binary SetFit model for each of five founder-side conditions: high problem articulation, high solution fit, belief formation, testing, and response to evidence. It uses three-fold stratified cross-validation and reports Cohen's kappa and macro-F1 against human codes before fitting final models.

Model probabilities are treated as graded evidence, not automatically as validated fuzzy memberships. Performance should be reviewed per condition, especially where the positive class is rare. The current public notebook leaves the evaluator-side SetFit stage empty; the feedback outcomes used by calibration therefore depend on the supplied human-coded feedback labels.

### Stage 4: fuzzy-set calibration

[`src/calibrate.py`](src/calibrate.py) joins the ordinal coding workbooks, binary condition labels, topic shares, and feedback labels. It creates one row per startup in `data/processed/qca_calibrated.csv`.

BERTopic shares are calibrated using fixed empirical anchors recorded in the script:

| Topic-derived set | Full non-membership | Crossover | Full membership |
|---|---:|---:|---:|
| Testing | 0.0085 | 0.0286 | 0.0635 |
| Customer/business model | 0.0386 | 0.0859 | 0.1812 |
| Market/competition | 0.0283 | 0.0619 | 0.1341 |
| Revenue/metrics | 0.0141 | 0.0578 | 0.1402 |

These anchors should be reconsidered when the corpus changes. They are sample-relative and should not be interpreted as universal substantive thresholds.

### Stage 5: fs-QCA and reporting

[`qca/fsqca_report.Rmd`](qca/fsqca_report.Rmd) implements the main analysis and report. The core acceptance analysis uses:

- necessity consistency threshold: `0.90`;
- necessity coverage threshold: `0.50`;
- sufficiency inclusion threshold: `0.80`;
- minimum frequency: one case per truth-table row; and
- positive directional expectations for the intermediate solution.

The feedback-challenge analysis uses a necessity threshold of `0.85` and a sufficiency threshold of `0.75`, reflecting the smaller number of high-challenge cases. The report also analyzes rejection separately, screens for veto-condition candidates, adds the BERTopic testing set as a robustness condition, and compares the mean and maximum feedback-ladder definitions.

## Recorded empirical checkpoints

The following values are embedded in the current code and report templates and should be checked after every full rerun:

| Checkpoint | Recorded value | Interpretation |
|---|---:|---|
| Total analyzed startups | 70 | Three program batches |
| Accepted | 27 | 38.6% of the sample |
| Rejected | 43 | 61.4% of the sample |
| Approximate founder segments | 9,600 | Development note in BERTopic validation code |
| Mean-ladder cases above 0.5 | 14 | 20.0%; limited outcome diversity |
| Maximum-ladder cases above 0.5 | 22 | 31.4%; sensitive to outcome construction |

The repository currently contains no shareable final truth table, minimized solution, or aggregate consistency/coverage export. Accordingly, this README does **not** label any particular condition or configuration as necessary or sufficient. A defensible results section should be added only after the private data are restored, the report is rendered, and aggregate outputs are reviewed for disclosure risk.

## Repository structure

```text
.
├── cf_nlp_pipeline.ipynb          # staged research notebook
├── CODEX_SPEC_nlp_fsqca_pipeline.md
├── requirements.txt
├── src/
│   ├── segment.py                 # transcript parsing and speaker roles
│   ├── bertopic_run.py            # guided topic model
│   ├── bertopic_validation.py     # coherence and substantive validation
│   ├── export_topic_words.py      # researcher-facing topic workbook
│   ├── setfit_conditions.py       # founder-condition classifiers
│   ├── calibrate.py               # fuzzy calibration and analytical join
│   ├── private_aliases.py         # local, ignored identity reconciliation
│   └── pipeline_app.py            # Streamlit runner
└── qca/
    ├── fsqca.R                    # concise command-line QCA script
    ├── fsqca_report.Rmd           # full analytical report
    ├── references.bib
    └── apa.csl
```

The following directories are deliberately absent from Git:

```text
data/raw/                         confidential transcripts
data/interim/                     row-level segmented text
data/processed/                   model outputs and calibrated case data
labels/                           confidential coding and identity files
libs/, runs/                      downloaded models and local artifacts
qca/*.html, qca/*.tex             reports that may expose case identifiers
```

## Private input contract

To run the project locally, restore these inputs without committing them:

```text
data/raw/Batch03/
data/raw/Batch04/
data/raw/Batch05/
labels/condition_labels.csv
labels/feedback_labels.csv
labels/Batch_3_CF_Coding.xlsx
labels/Batch_4_CF_Coding.xlsx
labels/Batch_5_CF_Coding.xlsx
```

`condition_labels.csv` must contain at least `startup_id`, `startup_name`, `decision`, `PA_high`, `SF_high`, `TG_belief`, `TG_test`, `TG_respond`, `TEAM_struct`, `PLATFORM`, and `B2B_pure`.

`feedback_labels.csv` must contain `startup_id`, `decision`, and the eight binary feedback codes listed above. Each coding workbook must expose the `CF_Coding_Batch04` sheet and the fields `startup_id`, `startup_name`, `problem_articulation`, and `solution_fit` after column-name normalization.

If startup names differ across sources, an optional local `labels/startup_aliases.json` file can map one spelling to another. The file is ignored by Git and may also be supplied through the `CF_STARTUP_ALIASES` environment variable.

## Installation

The project was designed for Python 3.11 and a local R installation. Create an isolated Python environment before installing the NLP stack:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current `requirements.txt` includes `ggrepel`, which is an R package rather than a Python dependency. If the Python installation stops on that line, remove or skip it and install `ggrepel` from R as shown below. The validation module also uses `numpy`, `scipy`, `gensim`, and `datasets`; install them explicitly if the resolver does not provide them transitively.

Install the report dependencies in R:

```r
install.packages(c(
  "QCA", "rmarkdown", "knitr", "kableExtra",
  "ggplot2", "dplyr", "tidyr", "ggrepel"
))
```

Model downloads require internet access on the first run. CPU execution is supported; BERTopic and SetFit can be substantially faster with a compatible CUDA-enabled PyTorch environment.

## Running the project

### Streamlit interface

From the repository root:

```powershell
streamlit run src/pipeline_app.py
```

The interface checks required files and dependencies, then exposes four stages: segmentation, BERTopic, calibration, and R Markdown report generation. Run the stages in order and inspect each intermediate output before proceeding.

### Notebook workflow

Open `cf_nlp_pipeline.ipynb` from the repository root. It provides the staged sequence for segmentation, topic modeling, topic validation, SetFit condition modeling, calibration, and QCA input preparation. The evaluator-side SetFit section is currently a placeholder and should not be presented as completed functionality.

### Command-line components

Segmentation can be run directly:

```powershell
python src/segment.py --raw-dir data/raw --output data/interim/segments.parquet
```

After the private inputs and processed topic file exist, render the analytical report with:

```powershell
Rscript -e "rmarkdown::render('qca/fsqca_report.Rmd')"
```

Do not publish the resulting HTML without checking it for startup names, quotations, case labels, and other indirect identifiers.

## Expected outputs

| Output | Purpose | Shareability |
|---|---|---|
| `data/interim/segments.parquet` | Sentence-level text with startup and speaker roles | Confidential |
| `data/processed/topic_membership.csv` | Topic shares per startup | Confidential row-level data |
| `data/processed/bertopic_model/` | Saved topic model | Review before sharing |
| `data/processed/topic_words.xlsx` | Topic words for researcher labeling | Aggregate, but review carefully |
| `data/processed/topic_decision_comparison.csv` | Topic means and one-sided tests by decision | Aggregate; disclosure review required |
| `data/processed/topic_representative_quotes.csv` | Representative founder quotations | Confidential |
| `data/processed/conditions_setfit.csv` | SetFit probabilities per startup | Confidential row-level data |
| `data/processed/qca_calibrated.csv` | Final case-level QCA matrix | Confidential |
| `qca/fsqca_report.html` | Tables, figures, paths, and case diagnostics | Confidential until anonymized |

## Validation and interpretation

A full analytical rerun should pass the following checks before results are reported:

- Review speaker-role assignments on a stratified sample from every batch.
- Confirm that all 70 case identifiers reconcile across transcript, label, topic, and feedback sources.
- Inspect topic coherence, topic words, and representative quotations; do not rely on a single coherence threshold.
- Report per-condition class balance, macro-F1, Cohen's kappa, and out-of-fold performance for SetFit.
- Re-estimate topic calibration anchors whenever the corpus or topic model changes.
- Report truth-table contradictions and limited-diversity rows.
- Compare inclusion thresholds of 0.75, 0.80, and 0.85 and document configuration changes.
- Compare mean, maximum, and dominant feedback-ladder outcomes.
- Inspect deviant cases in kind and degree without exposing identities.
- Export only aggregate results after a disclosure review.

fs-QCA identifies set-theoretic relations in this sample. It does not establish that changing a condition will cause acceptance. The sample is small relative to the eight-condition truth table, the outcome is an institutional decision rather than an objective measure of venture quality, and the feedback ladder is a researcher-defined construct. Results should therefore be described as configurational associations supported under stated calibration and inclusion choices.

## Known implementation boundaries

- Confidential data are not included, so a fresh clone cannot run end to end without authorized local inputs.
- The evaluator-side SetFit classifier described in the original design is not implemented in the public source snapshot.
- The Streamlit interface runs segmentation, BERTopic, calibration, and report rendering, but it does not expose the SetFit or BERTopic-validation stages.
- Cross-validation metrics are printed during SetFit execution but are not persisted to a shareable metrics file.
- The R Markdown report contains drafting placeholders; its narrative must be completed from the generated statistics before publication.
- Several hard-coded topic labels and calibration anchors depend on the original corpus and require revalidation after retraining.
- The binary acceptance outcome is represented fuzzily as 0.05/0.95 but remains substantively crisp.

These boundaries are part of the reproducibility record. They prevent the repository from overstating what has been implemented or independently verified.

## Recommended next steps

1. Move `ggrepel` from the Python requirements into an R dependency manifest and pin both Python and R environments.
2. Add unit tests for filename normalization, speaker-role assignment, calibration mappings, alias reconciliation, and analytical joins.
3. Implement or formally remove the evaluator-side SetFit stage so the documented architecture and executable pipeline agree.
4. Persist SetFit validation metrics and BERTopic diagnostics in aggregate, privacy-reviewed files.
5. Add threshold-sensitivity exports for every QCA outcome and document contradictory configurations.
6. Create an anonymized aggregate results table that can support a substantive public findings section.
7. Replace case identifiers in figures and reports with irreversible study IDs before any external release.

## References

- Dusa, A. (2019). *QCA with R: A Comprehensive Resource*. Springer. <https://doi.org/10.1007/978-3-319-75668-4>
- Grootendorst, M. (2022). BERTopic: Neural topic modelling with a class-based TF-IDF procedure. <https://arxiv.org/abs/2203.05794>
- Ragin, C. C. (2000). *Fuzzy-Set Social Science*. University of Chicago Press.
- Ragin, C. C. (2008). *Redesigning Social Inquiry: Fuzzy Sets and Beyond*. University of Chicago Press.
- Schneider, C. Q., & Wagemann, C. (2012). *Set-Theoretic Methods for the Social Sciences*. Cambridge University Press. <https://doi.org/10.1017/CBO9781139004244>
- Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., & Pereg, O. (2022). Efficient few-shot learning without prompts. <https://doi.org/10.18653/v1/2022.findings-emnlp.297>
- Zellweger, T., & Zenger, T. (2023). Entrepreneurs as scientists: A pragmatist approach to producing value out of uncertainty. *Academy of Management Review, 48*(3), 379–408. <https://doi.org/10.5465/amr.2020.0503>

## Responsible use and privacy

This repository is code-only by design. Do not commit raw transcripts, startup names, personal names, evaluator comments, hand-coded workbooks, row-level model scores, representative quotations, or rendered reports containing case labels. Before sharing aggregate findings, check small cells, rare configurations, quoted language, and model artifacts for re-identification risk.

The project should be used for research and methodological exploration. It should not be used as an automated screening or acceptance system. Historical decisions may encode institutional preferences and evaluator biases; a model that predicts those decisions can reproduce them without establishing that they are fair, valid, or desirable.
