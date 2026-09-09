# Run with:  streamlit run src/pipeline_app.py
# Install:   pip install streamlit

import streamlit as st, sys, subprocess
from pathlib import Path

import importlib
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
st.set_page_config(page_title="CF Pipeline Runner", layout="wide")

STAGES = (
    "Segmentation",
    "BERTopic",
    "Calibration",
    "QCA (R Markdown)",
)
STATUS_ICONS = {
    "Not started": "🔵",
    "Running": "🟡",
    "Done": "✅",
    "Error": "❌",
}
PACKAGE_IMPORTS = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "nltk": "nltk",
    "bertopic": "bertopic",
    "sentence-transformers": "sentence_transformers",
    "umap-learn": "umap",
    "scikit-learn": "sklearn",
    "openpyxl": "openpyxl",
    "numpy": "numpy",
}


def _folder_has_files(path):
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _required_files():
    return [
        (
            "labels/condition_labels.csv",
            (PROJECT_ROOT / "labels" / "condition_labels.csv").is_file(),
            3,
        ),
        (
            "labels/feedback_labels.csv",
            (PROJECT_ROOT / "labels" / "feedback_labels.csv").is_file(),
            3,
        ),
        (
            "labels/Batch_3_CF_Coding.xlsx",
            (PROJECT_ROOT / "labels" / "Batch_3_CF_Coding.xlsx").is_file(),
            3,
        ),
        (
            "labels/Batch_4_CF_Coding.xlsx",
            (PROJECT_ROOT / "labels" / "Batch_4_CF_Coding.xlsx").is_file(),
            3,
        ),
        (
            "labels/Batch_5_CF_Coding.xlsx",
            (PROJECT_ROOT / "labels" / "Batch_5_CF_Coding.xlsx").is_file(),
            3,
        ),
        (
            "data/raw/Batch03/",
            _folder_has_files(PROJECT_ROOT / "data" / "raw" / "Batch03"),
            1,
        ),
        (
            "data/raw/Batch04/",
            _folder_has_files(PROJECT_ROOT / "data" / "raw" / "Batch04"),
            1,
        ),
        (
            "data/raw/Batch05/",
            _folder_has_files(PROJECT_ROOT / "data" / "raw" / "Batch05"),
            1,
        ),
    ]


def check_dependencies():
    missing_pkgs = []
    for package_name, import_name in PACKAGE_IMPORTS.items():
        try:
            importlib.import_module(import_name)
        except Exception:
            missing_pkgs.append(package_name)

    try:
        r_check = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        r_missing = r_check.returncode != 0
    except (FileNotFoundError, OSError):
        r_missing = True

    file_checks = _required_files()
    any_missing = bool(
        missing_pkgs
        or r_missing
        or any(not exists for _, exists, _ in file_checks)
    )

    with st.expander("🔍 System check", expanded=any_missing):
        if not missing_pkgs and not r_missing:
            st.success("✅ All dependencies found. Ready to run.")

        if missing_pkgs:
            st.error("❌ Missing Python packages detected")
            st.write("Missing: " + ", ".join(missing_pkgs))
            st.code("pip install -r requirements.txt", language="bash")
            if st.button(
                "📦 Auto-install missing packages",
                key="install_dependencies",
            ):
                try:
                    with st.spinner("Installing packages..."):
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                "-r",
                                str(PROJECT_ROOT / "requirements.txt"),
                            ],
                            check=True,
                        )
                    st.rerun()
                except Exception as error:
                    st.exception(error)
            st.info("After installing, the app will refresh automatically.")

        if r_missing:
            st.warning("⚠️ Rscript not found on PATH")
            st.write(
                "Stage 4 (QCA Report) requires R to be installed.\n\n"
                "Download from: https://cran.r-project.org\n\n"
                "After installing R, restart this app.\n\n"
                "Stages 1–3 will still work without R."
            )

        st.markdown("#### 📁 Required files")
        for relative_path, exists, stage_number in file_checks:
            if exists:
                st.markdown(f"✅ `{relative_path}`")
            else:
                st.error(
                    f"❌ {relative_path}\n\n"
                    f"Copy this file into the project before running Stage "
                    f"{stage_number}."
                )

    return missing_pkgs, r_missing


def _initialize_session():
    for stage in STAGES:
        st.session_state.setdefault(f"status_{stage}", "Not started")
    st.session_state.setdefault("pipeline_run_log", [])
    st.session_state.setdefault("segmentation_metrics", None)
    st.session_state.setdefault("bertopic_metrics", None)
    st.session_state.setdefault("calibration_result", None)


def _status_text(stage):
    status = st.session_state[f"status_{stage}"]
    return f"{STATUS_ICONS[status]} **{stage}:** {status}"


def _set_status(stage, status, sidebar_slot=None, card_slot=None):
    st.session_state[f"status_{stage}"] = status
    text = _status_text(stage)
    if sidebar_slot is not None:
        sidebar_slot.markdown(text)
    if card_slot is not None:
        if status == "Done":
            card_slot.success(f"{stage} completed.")
        elif status == "Error":
            card_slot.error(f"{stage} failed. See the log below.")
        elif status == "Running":
            card_slot.warning(f"{stage} is running...")
        else:
            card_slot.info(f"{stage} has not started.")


def _append_log(stage, text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {stage}\n{text.rstrip()}"
    st.session_state["pipeline_run_log"].append(entry)


class LiveStringIO(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder

    def write(self, text):
        written = super().write(text)
        current = self.getvalue()
        self.placeholder.code(current[-15000:], language="text")
        return written

    def flush(self):
        return None


def _run_stage(stage, action, sidebar_slot, card_slot, live_slot):
    _set_status(stage, "Running", sidebar_slot, card_slot)
    capture = LiveStringIO(live_slot)
    try:
        with redirect_stdout(capture), redirect_stderr(capture):
            result = action()
    except Exception as error:
        traceback.print_exc(file=capture)
        _append_log(stage, capture.getvalue())
        _set_status(stage, "Error", sidebar_slot, card_slot)
        st.exception(error)
        return None

    _append_log(stage, capture.getvalue())
    _set_status(stage, "Done", sidebar_slot, card_slot)
    return result


def _render_segmentation_metrics(metrics):
    if not metrics:
        return
    columns = st.columns(3)
    for column, batch in zip(columns, (3, 4, 5)):
        values = metrics.get(batch, {"segments": 0, "startups": 0})
        with column:
            st.metric(f"Batch {batch:02d} segments", values["segments"])
            st.metric("Startups", values["startups"])


def _render_bertopic_metrics(metrics):
    if not metrics:
        return
    left, right = st.columns(2)
    left.metric("Topics", metrics["topics"])
    right.metric("Total segments", metrics["segments"])


def _render_calibration_result(result):
    if not result:
        return
    left, right = st.columns(2)
    left.metric("Rows", result["shape"][0])
    right.metric("Columns", result["shape"][1])
    st.dataframe(result["preview"], use_container_width=True)


_initialize_session()

st.title("Campus Founders — NLP + fs-QCA Pipeline")
st.caption("Run each stage in order. Each stage builds on the previous.")

with st.expander("ℹ️ First time on this computer? Start here", expanded=False):
    st.markdown(
        """
**Step 0 — One-time setup (new computer only)**

1. Install Python 3.11+: https://python.org
2. Install R: https://cran.r-project.org
3. Open a terminal in this folder and run:
```bash
pip install -r requirements.txt
```
4. Then launch the app:
```bash
streamlit run src/pipeline_app.py
```

**What this app does**

This app runs a 4-stage research pipeline for the Campus Founders study:
- Stage 1 reads pitch transcripts and segments them by speaker
- Stage 2 discovers topics in founder speech using BERTopic (~30 min)
- Stage 3 calibrates all variables into fuzzy sets for QCA
- Stage 4 runs the fs-QCA analysis and renders the HTML report

**Files you must provide**
- Transcript files in data/raw/Batch03/, Batch04/, Batch05/
- Coding files in labels/ (see checklist in System check above)
"""
    )

missing_pkgs, r_missing = check_dependencies()
missing_set = set(missing_pkgs)

st.sidebar.header("Pipeline status")
sidebar_slots = {}
for stage_name in STAGES:
    sidebar_slots[stage_name] = st.sidebar.empty()
    sidebar_slots[stage_name].markdown(_status_text(stage_name))

if st.sidebar.button("🔄 Reset all statuses", use_container_width=True):
    for stage_name in STAGES:
        st.session_state[f"status_{stage_name}"] = "Not started"
    st.rerun()

st.sidebar.info(
    "💡 Stage status resets when you close the browser.\n\n"
    "Your output files are saved to disk and persist across sessions."
)

with st.expander("STAGE 1 — Segmentation", expanded=True):
    st.write(
        "Reads all transcript files and splits them into labelled speaker "
        "segments."
    )
    st.write("**What it produces:** `data/interim/segments.parquet`")
    stage_status = st.empty()
    _set_status(
        "Segmentation",
        st.session_state["status_Segmentation"],
        sidebar_slots["Segmentation"],
        stage_status,
    )
    live_output = st.empty()
    segmentation_disabled = bool(
        missing_set.intersection({"pandas", "pyarrow", "nltk"})
    )
    if st.button(
        "▶ Run Segmentation",
        disabled=segmentation_disabled,
        key="run_segmentation",
    ):
        def run_segmentation():
            import segment

            return segment.run_segmentation(
                raw_dir=PROJECT_ROOT / "data" / "raw",
                output_path=(
                    PROJECT_ROOT / "data" / "interim" / "segments.parquet"
                ),
            )

        segments = _run_stage(
            "Segmentation",
            run_segmentation,
            sidebar_slots["Segmentation"],
            stage_status,
            live_output,
        )
        if segments is not None:
            metrics = {}
            for batch in (3, 4, 5):
                batch_rows = segments[segments["batch"] == batch]
                metrics[batch] = {
                    "segments": len(batch_rows),
                    "startups": batch_rows["startup_id"].nunique(),
                }
            st.session_state["segmentation_metrics"] = metrics
    _render_segmentation_metrics(st.session_state["segmentation_metrics"])

with st.expander("STAGE 2 — BERTopic"):
    st.write(
        "Discovers 12 discourse themes from founder speech using sentence "
        "embeddings and guided topic modelling."
    )
    st.warning(
        "This stage takes ~30 minutes. Run only if topic_membership.csv "
        "does not already exist."
    )
    topic_path = PROJECT_ROOT / "data" / "processed" / "topic_membership.csv"
    if topic_path.exists():
        st.success(
            "topic_membership.csv already exists — skip this stage unless "
            "you need to rebuild topics."
        )
    stage_status = st.empty()
    _set_status(
        "BERTopic",
        st.session_state["status_BERTopic"],
        sidebar_slots["BERTopic"],
        stage_status,
    )
    live_output = st.empty()
    bertopic_disabled = bool(
        missing_set.intersection(
            {
                "pandas",
                "pyarrow",
                "bertopic",
                "sentence-transformers",
                "umap-learn",
                "scikit-learn",
            }
        )
    )
    if st.button(
        "▶ Run BERTopic",
        disabled=bertopic_disabled,
        key="run_bertopic",
    ):
        def run_bertopic():
            import pandas as pd
            import bertopic_run

            result = bertopic_run.run_bertopic(
                segments_path=(
                    PROJECT_ROOT / "data" / "interim" / "segments.parquet"
                ),
                output_dir=str(PROJECT_ROOT / "data" / "processed"),
            )
            all_segments = pd.read_parquet(
                PROJECT_ROOT / "data" / "interim" / "segments.parquet"
            )
            topic_columns = [
                column
                for column in result.columns
                if column.startswith("topic_") and column.endswith("_share")
            ]
            return {
                "topics": len(topic_columns),
                "segments": len(all_segments),
            }

        metrics = _run_stage(
            "BERTopic",
            run_bertopic,
            sidebar_slots["BERTopic"],
            stage_status,
            live_output,
        )
        if metrics is not None:
            st.session_state["bertopic_metrics"] = metrics
    _render_bertopic_metrics(st.session_state["bertopic_metrics"])

with st.expander("STAGE 3 — Calibration"):
    st.write(
        "Calibrates all conditions to fuzzy sets and builds the QCA input "
        "file."
    )
    batch_paths = [
        PROJECT_ROOT / "labels" / "Batch_3_CF_Coding.xlsx",
        PROJECT_ROOT / "labels" / "Batch_4_CF_Coding.xlsx",
        PROJECT_ROOT / "labels" / "Batch_5_CF_Coding.xlsx",
    ]
    missing_batches = [path.name for path in batch_paths if not path.exists()]
    if missing_batches:
        st.error("Missing required batch files: " + ", ".join(missing_batches))
    stage_status = st.empty()
    _set_status(
        "Calibration",
        st.session_state["status_Calibration"],
        sidebar_slots["Calibration"],
        stage_status,
    )
    live_output = st.empty()
    calibration_disabled = bool(
        missing_batches
        or missing_set.intersection({"pandas", "openpyxl", "numpy"})
    )
    if st.button(
        "▶ Run Calibration",
        disabled=calibration_disabled,
        key="run_calibration",
    ):
        def run_calibration():
            import calibrate

            return calibrate.run_calibration(
                batch3_path=PROJECT_ROOT
                / "labels"
                / "Batch_3_CF_Coding.xlsx",
                batch4_path=PROJECT_ROOT
                / "labels"
                / "Batch_4_CF_Coding.xlsx",
                batch5_path=PROJECT_ROOT
                / "labels"
                / "Batch_5_CF_Coding.xlsx",
                condition_labels_path=PROJECT_ROOT
                / "labels"
                / "condition_labels.csv",
                feedback_labels_path=PROJECT_ROOT
                / "labels"
                / "feedback_labels.csv",
                topic_path=PROJECT_ROOT
                / "data"
                / "processed"
                / "topic_membership.csv",
                output_path=PROJECT_ROOT
                / "data"
                / "processed"
                / "qca_calibrated.csv",
                r_script_path=PROJECT_ROOT / "qca" / "fsqca.R",
            )

        calibrated = _run_stage(
            "Calibration",
            run_calibration,
            sidebar_slots["Calibration"],
            stage_status,
            live_output,
        )
        if calibrated is not None:
            st.session_state["calibration_result"] = {
                "shape": calibrated.shape,
                "preview": calibrated.head(),
            }
    _render_calibration_result(st.session_state["calibration_result"])

with st.expander("STAGE 4 — QCA Report (R Markdown)"):
    st.write("Runs the fs-QCA analysis and renders the HTML report.")
    stage_status = st.empty()
    _set_status(
        "QCA (R Markdown)",
        st.session_state["status_QCA (R Markdown)"],
        sidebar_slots["QCA (R Markdown)"],
        stage_status,
    )
    live_output = st.empty()
    if st.button(
        "▶ Knit HTML Report",
        disabled=r_missing,
        key="run_qca_report",
    ):
        def knit_report():
            command = [
                "Rscript",
                "-e",
                "rmarkdown::render('qca/fsqca_report.Rmd')",
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"R Markdown rendering failed with exit code {return_code}."
                )
            return PROJECT_ROOT / "qca" / "fsqca_report.html"

        report_path = _run_stage(
            "QCA (R Markdown)",
            knit_report,
            sidebar_slots["QCA (R Markdown)"],
            stage_status,
            live_output,
        )
        if report_path is not None and report_path.exists():
            st.success(
                f"HTML report created: [{report_path}]({report_path.as_uri()})"
            )

with st.expander("📋 Run log"):
    if st.session_state["pipeline_run_log"]:
        st.code(
            "\n\n".join(st.session_state["pipeline_run_log"]),
            language="text",
        )
    else:
        st.caption("No stages have run in this session.")
