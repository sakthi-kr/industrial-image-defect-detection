from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "patchcore_validation_report.json"
)

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "patchcore_validation.md"
)

README_PATH = PROJECT_ROOT / "README.md"
DOCS_INDEX_PATH = PROJECT_ROOT / "docs" / "README.md"

README_START_MARKER = (
    "<!-- PATCHCORE_VALIDATION_SECTION_START -->"
)

README_END_MARKER = (
    "<!-- PATCHCORE_VALIDATION_SECTION_END -->"
)

DOCS_START_MARKER = (
    "<!-- PATCHCORE_VALIDATION_DOC_START -->"
)

DOCS_END_MARKER = (
    "<!-- PATCHCORE_VALIDATION_DOC_END -->"
)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required validation report not found: {path}\n"
            "Run `python src/validate_patchcore_results.py` first."
        )

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a JSON object in: {path}"
        )

    return value


def format_metric(
    metrics: dict[str, Any],
    name: str,
) -> str:
    """Format one metric for Markdown."""
    value = metrics.get(name)

    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    return "Not available"


def markdown_cell(value: Any) -> str:
    """Escape text before inserting it into a Markdown table."""
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def build_check_rows(
    checks: list[dict[str, Any]],
) -> str:
    """Build Markdown table rows for validation checks."""
    rows: list[str] = []

    for check in checks:
        name = markdown_cell(
            check.get("name", "unknown")
        )
        status = markdown_cell(
            check.get("status", "UNKNOWN")
        )
        message = markdown_cell(
            check.get("message", "")
        )

        rows.append(
            f"| `{name}` | {status} | {message} |"
        )

    if not rows:
        rows.append(
            "| _No checks found_ | UNKNOWN | "
            "No validation checks were present. |"
        )

    return "\n".join(rows)


def replace_or_append_marked_section(
    text: str,
    section: str,
    start_marker: str,
    end_marker: str,
) -> str:
    """Replace an existing marked section or append a new one."""
    has_start = start_marker in text
    has_end = end_marker in text

    if has_start != has_end:
        raise ValueError(
            "The target document contains only one section marker. "
            "Remove the incomplete marked section and rerun this script."
        )

    if has_start and has_end:
        before, remainder = text.split(
            start_marker,
            1,
        )

        _, after = remainder.split(
            end_marker,
            1,
        )

        parts = [
            before.rstrip(),
            section.strip(),
        ]

        if after.strip():
            parts.append(after.strip())

        return "\n\n".join(parts) + "\n"

    return (
        text.rstrip()
        + "\n\n"
        + section.strip()
        + "\n"
    )


def build_validation_document(
    report: dict[str, Any],
) -> str:
    """Build the complete PatchCore validation document."""
    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    checks = report.get("checks", [])

    if not isinstance(summary, dict):
        summary = {}

    if not isinstance(metadata, dict):
        metadata = {}

    if not isinstance(checks, list):
        checks = []

    metrics = metadata.get(
        "calculated_metrics",
        {},
    )

    if not isinstance(metrics, dict):
        metrics = {}

    valid_checks = [
        check
        for check in checks
        if isinstance(check, dict)
    ]

    source_files = metadata.get(
        "source_files",
        [],
    )

    if not isinstance(source_files, list):
        source_files = []

    source_file_lines = "\n".join(
        str(path)
        for path in source_files
    )

    if not source_file_lines:
        source_file_lines = "Not available"

    threshold_interpretation = metadata.get(
        "threshold_interpretation",
        (
            "Metric thresholds are development regression checks, "
            "not production acceptance criteria."
        ),
    )

    check_rows = build_check_rows(
        valid_checks
    )

    return f"""# PatchCore Output Validation

## Purpose

This document records the reusable validation applied to the saved PatchCore prediction, metric, error-analysis, and confusion-matrix artifacts.

The validation is implemented with the separate `ml-testing-validation-toolkit` package. It checks the consistency of already generated outputs without rerunning the Anomalib model.

## Validation Scope

The workflow validates:

- required prediction-table columns
- missing and infinite values
- duplicate image records
- allowed defect types and labels
- numerical label and anomaly-score ranges
- target-class representation
- prediction lengths and label values
- metric regression thresholds
- confusion-matrix values and schema
- text-to-numeric label mapping
- correctness flags
- error-analysis table consistency
- source JSON report consistency
- project-relative output paths

## Current Validation Summary

| Item | Result |
|---|---:|
| Overall status | {summary.get("status", "UNKNOWN")} |
| Total checks | {summary.get("total_checks", "Not available")} |
| Passed checks | {summary.get("passed_checks", "Not available")} |
| Failed checks | {summary.get("failed_checks", "Not available")} |
| Prediction rows | {metadata.get("prediction_rows", "Not available")} |
| Error rows | {metadata.get("error_rows", "Not available")} |

## Recalculated Image-Level Metrics

| Metric | Result |
|---|---:|
| Accuracy | {format_metric(metrics, "accuracy")} |
| Precision | {format_metric(metrics, "precision")} |
| Recall | {format_metric(metrics, "recall")} |
| F1-score | {format_metric(metrics, "f1_score")} |

{threshold_interpretation}

## Individual Checks

| Check | Status | Message |
|---|---|---|
{check_rows}

## Validated Source Files

```text
{source_file_lines}
```

## Generated Validation Outputs

```text
results/patchcore_validation_report.json
results/patchcore_validation_checks.csv
```

## Interpretation

A passing report means the saved PatchCore artifacts are internally consistent, contain the expected schema, reproduce the reported image-level metrics, and do not expose private absolute paths.

It does not prove production readiness. The underlying model is still evaluated only on the controlled MVTec AD `bottle` benchmark. Threshold calibration, robustness testing, evaluation on additional object categories, and production monitoring remain future work.
"""


def build_readme_section(
    report: dict[str, Any],
) -> str:
    """Build the marked README validation section."""
    summary = report.get("summary", {})
    metadata = report.get("metadata", {})

    if not isinstance(summary, dict):
        summary = {}

    if not isinstance(metadata, dict):
        metadata = {}

    metrics = metadata.get(
        "calculated_metrics",
        {},
    )

    if not isinstance(metrics, dict):
        metrics = {}

    return f"""\
{README_START_MARKER}
## Reusable PatchCore Output Validation

The saved PatchCore artifacts are checked using the separate `ml-testing-validation-toolkit` package. This validation runs without loading Anomalib or rebuilding the PatchCore memory bank.

Checks cover:

- prediction-table schema and missing values
- label and anomaly-score ranges
- duplicate image records
- label-mapping consistency
- error-table consistency
- metric and confusion-matrix consistency
- project-relative paths in committed outputs

### Current Validation Result

| Item | Result |
|---|---:|
| Overall status | {summary.get("status", "UNKNOWN")} |
| Checks passed | {summary.get("passed_checks", "Not available")} / {summary.get("total_checks", "Not available")} |
| Accuracy | {format_metric(metrics, "accuracy")} |
| Precision | {format_metric(metrics, "precision")} |
| Recall | {format_metric(metrics, "recall")} |
| F1-score | {format_metric(metrics, "f1_score")} |

Run the validation in the project's normal Python environment:

```bash
python -m pip install -e ../ml-testing-validation-toolkit
python src/validate_patchcore_results.py
```

Generated outputs:

```text
results/patchcore_validation_report.json
results/patchcore_validation_checks.csv
```

Detailed documentation:

```text
docs/patchcore_validation.md
```

Passing these checks confirms internal artifact consistency. It does not replace robustness testing, independent threshold validation, evaluation on additional categories, or production deployment validation.
{README_END_MARKER}"""


def build_docs_index_section() -> str:
    """Build the marked documentation-index entry."""
    return f"""\
{DOCS_START_MARKER}
### PatchCore Output Validation

```text
patchcore_validation.md
```

Documents:

- reusable validation-toolkit integration
- saved prediction-table checks
- source-report and confusion-matrix consistency
- recalculated image-level metrics
- portable-path validation
- interpretation and remaining limitations
{DOCS_END_MARKER}"""


def main() -> None:
    report = load_json_object(
        REPORT_PATH
    )

    DOCUMENT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCUMENT_PATH.write_text(
        build_validation_document(report),
        encoding="utf-8",
    )

    if not README_PATH.exists():
        raise FileNotFoundError(
            f"README not found: {README_PATH}"
        )

    current_readme = README_PATH.read_text(
        encoding="utf-8"
    )

    updated_readme = replace_or_append_marked_section(
        current_readme,
        build_readme_section(report),
        README_START_MARKER,
        README_END_MARKER,
    )

    README_PATH.write_text(
        updated_readme,
        encoding="utf-8",
    )

    if DOCS_INDEX_PATH.exists():
        current_docs_index = DOCS_INDEX_PATH.read_text(
            encoding="utf-8"
        )
    else:
        current_docs_index = (
            "# Project Documentation\n"
        )

    updated_docs_index = replace_or_append_marked_section(
        current_docs_index,
        build_docs_index_section(),
        DOCS_START_MARKER,
        DOCS_END_MARKER,
    )

    DOCS_INDEX_PATH.write_text(
        updated_docs_index,
        encoding="utf-8",
    )

    print(f"Created: {DOCUMENT_PATH}")
    print(f"Updated: {README_PATH}")
    print(f"Updated: {DOCS_INDEX_PATH}")


if __name__ == "__main__":
    main()
