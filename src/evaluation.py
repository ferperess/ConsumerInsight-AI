from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.classification import CATEGORIES, SENTIMENTS


def evaluate_results(frame: pd.DataFrame) -> dict[str, Any]:
    """Avalia apenas previsões bem-sucedidas com rótulos de referência presentes."""
    metrics: list[dict[str, Any]] = []
    reports: dict[str, pd.DataFrame] = {}
    matrices: dict[str, pd.DataFrame] = {}

    for name, reference_column, prediction_column, labels in (
        ("categoria", "categoria_referencia", "categoria_principal", CATEGORIES),
        ("sentimento", "sentimento_referencia", "sentimento", SENTIMENTS),
    ):
        if reference_column not in frame.columns:
            continue
        valid = frame.loc[
            (frame["status"] == "success")
            & frame[reference_column].astype(str).str.strip().ne("")
            & frame[prediction_column].astype(str).str.strip().ne("")
        ]
        if valid.empty:
            continue

        truth = valid[reference_column].astype(str)
        predicted = valid[prediction_column].astype(str)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=labels, average="macro", zero_division=0
        )
        metrics.append(
            {
                "alvo": name,
                "registros_avaliados": len(valid),
                "acuracia": accuracy_score(truth, predicted),
                "precisao_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            }
        )
        report = classification_report(
            truth, predicted, labels=labels, output_dict=True, zero_division=0
        )
        reports[name] = pd.DataFrame(report).transpose().rename_axis("classe").reset_index()
        matrices[name] = pd.DataFrame(
            confusion_matrix(truth, predicted, labels=labels),
            index=pd.Index(labels, name="referencia"),
            columns=pd.Index(labels, name="previsto"),
        )

    return {
        "metrics": pd.DataFrame(metrics),
        "reports": reports,
        "confusion_matrices": matrices,
    }


def save_evaluation(evaluation: dict[str, Any], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    evaluation["metrics"].to_csv(
        directory / "metricas.csv", index=False, encoding="utf-8-sig"
    )
    for name, report in evaluation["reports"].items():
        report.to_csv(directory / f"relatorio_{name}.csv", index=False, encoding="utf-8-sig")
    for name, matrix in evaluation["confusion_matrices"].items():
        matrix.to_csv(directory / f"matriz_confusao_{name}.csv", encoding="utf-8-sig")
