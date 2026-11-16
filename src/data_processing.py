from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
PREDICTION_COLUMNS = [
    "categoria_principal",
    "categoria_secundaria",
    "sentimento",
    "resumo",
    "problema_principal",
    "assuntos",
    "status",
    "modelo",
    "hash_texto",
    "erro",
    "processado_em",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV, valida os campos e prepara o texto sem expor rótulos."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    frame = pd.read_csv(dataset_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "texto_reclamacao" not in frame.columns:
        raise ValueError("O dataset precisa conter a coluna obrigatória 'texto_reclamacao'.")

    if "id" not in frame.columns:
        frame.insert(0, "id", [f"REG-{index + 1:06d}" for index in range(len(frame))])

    frame["id"] = frame["id"].astype(str).str.strip()
    empty_ids = frame["id"].eq("")
    if empty_ids.any():
        frame.loc[empty_ids, "id"] = [
            f"REG-{index + 1:06d}" for index in frame.index[empty_ids]
        ]
    if frame["id"].duplicated().any():
        repeated = frame.loc[frame["id"].duplicated(), "id"].tolist()
        raise ValueError(f"IDs duplicados no dataset: {repeated[:5]}")

    frame["texto_reclamacao"] = (
        frame["texto_reclamacao"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return frame


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def successful_records(path: str | Path, model: str) -> dict[str, dict[str, Any]]:
    """Retorna resultados reutilizáveis, identificados por ID, texto e modelo."""
    result_path = Path(path)
    if not result_path.is_file():
        return {}

    previous = pd.read_csv(result_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"id", *PREDICTION_COLUMNS}
    if not required.issubset(previous.columns):
        LOGGER.warning("Arquivo de resultados existente sem colunas de retomada; será reprocessado.")
        return {}

    reusable: dict[str, dict[str, Any]] = {}
    for row in previous.to_dict(orient="records"):
        if row["status"] == "success" and row["modelo"] == model:
            reusable[row["id"]] = row
    return reusable


def save_results(frame: pd.DataFrame, path: str | Path) -> None:
    """Salva os resultados por substituição atômica para preservar checkpoints."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, output_path)


def result_record(source: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Combina uma linha fonte com resultado sem enviar os rótulos à API."""
    return {**source, **prediction}
