from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.classification import InvalidClassification, classify_complaints
from src.data_processing import (
    PREDICTION_COLUMNS,
    load_dataset,
    result_record,
    save_results,
    successful_records,
    text_hash,
)
from src.evaluation import evaluate_results, save_evaluation
from src.openrouter_client import OpenRouterClient, OpenRouterError
from src.visualization import generate_charts


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifica reclamações, avalia os resultados e gera gráficos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "consumerinsight_reclamacoes_sinteticas.csv",
        help="Caminho do CSV de entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "reclamacoes_classificadas.csv",
        help="Caminho do CSV de resultados e checkpoint.",
    )
    parser.add_argument("--limit", type=int, help="Quantidade máxima de registros selecionados.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Quantidade de reclamações enviadas em cada prompt.",
    )
    parser.add_argument(
        "--charts-dir", type=Path, default=ROOT / "outputs" / "charts"
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=ROOT / "outputs" / "reports"
    )
    return parser.parse_args()


def _configure_logging(output_path: Path) -> None:
    log_path = output_path.parent / "logs" / "consumerinsight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "Dependência ausente: instale os pacotes com 'python -m pip install -r requirements.txt'."
        ) from error
    load_dotenv(ROOT / ".env")


def _empty_prediction(model: str, digest: str, error: str) -> dict[str, Any]:
    return {
        "categoria_principal": "",
        "categoria_secundaria": "",
        "sentimento": "",
        "resumo": "",
        "problema_principal": "",
        "assuntos": "",
        "status": "error",
        "modelo": model,
        "hash_texto": digest,
        "erro": error,
        "processado_em": datetime.now(timezone.utc).isoformat(),
    }


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit precisa ser maior que zero.")
    if args.batch_size < 1:
        raise ValueError("--batch-size precisa ser maior que zero.")

    _load_environment()
    _configure_logging(args.output)
    logger = logging.getLogger(__name__)
    model = os.getenv("OPENROUTER_MODEL", "")
    dataset = load_dataset(args.input)
    selected = dataset.head(args.limit) if args.limit is not None else dataset
    existing = successful_records(args.output, model) if model else {}

    previous_rows: dict[str, dict[str, Any]] = {}
    if args.output.is_file():
        old = pd.read_csv(args.output, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        expected_columns = {"id", *PREDICTION_COLUMNS}
        if expected_columns.issubset(old.columns):
            previous_rows = {
                row["id"]: row for row in old.to_dict(orient="records") if row["id"]
            }
        else:
            logger.warning("O CSV de saída existente não tem o formato esperado; será substituído.")

    pending = []
    for source in selected.to_dict(orient="records"):
        cached = existing.get(source["id"])
        digest = text_hash(source["texto_reclamacao"])
        if (
            cached
            and cached.get("hash_texto") == digest
            and cached.get("modelo") == model
        ):
            prediction = {column: cached.get(column, "") for column in PREDICTION_COLUMNS}
            previous_rows[source["id"]] = result_record(source, prediction)
            logger.info("Registro %s já processado; reutilizando resultado.", source["id"])
        else:
            pending.append((source, digest))

    client: OpenRouterClient | None = None
    if any(source["texto_reclamacao"] for source, _ in pending):
        client = OpenRouterClient.from_environment()

    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        complaints = [
            (source["id"], source["texto_reclamacao"])
            for source, _ in batch
            if source["texto_reclamacao"]
        ]
        predictions: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        if complaints:
            try:
                assert client is not None
                predictions, errors = classify_complaints(complaints, client)
            except (OpenRouterError, InvalidClassification) as error:
                errors.update({record_id: str(error) for record_id, _ in complaints})

        for source, digest in batch:
            record_id = source["id"]
            if not source["texto_reclamacao"]:
                prediction = _empty_prediction(model, digest, "Texto da reclamação vazio.")
                logger.warning("Registro %s ignorado: texto vazio.", record_id)
            elif record_id in predictions:
                prediction = predictions[record_id]
                prediction.update(
                    {
                        "status": "success",
                        "modelo": model,
                        "hash_texto": digest,
                        "erro": "",
                        "processado_em": datetime.now(timezone.utc).isoformat(),
                    }
                )
                logger.info("Registro %s classificado.", record_id)
            else:
                error = errors.get(record_id, "A reclamação não recebeu uma classificação.")
                prediction = _empty_prediction(model, digest, error)
                logger.error("Falha ao classificar registro %s: %s", record_id, error)

            previous_rows[record_id] = result_record(source, prediction)
            save_results(pd.DataFrame(previous_rows.values()), args.output)

    results = pd.DataFrame(previous_rows.values())
    if results.empty:
        raise ValueError("Não há registros para analisar.")
    evaluation = evaluate_results(results)
    save_evaluation(evaluation, args.reports_dir)
    charts = generate_charts(results, evaluation, args.charts_dir)

    successes = int(results["status"].eq("success").sum())
    failures = len(results) - successes
    logger.info(
        "Concluído: %s sucesso(s), %s registro(s) sem classificação. Gráficos: %s",
        successes,
        failures,
        len(charts),
    )
    if not evaluation["metrics"].empty:
        print(evaluation["metrics"].to_string(index=False))
    print(f"Resultados: {args.output}")
    print(f"Relatórios: {args.reports_dir}")
    print(f"Gráficos: {args.charts_dir}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError, OpenRouterError) as error:
        logging.getLogger(__name__).error("%s", error)
        print(f"Erro: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
