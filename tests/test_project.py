from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import requests

from src.classification import (
    CATEGORIES,
    InvalidClassification,
    build_batch_messages,
    build_messages,
    classify_complaint,
    classify_complaints,
    redact_personal_data,
    validate_classification,
)
from src.data_processing import (
    PREDICTION_COLUMNS,
    load_dataset,
    save_results,
    successful_records,
    text_hash,
)
from src.evaluation import evaluate_results, save_evaluation
from src.openrouter_client import OpenRouterClient, OpenRouterError
from src.visualization import generate_charts


class ClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "categoria_principal": CATEGORIES[0],
            "categoria_secundaria": None,
            "sentimento": "Negativo",
            "resumo": "Cobrança não reconhecida.",
            "problema_principal": "Taxa não informada.",
            "assuntos": ["cobrança", "fatura"],
        }

    def test_prompt_contains_only_complaint_text_not_reference_fields(self) -> None:
        messages = build_messages("Texto de teste.")
        self.assertEqual(messages[1]["content"], "Texto de teste.")
        self.assertNotIn("categoria_referencia", json.dumps(messages))
        self.assertNotIn("sentimento_referencia", json.dumps(messages))

    def test_batch_prompt_contains_only_ids_and_redacted_complaint_text(self) -> None:
        messages = build_batch_messages([("REC-1", "Texto A"), ("REC-2", "Texto B")])
        request = json.loads(messages[1]["content"])
        self.assertEqual(
            request,
            {"reclamacoes": [{"id": "REC-1", "texto": "Texto A"}, {"id": "REC-2", "texto": "Texto B"}]},
        )
        self.assertIn("de forma independente", messages[0]["content"])
        self.assertNotIn("categoria_referencia", json.dumps(messages))

    def test_batch_validation_keeps_valid_items_and_marks_invalid_or_missing(self) -> None:
        client = Mock()
        client.chat.return_value = json.dumps(
            {
                "classificacoes": [
                    {"id": "1", **self.payload},
                    {
                        "id": "2",
                        **dict(self.payload, categoria_principal="Categoria inventada"),
                    },
                ]
            }
        )
        predictions, errors = classify_complaints(
            [("1", "Texto 1"), ("2", "Texto 2"), ("3", "Texto 3")],
            client,
        )
        self.assertEqual(set(predictions), {"1"})
        self.assertIn("taxonomia", errors["2"])
        self.assertIn("não incluiu", errors["3"])

    def test_prompt_masks_common_personal_identifiers(self) -> None:
        messages = build_messages(
            "Meu e-mail é pessoa@example.com, CPF 123.456.789-00, "
            "telefone (11) 91234-5678."
        )
        self.assertNotIn("pessoa@example.com", messages[1]["content"])
        self.assertNotIn("123.456.789-00", messages[1]["content"])
        self.assertNotIn("(11) 91234-5678", messages[1]["content"])
        self.assertEqual(
            messages[1]["content"],
            "Meu e-mail é <EMAIL>, CPF <DOCUMENTO>, telefone <TELEFONE>.",
        )

    def test_validation_rejects_unknown_category_and_bad_json(self) -> None:
        invalid = dict(self.payload, categoria_principal="Categoria inventada")
        with self.assertRaises(InvalidClassification):
            validate_classification(invalid)
        client = Mock()
        client.chat.return_value = "não é json"
        with self.assertRaises(InvalidClassification):
            classify_complaint("Reclamação", client)

    def test_validation_normalizes_topics_to_json(self) -> None:
        result = validate_classification(self.payload)
        self.assertEqual(json.loads(result["assuntos"]), ["cobrança", "fatura"])


class OpenRouterTests(unittest.TestCase):
    @staticmethod
    def response(status: int, content: str = "{}", headers: dict | None = None) -> Mock:
        response = Mock()
        response.status_code = status
        response.ok = 200 <= status < 400
        response.headers = headers or {}
        response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return response

    def test_retries_temporary_http_failure(self) -> None:
        session = Mock()
        session.post.side_effect = [
            self.response(429, headers={"Retry-After": "0"}),
            self.response(200, '{"ok": true}'),
        ]
        client = OpenRouterClient(
            "test-key", "test-model", session=session, sleeper=lambda _: None
        )
        self.assertEqual(client.chat([{"role": "user", "content": "x"}]), '{"ok": true}')
        self.assertEqual(session.post.call_count, 2)

    def test_does_not_retry_permanent_http_failure(self) -> None:
        session = Mock()
        session.post.return_value = self.response(401)
        client = OpenRouterClient(
            "test-key", "test-model", session=session, sleeper=lambda _: None
        )
        with self.assertRaises(OpenRouterError):
            client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(session.post.call_count, 1)

    def test_retries_connection_error_then_succeeds(self) -> None:
        session = Mock()
        session.post.side_effect = [
            requests.ConnectionError("offline"),
            self.response(200, '{"ok": true}'),
        ]
        client = OpenRouterClient(
            "test-key", "test-model", session=session, sleeper=lambda _: None
        )
        self.assertEqual(client.chat([{"role": "user", "content": "x"}]), '{"ok": true}')
        self.assertEqual(session.post.call_count, 2)


class DataAndEvaluationTests(unittest.TestCase):
    def test_cli_defaults_to_five_complaints_per_prompt(self) -> None:
        from unittest.mock import patch

        from main import parse_args

        with patch("sys.argv", ["main.py"]):
            self.assertEqual(parse_args().batch_size, 5)

    def test_loads_dataset_and_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            pd.DataFrame(
                [
                    {"id": "1", "texto_reclamacao": " Texto   normalizado "},
                    {"id": "1", "texto_reclamacao": "Outro texto"},
                ]
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_dataset(path)
            pd.DataFrame(
                [{"id": "", "texto_reclamacao": " Texto   normalizado "}]
            ).to_csv(path, index=False)
            loaded = load_dataset(path)
            self.assertEqual(loaded.loc[0, "texto_reclamacao"], "Texto normalizado")
            self.assertTrue(loaded.loc[0, "id"])

    def test_resume_only_reuses_success_for_same_model_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            first = dict.fromkeys(PREDICTION_COLUMNS, "")
            first.update(
                {
                    "id": "1",
                    "status": "success",
                    "modelo": "model-a",
                    "hash_texto": text_hash("texto"),
                }
            )
            second = dict.fromkeys(PREDICTION_COLUMNS, "")
            second.update(
                {
                    "id": "2",
                    "status": "error",
                    "modelo": "model-a",
                    "hash_texto": text_hash("texto"),
                }
            )
            save_results(
                pd.DataFrame([first, second]),
                path,
            )
            reusable = successful_records(path, "model-a")
            self.assertEqual(list(reusable), ["1"])

    def test_evaluation_excludes_failed_records_and_saves_outputs(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "status": "success",
                    "categoria_referencia": CATEGORIES[0],
                    "categoria_principal": CATEGORIES[0],
                    "sentimento_referencia": "Negativo",
                    "sentimento": "Negativo",
                    "problema_principal": "Taxa indevida",
                },
                {
                    "status": "error",
                    "categoria_referencia": CATEGORIES[1],
                    "categoria_principal": "",
                    "sentimento_referencia": "Neutro",
                    "sentimento": "",
                    "problema_principal": "",
                },
            ]
        )
        evaluation = evaluate_results(frame)
        self.assertEqual(evaluation["metrics"]["registros_avaliados"].tolist(), [1, 1])
        with tempfile.TemporaryDirectory() as temporary:
            save_evaluation(evaluation, temporary)
            charts = generate_charts(frame, evaluation, temporary)
            self.assertTrue((Path(temporary) / "metricas.csv").is_file())
            self.assertTrue(charts)
            self.assertTrue(all(path.is_file() for path in charts))

    def test_cli_flow_saves_and_reuses_success_without_an_api_call(self) -> None:
        from argparse import Namespace
        from unittest.mock import patch

        from main import run

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "input.csv"
            output_path = directory / "results.csv"
            pd.DataFrame(
                [
                    {
                        "id": "1",
                        "texto_reclamacao": "Cobrança não reconhecida.",
                        "categoria_referencia": CATEGORIES[0],
                        "sentimento_referencia": "Negativo",
                    }
                ]
            ).to_csv(input_path, index=False)
            args = Namespace(
                input=input_path,
                output=output_path,
                limit=None,
                batch_size=1,
                charts_dir=directory / "charts",
                reports_dir=directory / "reports",
            )
            response = {
                "categoria_principal": CATEGORIES[0],
                "categoria_secundaria": None,
                "sentimento": "Negativo",
                "resumo": "Cobrança não reconhecida.",
                "problema_principal": "Cobrança não reconhecida.",
                "assuntos": ["cobrança"],
            }
            client = Mock()
            client.chat.return_value = json.dumps(
                {"classificacoes": [{"id": "1", **response}]}
            )

            with patch("main._load_environment"), patch("main._configure_logging"), patch.dict(
                "os.environ", {"OPENROUTER_MODEL": "model-a"}
            ), patch("main.OpenRouterClient.from_environment", return_value=client) as factory:
                self.assertEqual(run(args), 0)
                self.assertEqual(client.chat.call_count, 1)
                self.assertEqual(run(args), 0)
                factory.assert_called_once()

            saved = pd.read_csv(output_path, dtype=str, keep_default_na=False)
            self.assertEqual(saved.loc[0, "status"], "success")

    def test_cli_sends_five_complaints_in_each_prompt(self) -> None:
        from argparse import Namespace
        from unittest.mock import patch

        from main import run

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "input.csv"
            output_path = directory / "results.csv"
            pd.DataFrame(
                [
                    {
                        "id": str(index),
                        "texto_reclamacao": f"Texto de reclamação {index}.",
                        "categoria_referencia": CATEGORIES[0],
                        "sentimento_referencia": "Negativo",
                    }
                    for index in range(6)
                ]
            ).to_csv(input_path, index=False)
            args = Namespace(
                input=input_path,
                output=output_path,
                limit=None,
                batch_size=5,
                charts_dir=directory / "charts",
                reports_dir=directory / "reports",
            )
            response = {
                "categoria_principal": CATEGORIES[0],
                "categoria_secundaria": None,
                "sentimento": "Negativo",
                "resumo": "Resumo.",
                "problema_principal": "Problema.",
                "assuntos": ["assunto"],
            }
            client = Mock()
            request_sizes = []

            def classify_batch(messages: list[dict[str, str]]) -> str:
                request = json.loads(messages[1]["content"])
                request_sizes.append(len(request["reclamacoes"]))
                self.assertNotIn("categoria_referencia", messages[1]["content"])
                return json.dumps(
                    {
                        "classificacoes": [
                            {"id": item["id"], **response}
                            for item in request["reclamacoes"]
                        ]
                    }
                )

            client.chat.side_effect = classify_batch
            with patch("main._load_environment"), patch("main._configure_logging"), patch.dict(
                "os.environ", {"OPENROUTER_MODEL": "model-a"}
            ), patch("main.OpenRouterClient.from_environment", return_value=client):
                self.assertEqual(run(args), 0)

            self.assertEqual(request_sizes, [5, 1])
            self.assertEqual(client.chat.call_count, 2)
            saved = pd.read_csv(output_path, dtype=str, keep_default_na=False)
            self.assertEqual(saved["status"].tolist(), ["success"] * 6)


if __name__ == "__main__":
    unittest.main()
