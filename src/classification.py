from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.openrouter_client import OpenRouterClient


LOGGER = logging.getLogger(__name__)
CATEGORIES = [
    "Cobranca e Pagamentos",
    "Entrega e Logistica",
    "Qualidade do Produto",
    "Atendimento ao Cliente",
    "Cancelamento e Reembolso",
    "Garantia e Assistencia Tecnica",
    "Servicos e Contratos",
    "Cadastro e Acesso",
    "Outros",
]
SENTIMENTS = ["Negativo", "Neutro", "Positivo"]
PII_PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "<EMAIL>"),
    (
        re.compile(
            r"(?<!\d)(?:\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|"
            r"\d{3}\.?\d{3}\.?\d{3}-?\d{2})(?!\d)"
        ),
        "<DOCUMENTO>",
    ),
    (
        re.compile(r"(?<!\d)(?:\+?55\s*)?\(?\d{2}\)?[\s.-]*9?\d{4}[\s.-]?\d{4}(?!\d)"),
        "<TELEFONE>",
    ),
)


class InvalidClassification(ValueError):
    """A resposta do modelo não corresponde ao contrato de classificação."""


def build_messages(text: str) -> list[dict[str, str]]:
    system_prompt = (
        _system_prompt()
        + " Retorne somente JSON valido com estas chaves: "
        + "categoria_principal, categoria_secundaria, sentimento, resumo, "
        + "problema_principal e assuntos (lista de strings)."
    )
    # O texto é o único dado do registro enviado ao provedor.
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redact_personal_data(text)},
    ]


def build_batch_messages(complaints: list[tuple[str, str]]) -> list[dict[str, str]]:
    if not complaints:
        raise ValueError("O lote precisa conter pelo menos uma reclamação.")
    if len({record_id for record_id, _ in complaints}) != len(complaints):
        raise ValueError("Os IDs das reclamações no lote precisam ser únicos.")

    system_prompt = (
        _system_prompt()
        + " Analise cada reclamação do lote de forma independente. "
        + "Retorne somente um objeto JSON com a chave classificacoes, contendo uma "
        + "classificação por reclamação e o respectivo id recebido. "
        + "Cada item deve conter id, categoria_principal, categoria_secundaria, "
        + "sentimento, resumo, problema_principal e assuntos. Não omita nem repita IDs."
    )
    complaint_payload = {
        "reclamacoes": [
            {"id": record_id, "texto": redact_personal_data(text)}
            for record_id, text in complaints
        ]
    }
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(complaint_payload, ensure_ascii=False),
        },
    ]


def _system_prompt() -> str:
    return (
        "Voce analisa reclamacoes de consumidores em portugues brasileiro. "
        "Classifique com base no problema efetivamente relatado, sem inferir fatos. "
        "Use categoria_principal exatamente de: "
        f"{json.dumps(CATEGORIES, ensure_ascii=False)}. "
        "Use uma categoria_secundaria da mesma lista ou null. "
        "Use sentimento exatamente de Negativo, Neutro ou Positivo. "
        "Se nao houver encaixe claro, use Outros."
    )


def redact_personal_data(text: str) -> str:
    """Mascara e-mail, CPF/CNPJ e telefones brasileiros em formatos comuns."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def validate_classification(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise InvalidClassification("A resposta precisa ser um objeto JSON.")

    category = payload.get("categoria_principal")
    secondary = payload.get("categoria_secundaria")
    sentiment = payload.get("sentimento")
    if category not in CATEGORIES:
        raise InvalidClassification("Categoria principal ausente ou fora da taxonomia.")
    if secondary is not None and secondary not in CATEGORIES:
        raise InvalidClassification("Categoria secundária fora da taxonomia.")
    if sentiment not in SENTIMENTS:
        raise InvalidClassification("Sentimento ausente ou inválido.")

    result: dict[str, Any] = {
        "categoria_principal": category,
        "categoria_secundaria": secondary,
        "sentimento": sentiment,
    }
    for field in ("resumo", "problema_principal"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise InvalidClassification(f"Campo obrigatório inválido: {field}.")
        result[field] = value.strip()

    topics = payload.get("assuntos")
    if not isinstance(topics, list) or any(
        not isinstance(item, str) or not item.strip() for item in topics
    ):
        raise InvalidClassification("O campo assuntos precisa ser uma lista de textos.")
    result["assuntos"] = json.dumps(
        [item.strip() for item in topics], ensure_ascii=False
    )
    return result


def classify_complaint(text: str, client: OpenRouterClient) -> dict[str, Any]:
    raw_response = client.chat(build_messages(text))
    payload = _parse_response(raw_response)
    return validate_classification(payload)


def classify_complaints(
    complaints: list[tuple[str, str]], client: OpenRouterClient
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Classifica várias reclamações em uma chamada e valida cada resultado por ID."""
    messages = build_batch_messages(complaints)
    raw_response = client.chat(messages)
    payload = _parse_response(raw_response)
    if not isinstance(payload, dict) or not isinstance(payload.get("classificacoes"), list):
        raise InvalidClassification(
            "A resposta do lote precisa conter uma lista em 'classificacoes'."
        )

    expected_ids = {record_id for record_id, _ in complaints}
    predictions: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    duplicated_ids: set[str] = set()

    for item in payload["classificacoes"]:
        if not isinstance(item, dict):
            continue
        record_id = item.get("id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            LOGGER.warning("Resposta em lote contém um ID ausente ou desconhecido; item ignorado.")
            continue
        if record_id in duplicated_ids:
            continue
        if record_id in predictions or record_id in errors:
            predictions.pop(record_id, None)
            errors[record_id] = "A resposta do modelo repetiu o ID da reclamação."
            duplicated_ids.add(record_id)
            continue

        classification = {key: value for key, value in item.items() if key != "id"}
        try:
            predictions[record_id] = validate_classification(classification)
        except InvalidClassification as error:
            errors[record_id] = str(error)

    for record_id in expected_ids:
        if record_id not in predictions and record_id not in errors:
            errors[record_id] = "A resposta do modelo não incluiu esta reclamação."

    return predictions, errors


def _parse_response(raw_response: str) -> Any:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as error:
        LOGGER.warning("Resposta do modelo não contém JSON válido.")
        raise InvalidClassification("Resposta do modelo não contém JSON válido.") from error
