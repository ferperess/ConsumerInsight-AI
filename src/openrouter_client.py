from __future__ import annotations

import os
import time
from typing import Callable

import requests


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """Erro explícito de configuração ou comunicação com a OpenRouter."""


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 45,
        max_retries: int = 3,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise OpenRouterError("OPENROUTER_API_KEY não foi configurada.")
        if not model.strip():
            raise OpenRouterError("OPENROUTER_MODEL não foi configurado.")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("Timeout deve ser positivo e max_retries não pode ser negativo.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()
        self.sleeper = sleeper

    @classmethod
    def from_environment(cls) -> "OpenRouterClient":
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model=os.getenv("OPENROUTER_MODEL", ""),
            timeout=float(os.getenv("OPENROUTER_TIMEOUT", "45")),
            max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "3")),
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    ENDPOINT, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.RequestException as error:
                if attempt >= self.max_retries:
                    raise OpenRouterError(
                        f"Falha de comunicação com a OpenRouter após {attempt + 1} tentativa(s)."
                    ) from error
                self.sleeper(min(2**attempt, 8))
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt >= self.max_retries:
                    raise OpenRouterError(
                        f"OpenRouter retornou HTTP {response.status_code} após "
                        f"{attempt + 1} tentativa(s)."
                    )
                self.sleeper(self._retry_delay(response, attempt))
                continue
            if not response.ok:
                raise OpenRouterError(f"OpenRouter retornou HTTP {response.status_code}.")

            try:
                body = response.json()
                content = body["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise OpenRouterError("Resposta da OpenRouter fora do formato esperado.") from error
            if not isinstance(content, str) or not content.strip():
                raise OpenRouterError("A OpenRouter retornou conteúdo vazio.")
            return content.strip()

        raise OpenRouterError("Não foi possível concluir a requisição à OpenRouter.")

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0), 30)
            except ValueError:
                pass
        return min(2**attempt, 8)
