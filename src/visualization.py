from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_charts(
    frame: pd.DataFrame, evaluation: dict, output_dir: str | Path
) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    successful = frame.loc[frame["status"] == "success"].copy()
    created: list[Path] = []

    created.append(
        _bar_chart(
            successful["categoria_principal"].value_counts(),
            "Reclamações por categoria prevista",
            "Categoria",
            "Quantidade",
            directory / "categorias_previstas.png",
            rotate=True,
        )
    )
    created.append(
        _bar_chart(
            successful["sentimento"].value_counts(),
            "Distribuição dos sentimentos previstos",
            "Sentimento",
            "Quantidade",
            directory / "sentimentos_previstos.png",
        )
    )

    if "categoria_referencia" in successful:
        comparison = pd.concat(
            [
                successful["categoria_referencia"].value_counts().rename("Referência"),
                successful["categoria_principal"].value_counts().rename("IA"),
            ],
            axis=1,
        ).fillna(0)
        axis = comparison.plot(kind="bar", figsize=(12, 6))
        axis.set(title="Categorias: referência e IA", xlabel="Categoria", ylabel="Quantidade")
        axis.legend(title="Origem")
        axis.tick_params(axis="x", rotation=45)
        created.append(_save_figure(axis.figure, directory / "comparacao_categorias.png"))

    if {"categoria_principal", "sentimento"}.issubset(successful.columns):
        cross = pd.crosstab(successful["categoria_principal"], successful["sentimento"])
        axis = cross.plot(kind="bar", stacked=True, figsize=(12, 6))
        axis.set(title="Sentimentos por categoria prevista", xlabel="Categoria", ylabel="Quantidade")
        axis.legend(title="Sentimento")
        axis.tick_params(axis="x", rotation=45)
        created.append(_save_figure(axis.figure, directory / "categoria_sentimento.png"))

    if "problema_principal" in successful:
        top_problems = successful["problema_principal"].value_counts().head(15).sort_values()
        created.append(
            _bar_chart(
                top_problems,
                "Problemas mais identificados pela IA (até 15)",
                "Menções",
                "Quantidade",
                directory / "principais_problemas.png",
                horizontal=True,
            )
        )

    if "data_reclamacao" in successful:
        dates = pd.to_datetime(successful["data_reclamacao"], errors="coerce")
        temporal = dates.dropna().dt.to_period("M").value_counts().sort_index()
        if not temporal.empty:
            figure, axis = plt.subplots(figsize=(11, 5))
            axis.plot(temporal.index.astype(str), temporal.values, marker="o")
            axis.set(title="Reclamações por mês", xlabel="Mês", ylabel="Quantidade")
            axis.tick_params(axis="x", rotation=45)
            created.append(_save_figure(figure, directory / "evolucao_temporal.png"))

    for name, matrix in evaluation["confusion_matrices"].items():
        figure, axis = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.9), 7))
        image = axis.imshow(matrix.values, cmap="Blues")
        axis.set(
            title=f"Matriz de confusão — {name}",
            xlabel="Classificação da IA",
            ylabel="Referência",
            xticks=range(len(matrix.columns)),
            xticklabels=matrix.columns,
            yticks=range(len(matrix.index)),
            yticklabels=matrix.index,
        )
        axis.tick_params(axis="x", rotation=45)
        figure.colorbar(image, ax=axis, label="Quantidade")
        if len(matrix) <= 12:
            for row in range(len(matrix.index)):
                for column in range(len(matrix.columns)):
                    axis.text(column, row, str(matrix.iloc[row, column]), ha="center", va="center")
        created.append(_save_figure(figure, directory / f"matriz_confusao_{name}.png"))

    return created


def _bar_chart(
    values: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    rotate: bool = False,
    horizontal: bool = False,
) -> Path:
    figure, axis = plt.subplots(figsize=(11, 6))
    if horizontal:
        values.plot(kind="barh", ax=axis)
    else:
        values.plot(kind="bar", ax=axis)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if rotate:
        axis.tick_params(axis="x", rotation=45)
    return _save_figure(figure, path)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
