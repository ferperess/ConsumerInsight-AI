# ConsumerInsight AI

O ConsumerInsight AI é um projeto acadêmico para estruturar e analisar reclamações de consumidores com apoio de modelos de linguagem acessados pela OpenRouter. A aplicação classifica cada texto, identifica sentimento, resume o problema, extrai assuntos e compara as previsões com os rótulos de referência do dataset.

> **Dataset:** os 720 registros fornecidos são sintéticos, criados por I.A e associados a empresas fictícias.

## Como a IA é utilizada

As reclamações são enviadas em lotes de cinco por prompt (configurável por `--batch-size`), e cada item é classificado de forma independente. O prompt contém apenas o ID e o texto de `texto_reclamacao`; informações contextuais e, principalmente, os campos de referência não são enviados. Formatos comuns de e-mail, CPF/CNPJ e telefone brasileiro são mascarados antes da chamada. A resposta JSON é validada por ID antes de ser salva. Respostas inválidas e falhas da API ficam marcadas como erro para possibilitar nova execução.

A classificação segue as categorias e sentimentos presentes no dataset. Os campos terminados em `_referencia` são mantidos no CSV de resultados para avaliação local, nunca no prompt.

## Tecnologias

- Python 3.10 ou superior
- Pandas e Requests
- OpenRouter API
- Matplotlib
- Scikit-learn
- python-dotenv

## Estrutura

```text
.
├── consumerinsight_reclamacoes_sinteticas.csv
├── README_dataset.md
├── main.py
├── requirements.txt
├── .env.example
├── src/
│   ├── classification.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── openrouter_client.py
│   └── visualization.py
├── notebooks/
│   └── analise_resultados.ipynb
├── tests/
│   └── test_project.py
└── outputs/                 # Gerado durante a execução
    ├── charts/
    ├── logs/
    └── reports/
```

## Instalação

No PowerShell, na pasta do projeto:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução

Para processar uma amostra de 30 registros:

```powershell
python main.py --limit 30
```

Para processar o dataset completo:

```powershell
python main.py
```

Também é possível indicar caminhos próprios e controlar quantas reclamações vão em cada prompt:

```powershell
python main.py --input caminho\reclamacoes.csv --output outputs\resultado.csv --limit 100 --batch-size 5
```

O CSV de saída é atualizado após cada registro do lote. Ao executar novamente com o mesmo arquivo de saída, os sucessos só são reutilizados quando o ID, o hash do texto e o modelo permanecem iguais. Registros que falharam são tentados novamente. O parâmetro `--limit` seleciona os primeiros registros do dataset de entrada; use-o para controlar o custo da API.

Para explorar o dataset e visualizar resultados já processados em um notebook:

```powershell
jupyter notebook notebooks\analise_resultados.ipynb
```

O notebook não chama a API; a classificação continua sendo executada pelo `main.py`.

## Arquivos gerados e interpretação

- `outputs/reclamacoes_classificadas.csv`: dados de entrada, previsões, status, modelo, hash e erro, se houver. Os campos de referência são preservados para avaliação local.
- `outputs/logs/consumerinsight.log`: andamento e erros sem registrar o texto das reclamações.
- `outputs/reports/metricas.csv`: acurácia, precisão macro, recall macro e F1 macro para categoria e sentimento.
- `outputs/reports/relatorio_*.csv`: métricas por classe.
- `outputs/reports/matriz_confusao_*.csv`: contagens de referência versus previsão.
- `outputs/charts/`: distribuição de categorias e sentimentos, comparação com referências, problemas recorrentes, evolução temporal quando houver datas e matrizes de confusão.

Registros sem classificação válida não entram nas métricas. As categorias de referência são rótulos sintéticos balanceados; não equivalem a uma anotação humana revisada. Recomenda-se revisar manualmente uma amostra antes de interpretar ou apresentar métricas.

## Testes

Os testes não chamam a OpenRouter nem consomem créditos:

```powershell
python -m unittest discover -v
```

Eles verificam validação e preparação do CSV, exclusão de campos de referência do prompt, validação do JSON e da taxonomia, repetição de requisições temporárias, retomada de resultados, métricas e geração de gráficos. A execução com API real só é feita ao executar `main.py` com credenciais válidas.

## Limitações

- O processamento depende de conectividade, disponibilidade, limites e custos da OpenRouter.
- Os rótulos de referência do dataset são sintéticos e podem refletir as próprias regras de geração do conjunto.
- Textos sintéticos e padronizados não representam diversidade linguística, ambiguidades ou ruído de reclamações reais.
- O modelo pode produzir classificações incorretas; valide respostas e revise os resultados antes de usá-los em decisões.
- A anonimização automática cobre formatos comuns de e-mail, CPF/CNPJ e telefones, mas não identifica com segurança nomes, endereços ou todos os formatos possíveis. Revise e anonimize manualmente os dados antes de enviar bases externas.

## Integrantes

- Fernando Peres
- João Victor Costa
- José Gabriel
- Lucas Sabino