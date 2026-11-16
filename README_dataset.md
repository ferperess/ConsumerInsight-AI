# Dataset ConsumerInsight AI

## Descricao
Dataset sintetico em portugues para experimentos academicos de analise e categorizacao de reclamacoes de consumidores com IA Generativa.

## Conteudo
- 720 registros sinteticos.
- 9 categorias de referencia, com 80 registros por categoria.
- 576 registros com sentimento de referencia negativo.
- 144 registros com sentimento de referencia neutro.
- Datas distribuidas entre 2025-01-01 e 2026-11-12.
- Empresas ficticias; nao representa reclamacoes reais.

## Arquivo
`consumerinsight_reclamacoes_sinteticas.csv`

## Dicionario de dados
| Campo | Descricao |
|---|---|
| id | Identificador unico sintetico |
| data_reclamacao | Data ficticia do registro |
| empresa_ficticia | Nome ficticio de empresa |
| canal | Canal ficticio de registro |
| produto_servico | Produto ou servico relacionado |
| texto_reclamacao | Texto sintetico a ser analisado pela IA |
| categoria_referencia | Categoria de referencia para avaliacao |
| sentimento_referencia | Sentimento de referencia para avaliacao |
| problema_referencia | Problema central de referencia |
| tags_referencia | Tags derivadas dos rotulos de referencia |

## Categorias
- Cobranca e Pagamentos
- Entrega e Logistica
- Qualidade do Produto
- Atendimento ao Cliente
- Cancelamento e Reembolso
- Garantia e Assistencia Tecnica
- Servicos e Contratos
- Cadastro e Acesso
- Outros

## Observacoes metodologicas
1. Os textos foram gerados artificialmente por templates combinados com variacoes. Nao sao relatos reais.
2. Os campos `categoria_referencia`, `sentimento_referencia` e `problema_referencia` sao rotulos sinteticos de referencia. Podem ser usados como apoio para avaliar a classificacao, mas nao equivalem a anotacao humana de reclamacoes reais.
3. O conjunto foi balanceado por categoria (80 registros cada), mas nao por sentimento (80% negativo e 20% neutro).
4. Recomenda-se reservar uma amostra para revisao humana antes de reportar metricas como resultado academico.
5. Remova os campos de referencia do texto enviado ao modelo. Eles devem permanecer apenas para avaliacao posterior.
6. Como os textos sao sinteticos e padronizados, os resultados nao devem ser generalizados diretamente para reclamacoes reais.

## Uso sugerido
- Piloto: 50 a 100 registros.
- Desenvolvimento e ajustes: 200 a 300 registros.
- Experimento final: todos os 720 registros, sujeito ao limite de custo e requisicoes da API.

## Reprodutibilidade
O dataset foi gerado com semente aleatoria fixa (42), mantendo a mesma estrutura e distribuicao.
