# Experimentos do TCC: classificação de sentimento em comentários do YouTube

Código e dados para replicar as duas abordagens comparadas no TCC. As duas classificam
comentários em português como **negativo**, **neutro** ou **positivo** e são avaliadas no
mesmo conjunto de teste (dataset 2: 500 comentários rotulados manualmente).

| Pasta | Abordagem | Modelos |
|---|---|---|
| [`encoders/`](encoders/README.md) | fine-tuning em duas etapas (Cardiff e depois dataset 1) | BERTimbau Large e Base, XLM-RoBERTa Base, DistilBERT multilíngue |
| [`claude/`](claude/README.md) | zero-shot via API, sem treino | Claude Haiku 5.5, Sonnet 5.5 e Opus 5.5 |

Cada pasta é autocontida: tem seus próprios dados, `requirements.txt` e README com o passo a
passo. Dá para replicar uma abordagem sem a outra.

## Resultados no dataset 2

| Modelo | Abordagem | Acurácia | F1 macro |
|---|---|---|---|
| Claude Opus 5.5 | zero-shot | 0,860 | 0,8463 |
| Claude Sonnet 5.5 | zero-shot | 0,838 | 0,8289 |
| Claude Haiku 5.5 | zero-shot | 0,833 | 0,8244 |
| BERTimbau Large | fine-tuning (Etapa 2) | 0,788 | 0,7799 |
| XLM-RoBERTa Base | fine-tuning (Etapa 2) | 0,784 | 0,7726 |
| BERTimbau Base | fine-tuning (Etapa 2) | 0,780 | 0,7658 |
| DistilBERT multilíngue | fine-tuning (Etapa 2) | 0,624 | 0,5902 |

Detalhes, custos, tempos e a variação esperada entre execuções estão nos READMEs de cada pasta.

## Dados

- `encoders/dataset/rotulos_finais.xlsx`: base de rotulagem completa, com os 1.000
  comentários, metadados (tema, canal, datas, curtidas), as pré-rotulagens por LLMs, os rótulos
  dos anotadores e o `rotulo_final`. Dela saem os datasets 1 e 2.
- `encoders/dataset/dataset1_finetuning.csv` e `dataset2_teste.csv`: as duas metades da base,
  com a mesma proporção de classes e sem comentários em comum.
- `encoders/dataset/cardiff_tweets_ptbr_full.csv`: tweets em português usados na Etapa 1.
- `claude/dataset/dataset2_teste.csv`: cópia idêntica do dataset 2, para a pasta do Claude
  rodar sozinha.

## Ambiente

As duas pastas usam Python 3.12 e versões compatíveis entre si, então uma venv única nesta pasta
atende às duas:

```bash
cd scripts
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # só para encoders/
pip install -r encoders/requirements.txt -r claude/requirements.txt
```

Os comandos de cada abordagem rodam de dentro da pasta dela (`cd encoders` ou `cd claude`).

## Relação com o sistema

O backend (`backend/`) não depende de nenhum código desta pasta. Ele só lê os modelos prontos
que o fine-tuning salva em `encoders/modelos/etapa2/` (caminho configurável em `MODELOS_DIR` no
`backend/.env`). Para os modelos Claude, o backend reproduz a configuração de `claude/` (prompt,
raciocínio, esforço e leitura da resposta) em `backend/app/services/classificador_claude.py`.

Não são versionados: `.venv/`, `claude/.env` (chave da API) e tudo o que é gerado ao rodar os
experimentos: `encoders/modelos/` (vários GB), `encoders/resultados/`, `encoders/cronometragem/`
e `claude/resultados/`. As predições e métricas de cada modelo são geradas depois do
treinamento (encoders) ou da classificação (Claude), seguindo o README de cada pasta.
