# Fine-tuning de encoders para análise de sentimento em comentários do YouTube

Código e dados para replicar a parte de fine-tuning do TCC: quatro modelos encoder
(BERTimbau Base, BERTimbau Large, DistilBERT multilíngue e XLM-RoBERTa Base) treinados
para classificar comentários em português como **negativo**, **neutro** ou **positivo**.

## Etapas

| Etapa | Parte de | Treino | Teste |
|---|---|---|---|
| Etapa 1 | modelo base do Hugging Face | Cardiff: 3.033 tweets em português | dataset 2 (500 comentários) |
| Etapa 2 | modelo da Etapa 1 | dataset 1 (500 comentários) | dataset 2 (500 comentários) |

Os quatro modelos usam a mesma receita em cada etapa (mesmos dados, seed 42 e
hiperparâmetros). Não há conjunto de validação: o número de épocas foi fixado antes de
rodar, e o dataset 2 só é usado na avaliação final.

| | Etapa 1 | Etapa 2 |
|---|---|---|
| Épocas | 4 | 4 |
| Learning rate | 2e-5 | 2e-5 |
| Batch | 16 × 2 (acumulação) = 32 | 16 |
| Weight decay | 0,01 | 0,01 |
| Warmup / scheduler | 10% / cosine | — / linear |
| Label smoothing | 0,05 | — |
| `max_length` | 256 | 256 |

## Estrutura

```
run.py                      fine-tuning + avaliação dos modelos de um config
validate_setup.py           confere config e dados sem treinar (segundos, sem GPU)
classificar_encoder.py      classifica o dataset 2 com um modelo salvo (predições + métricas)
cronometrar_finetuning.py   refaz as duas etapas cronometrando cada modelo
src/
  config.py                 leitura e validação dos configs
  data.py                   carregamento dos datasets e tokenização
  train.py                  fine-tuning e avaliação de um modelo
  metrics.py                métricas de qualidade, latência, parâmetros e tamanho
configs/
  etapa1_cardiff.yaml
  etapa2.yaml
dataset/
  cardiff_tweets_ptbr_full.csv   treino da Etapa 1 (text, sentimento)
  dataset1_finetuning.csv        dataset 1: treino da Etapa 2 (comment_id, text, sentimento)
  dataset2_teste.csv             dataset 2: teste das duas etapas (comment_id, text, sentimento)
  rotulos_finais.xlsx            base de rotulagem dos 1.000 comentários (anotadores, rotulo_final)
```

As pastas `modelos/` e `resultados/` são geradas ao rodar o fine-tuning e a classificação
(passos 1 a 3 abaixo) e não são versionadas.

### Dados

- **Cardiff**: subconjunto em português do *tweet_sentiment_multilingual* da Cardiff NLP
  (Barbieri et al., 2022, "XLM-T"), com os splits de treino, validação e teste juntos:
  3.033 tweets, 1.011 por classe.
- **Datasets 1 e 2**: 1.000 comentários do YouTube rotulados manualmente, divididos em
  duas metades com a mesma proporção de classes (122 negativos, 153 neutros e 225
  positivos em cada). Nenhum comentário aparece nas duas.

## Requisitos

- Python 3.12 (o experimento usou 3.12.3)
- GPU NVIDIA com CUDA (o experimento usou uma RTX 5070, CUDA 12.8)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Os modelos base são baixados do Hugging Face na primeira execução.

## Como replicar

Rode da raiz desta pasta, nesta ordem:

```bash
# 0) (opcional) conferir configs e dados
python validate_setup.py --config configs/etapa1_cardiff.yaml
python validate_setup.py --config configs/etapa2.yaml

# 1) Etapa 1 -> salva os modelos em modelos/etapa1_cardiff/
python run.py --config configs/etapa1_cardiff.yaml

# 2) Etapa 2 (parte dos modelos da Etapa 1) -> salva em modelos/etapa2/
python run.py --config configs/etapa2.yaml

# 3) Predições e métricas no dataset 2, para cada etapa e modelo
for etapa in etapa1_cardiff etapa2; do
  for m in neuralmind__bert-base-portuguese-cased neuralmind__bert-large-portuguese-cased \
           distilbert-base-multilingual-cased xlm-roberta-base; do
    python classificar_encoder.py modelos/$etapa/$m resultados/$etapa/predicoes/$m.csv
  done
done
```

O passo 3 grava, para cada modelo, um CSV com a predição e a confiança (maior
probabilidade do softmax) de cada comentário do dataset 2, e um `_relatorio.txt` com
acurácia, F1, precisão e recall macro, relatório por classe e matriz de confusão.
**As métricas do TCC são as desses arquivos.** Ele não sobrescreve arquivos existentes.

O `run.py` também grava, em `resultados/<etapa>/`, um `results.csv` com as métricas da
avaliação feita logo após o treino, tempo de treino, latência, nº de parâmetros e tamanho,
além de relatórios por classe em `reports/`.

### Tempos de fine-tuning

```bash
python cronometrar_finetuning.py
```

Refaz as duas etapas em `cronometragem/`, com cada modelo num processo separado, e grava
`cronometragem/tempos_finetuning.csv`:

| Coluna | O que mede |
|---|---|
| `tempo_total_s` | o comando inteiro de um modelo: iniciar o Python, carregar dados e modelo, treinar, avaliar, medir latência e salvar |
| `tempo_laco_treino_s` | só o laço de treino (`trainer.train()`) |
| `tempo_acumulado_s` | Etapa 1: igual ao total. Etapa 2: total da Etapa 1 + total da Etapa 2 do mesmo modelo |

## Resultados obtidos no TCC

Métricas no dataset 2 (passo 3), para comparar com uma nova execução:

| Modelo | Etapa 1: acurácia | Etapa 1: F1 macro | Etapa 2: acurácia | Etapa 2: F1 macro |
|---|---|---|---|---|
| BERTimbau Large | 0,754 | 0,7467 | 0,788 | 0,7799 |
| BERTimbau Base | 0,738 | 0,7297 | 0,780 | 0,7658 |
| XLM-RoBERTa Base | 0,724 | 0,7107 | 0,784 | 0,7726 |
| DistilBERT multilíngue | 0,506 | 0,4845 | 0,624 | 0,5902 |

## Variação esperada

O treino usa seed 42, mas operações em GPU não são totalmente determinísticas. Uma nova
execução pode diferir em alguns comentários. Na cronometragem, por exemplo, o XLM-RoBERTa
da Etapa 2 chegou a 0,786 de acurácia. Pela mesma razão, o `results.csv` do `run.py`
(avaliação em bf16, logo após o treino) pode diferir em um ou outro comentário das
predições do `classificar_encoder.py` (fp32).

No TCC, as predições da Etapa 1 foram geradas classificando, junto com o dataset 2, outros
500 comentários fora deste pacote. Os lotes de 64 tiveram outra composição, e uma nova
execução do passo 3 pode variar em algum comentário. Pelo mesmo motivo, os tempos totais
da Etapa 1 medidos no TCC incluíam a avaliação desse conjunto maior. O tempo do laço de
treino não é afetado.
