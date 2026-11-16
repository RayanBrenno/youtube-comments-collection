# Classificação de sentimento com Claude 5.5 (zero-shot)

Código, configuração e dados para rodar o Claude Haiku 5.5, o Sonnet 5.5 e o Opus 5.5 na
mesma tarefa dos encoders do TCC: classificar comentários do YouTube em português como
**negativo**, **neutro** ou **positivo**. Não há treino: cada modelo recebe só o prompt e o
comentário (zero-shot).

## Estrutura

```
run_claude.py             ponto de entrada
configs/claude_5_5.yaml   modelos, raciocínio, esforço, limites de saída, preços, teto de gasto
src/
  llm.py                  prompt, chamada à API, leitura da resposta, custo
  config.py               leitura do config
  data.py                 leitura do CSV
  metrics.py              métricas e relatório por classe
dataset/
  dataset2_teste.csv      500 comentários (comment_id, text, sentimento): 122 neg / 153 neu / 225 pos
requirements.txt
.env.example
```

## Como cada comentário é classificado

- **Uma chamada por comentário** com `client.messages.create` do SDK oficial `anthropic`,
  8 chamadas em paralelo. O SDK repete sozinho em caso de limite de requisições ou erro do
  servidor.
- **Prompt de sistema**, igual para os três modelos (os rótulos entram em ordem alfabética):

  ```
  You are a precise text classification model.
  Classify each input into EXACTLY one of these labels: negativo, neutro, positivo.
  Respond with ONLY the label word, exactly as written, and nothing else.
  ```

- **Mensagem do usuário**: só o texto do comentário, cortado em 1.500 caracteres (nenhum
  comentário do dataset chega a esse limite).
- **Sem exemplos no prompt** (`num_fewshot: 0`).
- **`temperature` e os outros parâmetros de amostragem não são enviados**: cada modelo usa o
  próprio padrão.
- **Leitura da resposta**: primeiro procura um rótulo idêntico à resposta. Se não houver,
  procura um rótulo dentro do texto da resposta. Se nenhum aparecer, a resposta é contada
  como inválida.

### Configuração por modelo

| Modelo | ID | Raciocínio (`thinking`) | Esforço (`effort`) | `max_tokens` | US$ por 1M tokens (entrada / saída) |
|---|---|---|---|---|---|
| Haiku 5.5 | `claude-haiku-5-5` | `{type: disabled}` | padrão do modelo (`medium`) | 16 | 0,10 / 0,50 |
| Sonnet 5.5 | `claude-sonnet-5-5` | `{type: between_tools}` (forma de desligar nesse modelo; `disabled` dá erro) | padrão do modelo (`high`) | 16 | 2,00 / 10,00 |
| Opus 5.5 | `claude-opus-5-5` | não enviado (não pode ser desligado) | `low` | 1024 | 4,00 / 20,00 |

O Opus 5.5 recebe mais espaço de saída para que o raciocínio não corte o rótulo. Na rodada
do TCC, a mediana foi de 5 tokens de saída (só o rótulo) e o máximo, 251 tokens.

## Requisitos

- Python 3.12 (o experimento usou 3.12.3)
- Uma chave da API da Anthropic (console.anthropic.com). O plano Claude Pro **não** dá
  acesso à API, que é cobrada à parte.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # e cole a chave em ANTHROPIC_API_KEY
```

O `run_claude.py` carrega o `.env` com `python-dotenv`, que também procura nas pastas acima
desta. Se houver um `.env` com outra chave numa pasta acima, ele pode ser usado.

## Como rodar

Tudo na raiz desta pasta, mudando o config entre as fases:

```bash
# Fase 0: estimar o custo sem classificar nada (count_tokens é gratuito)
#   configs/claude_5_5.yaml -> dry_run: true
python run_claude.py --config configs/claude_5_5.yaml

# Fase 1: piloto com 30 comentários (amostra estratificada, seed 42)
#   -> dry_run: false, limit_samples: 30
python run_claude.py --config configs/claude_5_5.yaml

# Fase 2: os 470 restantes
#   -> dry_run: false, limit_samples: null
python run_claude.py --config configs/claude_5_5.yaml
```

Cada resposta é gravada assim que chega. Se a execução for interrompida, rodar de novo
continua de onde parou, sem pagar outra vez pelos comentários já classificados. O teto
`max_cost_usd: 1.5` cancela o restante de um modelo se o gasto dele passar disso.

### Saídas (`resultados/claude_5_5/`)

Geradas ao rodar a classificação; a pasta não é versionada.

- `predictions/<modelo>.csv`: uma linha por comentário, com as colunas abaixo.
  - `id`: o `comment_id`
  - `text` e `label` (rótulo real)
  - `pred` (vazio = inválida) e `raw_output` (resposta bruta)
  - `stop_reason`, `input_tokens`, `output_tokens`
  - `latency_ms` (ida e volta da chamada) e `cost_usd`
- `reports/<modelo>_classification_report.txt` e `<modelo>_confusion_matrix.csv`.
- `results.csv`: uma linha por modelo, com o nº de respostas válidas e inválidas, a
  acurácia, F1/precisão/recall macro e weighted, a latência média, o custo total e os
  tokens. **As métricas são calculadas só sobre as respostas válidas.**

## Resultados obtidos no TCC

| Modelo | Válidas | Acurácia | F1 macro | F1 weighted | Custo (500 comentários) | Latência média |
|---|---|---|---|---|---|---|
| Haiku 5.5 | 498 de 500 | 0,8333 | 0,8244 | 0,8337 | US$ 0,0068 | 895 ms |
| Sonnet 5.5 | 500 de 500 | 0,8380 | 0,8289 | 0,8396 | US$ 0,1365 | 1.095 ms |
| Opus 5.5 | 500 de 500 | 0,8600 | 0,8463 | 0,8567 | US$ 0,3989 | 1.249 ms |

Nas duas respostas inválidas do Haiku, ele disse que o comentário parecia incompleto em vez
de dar um rótulo, e a resposta foi cortada no limite de 16 tokens. Nenhum modelo recusou
alguma chamada (`stop_reason: refusal`).

## Variação esperada

Os modelos não são determinísticos, porque usam a amostragem padrão e o raciocínio do Opus
varia. Uma nova execução pode diferir em alguns comentários, além da latência e do custo. O
código não usa os fallbacks automáticos da API (encaminhar uma recusa para outro modelo), de
propósito: cada resposta tem que vir do modelo avaliado.
