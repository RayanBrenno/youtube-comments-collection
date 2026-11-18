# Criação do conjunto de avaliação

Pipeline que constrói o **dataset de validação** do TCC: **1.000 comentários
do YouTube em português do Brasil**, de 10 temas, rotulados como `positivo`,
`negativo` ou `neutro`. O dataset serve para avaliar os modelos de
classificação de sentimento do trabalho (modelos baseados em Transformers e
o Claude) e não foi usado para treinar nenhum deles.

A justificativa metodológica, os números e as referências estão em
[`METODOLOGIA_DATASET.md`](METODOLOGIA_DATASET.md). Este README trata da
execução: o que rodar, em que ordem, o que cada etapa gera e quais passos
são manuais.

## Pasta crua e referência

Os dados ficam em duas pastas:

| Pasta | O que é |
|---|---|
| [`dataset_validacao/`](dataset_validacao/) | **Pasta crua, onde os scripts rodam.** Contém só o que não dá para regerar: os dados baixados da API, o prompt e as anotações feitas por pessoas e pelas LLMs. Todo o resto é gerado pelos scripts. |
| [`reference/`](reference/) | **Versão de referência**, com todos os arquivos gerados na execução original, inclusive o `dataset_rotulado.csv` final. Nenhum script lê ou escreve nela. Serve para comparar os resultados. |

O dataset final fica em `dataset_validacao/dataset_rotulado.csv` depois de
rodar o pipeline, e o rótulo de cada comentário está na coluna
**`rotulo_final`**. A cópia de referência está em
[`reference/dataset_rotulado.csv`](reference/dataset_rotulado.csv).

### Gerar os resultados a partir da pasta crua

Com as dependências instaladas (ver [Pré-requisitos](#pré-requisitos)), rode
a partir da raiz do repositório:

```bash
python evaluation-set-creation/select_videos.py
python evaluation-set-creation/collect_comments.py
python evaluation-set-creation/build_validation_dataset.py
python evaluation-set-creation/sample_blind_validation.py
python evaluation-set-creation/prepare_llm_batches.py
python evaluation-set-creation/collect_llm_labels.py
```

Nada disso usa a internet ou a cota da API. O `select_videos.py` lê a busca
salva em `cache/`, e o `collect_comments.py` encontra os 50 vídeos já
baixados e não baixa nada. Como as anotações já estão na pasta crua, os
passos manuais são pulados, e a última execução já grava o dataset final.
Ao final, os arquivos gerados devem ser **idênticos byte a byte** aos de
`reference/`.

### Voltar a pasta ao estado cru

Os arquivos gerados estão listados no [`.gitignore`](.gitignore) desta
pasta. Para apagar todos eles e manter só as entradas:

```bash
git clean -ndX -- evaluation-set-creation/dataset_validacao   # mostra o que seria apagado
git clean -fdX -- evaluation-set-creation/dataset_validacao   # apaga
```

O `-X` só remove arquivos ignorados, então os dados brutos e as anotações
nunca são apagados por esse comando.

### O que tem na pasta crua

| Arquivo | Origem | Passo |
|---|---|---|
| `cache/<category_id>.json` | busca na YouTube API | 1 |
| `comentarios_brutos/<video_id>.jsonl` | coleta na YouTube API | 2 |
| `coleta_comentarios.json` | registro da coleta (datas e contagens) | 2 |
| `exclusoes_revisao.csv` | revisão manual de idioma | 4 |
| `prompt_anotacao.md` | guia de anotação e prompt das LLMs | 6 e 8 |
| `amostra_cega_classificado.csv` | anotador 1 | 6 |
| `amostra_cega_classificado_2.csv` | anotador 2 | 6 |
| `adjudicacao.xlsx` | consenso entre os anotadores | 7 |
| `llm/respostas/{gpt,deepseek,gemini}.txt` | respostas das LLMs | 9 |
| `llm/modelos.txt` | registro de modelo, versão e data | 9 |
| `divergencias_classificado.xlsx` | classificação manual das divergências | 11 |

Para refazer uma anotação do zero, apague o arquivo correspondente da pasta
crua e siga o passo indicado. Por exemplo, sem os arquivos de
`llm/respostas/`, o `collect_llm_labels.py` pede os lotes de novo para cada
modelo.

## Visão geral

```
 1. select_videos.py            YouTube API → 50 vídeos (5 por tema)
 2. collect_comments.py         YouTube API → 115.167 comentários brutos
 3. build_validation_dataset.py filtros + sorteio → 1.000 comentários
 4. (manual) revisão de idioma  exclusoes_revisao.csv → roda o passo 3 de novo
 5. sample_blind_validation.py  sorteia a amostra cega (200)
 6. (manual) 2 anotadores       classificam a amostra cega, cada um por conta própria
 7. (manual) adjudicação        consenso nas 65 divergências entre anotadores
 8. prepare_llm_batches.py      20 lotes de 50 comentários para as LLMs
 9. (manual) GPT, DeepSeek e Gemini classificam os lotes no chat
10. collect_llm_labels.py       confere as respostas e gera divergencias.csv (130)
11. (manual) classificação das divergências entre LLMs
12. collect_llm_labels.py       → dataset_rotulado.csv (versão final)
```

Numa coleta do zero, os passos 1 e 2 usam a internet e a cota da API. A
partir do passo 3, tudo roda offline sobre os arquivos locais.

## Estrutura da pasta

```
evaluation-set-creation/
├── README.md
├── METODOLOGIA_DATASET.md         texto da metodologia
├── requirements.txt
├── .gitignore                     arquivos gerados dentro de dataset_validacao/
├── common.py                      cliente da YouTube API + normalização de texto
├── select_videos.py               passo 1
├── collect_comments.py            passo 2
├── build_validation_dataset.py    passos 3 e 4
├── sample_blind_validation.py     passo 5
├── prepare_llm_batches.py         passo 8
├── collect_llm_labels.py          passos 10 e 12
├── dataset_validacao/             pasta crua (entradas); os scripts geram aqui:
│   ├── videos_selecionados.csv    os 50 vídeos (passo 1)
│   ├── candidatos_avaliados.csv   os 928 candidatos e o motivo de cada exclusão (passo 1)
│   ├── coleta_metadados.json      parâmetros e critérios da seleção (passo 1)
│   ├── dataset_validacao.csv      os 1.000 comentários, sem rótulo (passo 3)
│   ├── reserva.csv                6 substitutos por vídeo (passo 3)
│   ├── exportacao_metadados.json  semente, filtros e contagens (passo 3)
│   ├── amostra_cega.csv           amostra de 200 em branco (passo 5)
│   ├── llm/lotes/lote_01..20.txt  mensagens para colar no chat (passo 8)
│   ├── llm/ids.csv                id curto (c0001...) → comment_id (passo 8)
│   ├── llm/refazer_<modelo>.txt   comentários faltantes, se houver (passo 10)
│   ├── divergencias.csv           130 comentários sem unanimidade, em branco (passo 10)
│   ├── dataset_rotulado.csv       ★ VERSÃO FINAL (passo 12)
│   └── metricas_anotacao.json     κ, unanimidade e distribuição (passo 12)
└── reference/                     execução original completa (somente leitura)
    ├── (as entradas e todos os arquivos gerados acima)
    ├── amostra_cega_classificado.xlsx / _2.xlsx  cópias de trabalho dos anotadores
    ├── llm/lotes.zip              os lotes compactados, como foram enviados
    ├── cache_v1_sem_filtro_duracao/       histórico: 1ª versão da seleção
    └── historico_v2_canal_por_categoria/  histórico: 2ª versão da seleção
```

## Pré-requisitos

- **Python 3.10 ou superior** (testado com 3.13).
- Dependências, instaladas a partir da raiz do repositório:

  ```bash
  python -m venv .venv
  .venv\Scripts\activate          # Windows
  # source .venv/bin/activate     # Linux/macOS
  pip install -r evaluation-set-creation/requirements.txt
  ```

- **Chave da YouTube Data API v3**, só para uma coleta nova nos passos 1 e 2.
  Ela é lida da variável `YOUTUBE_API_KEY` em `backend/.env`. Com a pasta
  crua atual, a chave não é necessária. O MongoDB da aplicação não é usado
  em nenhuma etapa.

Todos os comandos partem da **raiz do repositório**. Os scripts acham
`dataset_validacao/` pelo próprio caminho, então também funcionam quando
executados de outra pasta.

> **Atenção ao editar CSVs no Excel.** Os scripts leem os CSVs como UTF-8
> **sem BOM**. A opção "CSV UTF-8" do Excel grava um BOM, o que faz a
> coluna `comment_id` deixar de ser reconhecida. Edite a planilha `.xlsx` e
> exporte o CSV por um editor de texto ou pelo Python, ou confira que o
> arquivo não começa com BOM.

---

## Passo 1: seleção dos vídeos

```bash
python evaluation-set-creation/select_videos.py
```

Para cada uma das 10 categorias oficiais do YouTube, o script busca vídeos
com `search.list` (`regionCode=BR`, `relevanceLanguage=pt`, publicados entre
01/01/2025 e 30/09/2026, `order=viewCount`). Em seguida, percorre os
resultados **na ordem da busca** e aprova os 5 primeiros que atendem a todos
os critérios:

| Critério | Regra |
|---|---|
| Categoria | `categoryId` igual ao da busca |
| Formato | não pode ser live, conteúdo infantil (`madeForKids`) nem Short (até 3 min e vertical) |
| Duração na busca | `videoDuration=medium` (4 a 20 min); Música usa `any`, porque clipes costumam ser curtos |
| Música fora do tema | fora da categoria Música, exclui vídeos com tópico musical (`topicDetails`) |
| Idioma e país | idioma `pt`/`pt-BR` e canal do Brasil; exclui se nenhum dos dois puder ser verificado |
| Comentários | de 300 a 10.000, e comentários ativados |
| Visualizações | pelo menos 10.000 |
| Canal | no máximo um vídeo por canal no dataset inteiro |

Como a regra de canal vale para o dataset inteiro, a ordem das categorias em
`CATEGORIES` decide a prioridade quando o mesmo canal aparece em duas delas.

**Cache e cota.** Cada página de busca (50 resultados) custa cerca de 102
unidades da cota diária de 10.000. As páginas ficam salvas em
`dataset_validacao/cache/<category_id>.json`, e novas páginas só são pedidas
quando os candidatos em cache não completam a categoria (no máximo 5
páginas, ou 250 candidatos, por categoria). Se a cota acabar no meio, rode de
novo no dia seguinte: o progresso é mantido. Se os parâmetros de busca
mudarem, o script recusa o cache antigo.

**Saídas:** `videos_selecionados.csv` (o manifesto usado no passo 2),
`candidatos_avaliados.csv` (cada candidato com `status` e `motivo`) e
`coleta_metadados.json`.

**Resultado obtido:** 928 candidatos avaliados, **50 vídeos selecionados**
(5 por tema), com busca feita em 23/10/2026.

> Com o `cache/` da pasta crua, este passo não consome cota e reproduz a
> mesma seleção. Para tirar um snapshot novo da API, apague `cache/`. A
> busca do dia traz outros vídeos, e todos os passos seguintes, inclusive as
> anotações, precisam ser refeitos.

## Passo 2: coleta dos comentários

```bash
python evaluation-set-creation/collect_comments.py
```

Baixa **todos os comentários de primeiro nível** (sem respostas) de cada
vídeo de `videos_selecionados.csv` por `commentThreads.list`, em texto
simples. A coleta é completa porque a API não oferece ordem aleatória, só
por data ou relevância, e o sorteio do passo 3 precisa do conjunto inteiro
de cada vídeo. O nome do autor não é armazenado.

- Custo: 1 unidade a cada 100 comentários (cerca de 1.200 unidades no total).
- Cada vídeo só é gravado depois que todas as páginas dele foram baixadas,
  por isso nunca fica arquivo pela metade. Vídeos já salvos são pulados, e
  basta rodar de novo se a cota acabar.
- Vídeos removidos ou com comentários desativados são registrados como erro
  em `coleta_comentarios.json`.

**Saídas:** `comentarios_brutos/<video_id>.jsonl`, com um comentário por
linha (`comment_id`, `text`, `likes`, `reply_count`, `published_at`,
`updated_at`), e `coleta_comentarios.json`.

**Resultado obtido:** **115.167 comentários** de 50/50 vídeos, coletados em
25/10/2026.

> Na pasta crua os 50 arquivos já existem, então o script só confere e
> mostra o resumo, sem chamar a API.

## Passo 3: filtragem e sorteio

```bash
python evaluation-set-creation/build_validation_dataset.py
```

1. **Normalização:** quebras de linha e espaços repetidos viram um único
   espaço.
2. **Filtros**, aplicados nesta ordem sobre o corpus inteiro:

   | Filtro | Regra | Removidos |
   |---|---|---|
   | Link | URL, `www.` ou domínio (`.com`, `.br`, ...) | 135 |
   | Curto | menos de 3 palavras, sem contar menções `@` | 21.067 |
   | Longo | mais de 1.000 caracteres | 294 |
   | Duplicado | texto idêntico, sem diferenciar maiúsculas; fica o mais antigo | 3.567 |
   | Idioma | confiança de português < 0,05 no `lingua` (pt/es/en) | 1.572 |
   | **Aprovados** | | **88.532** |

   O limite de idioma é baixo de propósito. O detector confunde português
   informal ("que casa lindaaa", "SHOW DE BOLA") com espanhol ou inglês. Os
   estrangeiros que passarem por ele são removidos no passo 4.

3. **Sorteio:** em cada vídeo, os comentários aprovados são ordenados por
   `comment_id` e embaralhados com `random.Random(f"42-{video_id}")`. Os 20
   primeiros entram no dataset e os 6 seguintes vão para a reserva. Curtidas
   e data não influenciam o sorteio.
4. **Exclusões da revisão manual:** os comentários listados em
   `exclusoes_revisao.csv` (passo 4) são pulados, e o próximo do sorteio
   entra no lugar.

**Saídas:** `dataset_validacao.csv` (1.000 linhas, com `sentimento` vazio),
`reserva.csv` (300 linhas) e `exportacao_metadados.json`. A coluna
`ordem_sorteio` guarda a posição de cada comentário no sorteio do vídeo.

## Passo 4: revisão manual de idioma (manual)

> Na pasta crua, `exclusoes_revisao.csv` já está preenchido, e o passo 3
> já aplica as exclusões. Os itens abaixo descrevem como ele foi feito.

1. Rode o passo 3 sem `exclusoes_revisao.csv`.
2. Em `dataset_validacao.csv`, revise os comentários com
   `confianca_pt < 0.5`. Faça o mesmo em `reserva.csv`, porque os
   substitutos saem de lá.
3. Registre em `exclusoes_revisao.csv` os que estão **em outro idioma** ou
   **sem sentido**:

   ```csv
   comment_id,motivo
   UgzbwY6Ps34HjKWWuox4AaABAg,idioma estrangeiro (inglês)
   UgwZp9wzQQZd90UoLCN4AaABAg,texto sem sentido
   ```

4. Rode o passo 3 de novo. O sorteio **não é refeito**: o comentário
   excluído sai, os seguintes do mesmo vídeo sobem uma posição, e o primeiro
   da reserva entra no dataset.
5. Repita até não sobrar nenhum estrangeiro no dataset, já que um substituto
   também pode ser estrangeiro.

Comentários irônicos, ambíguos ou com gírias **não** são excluídos, porque
são casos difíceis legítimos para a avaliação.

**Resultado obtido:** **25 exclusões** (23 em outro idioma e 2 sem sentido),
todas repostas pela reserva. O dataset fechou com **1.000 comentários, 100
por tema**.

> **A partir daqui, `dataset_validacao.csv` precisa ficar estável.** As
> etapas seguintes se referem aos comentários pelo `comment_id`. Rodar o
> passo 3 de novo com as mesmas entradas gera um arquivo idêntico, mas
> qualquer mudança em filtros, semente ou exclusões invalida as anotações da
> pasta crua.

## Passo 5: sorteio da amostra cega

```bash
python evaluation-set-creation/sample_blind_validation.py
```

Sorteia **20 comentários por tema (200 no total)** do dataset, com semente
`42-amostra-cega`, e embaralha as linhas para os temas não aparecerem
agrupados. O arquivo tem só `comment_id`, `text` e `sentimento` (vazio), sem
tema, canal ou outra informação que possa influenciar quem anota.

**Saída:** `amostra_cega.csv`. Por segurança, o script se recusa a rodar se
esse arquivo já existir.

## Passo 6: anotação humana da amostra cega (manual)

> Na pasta crua, os dois arquivos anotados já existem.

Esta etapa precisa ser feita **antes** de executar as LLMs (passo 9), para
que nenhum anotador veja os rótulos delas.

1. Os dois anotadores leem o guia em
   [`prompt_anotacao.md`](dataset_validacao/prompt_anotacao.md): definições
   das três classes e regras para ironia, comentários mistos, perguntas,
   emojis, gírias e relatos negativos.
2. Cada anotador recebe uma cópia de `amostra_cega.csv` e preenche a coluna
   `sentimento` com `positivo`, `negativo` ou `neutro`, **sem consultar o
   outro**.
3. Salve as duas versões em `dataset_validacao/` como:
   - `amostra_cega_classificado.csv` para o anotador 1;
   - `amostra_cega_classificado_2.csv` para o anotador 2.

   Os scripts leem só os `.csv`. As planilhas `.xlsx` usadas como cópia de
   trabalho na execução original estão em `reference/`.

**Resultado obtido:** concordância de **κ = 0,50** entre os dois anotadores
(moderada), com **65 divergências**, concentradas na fronteira entre neutro
e polarizado.

## Passo 7: adjudicação das divergências humanas (manual)

> Na pasta crua, `adjudicacao.xlsx` já está preenchido.

1. Gere `adjudicacao.xlsx` com os comentários em que os dois anotadores
   discordaram:

   ```python
   import csv
   from openpyxl import Workbook

   d = "evaluation-set-creation/dataset_validacao/"
   a1 = {r["comment_id"]: r for r in csv.DictReader(open(d + "amostra_cega_classificado.csv", encoding="utf-8"))}
   a2 = {r["comment_id"]: r["sentimento"] for r in csv.DictReader(open(d + "amostra_cega_classificado_2.csv", encoding="utf-8"))}

   wb = Workbook()
   ws = wb.active
   ws.title = "adjudicacao"
   ws.append(["comment_id", "text", "anotador_1", "anotador_2", "consenso"])
   for cid, r in a1.items():
       if r["sentimento"] != a2[cid]:
           ws.append([cid, r["text"], r["sentimento"], a2[cid], None])
   wb.save(d + "adjudicacao.xlsx")
   ```

2. Os dois anotadores discutem cada caso com base no guia e preenchem a
   coluna **`consenso`** com `positivo`, `negativo` ou `neutro`.

O **gabarito humano** dos 200 comentários fica assim: onde os anotadores
concordaram, vale o rótulo deles; onde discordaram, vale o `consenso`. Quem
monta essa junção é o `collect_llm_labels.py`, e enquanto houver consenso em
branco ele avisa quantos comentários da amostra estão pendentes.

## Passo 8: preparação dos lotes para as LLMs

```bash
python evaluation-set-creation/prepare_llm_batches.py
```

- Embaralha os 1.000 comentários com semente `42-lotes-llm`, para cada lote
  misturar temas e vídeos.
- Troca o `comment_id` por ids curtos (`c0001` a `c1000`), para o modelo não
  errar ao copiá-los. A correspondência fica em `llm/ids.csv`.
- Monta **20 lotes de 50 comentários** em `llm/lotes/lote_XX.txt`. Cada lote
  já traz o prompt (o bloco entre as duas primeiras linhas `---` de
  `prompt_anotacao.md`) seguido dos comentários em JSON, pronto para colar.
- Cria `llm/respostas/gpt.txt`, `deepseek.txt` e `gemini.txt` se ainda não
  existirem. Arquivos de resposta que já existem não são alterados.

Por segurança, o script se recusa a rodar se `llm/ids.csv` já existir. Como
o embaralhamento usa semente fixa, os ids gerados a partir da pasta crua são
os mesmos usados nas respostas salvas.

## Passo 9: anotação pelas LLMs (manual)

> Na pasta crua, as três respostas já estão em `llm/respostas/`.

Os 1.000 comentários são classificados por **três LLMs de empresas
diferentes: GPT, DeepSeek e Gemini**. O Claude não participa da anotação
porque é um dos modelos avaliados, o que evita viés de autopreferência.

Para cada modelo:

1. Abra uma **conversa nova e sem memória** (modo temporário, quando
   existir) para **cada lote**.
2. Cole o conteúdo inteiro de `lote_XX.txt`.
3. Copie a resposta e cole no fim de `llm/respostas/<modelo>.txt`, uma
   resposta depois da outra. A ordem não importa, e o arquivo aceita texto,
   blocos de código ```` ```json ```` e respostas cortadas ao redor do JSON.
4. Anote em `llm/modelos.txt` o nome e a versão do modelo exibidos no chat,
   as datas de execução e observações, como lotes refeitos ou respostas
   cortadas.

## Passo 10: conferência e geração das divergências

```bash
python evaluation-set-creation/collect_llm_labels.py
```

O script lê as três respostas e verifica se cada modelo classificou os
1.000 comentários. Ids desconhecidos e rótulos inválidos são contados e
ignorados.

- **Se faltar algum comentário,** é gerado `llm/refazer_<modelo>.txt`, com o
  prompt e só os comentários faltantes. Cole esse arquivo numa conversa nova
  do mesmo modelo, acrescente a resposta em `respostas/<modelo>.txt` e rode
  o script de novo. Repita até os três modelos ficarem com 1000/1000. O
  arquivo de refazer é apagado sozinho quando o modelo fica completo.
- **Com os três completos,** o script gera `divergencias.csv`: os
  comentários **fora da amostra cega** em que as LLMs **não foram
  unânimes**, só com `comment_id`, `text` e `sentimento` vazio. O arquivo só
  é criado se ainda não existir.

**Resultado obtido:** unanimidade das três LLMs em **836 dos 1.000**
comentários (83,6%) e **130 divergências** fora da amostra cega.

## Passo 11: classificação manual das divergências (manual)

> Na pasta crua, `divergencias_classificado.xlsx` já está preenchido. Por
> isso, a execução do passo 10 já sai com o dataset final, e o passo 12 não
> precisa ser rodado de novo.

1. Classifique os 130 comentários de `divergencias.csv` com o mesmo guia,
   **sem consultar os rótulos das LLMs**. Por isso o arquivo não os mostra.
2. Salve o resultado como `divergencias_classificado.csv` ou
   `divergencias_classificado.xlsx`, com as colunas `comment_id` e
   `sentimento`. Se os dois existirem, o `.csv` tem prioridade.

## Passo 12: consolidação da versão final

```bash
python evaluation-set-creation/collect_llm_labels.py
```

Com todas as anotações presentes, o script grava o dataset final. O
**rótulo final** de cada comentário segue esta regra:

| Origem (`origem`) | Regra | Comentários |
|---|---|---|
| `cego` | gabarito humano da amostra cega (passos 6 e 7) | 200 |
| `unanime` | rótulo das três LLMs, quando unânimes | 670 |
| `divergente` | classificação manual do passo 11 | 130 |

A execução termina com `Rótulos finais pendentes: 0`. Se aparecer outro
número, falta consenso no passo 7 ou classificação no passo 11.

**Saídas:**

- **`dataset_rotulado.csv`**, a versão final, com todas as colunas de
  `dataset_validacao.csv` (menos `sentimento`) e mais:

  | Coluna | Conteúdo |
  |---|---|
  | `llm_gpt`, `llm_deepseek`, `llm_gemini` | rótulo de cada LLM |
  | `rotulo_anotador_1`, `rotulo_anotador_2` | rótulos humanos antes da adjudicação (só amostra cega) |
  | `rotulo_humano_cego` | gabarito humano após a adjudicação (só amostra cega) |
  | `rotulo_manual` | classificação manual (só divergências) |
  | **`rotulo_final`** | **rótulo a ser usado na avaliação dos modelos** |
  | `origem` | `cego`, `unanime` ou `divergente` |

- **`metricas_anotacao.json`**, com as métricas da anotação.

**Resultado obtido:**

| Indicador | Valor |
|---|---|
| κ entre os anotadores humanos (antes da adjudicação) | 0,501 |
| Unanimidade entre as três LLMs | 83,6% |
| κ de cada LLM contra o gabarito humano | GPT 0,744 · DeepSeek 0,735 · Gemini 0,748 |
| κ da maioria das três LLMs contra o gabarito | 0,764 |
| Distribuição final | 450 positivos (45,0%) · 306 neutros (30,6%) · 244 negativos (24,4%) |

Como as classes estão desbalanceadas, a métrica principal na avaliação dos
modelos deve ser o **F1-macro**.

---

## Reprodutibilidade

| Etapa | Reproduzível? |
|---|---|
| 1. Seleção | Sim, a partir de `cache/`. Uma busca nova na API gera outro snapshot. |
| 2. Coleta | Não: comentários são apagados, editados e adicionados com o tempo. O snapshot é `comentarios_brutos/`. |
| 3 e 4. Filtros e sorteio | Sim, com sementes fixas e `exclusoes_revisao.csv`. |
| 5 e 8. Amostra cega e lotes | Sim, com sementes fixas. Os scripts recusam sobrescrever as saídas. |
| 6, 7, 9 e 11. Anotações | Não: dependem de pessoas e das LLMs no chat (sem controle de temperatura). Os registros ficam na pasta crua. |
| 10 e 12. Consolidação | Sim. |

Rodando os seis scripts sobre a pasta crua, os 32 arquivos gerados
(incluindo os 20 lotes) ficam **idênticos byte a byte** aos de
`reference/`. Para conferir depois de rodar, no Git Bash ou no Linux:

```bash
cd evaluation-set-creation
for f in $(cd reference && ls *.csv *.json llm/ids.csv llm/lotes/*.txt); do
  cmp -s "reference/$f" "dataset_validacao/$f" && echo "OK   $f" || echo "DIFF $f"
done
```

## Histórico da seleção de vídeos

As duas pastas de histórico em `reference/` guardam versões anteriores do
passo 1, mantidas para documentar como os critérios evoluíram. Nenhum
script as lê.

- **`cache_v1_sem_filtro_duracao/`**: primeira versão, com busca sem o filtro
  `videoDuration` e no máximo um vídeo por canal **dentro de cada
  categoria**.
- **`historico_v2_canal_por_categoria/`**: segunda versão, já com o filtro de
  duração, mas ainda com um vídeo por canal **por categoria**. A versão
  atual passou a limitar a um vídeo por canal **no dataset inteiro**.

## Pendências de registro

- `dataset_validacao/llm/modelos.txt` ainda está em branco: falta preencher
  modelo, versão e datas de execução de cada LLM (ver passo 9).
