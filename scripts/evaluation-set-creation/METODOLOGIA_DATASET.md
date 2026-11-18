# Metodologia: construção e anotação do dataset de validação

> **Nota para o agente que vai reescrever este texto:** esta é uma das
> etapas da metodologia do TCC e deve ficar **enxuta**. Os números são
> definitivos e não devem ser alterados. O procedimento combina práticas
> já estabelecidas na literatura com parâmetros definidos neste trabalho,
> então não o apresente como um método novo. Os itens a confirmar estão no
> final.

## Objetivo

Construir um conjunto de **1.000 comentários do YouTube em português do
Brasil**, de temas variados, rotulados em **positivo, negativo ou neutro**.
Ele é a base principal de validação dos modelos de classificação de
sentimento avaliados no trabalho: modelos baseados em Transformers e o
Claude. O dataset não foi usado para treinar nenhum desses modelos.

## 1. Seleção dos vídeos

- **Amostragem estratificada** por **10 categorias oficiais do YouTube**,
  com **5 vídeos por categoria**: Notícias e Política, Esportes, Games,
  Música, Entretenimento, Ciência e Tecnologia, Educação, Comédia, Pessoas e
  Blogs, e Estilo e Tutoriais. Usar as categorias do YouTube como estratos
  segue corpora multicategoria de comentários (Dutta, 2024).
- **Busca pela YouTube Data API v3**, como em Thelwall (2018): vídeos em
  português do Brasil, publicados entre janeiro de 2025 e setembro de 2026,
  ordenados por número de visualizações.
- **Critérios de inclusão:** de 300 a 10.000 comentários e pelo menos 10 mil
  visualizações. O piso garante comentários suficientes para o sorteio; o
  teto evita vídeos virais atípicos.
- **Critérios de exclusão:**
  - Shorts, lives e conteúdo infantil;
  - vídeos de outros idiomas ou de canais estrangeiros;
  - clipes musicais fora da categoria Música, identificados pela
    classificação de tópicos do próprio YouTube. A categoria é declarada por
    quem publica o vídeo, e muitos clipes apareciam em outros temas;
  - mais de um vídeo do mesmo canal no dataset.
- A seleção foi **automatizada e reproduzível**: os critérios foram
  aplicados na ordem dos resultados da busca até completar cada categoria.
  Foram avaliados **928 vídeos candidatos** para chegar aos **50
  selecionados**.

## 2. Coleta e amostragem dos comentários

- **Coleta:** foram coletados todos os **115.167 comentários de primeiro
  nível** dos 50 vídeos. As respostas ficaram de fora. A coleta completa foi
  necessária porque a API não oferece ordem aleatória, e o sorteio precisava
  do conjunto inteiro de cada vídeo. O nome dos autores não foi armazenado.
- **Filtragem:** foram removidos os comentários com links, com menos de 3
  palavras (por exemplo, só emojis ou "kkkk"), com mais de 1.000
  caracteres, duplicados e em outro idioma. Restaram **88.532 comentários**.
  - O filtro de idioma foi **deliberadamente tolerante**, para não descartar
    português informal ("que casa lindaaa", "SHOW DE BOLA"), que o detector
    automático confundia com espanhol ou inglês.
- **Sorteio:** **20 comentários por vídeo**, com semente fixa, totalizando
  **1.000 comentários** (100 por categoria). O sorteio não considerou
  curtidas nem data, para evitar o viés de selecionar só os comentários
  mais populares.
- **Revisão:** uma revisão manual identificou **25 comentários
  estrangeiros**, que foram substituídos pelo próximo comentário sorteado do
  mesmo vídeo.
  - Comentários irônicos ou ambíguos **foram mantidos**, porque são casos
    difíceis legítimos para a avaliação dos modelos.

## 3. Anotação

A anotação combinou **classificação por LLMs com validação humana**,
abordagem apoiada por Gilardi, Alizadeh e Kubli (2023), que mostraram que
LLMs superam anotadores de crowdsourcing, e por Pangakis, Wolken e Fasching
(2023), que defendem que toda anotação automática seja validada contra
rótulos humanos.

1. **Guia de anotação.** Foi elaborado com base em Mohammad (2016). Ele
   define as três classes e regras para os casos difíceis: ironia,
   comentários mistos, perguntas, emojis e gírias. O mesmo guia orientou os
   anotadores humanos e serviu de prompt para as LLMs.
2. **Amostra de validação (n = 200).** Foram sorteados 200 comentários do
   próprio dataset, 20 por categoria. **Dois anotadores humanos** os
   classificaram de forma independente, **antes** da execução das LLMs.
   - A concordância foi **κ = 0,50**, moderada segundo Landis e Koch
     (1977), com as discordâncias concentradas na fronteira entre neutro e
     polarizado.
   - As **65 divergências** foram resolvidas por **adjudicação** entre os
     dois anotadores, com base no guia. O resultado forma o gabarito
     humano.
3. **Anotação por LLMs.** Os 1.000 comentários foram classificados por
   **três LLMs de empresas distintas: GPT, DeepSeek e Gemini**, com o mesmo
   prompt e cada lote em uma conversa nova.
   - Usar modelos de empresas diferentes reduz a chance de erros em comum.
   - O **Claude não participou da anotação** por ser um dos modelos
     avaliados no trabalho. Isso evita o viés de autopreferência
     (Panickssery, Bowman e Feng, 2024).
4. **Rótulo final:**
   - **Amostra de validação:** vale o gabarito humano (200 comentários).
   - **Três LLMs unânimes:** vale o rótulo delas (670 comentários).
   - **LLMs divergentes:** classificação manual, feita sem ver os rótulos
     das LLMs (130 comentários).

## 4. Resultados da anotação

| Indicador | Valor |
|---|---|
| Concordância entre os anotadores humanos (κ de Cohen) | 0,50 |
| Unanimidade entre as três LLMs | 83,6% |
| Concordância de cada LLM com o gabarito humano (κ) | 0,74 a 0,75 |
| Acerto dos casos unânimes na amostra de validação | 90,4% |

- As LLMs alcançaram concordância **substancial** com o gabarito humano
  (Landis e Koch, 1977).
- O acerto de 90,4% nos casos unânimes justifica aceitá-los sem revisão
  humana.
- **Distribuição final:** **45,0% positivos, 30,6% neutros e 24,4%
  negativos.** Como as classes estão desbalanceadas, os modelos devem ser
  avaliados com **F1-macro** como métrica principal.
- **A distribuição varia bastante entre os temas:**
  - Música: 72% positivos;
  - Notícias e Política: 45% negativos;
  - Games: 53% neutros.

  Isso confirma a importância da estratificação por tema.

## 5. Limitações

- A **categoria do vídeo é declarada por quem publica**, então há temas que
  se sobrepõem. Por exemplo, culinária aparece em mais de uma categoria.
- A **busca do YouTube não é uma amostra aleatória**: a ordenação é definida
  pela plataforma e favorece conteúdo popular.
- Os **casos unânimes não foram revisados um a um**. Pela taxa de acerto
  medida, cerca de 10% deles podem estar incorretos.
- As LLMs foram usadas **pelas interfaces de chat**, e não por API, sem
  controle de parâmetros como a temperatura.

## Referências

- DUTTA, H. S. YT-30M: a multi-lingual multi-category dataset of YouTube
  comments. arXiv:2412.03465, 2024.
- GILARDI, F.; ALIZADEH, M.; KUBLI, M. ChatGPT outperforms crowd workers for
  text-annotation tasks. *Proceedings of the National Academy of Sciences*,
  v. 120, n. 30, e2305016120, 2023.
- LANDIS, J. R.; KOCH, G. G. The measurement of observer agreement for
  categorical data. *Biometrics*, 1977.
- MOHAMMAD, S. A practical guide to sentiment annotation: challenges and
  solutions. In: *Workshop on Computational Approaches to Subjectivity,
  Sentiment and Social Media Analysis (WASSA)*, p. 174–179, 2016.
- PANGAKIS, N.; WOLKEN, S.; FASCHING, N. Automated annotation with
  generative AI requires validation. arXiv:2306.00176, 2023.
- PANICKSSERY, A.; BOWMAN, S. R.; FENG, S. LLM evaluators recognize and
  favor their own generations. In: *Advances in Neural Information
  Processing Systems (NeurIPS)*, 2024.
- THELWALL, M. Social media analytics for YouTube comments: potential and
  limitations. *International Journal of Social Research Methodology*,
  v. 21, n. 3, p. 303–316, 2018.

## A confirmar antes de fechar o texto

1. **O nome e a versão de cada LLM** exibidos no chat, e as datas em que a
   anotação foi feita.
2. **Se o segundo anotador recebeu o guia antes de classificar**, e se ele
   será nomeado no texto ou citado só como "segundo anotador".
3. **Volume e páginas de Landis e Koch (1977)**, a única referência citada
   de memória; as demais foram verificadas.
