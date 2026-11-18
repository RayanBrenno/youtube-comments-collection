# Análise de sentimento em comentários do YouTube em português: encoders ajustados e LLMs em modo zero-shot

Repositório do Trabalho de Conclusão de Curso (TCC) que compara duas abordagens de
classificação de sentimento em comentários do YouTube escritos em português do Brasil,
e da aplicação web desenvolvida para aplicar os modelos avaliados a vídeos reais.

## Resumo

Comentários do YouTube são uma fonte abundante de opinião pública, mas o volume torna
inviável a leitura manual. Este trabalho classifica cada comentário em **negativo**,
**neutro** ou **positivo** e compara duas abordagens na mesma tarefa e no mesmo conjunto
de teste:

1. **Fine-tuning de modelos encoder** (BERTimbau Large e Base, XLM-RoBERTa Base e
   DistilBERT multilíngue), em duas etapas: primeiro com tweets em português do corpus
   da Cardiff NLP e depois com comentários do YouTube.
2. **Modelos de linguagem de grande porte (LLMs) em modo zero-shot**, sem treino:
   Claude Haiku 5.5, Sonnet 5.5 e Opus 5.5, acessados pela API da Anthropic.

Para a avaliação foi construído um conjunto de **1.000 comentários de 50 vídeos e 10
temas**, rotulado por dois anotadores humanos e por três LLMs de outras empresas, com
revisão manual dos casos em que elas divergiram. O conjunto foi dividido em duas
metades estratificadas: o **dataset 1** (500 comentários), usado no fine-tuning, e o
**dataset 2** (500 comentários), usado como teste de todos os modelos.

No dataset 2, o melhor encoder (BERTimbau Large) obteve **F1 macro de 0,780** e o melhor
LLM (Claude Opus 5.5) obteve **0,846**. Os três modelos Claude superaram todos os
encoders, mesmo sem nenhum exemplo de treino.

## Objetivos

### Objetivo geral

Avaliar se LLMs usados em modo zero-shot classificam o sentimento de comentários do
YouTube em português com desempenho comparável ou superior ao de encoders ajustados
para a tarefa, considerando também custo e tempo de resposta.

### Objetivos específicos

- Construir um conjunto rotulado de comentários do YouTube em português, estratificado
  por tema e com procedimento de anotação documentado e reproduzível.
- Ajustar quatro encoders pré-treinados em duas etapas (domínio próximo e domínio-alvo)
  e medir o ganho de cada etapa.
- Avaliar três modelos Claude de portes diferentes na mesma tarefa, sem treino.
- Comparar as abordagens por acurácia e F1 macro, além de custo, latência e tamanho
  dos modelos.
- Integrar os modelos avaliados a uma aplicação web de coleta e análise de comentários.

## Metodologia

O trabalho segue quatro etapas. O código, os dados e o passo a passo de cada uma ficam
em [`scripts/`](scripts/README.md).

| Etapa | Descrição | Pasta |
|---|---|---|
| 1. Construção do conjunto de dados | Seleção de 50 vídeos (5 por tema), coleta de 115.167 comentários, filtragem, sorteio de 1.000 comentários e anotação | [`scripts/evaluation-set-creation/`](scripts/evaluation-set-creation/README.md) |
| 2. Fine-tuning dos encoders | Etapa 1 com 3.033 tweets da Cardiff NLP; Etapa 2 com o dataset 1 | [`scripts/encoders/`](scripts/encoders/README.md) |
| 3. Classificação zero-shot | Uma chamada à API por comentário, com o mesmo prompt para os três modelos | [`scripts/claude/`](scripts/claude/README.md) |
| 4. Avaliação | Todos os modelos no dataset 2, com F1 macro como métrica principal | resultados em cada pasta |

A justificativa das escolhas de coleta e anotação, com as referências, está em
[`METODOLOGIA_DATASET.md`](scripts/evaluation-set-creation/METODOLOGIA_DATASET.md).

### Conjunto de dados

| Indicador | Valor |
|---|---|
| Vídeos | 50 (5 por categoria do YouTube, um por canal) |
| Comentários coletados | 115.167 (primeiro nível, sem respostas) |
| Comentários após os filtros | 88.532 |
| Comentários rotulados | 1.000 (20 por vídeo, 100 por tema) |
| Concordância entre anotadores humanos (κ de Cohen) | 0,50 |
| Concordância de cada LLM anotadora com o gabarito humano (κ) | 0,74 a 0,75 |
| Distribuição das classes | 45,0% positivos, 30,6% neutros, 24,4% negativos |

As classes são desbalanceadas, por isso a métrica principal é o F1 macro, que dá o
mesmo peso às três classes.

## Resultados

Desempenho no dataset 2 (500 comentários: 122 negativos, 153 neutros e 225 positivos).

| Modelo | Abordagem | Acurácia | F1 macro |
|---|---|---|---|
| Claude Opus 5.5 | zero-shot | 0,860 | 0,8463 |
| Claude Sonnet 5.5 | zero-shot | 0,838 | 0,8289 |
| Claude Haiku 5.5 | zero-shot | 0,833 | 0,8244 |
| BERTimbau Large | fine-tuning (Etapas 1 e 2) | 0,788 | 0,7799 |
| XLM-RoBERTa Base | fine-tuning (Etapas 1 e 2) | 0,784 | 0,7726 |
| BERTimbau Base | fine-tuning (Etapas 1 e 2) | 0,780 | 0,7658 |
| DistilBERT multilíngue | fine-tuning (Etapas 1 e 2) | 0,624 | 0,5902 |

Principais observações:

- A Etapa 2 do fine-tuning (comentários do YouTube) melhorou todos os encoders em
  relação à Etapa 1 (só tweets). O BERTimbau Large, por exemplo, passou de 0,747 para
  0,780 de F1 macro.
- O Claude Haiku 5.5, o menor e mais barato dos três, superou o melhor encoder em 4,4
  pontos de F1 macro, ao custo de US$ 0,0068 para os 500 comentários.
- Os encoders rodam localmente e não têm custo por uso, mas exigem dados rotulados,
  treino e GPU.

Métricas por etapa, custos, latências e a variação esperada entre execuções estão nos
READMEs de [`scripts/encoders/`](scripts/encoders/README.md) e
[`scripts/claude/`](scripts/claude/README.md).

## Sistema desenvolvido

Além dos experimentos, o trabalho inclui uma aplicação web que coleta os comentários de
um vídeo do YouTube a partir da URL e classifica cada um com o modelo escolhido pelo
usuário. A aplicação usa os mesmos modelos e a mesma configuração dos experimentos.

### Arquitetura

```
Navegador (React)  --HTTP/JSON-->  API (FastAPI)  --->  YouTube Data API v3
                                        |
                                        +--->  Classificador
                                        |        - encoders ajustados (locais, PyTorch)
                                        |        - Claude (API da Anthropic)
                                        |
                                        +--->  MongoDB (usuários, vídeos, comentários)
```

O backend não depende do código de `scripts/`: ele só lê os encoders já treinados em
`scripts/encoders/modelos/etapa2/`. Para os modelos Claude, reproduz a configuração do
experimento (prompt, raciocínio, esforço e leitura da resposta) em
[`classificador_claude.py`](backend/app/services/classificador_claude.py).

### Funcionalidades

- Cadastro e login de usuários, com sessão por token JWT.
- Consulta de vídeo pela URL: título, canal, data de publicação, visualizações,
  curtidas, número de comentários e thumbnail.
- Coleta automática dos comentários pela YouTube Data API v3.
- Classificação de sentimento de cada comentário, com escolha entre os quatro encoders
  e os três modelos Claude. Modelos sem os arquivos ou sem a chave da API aparecem como
  indisponíveis, com o motivo.
- Armazenamento de vídeos e comentários no MongoDB e histórico de consultas por usuário.
- Atualização de vídeos já consultados, com comparação das métricas antigas e novas.
  Só os comentários novos são classificados, com o mesmo modelo da primeira consulta.
- Painel com o resumo das consultas do usuário e lista de comentários com busca,
  filtro por sentimento e ordenação.

### Classificadores disponíveis

| Modelo | Execução | Score exibido |
|---|---|---|
| BERTimbau Large (padrão) | local | confiança do modelo (maior probabilidade do softmax) |
| BERTimbau Base | local | idem |
| XLM-RoBERTa Base | local | idem |
| DistilBERT multilíngue | local | idem |
| Claude Opus 5.5, Sonnet 5.5 e Haiku 5.5 | API da Anthropic, cobrada por uso | sem score (o modelo devolve só o rótulo) |

A inferência local é a mesma do
[`classificar_encoder.py`](scripts/encoders/classificar_encoder.py) usado nos
experimentos.

### Rotas da API

| Método | Rota | Função |
|---|---|---|
| POST | `/auth/register` | cadastro de usuário |
| POST | `/auth/login` | login, devolve o token JWT |
| GET | `/auth/me` | dados do usuário autenticado |
| GET | `/youtube/classificadores` | modelos disponíveis e motivo dos indisponíveis |
| POST | `/youtube/check-video` | verifica se o vídeo já foi consultado pelo usuário |
| POST | `/youtube/full-data` | coleta, classifica e salva o vídeo e os comentários |
| POST | `/youtube/videos/{video_id}/update` | atualiza um vídeo já salvo |
| GET | `/youtube/history/{user_id}` | histórico de consultas |
| GET | `/youtube/dashboard/{user_id}` | resumo para o painel inicial |

A documentação interativa fica em `/docs` com o backend em execução.

### Tecnologias

| Camada | Tecnologias |
|---|---|
| Frontend | React (Vite), TypeScript, TailwindCSS, React Router, Axios |
| Backend | Python, FastAPI, Pydantic, JWT, PyTorch, Transformers, SDK `anthropic` |
| Dados | MongoDB (MongoDB Atlas) |
| Serviços externos | YouTube Data API v3, API da Anthropic |
| Experimentos | PyTorch, Transformers, scikit-learn, pandas |

A aplicação foi hospedada primeiro na AWS (EC2 para o backend, Amplify para o frontend
e ngrok para o HTTPS) e depois no Render e no Vercel. Com a inclusão dos encoders
ajustados, que ocupam vários GB (o BERTimbau Large sozinho tem 1,3 GB), ela passou a ser
executada localmente, já que os modelos não cabem nos planos gratuitos de hospedagem.

## Estrutura do repositório

```
backend/                          API FastAPI
  app/routes/                     rotas de autenticação e do YouTube
  app/services/                   coleta no YouTube e classificadores (encoder e Claude)
  app/youtube_repository.py       acesso ao MongoDB
frontend/                         interface React + Vite
  src/pages/                      login, cadastro, início, consulta e atualização
  src/components/                 componentes de interface
scripts/                          experimentos do TCC
  evaluation-set-creation/        construção e anotação do conjunto de dados
  encoders/                       fine-tuning e avaliação dos encoders
  claude/                         classificação zero-shot com Claude 5.5
```

## Como executar a aplicação

Para replicar os experimentos, siga o [`scripts/README.md`](scripts/README.md). Esta
seção trata só da aplicação web.

### Pré-requisitos

- Python 3.12 (as versões fixadas no backend não têm pacotes prontos para 3.13).
- Node.js 18 ou superior.
- Um banco MongoDB (local ou Atlas) e uma chave da YouTube Data API v3.
- Os encoders ajustados em `scripts/encoders/modelos/etapa2/`, gerados pelas Etapas 1 e
  2 do [`scripts/encoders/README.md`](scripts/encoders/README.md). Sem eles, só os
  modelos Claude ficam disponíveis.
- GPU NVIDIA com CUDA, recomendada para os encoders. Sem GPU eles rodam na CPU, mais
  devagar.
- Opcional: uma chave da API da Anthropic, para os modelos Claude.

### Backend

```bash
cd backend

python3.12 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# torch com CUDA 12.8 vem do índice do PyTorch, antes do requirements
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cp .env.example .env           # e preencha as variáveis

uvicorn app.main:app --reload
```

O servidor fica em http://127.0.0.1:8000. Variáveis do `backend/.env` (modelo completo em
[`backend/.env.example`](backend/.env.example)):

| Variável | Uso |
|---|---|
| `MONGO_URL`, `MONGO_DB` | conexão com o MongoDB |
| `YOUTUBE_API_KEY` | chave da YouTube Data API v3 |
| `SECRET_KEY` | assinatura dos tokens JWT |
| `API_FRONT_URL` | origem do frontend liberada no CORS (ex.: `http://localhost:5173`) |
| `CLASSIFICADOR_PADRAO` | modelo usado quando nenhum é escolhido (padrão: `bertimbau-large`) |
| `MODELOS_DIR` | pasta dos encoders ajustados (padrão: `../scripts/encoders/modelos/etapa2`) |
| `ANTHROPIC_API_KEY` | opcional; habilita os modelos Claude |

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Antes de rodar, crie `frontend/.env` com o endereço do backend:

```
VITE_API_URL=http://127.0.0.1:8000
```

A interface fica em http://localhost:5173.

## Limitações

- A categoria de cada vídeo é declarada por quem publica, e a busca do YouTube favorece
  conteúdo popular. O conjunto não é uma amostra aleatória da plataforma.
- A concordância entre os anotadores humanos foi moderada (κ = 0,50), o que mostra que a
  fronteira entre neutro e polarizado é ambígua mesmo para pessoas.
- Os 670 comentários em que as três LLMs anotadoras foram unânimes não foram revisados um
  a um. Pela taxa de acerto medida na amostra humana, cerca de 10% deles podem estar
  incorretos.
- O conjunto de teste tem 500 comentários, e diferenças pequenas entre modelos (por
  exemplo, entre BERTimbau Large e XLM-RoBERTa Base) estão dentro da variação esperada
  entre execuções.
- Os modelos Claude não são determinísticos e dependem de um serviço pago e externo.
