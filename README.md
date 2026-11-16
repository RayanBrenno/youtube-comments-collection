## 📌 Visão Geral

Este projeto consiste em uma aplicação web fullstack desenvolvida para coleta, armazenamento e análise de comentários de vídeos do YouTube.

A plataforma permite que usuários consultem vídeos a partir de uma URL, recuperem informações públicas do vídeo e realizem a extração automática dos comentários utilizando a API oficial do YouTube Data API v3.

Além da coleta dos dados, o sistema mantém um histórico de consultas por usuário, possibilitando atualização dos dados de vídeos já armazenados, e classifica o sentimento de cada comentário com modelos de inteligência artificial (encoders fine-tuned no TCC ou modelos Claude).

O projeto foi idealizado tanto como aplicação prática quanto como base para desenvolvimento acadêmico relacionado à análise de sentimentos em comentários de redes sociais.

---

## 🎯 Objetivo

O principal objetivo do projeto é construir uma plataforma capaz de coletar e organizar comentários de vídeos do YouTube de forma automatizada.

Além disso, o projeto busca:

- Aplicar conceitos de desenvolvimento fullstack
- Integrar APIs externas (YouTube Data API v3)
- Trabalhar com persistência de dados em MongoDB
- Implementar autenticação segura com JWT
- Estruturar uma aplicação escalável baseada em serviços
- Classificar o sentimento dos comentários com modelos de IA/NLP
- Comparar encoders fine-tuned com LLMs na mesma tarefa

---

## 🧠 Funcionalidades

- 🔐 Autenticação de usuários (Login / Register)
- 👤 Gerenciamento de sessão com JWT
- 🎥 Consulta de vídeos via URL do YouTube
- 📊 Coleta de informações públicas do vídeo
- 💬 Extração automática de comentários
- 🗂️ Armazenamento de vídeos e comentários no MongoDB
- 🔄 Atualização de vídeos previamente consultados
- 🕒 Histórico de consultas por usuário
- 🖼️ Exibição de thumbnails dos vídeos
- 📈 Comparação de métricas após atualização
- ⚡ Interface dinâmica e responsiva
- 🤖 Classificação de sentimento com IA, com escolha do modelo (4 encoders locais ou 3 modelos Claude)

---

## 🛠️ Tecnologias utilizadas

### Frontend

- React (Vite)
- TypeScript
- TailwindCSS
- React Router
- Axios

### Backend

- FastAPI
- Python
- JWT (Autenticação)
- MongoDB
- Pydantic

### Infraestrutura e Serviços

- YouTube Data API v3
- MongoDB Atlas
- Render (backend)
- Vercel (frontend)

<details>
<summary>📦 Infraestrutura anterior (AWS Free Tier)</summary>

O projeto rodou inicialmente 100% na AWS:

- AWS EC2 (backend)
- AWS Amplify (frontend)
- ngrok (túnel HTTPS gratuito, usado pra dar HTTPS à instância EC2)

Após o fim do período gratuito da AWS, o projeto foi migrado para Render (backend) e Vercel (frontend), que possuem planos gratuitos permanentes.

</details>

---

## 📁 Estrutura do repositório

```
backend/        API FastAPI (coleta, classificação, autenticação, MongoDB)
frontend/       interface React + Vite
scripts/        experimentos do TCC
├── encoders/   fine-tuning dos encoders; os modelos treinados ficam em encoders/modelos/
└── claude/     classificação zero-shot com Claude 5.5 via API
```

O backend e o frontend não dependem do código de `scripts/`: o backend só lê os modelos prontos
em `scripts/encoders/modelos/etapa2/`.

---

## ⚙️ Como rodar o projeto

O sistema roda localmente. Os modelos de IA fine-tuned ocupam vários GB (o BERTimbau Large
sozinho tem 1,3 GB) e não cabem no plano gratuito de hospedagem.

### Pré-requisitos

- Python 3.12 (as versões fixadas no backend não têm pacotes prontos para 3.13)
- Node.js 18+
- GPU NVIDIA com CUDA é recomendada para os modelos locais (sem GPU eles rodam na CPU, mais devagar)
- Os encoders fine-tuned em `scripts/encoders/modelos/etapa2/`. Para gerá-los, siga o
  [scripts/encoders/README.md](scripts/encoders/README.md) (Etapas 1 e 2)
- Opcional: uma chave da API da Anthropic, para os modelos Claude

---

### 🔙 Backend (FastAPI)

```bash
cd backend

python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# torch com CUDA 12.8 vem do índice do PyTorch, antes do requirements
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cp .env.example .env       # e preencha as variáveis

uvicorn app.main:app --reload
```

Servidor disponível em: http://127.0.0.1:8000 (documentação interativa em `/docs`)

Variáveis do `backend/.env` (modelo completo em [backend/.env.example](backend/.env.example)):

| Variável | Para quê |
|---|---|
| `MONGO_URL`, `MONGO_DB` | conexão com o MongoDB |
| `YOUTUBE_API_KEY` | chave da YouTube Data API v3 |
| `SECRET_KEY` | assinatura dos tokens JWT |
| `API_FRONT_URL` | origem do frontend liberada no CORS (ex.: `http://localhost:5173`) |
| `CLASSIFICADOR_PADRAO` | modelo usado quando nenhum é escolhido (padrão: `bertimbau-large`) |
| `MODELOS_DIR` | pasta dos encoders fine-tuned (padrão: `../scripts/encoders/modelos/etapa2`) |
| `ANTHROPIC_API_KEY` | opcional; libera os modelos Claude |

### 🔜 Frontend (React)

```bash
cd frontend

npm install
npm run dev
```

App disponível em: http://localhost:5173

Crie um arquivo `.env` dentro de `frontend/` com a seguinte variável:

```
VITE_API_URL=http://127.0.0.1:8000
```

---

## 🤖 Classificação de sentimento

Cada comentário coletado é classificado como **negativo**, **neutro** ou **positivo**. Na
tela de consulta, o usuário escolhe o modelo:

| Modelo | Tipo | Acurácia no TCC | F1 macro no TCC |
|---|---|---|---|
| BERTimbau Large (padrão) | local, fine-tuned | 0,788 | 0,7799 |
| XLM-RoBERTa Base | local, fine-tuned | 0,784 | 0,7726 |
| BERTimbau Base | local, fine-tuned | 0,780 | 0,7658 |
| DistilBERT multilíngue | local, fine-tuned | 0,624 | 0,5902 |
| Claude Opus 5.5 | API, zero-shot | 0,860 | 0,8463 |
| Claude Sonnet 5.5 | API, zero-shot | 0,838 | 0,8289 |
| Claude Haiku 5.5 | API, zero-shot | 0,833 | 0,8244 |

Métricas no dataset 2 (500 comentários do YouTube rotulados manualmente). Detalhes em
[scripts/README.md](scripts/README.md), que reúne os experimentos do TCC:
[scripts/encoders/](scripts/encoders/README.md) (fine-tuning) e
[scripts/claude/](scripts/claude/README.md) (Claude zero-shot).

- **Modelos locais**: os encoders fine-tuned do TCC, carregados de `MODELOS_DIR`. A inferência
  é a mesma do `scripts/encoders/classificar_encoder.py`, e o score mostrado é a confiança do modelo
  (maior probabilidade do softmax).
- **Claude**: mesma configuração do experimento do TCC (prompt, raciocínio, esforço e leitura da
  resposta), com uma chamada por comentário. Cobrado por uso na API da Anthropic. O Claude devolve
  só o rótulo, então esses comentários não têm score.
- Modelos sem a pasta ou sem a chave aparecem como indisponíveis, com o motivo.
- Cada vídeo guarda o modelo da primeira consulta. As atualizações classificam só os comentários
  novos, com esse mesmo modelo.

## 🔄 Fluxo de funcionamento

1. O usuário acessa a aplicação através do frontend.

2. Na tela inicial, o usuário pode:
   - Fazer login
   - Criar uma conta

3. Após autenticação:
   - O sistema gera um token JWT
   - A sessão do usuário é armazenada

4. O usuário informa a URL de um vídeo do YouTube.

5. O backend:
   - Extrai o ID do vídeo
   - Consulta a YouTube Data API v3
   - Recupera informações públicas do vídeo

6. O sistema:
   - Coleta os comentários do vídeo
   - Classifica o sentimento de cada comentário com o modelo escolhido
   - Armazena os dados no MongoDB

7. Os dados exibidos incluem:
   - Título
   - Canal
   - Data de publicação
   - Views
   - Likes
   - Quantidade de comentários
   - Thumbnail
   - Comentários coletados

8. Caso o vídeo já exista:
   - O sistema identifica o histórico do usuário
   - Permite atualização dos dados

9. Durante a atualização:
   - O sistema compara métricas anteriores e atuais
   - Atualiza os comentários armazenados

10. O usuário pode:
   - Consultar múltiplos vídeos
   - Visualizar histórico
   - Atualizar consultas anteriores