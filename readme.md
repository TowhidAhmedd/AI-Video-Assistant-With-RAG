# 🎥 AI Video Assistant — Meeting Intelligence with RAG

> Ingest any YouTube video or local recording, get an AI-generated summary, action items, and key decisions — then chat over the full transcript with a RAG-powered Q&A engine. Built with Whisper, Mistral, LangChain, and Chroma. Deployed on Render.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Tech Stack](#tech-stack)
- [Environment Variables](#environment-variables)
- [Local Development](#local-development)
- [Deploying to Render](#deploying-to-render)
  - [Prerequisites](#prerequisites)
  - [1. Deploy as a Web Service](#1-deploy-as-a-web-service)
  - [2. render.yaml — Infrastructure as Code](#2-renderyaml--infrastructure-as-code)
  - [Deployment Notes & Gotchas](#deployment-notes--gotchas)
- [How to Use](#how-to-use)
- [Troubleshooting](#troubleshooting)
- [Licence](#licence)

---

## What It Does

| Stage | Capability |
|---|---|
| **Input** | YouTube URL or local file path; language (`english` / `hinglish`) |
| **Audio** | Download or load media, normalise, and chunk for transcription |
| **Transcription** | Local OpenAI Whisper over audio chunks |
| **Understanding** | Mistral (via LangChain) for title, summary, action items, key decisions, and open questions |
| **RAG** | Chroma + SentenceTransformers embeddings for Q&A grounded in the transcript |
| **UI** | Streamlit — wide layout, pipeline status sidebar, output cards, conversational chat |

---

## Architecture

```
YouTube URL / Local File
        │
        ▼
┌───────────────────┐
│  utils/           │
│  audio_processor  │  ← yt-dlp, pydub, ffmpeg (download + chunk)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  core/            │
│  transcriber      │  ← OpenAI Whisper (local, chunked)
└────────┬──────────┘
         │
         ├──────────────────────────┐
         ▼                          ▼
┌───────────────────┐    ┌──────────────────────┐
│  core/summarizer  │    │  core/rag_engine      │
│  core/extractor   │    │  Chroma + Embeddings  │
│  Mistral API      │    │  (vector_db/)         │
└───────────────────┘    └──────────────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
           Streamlit UI (app.py)
           Summary · Extraction · Chat
```

**Pipeline order**: audio → transcript → title → summary → structured extraction → RAG chain build → chat

---

## Repository Layout

| Path | Role |
|---|---|
| `app.py` | Streamlit page config, sidebar, pipeline orchestration, results layout, RAG chat |
| `core/transcriber` | Whisper-based chunked transcription |
| `core/summarizer` | LangChain + Mistral summary generation |
| `core/extractor` | Action items, key decisions, open questions extraction |
| `core/rag_engine` | Chroma chain construction and Q&A |
| `utils/audio_processor` | Media acquisition (yt-dlp), conversion (pydub/ffmpeg), chunking |
| `vector_db/` | On-disk Chroma persistence (SQLite + segment files) |
| `downloades/` | Working directory for downloaded/test audio files |
| `Requirements.txt` | Pinned dependencies: audio, Whisper, LangChain, Chroma, Streamlit |

> **Note**: Commit the `.py` sources in `core/` and `utils/` — do not rely on checked-in `__pycache__/*.pyc`. Add `__pycache__/`, `vector_db/`, and `.env` to `.gitignore`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit, streamlit-extras, watchdog |
| **Media** | yt-dlp, pydub, ffmpeg-python (requires FFmpeg on PATH) |
| **Speech-to-Text** | OpenAI Whisper, PyTorch, torchaudio |
| **Translation** | deep-translator (English / Hinglish workflows) |
| **LLM Orchestration** | LangChain, langchain-mistralai, Mistral API |
| **RAG** | ChromaDB, langchain-chroma, sentence-transformers, langchain-huggingface |
| **Exports** | reportlab, fpdf2 (PDF output) |
| **Hosting** | Render |

**Python:** 3.10+ recommended (3.11 matches current bytecode artefacts)

---

## Environment Variables

Create a `.env` file at the project root (already gitignored):

```env
MISTRAL_API_KEY=your_mistral_key_here

# Optional — required only for gated HuggingFace embedding models
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | ✅ Yes | Powers summarisation and extraction via Mistral + LangChain |
| `HUGGINGFACEHUB_API_TOKEN` | ⚡ Optional | Required only if pulling gated HuggingFace embedding models |

> **Security**: Never commit `.env`. Render secrets are set via the dashboard Environment panel.

---

## Local Development

```bash
# Windows (PowerShell)
cd path\to\AI-Video-Assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
```

Ensure **FFmpeg** is installed and available on `PATH` before running:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

Start the app:

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Deploying to Render

The AI Video Assistant deploys as a single **Web Service** on Render. Streamlit serves the full UI and runs the backend pipeline from one process — no separate service needed.

### Prerequisites

- A [Render account](https://render.com) (Starter instance recommended — free tier will struggle with Whisper + PyTorch)
- Your project pushed to a GitHub or GitLab repository
- `MISTRAL_API_KEY` ready
- **FFmpeg** handled via Docker (see below) — Render's Python runtime does not include it by default

---

### 1. Deploy as a Web Service

#### Option A — Python Runtime (simpler, no FFmpeg pre-installed)

1. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**
2. Connect your GitHub repository
3. Configure the service:

| Setting | Value |
|---|---|
| **Name** | `ai-video-assistant` |
| **Region** | Choose closest to your users |
| **Branch** | `main` |
| **Root Directory** | *(leave blank — repo root)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r Requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |
| **Instance Type** | Starter or higher |

4. Under **Environment Variables**, add:

```
MISTRAL_API_KEY          = your_mistral_key_here
HUGGINGFACEHUB_API_TOKEN = your_hf_token_here    # if needed
```

5. Click **Create Web Service**

#### Option B — Docker Runtime (recommended — includes FFmpeg)

Add a `Dockerfile` at the repo root:

```dockerfile
FROM python:3.11-slim

# Install FFmpeg and system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

On Render, set **Runtime** to `Docker` — Render will build from your `Dockerfile` and FFmpeg will be available at runtime. This is the most reliable deployment path for this stack.

Your app will be live at:
```
https://ai-video-assistant.onrender.com
```

---

### 2. `render.yaml` — Infrastructure as Code

Add a `render.yaml` at the repo root for one-click Blueprint deployments:

```yaml
# render.yaml  (place at repo root)

services:
  - type: web
    name: ai-video-assistant
    runtime: docker              # Use docker to get FFmpeg support
    envVars:
      - key: MISTRAL_API_KEY
        sync: false              # Render prompts for this secret on first deploy
      - key: HUGGINGFACEHUB_API_TOKEN
        sync: false
    healthCheckPath: /

    # Uncomment to persist vector_db/ and downloads across redeploys
    # disk:
    #   name: ai-video-data
    #   mountPath: /app/vector_db
    #   sizeGB: 5
```

To deploy via Blueprint:

1. Push `render.yaml` and `Dockerfile` to your repository
2. Go to **Render Dashboard → New → Blueprint**
3. Connect your repository — Render provisions everything automatically

---

### Deployment Notes & Gotchas

| Topic | Notes |
|---|---|
| **FFmpeg not pre-installed** | Render's Python runtime has no FFmpeg. The app will crash on audio processing without it. Use the **Docker** deployment option above — this is the single most common failure point for this stack on Render. |
| **PyTorch + Whisper build size** | The full dependency set exceeds 2–3GB. Free tier builds will time out. Use a **Starter** instance or the Docker image with CPU-optimised wheels (`--extra-index-url https://download.pytorch.org/whl/cpu`). |
| **Ephemeral Filesystem** | `vector_db/` (Chroma) and `downloades/` are wiped on every redeploy or restart. Uncomment the `disk` block in `render.yaml` to attach a **Render Disk** (from $0.25/GB/month) for persistence. |
| **Whisper model download** | Whisper pulls model weights from OpenAI on first run (~140MB for `base`, ~1.5GB for `large`). Pin to `base` or `small` for Render to keep cold starts under 2 minutes. |
| **Embedding model download** | SentenceTransformer models (~90MB) are downloaded from HuggingFace on first startup and cached in the build layer on subsequent deploys. |
| **Port binding** | The start command must pass `--server.port=$PORT`. Streamlit defaults to 8501 — hardcoding this causes the service to fail since Render assigns the port dynamically via `$PORT`. |
| **Requirements.txt casing** | Render's Python detector expects `requirements.txt` (lowercase). Either rename the file or set the build command to `pip install -r Requirements.txt` explicitly. |
| **Free Tier Spin-Down** | Free services sleep after 15 min of inactivity. First wake-up takes 30–60s plus model load time. Use [UptimeRobot](https://uptimerobot.com) to ping the root URL every 10 minutes. |
| **Secrets** | Never commit `.env`. Set all keys in Render's **Environment** panel. `sync: false` in `render.yaml` means Render prompts you securely on first deploy. |

---

## How to Use

1. Open the **sidebar**
2. Paste a **YouTube URL** or a **local file path** to a supported video/audio file
3. Choose **Language** (`english` or `hinglish`)
4. Click **Analyse** and wait for the pipeline — audio → transcript → title → summary → extraction → RAG
5. Read the **summary**, **transcript**, and structured fields
6. Use **Chat with your Meeting** to ask questions grounded in the transcript

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Import errors for `core` / `utils` | Ensure `.py` sources exist — don't rely solely on `.pyc` bytecode |
| FFmpeg / decode errors | Use Docker deployment on Render; locally, install FFmpeg and confirm it's on `PATH` |
| Mistral / LLM errors | Check `MISTRAL_API_KEY`, billing status, and rate limits |
| Out of memory | Switch to Whisper `base` or `small`; use a Render instance with more RAM |
| Empty RAG answers | Confirm Chroma ingest completed and `vector_db/` is writable |
| Render build timeout | Use Docker deployment or upgrade to a Starter/Standard instance |
| Streamlit not binding | Ensure `--server.port=$PORT` is in the start command — never hardcode 8501 |

---

## Licence

No `LICENSE` file is present in this repository. Add one before distributing or open-sourcing the project. MIT is recommended for a learning and demo project.

---

Built as a **generative-AI + RAG** learning and demo project: **Streamlit** frontend, **Whisper** transcription, **Mistral** reasoning, and **Chroma** retrieval-augmented chat over meeting content.
