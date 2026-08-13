# Deploying MentalCare AI with Qwen

## What you need

The chatbot has two containers:

- `mentalcare-ai` runs Flask, the safety rules, and the emotion model.
- `ollama` downloads and runs `qwen3:4b`, then creates the `my-qwen-chatbot` model from `deploy/ollama/Modelfile`.

This cannot run on Streamlit Community Cloud because it cannot start a second Ollama container or keep the multi-gigabyte Qwen model on persistent disk. Deploy it to a Docker-capable Linux VM with a persistent disk. An NVIDIA GPU VM is strongly recommended; CPU-only Qwen responses can be slow.

## Before deploying

1. Install Docker Engine and the Docker Compose plugin on the Linux VM.
2. For GPU inference, install NVIDIA drivers and NVIDIA Container Toolkit, then confirm that Docker can access the GPU. Ollama's official Docker instructions cover the required host setup.
3. Clone this repository on the VM and create a `.env` file from `.env.example`.
4. Set a unique secret in `.env`:

   ```env
   FLASK_SECRET_KEY=replace-with-a-long-random-secret
   ```

   Do not commit `.env`.

## Start the app

From the repository root, run one of the following:

**NVIDIA GPU host (recommended):**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

**CPU-only host (works, but Qwen may respond slowly):**

```bash
docker compose up --build -d
docker compose logs -f ollama
```

The first start downloads Qwen and may take several minutes. Wait until the Ollama service is healthy, then verify the app:

```bash
curl http://localhost:5000/health
```

If you place the VM behind a domain, use a reverse proxy such as Caddy or Nginx to provide HTTPS and forward traffic to port `5000`. Do not expose port `11434`: Ollama is intentionally available only to the app's internal Docker network.

## Persistent data

Docker volumes retain both the SQLite database and Qwen model between restarts:

- `mentalcare_data`: chat/feedback database.
- `ollama_models`: downloaded Qwen files and the custom model.

Back up `mentalcare_data` if you choose to retain conversation data. The current database is appropriate for a portfolio demo; production handling of health-related messages needs a privacy, security, retention, and safety review.

## Environment variables used in production

The Compose file sets these automatically for the web app:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=my-qwen-chatbot
OLLAMA_BASE_URL=http://ollama:11434
DATABASE_PATH=/app/data/mentalcare.db
```

Do not set `OLLAMA_BASE_URL` to `localhost` in cloud deployment. Inside the web container, `localhost` means the web container itself, not the Ollama container.

## Common failures

- **Ollama never becomes healthy:** check `docker compose logs ollama`; the initial Qwen download may still be in progress, or the server may lack disk space/RAM.
- **Replies use a fallback template:** the web app could not reach Ollama. Confirm `OLLAMA_BASE_URL=http://ollama:11434` and that `docker compose ps` shows Ollama as healthy.
- **Out of memory or very slow replies:** use a GPU-enabled VM or a smaller Qwen model, and retain the model volume so it is not downloaded on every deployment.
- **A cloud platform reports a build timeout:** use a persistent VM or a platform that supports multi-container Docker services and persistent volumes. Do not use Streamlit Community Cloud for this Qwen deployment.
