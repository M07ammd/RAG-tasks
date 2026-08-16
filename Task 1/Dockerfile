# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        jupyter \
        nbconvert \
        ipykernel \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Expose Jupyter port ───────────────────────────────────────────────────────
EXPOSE 8888

# ── Default: launch Jupyter Notebook (no token / password for local dev) ──────
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--allow-root", \
     "--notebook-dir=/app"]
