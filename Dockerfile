FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FILE_AGENT_WEB_HOST=0.0.0.0 \
    FILE_AGENT_WEB_SESSION_ROOT=/tmp/file-agent/sessions \
    FILE_AGENT_WEB_RUNS_ROOT=/tmp/file-agent/runs

WORKDIR /opt/file-agent

RUN python -m pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md NOTES.md ./
COPY app/ ./app/
RUN python -m pip install --no-build-isolation .

COPY web/ ./web/
COPY workspace/ ./workspace/
COPY agent.py web_server.py ./

RUN groupadd --system fileagent \
    && useradd --system --gid fileagent --no-create-home \
        --home-dir /opt/file-agent --shell /usr/sbin/nologin fileagent \
    && install -d -o fileagent -g fileagent /tmp/file-agent/sessions /tmp/file-agent/runs

USER fileagent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; port=os.environ.get('FILE_AGENT_WEB_PORT') or os.environ.get('PORT','8000'); response=urllib.request.urlopen('http://127.0.0.1:'+port+'/healthz',timeout=3); response.close()"]

CMD ["python", "web_server.py"]
