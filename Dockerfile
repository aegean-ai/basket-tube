FROM python:3.11-slim

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY --chown=$USERNAME:$USERNAME pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project && \
    chown -R $USERNAME:$USERNAME /app

COPY --chown=$USERNAME:$USERNAME . .

USER $USERNAME

CMD ["uv", "run", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8080"]
