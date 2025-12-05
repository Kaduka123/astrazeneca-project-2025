# docker/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran libopenblas-dev pkg-config \
    git zlib1g-dev libffi-dev libzmq3-dev libblas-dev liblapack-dev \
    curl gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Use Fixed IDs ---
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME || echo "Group ${USERNAME} ($USER_GID) already exists" \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME || echo "User ${USERNAME} ($USER_UID) already exists"

RUN mkdir -p /app && chown $USERNAME:$USERNAME /app
WORKDIR /app

RUN mkdir -p /app/logs /app/results /app/data/preprocessed /app/data/processed /app/data/raw && \
    chown -R $USERNAME:$USERNAME /app/logs /app/results /app/data

COPY --chown=$USERNAME:$USERNAME pyproject.toml poetry.lock ./

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev --no-interaction

COPY --chown=$USERNAME:$USERNAME . /app

COPY --chown=root:root ./docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "scripts.run_analysis"]