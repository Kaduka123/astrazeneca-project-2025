#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

APP_USER=appuser
APP_GROUP=appuser
APP_DIR=/app
LOGS_DIR=${APP_DIR}/logs
RESULTS_DIR=${APP_DIR}/results
DATA_PREPROCESSED_DIR=${APP_DIR}/data/preprocessed
DATA_PROCESSED_DIR=${APP_DIR}/data/processed

# Running as root initially
echo "Entrypoint: Running as UID=$(id -u), GID=$(id -g)"

# --- Create directories IF THEY DON'T EXIST and set ownership ---
echo "Entrypoint: Ensuring output directories exist and are owned by ${APP_USER}:${APP_GROUP}..."
mkdir -p "$LOGS_DIR"
chown "${APP_USER}:${APP_GROUP}" "$LOGS_DIR"
mkdir -p "$RESULTS_DIR"
chown "${APP_USER}:${APP_GROUP}" "$RESULTS_DIR"
mkdir -p "$DATA_PREPROCESSED_DIR"
chown "${APP_USER}:${APP_GROUP}" "$DATA_PREPROCESSED_DIR"
mkdir -p "$DATA_PROCESSED_DIR"
chown "${APP_USER}:${APP_GROUP}" "$DATA_PROCESSED_DIR"
# DON'T chown /app/data or /app/data/raw

echo "Entrypoint: Directory checks complete."

# --- Execute command as appuser ---
echo "Entrypoint: Switching to user ${APP_USER} ($(id -u ${APP_USER}):$(id -g ${APP_USER})) and executing command: $@"

# Use gosu (installed via Dockerfile) to switch user and execute the command
exec gosu "${APP_USER}" "$@"