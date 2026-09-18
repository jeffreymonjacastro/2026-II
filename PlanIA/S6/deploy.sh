#!/usr/bin/env bash

set -euo pipefail

# Uso:
#   ./deploy.sh [PROJECT_ID] [REGION] [SERVICE_NAME]
# Ejemplo:
#   ./deploy.sh plania-workshop us-central1 bank-marketing-api

PROJECT_ID="${1:-${PROJECT_ID:-plania-workshop}}"
REGION="${2:-${REGION:-us-central1}}"
SERVICE_NAME="${3:-${SERVICE_NAME:-bank-marketing-api}}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud CLI no está instalado o no está disponible en PATH." >&2
  exit 1
fi

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
if [[ -z "${ACTIVE_ACCOUNT}" ]]; then
  echo "Error: no hay una cuenta activa. Ejecute: gcloud auth login" >&2
  exit 1
fi

echo "Cuenta activa: ${ACTIVE_ACCOUNT}"
echo "Proyecto: ${PROJECT_ID}"
echo "Región: ${REGION}"
echo "Servicio: ${SERVICE_NAME}"

gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "${PROJECT_ID}"

gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --quiet

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format='value(status.url)')"

echo "SERVICE_URL=${SERVICE_URL}"

if command -v curl >/dev/null 2>&1; then
  echo "Verificando /health..."
  curl --fail --silent --show-error "${SERVICE_URL}/health"
  echo
else
  echo "Aviso: curl no está instalado; abra ${SERVICE_URL}/health para verificar el servicio."
fi
