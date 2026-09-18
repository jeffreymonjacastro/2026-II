param(
    [string]$ProjectId = "plania-workshop",
    [string]$Region = "us-central1",
    [string]$ServiceName = "bank-marketing-api"
)

$ErrorActionPreference = "Stop"

gcloud config set project $ProjectId
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --allow-unauthenticated `
    --memory 1Gi `
    --cpu 1 `
    --min-instances 0 `
    --max-instances 2 `
    --quiet

$ServiceUrl = gcloud run services describe $ServiceName `
    --region $Region `
    --format "value(status.url)"

Write-Output "SERVICE_URL=$ServiceUrl"
Invoke-RestMethod -Uri "$ServiceUrl/health" -Method Get | ConvertTo-Json
