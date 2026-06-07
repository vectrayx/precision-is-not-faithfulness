#!/usr/bin/env bash
# Create an Azure OpenAI resource + deploy a frontier model, then print the env vars
# the generators need. Uses Azure credit. Review before running.
#
# Prereqs: az login (already done), a subscription with Azure OpenAI access.
set -euo pipefail

RG="${RG:-f1-paper-rg}"
LOCATION="${LOCATION:-eastus2}"
ACCOUNT="${ACCOUNT:-f1paper-aoai-$RANDOM}"
DEPLOYMENT="${DEPLOYMENT:-gpt-4o}"
MODEL="${MODEL:-gpt-4o}"
MODEL_VERSION="${MODEL_VERSION:-2024-08-06}"
SKU_CAP="${SKU_CAP:-10}"   # tokens-per-minute in thousands; keep small to cap spend

echo "Creating resource group $RG in $LOCATION ..."
az group create -n "$RG" -l "$LOCATION" -o none

echo "Creating Azure OpenAI account $ACCOUNT ..."
az cognitiveservices account create \
  -n "$ACCOUNT" -g "$RG" -l "$LOCATION" \
  --kind OpenAI --sku S0 --yes -o none

echo "Deploying $MODEL ($MODEL_VERSION) as deployment '$DEPLOYMENT' ..."
az cognitiveservices account deployment create \
  -n "$ACCOUNT" -g "$RG" \
  --deployment-name "$DEPLOYMENT" \
  --model-name "$MODEL" --model-version "$MODEL_VERSION" \
  --model-format OpenAI \
  --sku-capacity "$SKU_CAP" --sku-name "Standard" -o none

ENDPOINT=$(az cognitiveservices account show -n "$ACCOUNT" -g "$RG" --query properties.endpoint -o tsv)
KEY=$(az cognitiveservices account keys list -n "$ACCOUNT" -g "$RG" --query key1 -o tsv)

cat <<EOF

# ---- add to your environment (do NOT commit) ----
export AZURE_OPENAI_ENDPOINT="$ENDPOINT"
export AZURE_OPENAI_API_KEY="$KEY"
export AZURE_OPENAI_DEPLOYMENT="$DEPLOYMENT"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"

# then: python experiments/run_pilot.py --generators azure_openai --lang en
# teardown to stop spend: az group delete -n $RG --yes
EOF
