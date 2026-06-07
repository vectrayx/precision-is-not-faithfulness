#!/usr/bin/env bash
# Create a GCP GPU VM for (a) serving an open model with vLLM as an OpenAI-compatible
# endpoint, and (b) LoRA fine-tuning. Uses GCP credit. Review before running.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project)}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-f1-paper-gpu}"
MACHINE="${MACHINE:-g2-standard-8}"      # 1x L4 (24GB) — enough for 7B inference/LoRA
GPU="${GPU:-nvidia-l4}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu124-ubuntu-2204}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

echo "Creating GPU VM $VM ($MACHINE + $GPU) in $ZONE ..."
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="type=$GPU,count=1" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB --metadata="install-nvidia-driver=True"

cat <<'EOF'

# ---- on the VM ----
# SSH:   gcloud compute ssh f1-paper-gpu --zone us-central1-a
# Serve an open model (OpenAI-compatible) with vLLM:
#   pip install vllm
#   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
# From the client set:
#   export OPENAI_BASE_URL="http://<VM_IP>:8000/v1"
#   export OPENAI_MODEL="Qwen/Qwen2.5-7B-Instruct"
#   python experiments/run_pilot.py --generators openai_compatible --lang en
#
# Fine-tuning (LoRA):  python src/models/finetune.py --help
#
# IMPORTANT: stop the VM to halt spend:
#   gcloud compute instances stop f1-paper-gpu --zone us-central1-a
EOF
