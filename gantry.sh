#!/usr/bin/env bash

set -euo pipefail

beaker_whoami="${BEAKER_WHOAMI:-$(beaker account whoami --format json | jq -r '.[0].name')}"

if (($# == 0)); then
    cmd=(
        env
        NNODES=1
        LOSS_MODE=vanilla
        CLIP_LOW=0.2
        CLIP_HIGH=0.28
        bash
        examples/dppo_trainer/run_qwen30b_dppo.sh
    )
else
    cmd=(env "$@")
fi

# Run Gantry outside the repo so it doesn't stage the current git checkout.
gantry run \
    --workspace ai2/oe-adapt-code \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --priority high \
    --gpus 8 \
    --timeout 5h \
    --task-name gantry-verl \
    --docker-image verlai/verl:vllm018.dev1 \
    --system-python \
    --no-logs \
    --env WANDB_ENTITY=mnoukhov \
    --secret-env HF_TOKEN=michaeln_HF_TOKEN \
    --secret-env WANDB_API_KEY=michaeln_WANDB_API_KEY \
    --install "echo 'Skipping Gantry Python setup'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    -- "${cmd[@]}"
