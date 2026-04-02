#!/usr/bin/env bash

set -euo pipefail

beaker_whoami="${BEAKER_WHOAMI:-$(beaker account whoami --format json | jq -r '.[0].name')}"

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
    --install "echo 'Skipping Gantry Python setup'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    -- LOSS_MODE=vanilla CLIP_LOW=0.2 CLIP_HIGH=0.28 bash examples/dppo_trainer/run_qwen30b_dppo.sh
