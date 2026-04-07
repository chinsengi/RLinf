#!/bin/bash
#
# submit_yam_beaker_cluster.sh — Submit a Beaker job that idles with GPUs.
#
# Submits a gantry job that idles forever. The repo and .venv are on Weka.
# Attach to the container to start Ray and run training manually.
#
# Prerequisites:
#   gantry installed: pip install beaker-gantry
#
# Usage:
#   bash scripts/submit_yam_beaker_cluster.sh [OPTIONS]

set -euo pipefail

# --- Defaults ---
EXP_NAME="rlinf-cluster"
GPUS=1
CLUSTER="ai2/ceres-cirrascale"
WORKSPACE="ai2/molmo-act"
BUDGET=""
PRIORITY="urgent"
DRY_RUN=""
SHOW_LOGS=""
ALLOW_DIRTY=""

BEAKER_IMAGE="shiruic/shirui-torch2.8.0_cuda12.8"
WEKA_MOUNT="oe-training-default:/weka/oe-training-default"

usage() {
    cat <<'EOF'
Usage: bash scripts/submit_yam_beaker_cluster.sh [OPTIONS]

Submit a Beaker job that idles with GPUs.
The repo and .venv are on Weka. Attach to start Ray and training manually.

Options:
  --gpus N              GPUs (default: 1)
  --name NAME           Experiment name (default: rlinf-cluster)
  --cluster CLUSTER     Beaker cluster (default: ai2/ceres-cirrascale)
  --workspace WORKSPACE Beaker workspace (default: ai2/molmo-act)
  --budget BUDGET       Beaker budget account
  --priority PRIORITY   Job priority (default: urgent)
  --show-logs           Stream Beaker logs after submission
  --allow-dirty         Allow dirty git working directory
  --dry-run             Print command without executing
  --help                Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)         usage ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --name)         EXP_NAME="$2"; shift 2 ;;
        --cluster)      CLUSTER="$2"; shift 2 ;;
        --workspace)    WORKSPACE="$2"; shift 2 ;;
        --budget)       BUDGET="$2"; shift 2 ;;
        --priority)     PRIORITY="$2"; shift 2 ;;
        --show-logs)    SHOW_LOGS="true"; shift ;;
        --allow-dirty)  ALLOW_DIRTY="true"; shift ;;
        --dry-run)      DRY_RUN="true"; shift ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

# --- Build gantry command ---
gantry_args=(
    gantry run --yes --no-python
    --replicas 1
    --gpus "${GPUS}"
    --shared-memory "300G"
    --host-networking
    --beaker-image "${BEAKER_IMAGE}"
    --workspace "${WORKSPACE}"
    --cluster "${CLUSTER}"
    --name "${EXP_NAME}"
    --priority "${PRIORITY}"
    --weka "${WEKA_MOUNT}"
    --env "HF_HOME=/weka/oe-training-default/shiruic/hf_cache"
    --env "EMBODIED_PATH=examples/embodiment"
    --env-secret "HF_TOKEN=hf_token_shirui"
)

[ -n "$BUDGET" ]      && gantry_args+=("--budget" "$BUDGET")
[ -n "$SHOW_LOGS" ]   && gantry_args+=("--show-logs")
[ -n "$ALLOW_DIRTY" ] && gantry_args+=("--allow-dirty")

gantry_args+=("--" "bash" "-c" "echo '=== Container idle. ===' && tail -f /dev/null")

echo "=== Submit Beaker Cluster (idle) ==="
echo "GPUs:         ${GPUS}"
echo "Cluster:      ${CLUSTER}"
echo "Workspace:    ${WORKSPACE}"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "[dry-run] Would execute:"
    printf '  %s\n' "${gantry_args[@]}"
else
    "${gantry_args[@]}"
fi
