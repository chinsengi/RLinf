#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"

# Auto-select the staged YAM entrypoints only for configs that actually opt in
# to the staged runtime (`vlm_planner:` block) and use the explicit `_async` or
# `_sync` suffix. All other configs keep the generic simulation launcher path.
if [[ -f "${CONFIG_FILE}" ]] && rg -q '^vlm_planner:' "${CONFIG_FILE}"; then
    if rg -q 'env/yam_pi05_follower@env\.train' "${CONFIG_FILE}"; then
        echo "Error: ${CONFIG_NAME} uses the desktop-driven topology."
        echo "Use scripts/submit_yam_beaker_cluster.sh and scripts/join_beaker_cluster.sh instead."
        exit 1
    fi

    case "$CONFIG_NAME" in
        *_async)
            SRC_FILE="${EMBODIED_PATH}/train_embodied_agent_staged_async.py"
            ;;
        *_sync)
            SRC_FILE="${EMBODIED_PATH}/train_embodied_agent_staged.py"
            ;;
        *)
            SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
            ;;
    esac
else
    SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
fi
export SRC_FILE

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
