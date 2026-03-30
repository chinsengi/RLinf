#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

if [ -z "$1" ]; then
    CONFIG_NAME="realworld_dummy_franka_sac_cnn"
else
    CONFIG_NAME=$1
fi

# Auto-select entry script: only the explicit staged YAM configs use the
# dedicated staged entrypoints. All other realworld configs keep their existing
# launch path.
case "$CONFIG_NAME" in
    yam_ppo_openpi_async|yam_ppo_openpi_subtask_async|yam_ppo_openpi_desktop_async)
        SRC_FILE="${EMBODIED_PATH}/train_embodied_agent_staged_async.py"
        ;;
    yam_ppo_openpi_sync|yam_ppo_openpi_subtask_sync|yam_ppo_openpi_desktop_sync)
        SRC_FILE="${EMBODIED_PATH}/train_embodied_agent_staged.py"
        ;;
    *)
        SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
        ;;
esac
export SRC_FILE

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
