#!/bin/bash
#
# start_robot_server.sh — Launch RobotServer + persistent reverse SSH tunnel to Beaker.
#
# The robot desktop can reach the Beaker container via Tailscale, but not
# the other way around.  A reverse SSH tunnel exposes the local gRPC port
# on the container so that RemoteEnv can connect to localhost:<port>.
#
# Uses autossh to keep the tunnel alive across Beaker job restarts.
# Every Beaker job registers the Tailscale hostname "beaker-0", so the
# tunnel automatically reconnects when a new job starts — no IP needed.
#
# Because Beaker jobs are ephemeral, the SSH host key behind "beaker-0" also
# changes across preemptions / restarts.  The tunnel therefore disables
# known_hosts persistence for this connection so stale host keys do not block
# reverse port forwarding on the desktop.
#
# RemoteEnv retries the gRPC connection for grpc_connect_timeout seconds
# (default 300s), giving the tunnel time to establish after job submission.
#
# Usage:
#   bash scripts/start_robot_server.sh --config /path/to/yam_env.yaml [OPTIONS]
#
# Options:
#   --config PATH         Path to YAM env YAML config (required)
#   --train-config PATH   Optional top-level training YAML to source timing from
#                         (default: examples/embodiment/config/yam_ppo_openpi_async.yaml;
#                         sync configs are also supported)
#   --port PORT           gRPC server port (default: 50051)
#   --remote-host HOST    Beaker Tailscale hostname or IP (default: beaker-0)
#   --remote-user USER    SSH user on the Beaker container (default: shiruic)
#   --return-home-minutes MIN  Override desktop episode duration from CLI
#   --cooldown-minutes MIN     Override desktop cooldown from CLI
#   --use-follower-servers  Launch YAM follower servers on 1234/1235 first (default)
#   --no-follower-servers   Skip follower servers (auto-set by --dummy)
#   --gripper-open VAL    Optional gripper open limit for follower startup
#   --gripper-close VAL   Optional gripper close limit for follower startup
#   --allow-plain-ssh     Allow fallback to plain ssh if autossh is unavailable
#   --kill-video-holders  Kill stale processes holding /dev/video* before startup
#   --no-kill-video-holders  Do not kill stale /dev/video* holders before startup
#   --reset-can           Reset CAN interfaces before startup
#   --no-reset-can        Do not reset CAN interfaces before startup (default)
#   --no-tunnel           Start RobotServer only, no SSH tunnel
#   --dummy               Run without real hardware (zero observations)
#   --verbose             Show robot state before serving and log every chunk step
#   --sbs                 Step-by-step: wait for Enter before each chunk (implies --verbose)
#   --help                Show this help

set -euo pipefail

CONFIG=""
TRAIN_CONFIG=""
PORT=50051
MAX_MESSAGE_SIZE=67108864
REMOTE_HOST="beaker-0"
REMOTE_USER="shiruic"
RETURN_HOME_MINUTES_OVERRIDE=""
COOLDOWN_MINUTES_OVERRIDE=""
USE_FOLLOWER_SERVERS=true
GRIPPER_OPEN=""
GRIPPER_CLOSE=""
ALLOW_PLAIN_SSH=false
KILL_VIDEO_HOLDERS=true
RESET_CAN=false
NO_TUNNEL=false
DUMMY=false
VERBOSE=false
SBS=false
FOLLOWER_PID=""
TUNNEL_PID=""
TUNNEL_LOG=""
TUNNEL_LAUNCHER=""

usage() {
    cat <<'EOF'
Usage: bash scripts/start_robot_server.sh --config PATH [OPTIONS]

Launch the gRPC RobotServer wrapping YAMEnv, with a persistent autossh reverse
tunnel to Beaker. The tunnel reconnects automatically when a new Beaker job
starts (all jobs register the Tailscale hostname "beaker-0").

Options:
  --config PATH         Path to YAM env YAML config (required)
  --train-config PATH   Optional top-level training YAML to source timing from
                        (default: examples/embodiment/config/yam_ppo_openpi_async.yaml;
                        sync configs are also supported)
  --port PORT           gRPC server port (default: 50051)
  --remote-host HOST    Beaker Tailscale hostname or IP (default: beaker-0)
  --remote-user USER    SSH user on the Beaker container (default: shiruic)
  --return-home-minutes MIN  Override desktop episode duration from CLI
  --cooldown-minutes MIN     Override desktop cooldown from CLI
  --use-follower-servers  Launch YAM follower servers on 1234/1235 first (default)
  --no-follower-servers   Skip follower servers (auto-set by --dummy)
  --gripper-open VAL    Optional gripper open limit for follower startup
  --gripper-close VAL   Optional gripper close limit for follower startup
  --allow-plain-ssh     Allow fallback to plain ssh if autossh is unavailable
  --kill-video-holders  Kill stale processes holding /dev/video* before startup (default: on)
  --no-kill-video-holders  Do not kill stale /dev/video* holders before startup
  --reset-can           Reset CAN interfaces before startup
  --no-reset-can        Do not reset CAN interfaces before startup (default)
  --no-tunnel           Start RobotServer only, without SSH tunnel
  --dummy               Run without real hardware (zero observations)
  --verbose             Show robot state before serving and log every chunk step
  --sbs                 Step-by-step: wait for Enter before each chunk (implies --verbose)
  --help                Show this help

Examples:
  # Persistent server + auto-reconnecting tunnel (default beaker-0 hostname):
  bash scripts/start_robot_server.sh \
      --config examples/embodiment/config/env/yam_pi05_follower.yaml \
      --train-config examples/embodiment/config/yam_ppo_openpi_async.yaml \
      --use-follower-servers

  # Same timing-sharing flow with the sync staged runtime:
  bash scripts/start_robot_server.sh \
      --config examples/embodiment/config/env/yam_pi05_follower.yaml \
      --train-config examples/embodiment/config/yam_ppo_openpi_sync.yaml \
      --use-follower-servers

  # Local only (no tunnel, for testing):
  bash scripts/start_robot_server.sh --config examples/embodiment/config/env/yam_pi05_follower.yaml \
      --no-tunnel --dummy

  # YAM follower servers + RobotServer:
  bash scripts/start_robot_server.sh \
      --config examples/embodiment/config/env/yam_pi05_follower.yaml \
      --use-follower-servers --no-tunnel

  # Explicit IP instead of hostname (e.g. for one-off debugging):
  bash scripts/start_robot_server.sh \
      --config examples/embodiment/config/env/yam_pi05_follower.yaml \
      --use-follower-servers \
      --remote-host 100.87.5.72
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)         usage ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --remote-host)  REMOTE_HOST="$2"; shift 2 ;;
        --remote-user)  REMOTE_USER="$2"; shift 2 ;;
        --return-home-minutes) RETURN_HOME_MINUTES_OVERRIDE="$2"; shift 2 ;;
        --cooldown-minutes) COOLDOWN_MINUTES_OVERRIDE="$2"; shift 2 ;;
        --use-follower-servers) USE_FOLLOWER_SERVERS=true; shift ;;
        --no-follower-servers) USE_FOLLOWER_SERVERS=false; shift ;;
        --gripper-open) GRIPPER_OPEN="$2"; shift 2 ;;
        --gripper-close) GRIPPER_CLOSE="$2"; shift 2 ;;
        --allow-plain-ssh) ALLOW_PLAIN_SSH=true; shift ;;
        --kill-video-holders) KILL_VIDEO_HOLDERS=true; shift ;;
        --no-kill-video-holders) KILL_VIDEO_HOLDERS=false; shift ;;
        --reset-can)    RESET_CAN=true; shift ;;
        --no-reset-can) RESET_CAN=false; shift ;;
        --no-tunnel)    NO_TUNNEL=true; shift ;;
        --dummy)        DUMMY=true; shift ;;
        --verbose)      VERBOSE=true; shift ;;
        --sbs)          SBS=true; shift ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    exit 1
fi

# Dummy mode does not need real follower servers.
if [ "$DUMMY" = true ]; then
    USE_FOLLOWER_SERVERS=false
fi

if [ -z "$TRAIN_CONFIG" ]; then
    DEFAULT_TRAIN_CONFIG="examples/embodiment/config/yam_ppo_openpi_async.yaml"
    if [ -f "$DEFAULT_TRAIN_CONFIG" ]; then
        TRAIN_CONFIG="$DEFAULT_TRAIN_CONFIG"
    fi
fi

resolve_shared_timing() {
    local train_cfg="$1"
    local return_home_override="$2"
    local cooldown_override="$3"
    python - "$train_cfg" "$return_home_override" "$cooldown_override" <<'PY'
import sys
from pathlib import Path

import yaml

train_cfg = sys.argv[1]
return_home_override = sys.argv[2]
cooldown_override = sys.argv[3]

return_home = None
cooldown = None

if train_cfg:
    path = Path(train_cfg).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    env_cfg = data.get("env", {}) or {}
    if env_cfg.get("return_home_minutes") is not None:
        return_home = float(env_cfg["return_home_minutes"])
    if env_cfg.get("server_cooldown_minutes") is not None:
        cooldown = float(env_cfg["server_cooldown_minutes"])

if return_home_override:
    return_home = float(return_home_override)
if cooldown_override:
    cooldown = float(cooldown_override)

if return_home is not None:
    print(f"RLINF_EPISODE_DURATION_S={return_home * 60.0}")
if cooldown is not None:
    print(f"RLINF_EPISODE_COOLDOWN_MINUTES={cooldown}")
PY
}

if [ -n "$TRAIN_CONFIG" ] || [ -n "$RETURN_HOME_MINUTES_OVERRIDE" ] || [ -n "$COOLDOWN_MINUTES_OVERRIDE" ]; then
    while IFS= read -r kv; do
        [ -n "$kv" ] || continue
        export "$kv"
    done < <(resolve_shared_timing "$TRAIN_CONFIG" "$RETURN_HOME_MINUTES_OVERRIDE" "$COOLDOWN_MINUTES_OVERRIDE")
fi

CLEANING_UP=false
SERVER_SHUTDOWN_WAIT_S=8

video_devices_exist() {
    compgen -G "/dev/video*" >/dev/null 2>&1
}

video_device_holder_pids() {
    if ! video_devices_exist; then
        return 0
    fi

    if command -v lsof >/dev/null 2>&1; then
        lsof -t /dev/video* 2>/dev/null | sort -u | tr '\n' ' '
        return 0
    fi

    if command -v fuser >/dev/null 2>&1; then
        fuser /dev/video* 2>/dev/null | tr ' ' '\n' | awk 'NF' | sort -u | tr '\n' ' '
        return 0
    fi
}

report_video_device_holders() {
    if ! video_devices_exist; then
        return 0
    fi
    local holders
    if command -v lsof >/dev/null 2>&1; then
        holders="$(lsof /dev/video* 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        holders="$(fuser -v /dev/video* 2>/dev/null || true)"
    else
        return 0
    fi
    if [ -n "$holders" ]; then
        echo "Video device holders detected:"
        echo "$holders"
        echo ""
    fi
}

kill_video_device_holders() {
    local pids
    pids="$(video_device_holder_pids)"
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "Killing video device holder(s): $pids"
    kill $pids 2>/dev/null || true
    sleep 2
    kill -9 $pids 2>/dev/null || true
}

cleanup() {
    if [ "$CLEANING_UP" = true ]; then return; fi
    CLEANING_UP=true
    echo "Shutting down..."

    # Stop the RobotServer first so it can return the arms home while the
    # follower servers are still alive.
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        for ((i=0; i<SERVER_SHUTDOWN_WAIT_S; i++)); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
    fi

    [ -n "${FOLLOWER_PID:-}" ] && kill "$FOLLOWER_PID" 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true

    sleep 2

    [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
    [ -n "${FOLLOWER_PID:-}" ] && kill -9 "$FOLLOWER_PID" 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && kill -9 "$TUNNEL_PID" 2>/dev/null || true

    wait 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

is_ipv4_address() {
    [[ "$1" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

wait_for_local_port() {
    local port="$1"
    local timeout_s="$2"
    python - "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
timeout_s = float(sys.argv[2])
deadline = time.time() + timeout_s

while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            time.sleep(0.25)
            continue
        sys.exit(0)
sys.exit(1)
PY
}

validate_tunnel_prereqs() {
    if command -v autossh &>/dev/null; then
        return 0
    fi

    if [ "$ALLOW_PLAIN_SSH" = true ]; then
        echo "WARNING: autossh not found; falling back to plain ssh."
        echo "         The tunnel will not auto-reconnect if the Beaker node restarts."
        return 0
    fi

    echo "ERROR: autossh is required for a persistent reverse tunnel but was not found."
    echo "Install autossh, or rerun with --allow-plain-ssh for a one-shot tunnel."
    exit 1
}

start_tunnel() {
    local ssh_args=(
        -N
        -R "${PORT}:localhost:${PORT}"
        -o ServerAliveInterval=10
        -o ServerAliveCountMax=3
        -o ExitOnForwardFailure=yes
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o GlobalKnownHostsFile=/dev/null
        -o ConnectTimeout=10
        -o BatchMode=yes
    )

    if command -v autossh &>/dev/null; then
        TUNNEL_LAUNCHER="autossh"
        export AUTOSSH_GATETIME=0
        export AUTOSSH_POLL=10
        export AUTOSSH_LOGLEVEL=7
        autossh -M 0 "${ssh_args[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            >"${TUNNEL_LOG}" 2>&1 &
    else
        TUNNEL_LAUNCHER="ssh"
        echo "WARNING: autossh not found; using plain ssh because --allow-plain-ssh was set." | tee -a "${TUNNEL_LOG}"
        echo "         The tunnel will work, but it will not auto-reconnect if it drops." | tee -a "${TUNNEL_LOG}"
        ssh "${ssh_args[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            >>"${TUNNEL_LOG}" 2>&1 &
    fi

    TUNNEL_PID=$!
}

print_tunnel_failure_hint() {
    echo ""
    echo "ERROR: ${TUNNEL_LAUNCHER} (PID ${TUNNEL_PID}) died within 3 seconds."
    echo "  Try manually:  ssh -v -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null ${REMOTE_USER}@${REMOTE_HOST} echo ok"
    echo "  Common causes: host unreachable, SSH key rejected, autossh not installed"
    echo "  Tunnel log: ${TUNNEL_LOG}"
    if [ -s "${TUNNEL_LOG}" ]; then
        echo "  Last tunnel log lines:"
        tail -n 20 "${TUNNEL_LOG}" | sed 's/^/    /'
    fi
    echo ""
    echo "RobotServer is still running (PID ${SERVER_PID}). Tunnel is NOT active."
}

if [ "$KILL_VIDEO_HOLDERS" = true ]; then
    echo "=== Releasing /dev/video* holders ==="
    kill_video_device_holders
    echo ""
elif [ -n "$(video_device_holder_pids)" ]; then
    report_video_device_holders
    echo "TIP: rerun with --kill-video-holders to terminate stale /dev/video* holders."
    echo ""
fi

if [ "$DUMMY" = false ] && [ "$RESET_CAN" = true ]; then
    echo "=== Resetting CAN interfaces ==="
    bash third_party/yam_realtime/yam_realtime/scripts/reset_all_can.sh
    echo ""
fi

if [ "$USE_FOLLOWER_SERVERS" = true ] && [ "$DUMMY" = false ]; then
    echo "=== Starting YAM follower servers ==="
    FOLLOWER_ARGS=()
    [ -n "$GRIPPER_OPEN" ] && FOLLOWER_ARGS+=(--gripper-open "$GRIPPER_OPEN")
    [ -n "$GRIPPER_CLOSE" ] && FOLLOWER_ARGS+=(--gripper-close "$GRIPPER_CLOSE")

    # Start follower servers immune to SIGINT so that Ctrl+C on the
    # terminal only reaches the robot server.  The robot server returns
    # the arms home while the follower servers are still alive, then the
    # cleanup() function terminates the follower servers afterwards.
    # (Python preserves SIG_IGN across exec, so the launcher and its
    # children stay immune to the terminal interrupt.)
    (trap '' INT; exec python scripts/start_yam_follower_servers.py "${FOLLOWER_ARGS[@]}") &
    FOLLOWER_PID=$!
    echo "Follower launcher PID: ${FOLLOWER_PID}"
    echo "Waiting for follower servers on localhost:1234 and localhost:1235..."

    if ! wait_for_local_port 1234 20 || ! wait_for_local_port 1235 20; then
        echo "ERROR: follower servers did not become ready within 20 seconds."
        exit 1
    fi
    echo "Follower servers are ready."
    echo ""
fi

echo "=== Starting RobotServer ==="
echo "Config: ${CONFIG}"
if [ -n "${TRAIN_CONFIG}" ]; then
    echo "Train timing config: ${TRAIN_CONFIG}"
fi
if [ -n "${RLINF_EPISODE_DURATION_S:-}" ]; then
    echo "Episode duration: ${RLINF_EPISODE_DURATION_S}s"
fi
if [ -n "${RLINF_EPISODE_COOLDOWN_MINUTES:-}" ]; then
    echo "Episode cooldown: ${RLINF_EPISODE_COOLDOWN_MINUTES} min"
fi
echo "Port:   ${PORT}"
echo "Max message size: ${MAX_MESSAGE_SIZE}"
echo ""

SERVER_ARGS=(
    --config-path "${CONFIG}"
    --port "${PORT}"
    --max-message-size "${MAX_MESSAGE_SIZE}"
)
[ "$DUMMY" = true ] && SERVER_ARGS+=(--dummy)
[ "$VERBOSE" = true ] && SERVER_ARGS+=(--verbose)
[ "$SBS" = true ] && SERVER_ARGS+=(--sbs)

if [ "$VERBOSE" = true ]; then
    READY_FLAG=$(mktemp /tmp/rlinf_robot_ready.XXXX)
    rm -f "$READY_FLAG"
    RLINF_ROBOT_SERVER_READY_FLAG="$READY_FLAG" \
        python -m rlinf.envs.remote.robot_server "${SERVER_ARGS[@]}" &
    SERVER_PID=$!

    while [ ! -f "$READY_FLAG" ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: RobotServer exited before becoming ready."
            report_video_device_holders
            exit 1
        fi
        sleep 0.5
    done
    rm -f "$READY_FLAG"
else
    python -m rlinf.envs.remote.robot_server "${SERVER_ARGS[@]}" &
    SERVER_PID=$!
fi

sleep 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: RobotServer exited shortly after launch."
    report_video_device_holders
    wait "$SERVER_PID" || true
    exit 1
fi

if [ "$NO_TUNNEL" = false ]; then
    validate_tunnel_prereqs
    TUNNEL_LOG=$(mktemp "/tmp/rlinf_robot_tunnel.${PORT}.XXXX.log")
    echo "=== Starting persistent reverse SSH tunnel ==="
    echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
    echo "Tunnel: ${REMOTE_HOST}:localhost:${PORT} -> this machine:${PORT}"
    echo "Log:    ${TUNNEL_LOG}"
    echo ""
    if is_ipv4_address "${REMOTE_HOST}"; then
        echo "WARNING: --remote-host is an IP address."
        echo "         Reconnect after Beaker replacement is less reliable because the new job may get a new IP."
        echo "         Prefer the stable Tailscale hostname (default: beaker-0)."
        echo ""
    fi
    echo "autossh will reconnect automatically when a new Beaker job starts."
    echo "This works best when using the stable Tailscale hostname (default: beaker-0)."
    echo "ROBOT_SERVER_URL=localhost:${PORT} is set in submit_yam_training.sh."
    echo ""

    start_tunnel
    echo "${TUNNEL_LAUNCHER} PID: ${TUNNEL_PID}"

    # Give the tunnel a moment to start, then verify it's still alive.
    sleep 3
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        print_tunnel_failure_hint
    else
        echo "${TUNNEL_LAUNCHER} is running."
        echo ""
    fi
fi

# Monitor both processes: warn if the tunnel dies while robot server is still up.
while true; do
    if [ "$USE_FOLLOWER_SERVERS" = true ] && [ -n "${FOLLOWER_PID:-}" ]; then
        if ! kill -0 "$FOLLOWER_PID" 2>/dev/null; then
            echo "Follower server launcher (PID ${FOLLOWER_PID}) exited."
            break
        fi
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "RobotServer (PID ${SERVER_PID}) exited."
        break
    fi
    if [ "$NO_TUNNEL" = false ] && [ -n "${TUNNEL_PID:-}" ]; then
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            echo ""
            echo "WARNING: ${TUNNEL_LAUNCHER} (PID ${TUNNEL_PID}) died. Restarting tunnel..."
            start_tunnel
            echo "Restarted ${TUNNEL_LAUNCHER} with PID ${TUNNEL_PID}"
            sleep 1
            if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
                print_tunnel_failure_hint
            fi
        fi
    fi
    sleep 10
done

exit 0
