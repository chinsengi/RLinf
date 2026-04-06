#!/usr/bin/env bash
# Measure network bandwidth between this workstation and a remote host.
# Usage: bash scripts/measure_bandwidth.sh [user@host] [size_mb]

set -euo pipefail

REMOTE="${1:-shiruic@beaker-0}"
SIZE_MB="${2:-256}"
IPERF_PORT=5201
IPERF_DURATION=10

echo "=== Network Bandwidth Test: localhost <-> $REMOTE ==="
echo "Date: $(date)"
echo ""

# --- Latency ---
echo "--- Latency (ping) ---"
HOST_ONLY="${REMOTE#*@}"
ping -c 5 "$HOST_ONLY" 2>/dev/null || echo "(ping blocked or host unreachable)"
echo ""

# --- iperf3 test (preferred) ---
run_iperf() {
    echo "--- iperf3 bandwidth test (${IPERF_DURATION}s, port $IPERF_PORT) ---"

    # Start iperf3 server on remote
    ssh "$REMOTE" "pkill -f 'iperf3 -s -p $IPERF_PORT' 2>/dev/null; nohup iperf3 -s -p $IPERF_PORT -1 > /dev/null 2>&1 &"
    sleep 1

    echo ""
    echo ">> Upload (local -> remote):"
    iperf3 -c "$HOST_ONLY" -p "$IPERF_PORT" -t "$IPERF_DURATION" -f m 2>&1 | tail -3
    echo ""

    # Restart server for reverse test
    ssh "$REMOTE" "pkill -f 'iperf3 -s -p $IPERF_PORT' 2>/dev/null; nohup iperf3 -s -p $IPERF_PORT -1 > /dev/null 2>&1 &"
    sleep 1

    echo ">> Download (remote -> local):"
    iperf3 -c "$HOST_ONLY" -p "$IPERF_PORT" -t "$IPERF_DURATION" -R -f m 2>&1 | tail -3
    echo ""
}

# --- SSH+dd fallback ---
run_ssh_dd() {
    echo "--- SSH throughput test (${SIZE_MB} MB) ---"
    echo ""

    echo ">> Upload (local -> remote):"
    dd if=/dev/zero bs=1M count="$SIZE_MB" 2>/dev/null \
        | ssh "$REMOTE" "cat > /dev/null" \
        2>&1
    # Use pv if available, otherwise time it manually
    START=$(date +%s%N)
    dd if=/dev/zero bs=1M count="$SIZE_MB" 2>/dev/null \
        | ssh "$REMOTE" "cat > /dev/null"
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    SPEED=$(echo "scale=1; $SIZE_MB / $ELAPSED" | bc)
    echo "  Transferred ${SIZE_MB} MB in ${ELAPSED}s -> ${SPEED} MB/s"
    echo ""

    echo ">> Download (remote -> local):"
    START=$(date +%s%N)
    ssh "$REMOTE" "dd if=/dev/zero bs=1M count=$SIZE_MB 2>/dev/null" > /dev/null
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    SPEED=$(echo "scale=1; $SIZE_MB / $ELAPSED" | bc)
    echo "  Transferred ${SIZE_MB} MB in ${ELAPSED}s -> ${SPEED} MB/s"
    echo ""
}

# --- SCP throughput ---
run_scp() {
    echo "--- SCP throughput test (${SIZE_MB} MB) ---"
    TMPFILE=$(mktemp)
    dd if=/dev/urandom bs=1M count="$SIZE_MB" of="$TMPFILE" 2>/dev/null

    echo ">> Upload (scp local -> remote):"
    START=$(date +%s%N)
    scp -q "$TMPFILE" "$REMOTE:/tmp/_bw_test_upload"
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    SPEED=$(echo "scale=1; $SIZE_MB / $ELAPSED" | bc)
    echo "  ${SIZE_MB} MB in ${ELAPSED}s -> ${SPEED} MB/s"

    echo ">> Download (scp remote -> local):"
    START=$(date +%s%N)
    scp -q "$REMOTE:/tmp/_bw_test_upload" "$TMPFILE"
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    SPEED=$(echo "scale=1; $SIZE_MB / $ELAPSED" | bc)
    echo "  ${SIZE_MB} MB in ${ELAPSED}s -> ${SPEED} MB/s"

    # Cleanup
    rm -f "$TMPFILE"
    ssh "$REMOTE" "rm -f /tmp/_bw_test_upload" 2>/dev/null
    echo ""
}

# --- Run tests ---
if command -v iperf3 &>/dev/null && ssh "$REMOTE" "command -v iperf3" &>/dev/null; then
    run_iperf
else
    echo "(iperf3 not available on both ends, using SSH-based tests)"
    echo ""
fi

run_ssh_dd
run_scp

echo "=== Done ==="
