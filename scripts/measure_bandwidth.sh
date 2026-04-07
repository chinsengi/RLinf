#!/usr/bin/env bash
# Measure network bandwidth between this workstation and a remote host.
# Usage: bash scripts/measure_bandwidth.sh [user@host] [size_mb]

set -euo pipefail

REMOTE="${1:-shiruic@beaker-0}"
SIZE_MB="${2:-32}"
echo "=== Network Bandwidth Test: localhost <-> $REMOTE ==="
echo "Date: $(date)"
echo ""

# --- Latency ---
echo "--- Latency (ping) ---"
HOST_ONLY="${REMOTE#*@}"
ping -c 5 "$HOST_ONLY" 2>/dev/null || echo "(ping blocked or host unreachable)"
echo ""

# --- SSH+dd throughput ---
run_ssh_dd() {
    echo "--- SSH throughput test (${SIZE_MB} MB) ---"
    echo ""

    echo ">> Upload (local -> remote):"
    START=$(date +%s%N)
    dd if=/dev/zero bs=1M count="$SIZE_MB" 2>/dev/null \
        | ssh -q "$REMOTE" "cat > /dev/null"
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    SPEED=$(echo "scale=1; $SIZE_MB / $ELAPSED" | bc)
    echo "  Transferred ${SIZE_MB} MB in ${ELAPSED}s -> ${SPEED} MB/s"
    echo ""

    echo ">> Download (remote -> local):"
    START=$(date +%s%N)
    ssh -q "$REMOTE" "dd if=/dev/zero bs=1M count=$SIZE_MB 2>/dev/null" > /dev/null
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    SPEED=$(echo "scale=1; $SIZE_MB / $ELAPSED" | bc)
    echo "  Transferred ${SIZE_MB} MB in ${ELAPSED}s -> ${SPEED} MB/s"
    echo ""
}

run_ssh_dd

echo "=== Done ==="
