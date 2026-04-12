#!/bin/bash
# Autonomous training monitor. Appends a snapshot every 5 minutes.
# Auto-exits if training process dies, capturing the post-mortem.

MONITOR_LOG=/home/croco/SpaMo-PJM/logs/monitor.log
RUN_LOG=/home/croco/SpaMo-PJM/logs/run.log
TRAIN_PATTERN="main.py -c configs/finetune.yaml"

echo "=== monitor started $(date -Iseconds) ===" >> "$MONITOR_LOG"

while true; do
    ts=$(date -Iseconds)

    if ! pgrep -f "$TRAIN_PATTERN" > /dev/null; then
        echo "" >> "$MONITOR_LOG"
        echo "=== [$ts] TRAINING PROCESS DIED ===" >> "$MONITOR_LOG"
        echo "--- Last 40 lines of run.log ---" >> "$MONITOR_LOG"
        tail -40 "$RUN_LOG" >> "$MONITOR_LOG"
        echo "=== monitor exiting ===" >> "$MONITOR_LOG"
        exit 0
    fi

    {
        echo ""
        echo "=== [$ts] tick ==="
        echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader)"
        echo "disk_free: $(df -h /home/croco/SpaMo-PJM | tail -1 | awk '{print $4}')"
        echo "--- recent log ---"
        tail -15 "$RUN_LOG"
    } >> "$MONITOR_LOG"

    sleep 300
done
