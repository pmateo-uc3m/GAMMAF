#!/bin/bash
# EXP-07: v5 with influence_ema_decay=0.5 on the full seed-28 sample.
# EXP-04: v0 baseline on the seed-29 sample.
# Temporarily swaps defense-models/CASPIAN.py and restores v5 at the end.
set -u
cd /project_antwerp/GAMMAF
PY=/project_antwerp/gammaf-env/bin/python3
ACTIVE=defense-models/CASPIAN.py
BACKUP=/tmp/opencode/CASPIAN-v5-active.py

cp "$ACTIVE" "$BACKUP"

echo "[harness] EXP-07 start $(date -u +%H:%M:%S)"
$PY -u MainEvaluation.py CASPIAN-tests/configs/evaluation-caspian-tuned.yaml \
    > CASPIAN-tests/logs/EXP-07-v5-tuned.log 2>&1

cp CASPIAN-tests/snapshots/CASPIAN-v0-baseline.py "$ACTIVE"
echo "[harness] EXP-04 start $(date -u +%H:%M:%S)"
$PY -u MainEvaluation.py CASPIAN-tests/configs/evaluation-caspian-baseline-seed29.yaml \
    > CASPIAN-tests/logs/EXP-04-baseline-seed29.log 2>&1

cp "$BACKUP" "$ACTIVE"
echo "[harness] restored v5 $(date -u +%H:%M:%S)"
sha256sum "$ACTIVE" >> CASPIAN-tests/logs/EXP-04-baseline-seed29.log
