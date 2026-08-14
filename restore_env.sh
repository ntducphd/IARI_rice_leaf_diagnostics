#!/usr/bin/env bash
# restore_env.sh — build the pinned environment for this compendium.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
if command -v mamba >/dev/null 2>&1; then SOLVER=mamba; else SOLVER=conda; fi
$SOLVER env create -f environment.yml -n ldapaper 2>/dev/null || \
  $SOLVER env update -f environment.yml -n ldapaper
echo "done — activate with:  conda activate ldapaper"
