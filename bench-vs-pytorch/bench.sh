#!/usr/bin/env bash
# Obrain-grade bench runner — produces dated baselines for diffing.
#
# Outputs:
#   baselines/rustorch-YYYY-MM-DD.json
#   baselines/pytorch-1thread-YYYY-MM-DD.json
#   baselines/pytorch-12thread-YYYY-MM-DD.json
#
# Usage:
#   ./bench.sh                  # full run
#   ./bench.sh --label foo      # tag the date with a label (e.g. --label post-T2)
#   ./bench.sh --skip-pytorch   # skip PyTorch (faster, when only Rust changed)
#   ./bench.sh --skip-rust      # skip Rust (when only re-checking PyTorch)

set -euo pipefail

cd "$(dirname "$0")"

DATE=$(date -u +%Y-%m-%d)
LABEL=""
SKIP_PYTORCH=false
SKIP_RUST=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --label) LABEL="-$2"; shift 2 ;;
    --skip-pytorch) SKIP_PYTORCH=true; shift ;;
    --skip-rust) SKIP_RUST=true; shift ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

mkdir -p baselines

if [[ "$SKIP_RUST" != "true" ]]; then
  echo "==> RusTorch criterion bench (target-cpu=native, taskpolicy QoS utility)"
  taskpolicy -c utility cargo bench --bench compare 2>&1 | tail -3
  python3 parse_criterion.py > "baselines/rustorch-${DATE}${LABEL}.json"
  echo "  -> baselines/rustorch-${DATE}${LABEL}.json"
fi

if [[ "$SKIP_PYTORCH" != "true" ]]; then
  echo "==> PyTorch 1-thread"
  taskpolicy -c utility python3 pytorch_bench.py 1 > "baselines/pytorch-1thread-${DATE}${LABEL}.json"
  echo "  -> baselines/pytorch-1thread-${DATE}${LABEL}.json"

  echo "==> PyTorch 12-thread (M4 Max, all P-cores)"
  taskpolicy -c utility python3 pytorch_bench.py 12 > "baselines/pytorch-12thread-${DATE}${LABEL}.json"
  echo "  -> baselines/pytorch-12thread-${DATE}${LABEL}.json"
fi

echo
echo "==> Summary table"
python3 compare.py \
  --rustorch "baselines/rustorch-${DATE}${LABEL}.json" \
  --pytorch1 "baselines/pytorch-1thread-${DATE}${LABEL}.json" \
  --pytorch12 "baselines/pytorch-12thread-${DATE}${LABEL}.json"
