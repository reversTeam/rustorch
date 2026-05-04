#!/usr/bin/env bash
# Obrain-grade bench runner — produces dated baselines for diffing.
#
# Outputs:
#   baselines/rustorch-YYYY-MM-DD.json
#   baselines/pytorch-1thread-YYYY-MM-DD.json
#   baselines/pytorch-12thread-YYYY-MM-DD.json
#
# Default scheduling: NO `taskpolicy` → macOS lets QoS=user-initiated
# (criterion's default) and the scheduler keeps the bench on P-cores
# of M-series Macs. This matches transformer runtime conditions.
#
# `--e-cores` opt-in: forces QoS=utility via `taskpolicy -c utility`,
# which on M-series Macs reroutes the process to the Efficiency cores.
# Useful as a stress-test for edge / efficiency deployment, but the
# resulting baseline systematically under-reports performance on
# bandwidth-bound ops (vDSP, rayon-shard) — do NOT use as canonical.
#
# See note 90fdc88a-178c-4514-88d4-38eaf2816990 (Knowledge Fabric)
# for the full root-cause analysis of why E-core baselines lie.
#
# Usage:
#   ./bench.sh                  # full run on P-cores (canonical)
#   ./bench.sh --label foo      # tag the date with a label (e.g. --label post-T2)
#   ./bench.sh --skip-pytorch   # skip PyTorch (faster, when only Rust changed)
#   ./bench.sh --skip-rust      # skip Rust (when only re-checking PyTorch)
#   ./bench.sh --e-cores        # OPT-IN: force E-core run via taskpolicy utility

set -euo pipefail

cd "$(dirname "$0")"

DATE=$(date -u +%Y-%m-%d)
LABEL=""
SKIP_PYTORCH=false
SKIP_RUST=false
TASKPOLICY=""
QOS_LABEL="P-cores (default QoS)"

while [[ $# -gt 0 ]]; do
  case $1 in
    --label) LABEL="-$2"; shift 2 ;;
    --skip-pytorch) SKIP_PYTORCH=true; shift ;;
    --skip-rust) SKIP_RUST=true; shift ;;
    --e-cores)
      TASKPOLICY="taskpolicy -c utility"
      QOS_LABEL="E-cores (taskpolicy QoS utility)"
      shift
      ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

mkdir -p baselines

if [[ "$SKIP_RUST" != "true" ]]; then
  echo "==> RusTorch criterion bench (target-cpu=native, ${QOS_LABEL})"
  ${TASKPOLICY} cargo bench --bench compare 2>&1 | tail -3
  python3 parse_criterion.py > "baselines/rustorch-${DATE}${LABEL}.json"
  echo "  -> baselines/rustorch-${DATE}${LABEL}.json"
fi

if [[ "$SKIP_PYTORCH" != "true" ]]; then
  echo "==> PyTorch 1-thread (${QOS_LABEL})"
  ${TASKPOLICY} python3 pytorch_bench.py 1 > "baselines/pytorch-1thread-${DATE}${LABEL}.json"
  echo "  -> baselines/pytorch-1thread-${DATE}${LABEL}.json"

  echo "==> PyTorch 12-thread (${QOS_LABEL})"
  ${TASKPOLICY} python3 pytorch_bench.py 12 > "baselines/pytorch-12thread-${DATE}${LABEL}.json"
  echo "  -> baselines/pytorch-12thread-${DATE}${LABEL}.json"
fi

echo
echo "==> Summary table"
python3 compare.py \
  --rustorch "baselines/rustorch-${DATE}${LABEL}.json" \
  --pytorch1 "baselines/pytorch-1thread-${DATE}${LABEL}.json" \
  --pytorch12 "baselines/pytorch-12thread-${DATE}${LABEL}.json"
