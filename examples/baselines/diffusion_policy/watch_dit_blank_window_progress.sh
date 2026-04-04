#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="${ROOT_DIR}/runs/dit_blank_window"
PYTHON_BIN="${PYTHON_BIN:-/home/hebu/miniconda3/envs/maniskill_py311/bin/python}"
REFRESH_SECS="${REFRESH_SECS:-5}"

while true; do
  if [[ -n "${TERM:-}" ]]; then
    clear || true
  fi
  EVENT_FILE="$(ls -1t "${RUN_DIR}"/events.out.tfevents.* 2>/dev/null | head -n 1 || true)"
  date '+%F %T'
  echo "run: dit_blank_window"
  echo
  if [[ -n "${EVENT_FILE}" ]]; then
    "${PYTHON_BIN}" - "${EVENT_FILE}" <<'PY'
from tensorboard.backend.event_processing import event_accumulator
import sys

path = sys.argv[1]

try:
    ea = event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
except Exception as exc:
    print(f"读取失败: {exc}")
    raise SystemExit(0)

scalar_tags = ea.Tags().get("scalars", [])
if not scalar_tags:
    print("current_iter: N/A")
    print("best_success_once: N/A")
    print("best_success_at_end: N/A")
    raise SystemExit(0)

iter_tags = ["losses/total_loss", "charts/learning_rate"]
current_iter = None
for tag in iter_tags:
    if tag in scalar_tags and ea.Scalars(tag):
        current_iter = ea.Scalars(tag)[-1].step
        break
if current_iter is None:
    current_iter = max(ea.Scalars(tag)[-1].step for tag in scalar_tags if ea.Scalars(tag))

def format_best(tag: str) -> str:
    if tag not in scalar_tags or not ea.Scalars(tag):
        return "N/A"
    values = [event.value for event in ea.Scalars(tag)]
    return f"{max(values):.6f}"

print(f"current_iter: {current_iter}")
print(f"best_success_once: {format_best('eval/success_once')}")
print(f"best_success_at_end: {format_best('eval/success_at_end')}")
PY
  else
    echo "current_iter: N/A"
    echo "best_success_once: N/A"
    echo "best_success_at_end: N/A"
  fi

  sleep "${REFRESH_SECS}"
done
