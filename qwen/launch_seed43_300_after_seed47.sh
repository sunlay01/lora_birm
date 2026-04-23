#!/usr/bin/env bash
set -euo pipefail

SUPERVISOR_PID="${SUPERVISOR_PID:-729511}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-/root/qwen_seed_sweep_300_supervisor.log}"
SWEEP_DONE_MARKER='[supervisor] finished seed=47 rc=0'
RUN_NAME="qwen_lora_birm_stable_300_lr2e-5_pen3_ema98_seed43"
OUT_ROOT="/root/qwen_lora_birm_tuning"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
RUN_LOG="/root/lora_birm/qwen/seed43_300.log"

echo "[launcher] waiting for seed47 completion from pid=${SUPERVISOR_PID} at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

while true; do
  if grep -Fq "${SWEEP_DONE_MARKER}" "${SUPERVISOR_LOG}"; then
    break
  fi

  if ! ps -p "${SUPERVISOR_PID}" >/dev/null 2>&1; then
    echo "[launcher] supervisor pid=${SUPERVISOR_PID} exited before seed47 completion marker appeared" >&2
    exit 1
  fi

  sleep 60
done

echo "[launcher] seed47 finished; preparing seed43 300-step run at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

if [[ -f "${OUT_DIR}/summary.csv" ]]; then
  echo "[launcher] ${OUT_DIR}/summary.csv already exists, skipping rerun"
  exit 0
fi

if [[ -e "${OUT_DIR}" ]]; then
  backup_dir="${OUT_DIR}_preexisting_backup_$(date -u +'%Y%m%dT%H%M%SZ')"
  mv "${OUT_DIR}" "${backup_dir}"
  echo "[launcher] backed up preexisting output dir to ${backup_dir}"
fi

if [[ -f "${RUN_LOG}" ]]; then
  backup_log="${RUN_LOG%.log}_preexisting_backup_$(date -u +'%Y%m%dT%H%M%SZ').log"
  mv "${RUN_LOG}" "${backup_log}"
  echo "[launcher] backed up preexisting run log to ${backup_log}"
fi

cd /root/lora_birm/qwen
QWEN_SEED=43 \
QWEN_MAX_STEPS=300 \
QWEN_RUN_NAME="${RUN_NAME}" \
QWEN_OUT_ROOT="${OUT_ROOT}" \
QWEN_MODEL_CACHE_DIR=/root/autodl-tmp/modelscope_cache \
QWEN_DATA_PARQUET=/root/train-00000-of-00001.parquet \
python qwen_lora_birm_stable_run.py | tee "${RUN_LOG}"
rc=${PIPESTATUS[0]}

echo "[launcher] seed43 300-step finished rc=${rc} at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
exit "${rc}"
