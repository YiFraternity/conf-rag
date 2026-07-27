#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONF_RAG_DIR="$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/yhliu/miniconda3/envs/myenv/bin/python}"
CONDA_SH="${CONDA_SH:-/home/yhliu/miniconda3/etc/profile.d/conda.sh}"
LOG_DIR="$ROOT_DIR/logs/confidence_conditions"
GEN_CONFIG_DIR="$ROOT_DIR/tmp_generated_configs/confidence_conditions"

mkdir -p "$LOG_DIR" "$GEN_CONFIG_DIR"
cd "$ROOT_DIR"
source "$CONDA_SH"
conda activate myenv

MODELS=(
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
)

DATASETS=(
  "HotpotQA"
  "StrategyQA"
  "2WikiMultihopQA"
)

CONDITIONS=(
  "baseline"
  "low_only"
  "mid_only"
  "high_only"
)

GPU_SLOTS=("0" "1" "2" "3")

dataset_slug() {
  case "$1" in
    HotpotQA) echo "hotpotqa" ;;
    StrategyQA) echo "strategyqa" ;;
    2WikiMultihopQA) echo "2wikimultihopqa" ;;
    *)
      echo "unknown dataset: $1" >&2
      return 1
      ;;
  esac
}

build_generated_config() {
  local model="$1"
  local dataset="$2"
  local condition="$3"
  local config_path="$GEN_CONFIG_DIR/${model}_${dataset}_${condition}.json"

  "$PYTHON_BIN" - "$CONF_RAG_DIR" "$model" "$dataset" "$condition" "$config_path" <<'PY'
import json
import os
import sys

conf_rag_dir, model, dataset, condition, output_path = sys.argv[1:]
base_config_path = os.path.join(
    conf_rag_dir,
    "config",
    model,
    dataset,
    "SeqValueNoReftQueryLastSentTrace.json",
)

with open(base_config_path, "r") as f:
    config = json.load(f)

dataset_slug_map = {
    "HotpotQA": "hotpotqa",
    "StrategyQA": "strategyqa",
    "2WikiMultihopQA": "2wikimultihopqa",
}

config["experiment_condition"] = condition
config["use_reflect"] = False
config["disable_retrieval"] = condition == "baseline"
config["save_trace"] = True
config["output_dir"] = os.path.join(
    "results",
    "confidence_conditions",
    condition,
    model,
    "SeqValueNoReft",
    "query_and_last_sentence",
    dataset_slug_map[dataset],
)

if model == "Qwen2.5-7B-Instruct":
    if dataset == "2WikiMultihopQA":
        config["tensor_parallel_size"] = 1
        config["gpu_memory_utilization"] = 0.72
        config["max_num_seqs"] = 32
    else:
        config.setdefault("tensor_parallel_size", 1)
        config.setdefault("gpu_memory_utilization", 0.72)
        config.setdefault("max_num_seqs", 32)

with open(output_path, "w") as f:
    json.dump(config, f, indent=2)
PY

  echo "$config_path"
}

launch_job() {
  local slot="$1"
  local model="$2"
  local dataset="$3"
  local condition="$4"

  local generated_config
  generated_config="$(build_generated_config "$model" "$dataset" "$condition")"

  local dataset_slug_value
  dataset_slug_value="$(dataset_slug "$dataset")"
  local model_slug="${model//./_}"
  local job_name="${model_slug}_${dataset_slug_value}_${condition}"
  local log_path="$LOG_DIR/${job_name}.out"
  local pid_path="$LOG_DIR/${job_name}.pid"

  env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$slot" \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u "$CONF_RAG_DIR/src/main.py" --config-path "$generated_config" \
    > "$log_path" 2>&1 &

  local pid="$!"
  echo "$pid" > "$pid_path"
  printf 'launched %-55s gpu=%s pid=%s log=%s\n' "$job_name" "$slot" "$pid" "$log_path"

  wait "$pid"
}

tasks=()
for condition in "${CONDITIONS[@]}"; do
  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      tasks+=("$model|$dataset|$condition")
    done
  done
done

for slot_index in "${!GPU_SLOTS[@]}"; do
  (
    slot="${GPU_SLOTS[$slot_index]}"
    task_pos="$slot_index"
    while (( task_pos < ${#tasks[@]} )); do
      IFS='|' read -r model dataset condition <<<"${tasks[$task_pos]}"
      launch_job "$slot" "$model" "$dataset" "$condition"
      ((task_pos += ${#GPU_SLOTS[@]}))
    done
  ) &
done

wait

echo "all confidence-condition jobs finished"
