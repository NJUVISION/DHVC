#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}
result_root=${RESULTS_ROOT:-"$project_root/results"}
model=${MODEL_PATH:-"$project_root/weights/model.pth"}
: "${CLIC_ROOT:?set CLIC_ROOT}"
: "${KODAK_ROOT:?set KODAK_ROOT}"
: "${TECNICK_ROOT:?set TECNICK_ROOT}"
mkdir -p "$result_root"

pids=()
for shard in 0 1; do
  "$python_bin" -u "$project_root/eval.py" --model "$model" \
    --dataset clic --root "$CLIC_ROOT" --device "cuda:$shard" \
    --num-shards 2 --shard-id "$shard" \
    --output "$result_root/clic_${shard}.jsonl" \
    >"$result_root/clic_${shard}.log" 2>&1 & pids+=("$!")
done
for shard in 0 1; do
  gpu=$((shard + 2))
  "$python_bin" -u "$project_root/eval.py" --model "$model" \
    --dataset kodak --root "$KODAK_ROOT" --device "cuda:$gpu" \
    --num-shards 2 --shard-id "$shard" \
    --output "$result_root/kodak_${shard}.jsonl" \
    >"$result_root/kodak_${shard}.log" 2>&1 & pids+=("$!")
done
for shard in 0 1 2 3; do
  gpu=$((shard + 4))
  "$python_bin" -u "$project_root/eval.py" --model "$model" \
    --dataset tecnick --root "$TECNICK_ROOT" --device "cuda:$gpu" \
    --num-shards 4 --shard-id "$shard" \
    --output "$result_root/tecnick_${shard}.jsonl" \
    >"$result_root/tecnick_${shard}.log" 2>&1 & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for dataset in clic kodak tecnick; do
  cat "$result_root/${dataset}_"*.jsonl >"$result_root/${dataset}.jsonl"
  "$python_bin" "$project_root/score.py" "$result_root/${dataset}.jsonl" \
    --output "$result_root/${dataset}_score.json"
done
