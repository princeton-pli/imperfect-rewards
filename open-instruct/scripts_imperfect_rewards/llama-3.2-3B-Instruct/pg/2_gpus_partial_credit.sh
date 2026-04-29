#!/bin/bash

CONSTRAINTS_STR="${1:?Missing constraints arg. Example: \"count:pronouns;custom:multiples\"}"
LEARNING_RATE="${2:-1e-6}"

IFS=';' read -r -a CONSTRAINTS <<< "$CONSTRAINTS_STR"

SLUGS=()
for c in "${CONSTRAINTS[@]}"; do
  SLUGS+=("${c//:/_}")
done

PAIR_SLUG="$(printf "%s__" "${SLUGS[@]}")"
PAIR_SLUG="${PAIR_SLUG%__}"   # e.g. count_pronouns__custom_multiples

EXP_NAME="$PAIR_SLUG"
RUN_DATE="$(date +%m_%d_%Y)"
OUT_DIR="pg_vr/llama3B_instruct/${PAIR_SLUG}/${RUN_DATE}/partial_credit/"
mkdir -p "$OUT_DIR"

TRAIN_DS="<YOUR_OUTPUT_DATA_DIR>/${PAIR_SLUG}__train_prefs/dataset.jsonl"
EVAL_DS="<YOUR_OUTPUT_DATA_DIR>/${PAIR_SLUG}__test_prefs/dataset.jsonl"

VERIFIER_WEIGHTS_JSON="$(
python - "$CONSTRAINTS_STR" <<'PY'
import json, sys
constraints = sys.argv[1].split(";")
w = 1.0 / len(constraints)
print(json.dumps({c: w for c in constraints}))
PY
)"

echo "Pre-building DeepSpeed cpu_adam extension..."
python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load()" || { echo "FATAL: cpu_adam build failed"; exit 1; }

python open_instruct/grpo_fast.py \
  --exp_name "$EXP_NAME" \
  --output_dir "$OUT_DIR" \
  --beta 0.0 \
  --num_unique_prompts_rollout 16 \
  --num_samples_per_prompt_rollout 64 \
  --temperature 1 \
  --num_mini_batches 1 \
  --kl_estimator 2 \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --dataset_cache_mode local \
  --dataset_mixer_list "$TRAIN_DS" 4000 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list "$EVAL_DS" 50 \
  --dataset_mixer_eval_list_splits train \
  --dataset_config_seed 42 \
  --max_prompt_token_length 512 \
  --response_length 512 \
  --pack_length 1024 \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --apply_verifiable_reward True \
  --verifier_weights "$VERIFIER_WEIGHTS_JSON" \
  --verification_reward 1 \
  --non_stop_penalty False \
  --total_episodes 2048000 \
  --deepspeed_stage 2 \
  --per_device_train_batch_size 8 \
  --single_gpu_mode False \
  --vllm_sync_backend nccl \
  --num_learners_per_node 1 \
  --num_epochs 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_num_engines 1 \
  --vllm_enforce_eager True \
  --seed 1 \
  --local_eval_every 100 \
  --gradient_checkpointing \
  --with_tracking True \
  --tensorboard_project_name "${EXP_NAME}_weighted" \
  --push_to_hub False \
  --vllm_gpu_memory_utilization 0.75 \
  --save_freq -1 \
  --eval_on_step_0 True \
  --save_traces True \
  --save_traces_freq 1 \
  --save_traces_generations False \
  --filter_zero_std_samples False \
  --use_vllm_logprobs False \
  --enable_thinking False \
  --load_ref_policy False \
  --deepspeed_offload_optimizer True \