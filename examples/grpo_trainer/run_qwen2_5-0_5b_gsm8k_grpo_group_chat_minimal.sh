#!/bin/bash

set -euo pipefail
set -x

# Minimal GRPO + group-chat example (single GPU)
# 1) Preprocess GSM8K:
#    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k"
# 2) Run:
#    DATA_DIR="$HOME/data/gsm8k" bash examples/grpo_trainer/run_qwen2_5-0_5b_gsm8k_grpo_group_chat_minimal.sh

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.agent_loop.group_chat.GroupChatAgentLoopManager \
    actor_rollout_ref.rollout.custom.group_chat.enable=True \
    actor_rollout_ref.rollout.custom.group_chat.num_rounds=2 \
    actor_rollout_ref.rollout.custom.group_chat.strip_think=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_gsm8k_group_chat_minimal' \
    trainer.experiment_name='qwen2_5_0_5b_group_chat' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    trainer.save_freq=0 \
    "$@"
