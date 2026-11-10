#!/usr/bin/env bash
# Stage 0 sequence inside the container. Finished runs are skipped on restart.
set -u

python scripts/stage0_benchmark.py

run_eval() {
    local suite=$1 steps=$2 episodes=$3
    local out="results/stage0/eval_${suite}_n${steps}"
    if [ -f "$out/eval_info.json" ]; then
        echo "skip $out"
        return
    fi
    rm -rf "$out"
    python scripts/stage0_eval.py \
        --policy.path=HuggingFaceVLA/smolvla_libero \
        --policy.n_action_steps="$steps" \
        --env.type=libero \
        --env.task="$suite" \
        --env.task_ids='[0,1,2,3,4]' \
        --env.observation_height=256 \
        --env.observation_width=256 \
        --eval.batch_size=1 \
        --eval.n_episodes="$episodes" \
        --eval.use_async_envs=false \
        --seed=1000 \
        --output_dir="$out"
    echo "finished $out exit=$?"
}

run_eval libero_spatial 10 2
run_eval libero_object 10 2
run_eval libero_spatial 1 1
run_eval libero_object 1 1
