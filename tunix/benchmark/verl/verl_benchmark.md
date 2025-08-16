# veRL Benchmark Report: GRPO + LLaMA3.2-1B on GSM8K

## Experimental Configuration

| Item | Configuration |
|------|---------------|
| **Environment** | GPU H100 × 8 |
| **Framework** | veRL |
| **Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Model** | LLaMA3.2-1B |
| **Task** | GSM8K |
| **Training Steps** | 105 global steps |
| **Training Time** | Approximately 2h10min |
| **Wandb** |  [Wandb log-LLama 3.2 1B](https://wandb.ai/haoyugao-google/verl_grpo_example_gsm8k/runs/w73j5mdv?nw=nwuserhaoyugao)  |

---

## Training Results Analysis

### 1. Convergence Metrics

| Metric | Value | Conclusion |
|--------|-------|------------|
| `actor/lr` | 0.0 | Learning rate decayed to 0, training ended |
| **`actor/pg_loss`** | **-0.00012** | Policy updates are minimal, indicates convergence |
| `actor/pg_clipfrac` | 0.00083 | Almost no updates were clipped, stable policy |
| **`actor/entropy`** | **0.05091** | Low entropy, policy is deterministic |
| `actor/kl_loss` | 0.00329 | Very small KL divergence, minimal deviation from reference policy |

**Conclusion**: Model has converged, policy is stable and learning has mostly stopped.

### 2. Performance Metrics

| Metric | Value | Conclusion |
|--------|-------|------------|
| `val-core/gsm8k/reward/mean@1` | 0.64291 | Final validation reward |
| `critic/rewards/mean` | 0.89863 | Final average reward, quality generation |
| `critic/advantages/mean` | -0.01817 | Advantage is near 0, close to reference policy |
| `critic/returns/mean` | -0.01817 | Return is stable, no drastic change |

**Conclusion**: Training significantly improved model performance.

### 3. Efficiency Metrics

| Metric | Value |
|--------|-------|
| `perf/mfu/actor` | 0.0654 |
| `perf/throughput` | 2,092.46 tokens/s |
| **`perf/time_per_step`** | **88.7s** |
| `timing_s/generate_sequences` | 18.6s |
| `timing_s/update_actor` | 27.3s |

### 4. Input/Output Length

| Metric | Value |
|--------|-------|
| `prompt_length/mean` | 106.77 |
| `response_length/mean` | 183.24 |
| `response_length/clip_ratio` | 0.00293 |

---

## Metric Definitions

### Convergence Metrics
- **`actor/lr`**: Current learning rate of the policy model
- **`actor/pg_loss`**: Policy gradient loss; near-zero means little update
- **`actor/pg_clipfrac`**: Fraction of updates that were clipped
- **`actor/entropy`**: Entropy of the policy distribution; low = deterministic output
- **`actor/kl_loss`**: KL divergence from the reference policy

### Performance Metrics
- **`val-core/gsm8k/reward/mean@1`**: Average reward on the validation set
- **`critic/rewards/mean`**: Average reward of generated samples
- **`critic/advantages/mean`**: Mean advantage; positive = better than baseline
- **`critic/returns/mean`**: Mean return (cumulative discounted reward)

### Efficiency Metrics
- **`perf/mfu/actor`**: GPU Memory-FLOP utilization
- **`perf/throughput`**: Tokens processed per second
- **`perf/time_per_step`**: Wall time per training step
- **`timing_s/generate_sequences`**: Time spent generating model outputs
- **`timing_s/update_actor`**: Time spent updating the actor model

### Length Metrics
- **`prompt_length/mean`**: Average prompt token length
- **`response_length/mean`**: Average generated response length
- **`response_length/clip_ratio`**: Fraction of responses that were truncated




## Test config

```shell
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-1B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='llama3_1b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
```



## Training result log

```
Training Progress: 100%|██████████| 105/105 [2:10:46<00:00, 74.73s/it]
(TaskRunner pid=67763) wandb:                                                                                
(TaskRunner pid=67763) wandb: 
(TaskRunner pid=67763) wandb: Run history:
(TaskRunner pid=67763) wandb:                       actor/entropy █▇▆▄▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                     actor/grad_norm ▁▅▇█▇██████▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▇
(TaskRunner pid=67763) wandb:                       actor/kl_coef ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                       actor/kl_loss ▁▁▂▃▃▄▄▅▅▅▆▆▆▆▇▆▇▇▇▇▇▇▇▇▇▇▇█▇▇▇█████████
(TaskRunner pid=67763) wandb:                            actor/lr ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                   actor/pg_clipfrac ▁▁▇█▆▅▅▆▆▆▅▅▅▅▆▆▅▅▅▅▅▅▅▄▅▆▄▅▄▆▅▄▄▆▅▄▆▄▄▅
(TaskRunner pid=67763) wandb:             actor/pg_clipfrac_lower ▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁▅▁▁▁▄▁▁▆▁▁▁▁▁▁▁▁█▅▁
(TaskRunner pid=67763) wandb:                       actor/pg_loss ▄▆▇█▂▃▆▄▂▅▅▆▆▇▄▂▃▃▆▇▆▄▅▄▆▆▄▄▄▅▄▄▆▃▅▄▁▄▅▄
(TaskRunner pid=67763) wandb:                        actor/ppo_kl ▃█▄▅▅▃▂▂▂▃▃▃▂▁▂▃▂▄▃▃▁▂▃▂▂▁▁▁▂▂▃▂▂▄▄▃▃▅▃▃
(TaskRunner pid=67763) wandb:               critic/advantages/max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:              critic/advantages/mean █▁▂▂▂▃▃▃▂▅▂▃▄▄▃▄▅▄▄▅▃▃▃▃▄▃▄▃▄▃▄▅▄▅▅▆▄▃▅▃
(TaskRunner pid=67763) wandb:               critic/advantages/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                  critic/returns/max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                 critic/returns/mean █▁▁▂▂▃▄▄▃▃▄▃▃▄▄▃▄▅▄▄▃▄▄▃▄▃▄▄▄▄▅▅▃▅▆▆▅▆▅▄
(TaskRunner pid=67763) wandb:                  critic/returns/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                  critic/rewards/max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                 critic/rewards/mean ▁▁▅▆▆▆▇▆▇▇▇▇▇▇▇▇▇▇▇▇████████████████████
(TaskRunner pid=67763) wandb:                  critic/rewards/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                    critic/score/max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                   critic/score/mean ▁▃▄▅▆▆▆▇▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇███████████████
(TaskRunner pid=67763) wandb:                    critic/score/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:          global_seqlen/balanced_max ██▇▇▄▂▁▂▁▁▃▃▃▄▃▂▄▄▅▆▆▅▆▆▇▅▅▆▆▅▇▅▆▅▅▅▅▅▆▄
(TaskRunner pid=67763) wandb:          global_seqlen/balanced_min ██▇▅▄▁▂▁▂▂▄▄▃▄▅▅▅▆▆▆▅▆▆▆▇▅█▆▆▆▅▆▇▅▅▆▅▆▆▅
(TaskRunner pid=67763) wandb:                   global_seqlen/max ▆▂▃▂▁▂▁▅▂▂▂▃▂▂▄▇▅▃▅▄▅▆▆▅▄▇▄▃▅▅▄▃▅▃▄█▄▅▅▄
(TaskRunner pid=67763) wandb:                  global_seqlen/mean ▇█▅▄▄▁▁▂▁▃▄▃▄▂▅▅▅▅▆▆▇▆▅▇█▇▆▆▆▅▇▄▆▄▅▄▅▇▅▇
(TaskRunner pid=67763) wandb:                   global_seqlen/min ▅█▅▆▅▃▄▄▁▅▄▄▅▆▅▆▇▇▅▇▆▄▅▆▅▅▆▇▅▄██▅▇▆▇▆▆▆▄
(TaskRunner pid=67763) wandb:           global_seqlen/minmax_diff ▃▂▃▅▆▅▄▁▃▇▃▂▆▅▆▃▅▅▅█▅▆▆▅▆▅▇▅▅▃▇▅▆▄▃▂▄▄▅▂
(TaskRunner pid=67763) wandb:             perf/cpu_memory_used_gb ▄▁▂▃▃▄▂▁▆▃▇▅▂▇█▅▂▂▃▃▃█▇▂▅▆▇▆▆▇▃▄▆▄▇█▇▇▇▇
(TaskRunner pid=67763) wandb:        perf/max_memory_allocated_gb ▁▄▄▄▄▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅███
(TaskRunner pid=67763) wandb:         perf/max_memory_reserved_gb ▁▅██████████████████████████████████████
(TaskRunner pid=67763) wandb:                      perf/mfu/actor ▅█▆▄▂▃▁▂▅▅▂▅▃▄▃▅▅▅▅▅▅▆▆▆▄▇▅▆▆▇▅▆▆▅▄▅▄▅▇▅
(TaskRunner pid=67763) wandb:                     perf/throughput █▇▆▇▇▆▆▁▆▆▆▆▇▇▇▆▂▇▆▂▇▇▂▇██▇▂▇▇▇▂▇▇█▆▇▇▆▁
(TaskRunner pid=67763) wandb:                  perf/time_per_step ▃▁▂▂▁█▁▂▁▃▂▂█▂▂▁▇▂▂▂▁▁▂▂▂▂▂▆▂▂▂▂▂▂▂▂▂▁▃▂
(TaskRunner pid=67763) wandb:               perf/total_num_tokens █▆█▃▄▁▁▃▃▂▄▃▅▄▅▅▅▆▆▆▆▇▅▅▅▆▆▅▇▆▅▅▄▄▄▆▆▅▅▄
(TaskRunner pid=67763) wandb:            prompt_length/clip_ratio ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                   prompt_length/max ▄▄▃▃▁▄▄▅▃▅▂▅▃▄▅█▅▂▅▂▅▄█▅▅▃▂▄█▅▄▅▂▂▅▄▁▄▅▅
(TaskRunner pid=67763) wandb:                  prompt_length/mean ▃▅▃▄▆▄▃▇▅▃▄▃▄▆▁▄▅▄▅▄▅▅▄▅▄▃▇▃▅▄▆▆▄█▅▂▆▅▄▄
(TaskRunner pid=67763) wandb:                   prompt_length/min ▆▅▇▅▅▅▆▁▅▆▆▅█▆▆▇▁▆█▅█▁▇▆▆▇▆▅▅▇▁▇▁▆▅▆▁▇▇▆
(TaskRunner pid=67763) wandb:          response_length/clip_ratio ▃█▃▂▁▁▂▁▁▁▁▂▁▂▁▂▂▁▂▂▄▄▂▅▂▄▂▂▄▃▄▄▃▄▆▄▃▅▃▄
(TaskRunner pid=67763) wandb:                 response_length/max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=67763) wandb:                response_length/mean ▅▄▄▅▁▁▂▃▄▄▃▄▅▃▅▅▅▅▆▇▆▆█▇▆▆▇▇▇▇▅█▅▆▆▅██▇▅
(TaskRunner pid=67763) wandb:                 response_length/min ▁▃▅▄▂▃▅▄▆▆▅▆▆▆▄▅▅▅▄▆▅▆▄▄▆▄▃▆▃▆█▇▆█▆▇▇▆▇▅
(TaskRunner pid=67763) wandb:             timing_per_token_ms/adv ▁▂▂▂▂▁▁▂▁▁▁▂▁█▂▁▁▂▁▁▁▁▁▁▂▁▁▁▁▁▁▂▁▂▁▂▂▁▂▁
(TaskRunner pid=67763) wandb:             timing_per_token_ms/gen ▂▁▂▆▂▅▅▇▆▂▅▆▄▄▅▃▄▇▇▅▄▁▅▄▇▃▆▃▁▇█▄▄▃▄▇█▅▄▄
(TaskRunner pid=67763) wandb:             timing_per_token_ms/ref ▁▂▄▅▄▆█▇▇▇▅▆▅▆▅▇▃▄▄▃▃▂▃▃▄▃▂▃▃▃▄▃▂▂▄▄▄▃▂▄
(TaskRunner pid=67763) wandb:    timing_per_token_ms/update_actor ▄▇█▇▇▄▅▇▇█▇▆▅▄▅▅▆▅▄▂▃▃▂▃▃▅▁▂▄▃▄▃▃▅▄▅▄▃▄▄
(TaskRunner pid=67763) wandb:                        timing_s/adv ▁▂▂▁▂▁▁▂▂▂▂▁▁█▁▁▁▂▁▁▁▁▂▁▂▁▁▁▁▂▁▁▁▂▁▁▁▁▁▂
(TaskRunner pid=67763) wandb:                        timing_s/gen ▃▃▃▂▃▃▄▂▇▃▁▃▅▃▃▇▅▇█▂▃▅▇▄▆▁▄▄▇▃▇▂▇▆▆▄▄█▄▅
(TaskRunner pid=67763) wandb:         timing_s/generate_sequences ▅▃▃▅▃▃▃▁▂▃▃▄▄▄▅▅▃▇▅▅▆▆▅▇▆▆▄▄▇▅▄▄▄▆▇▇▅█▅▇
(TaskRunner pid=67763) wandb:               timing_s/old_log_prob ▆▅▅▂▆▆▆▃▄▆▁▂▆▇▅▂▆▆▆▃▂▅▆█▃▆▅▃▂▅▃▅▄▄▁▅▄▃▂▃
(TaskRunner pid=67763) wandb:                        timing_s/ref ▇▆▅▁▃█▆▆▄▃▆▆▆▆▆▆▅▅▆▅▇▄▆▆▆▆▄▆▆▅█▅▃▇▃▆▅▆▅▇
(TaskRunner pid=67763) wandb:                    timing_s/reshard ▁▁▃▃▃▃▄▁▃▃▂▃▃▁▃▁▃▃▃▅▅▃▃▃▅▃▂▃▂▃▃▇▄▂█▃▄▁▃▃
(TaskRunner pid=67763) wandb:                     timing_s/reward ▃▂▂▂▁▁▁▁▂▂▁▂▁▁▂▂▂▃█▃▂▂▁▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂
(TaskRunner pid=67763) wandb:            timing_s/save_checkpoint █▃▃▂▁▅
(TaskRunner pid=67763) wandb:              timing_s/start_profile ▁▁▁█▁▁▁▁▁▁▁▁▁▁▇▁▂█▁▁▁▂▁▁▁▁▂▁▁▁▁▁▁▁█▁▁▁▂▂
(TaskRunner pid=67763) wandb:                       timing_s/step ▃▁▆▁▇▁▁▂▂█▁▂▂▇▂▁▂▁▂▂▂▂▂▇▂▂▆▂▂▁▂▁▇▂▆▂▂█▂█
(TaskRunner pid=67763) wandb:               timing_s/stop_profile █▂▂▃█▂▄▃▃▇▅▃▂▂▂▂▂▂▁▃▂▂▄▁▁▃▁▁▂▇▂▇▁▂▂▂▃▂▅▇
(TaskRunner pid=67763) wandb:                    timing_s/testing ▁▄▅▄█▆▄▂▄▃▄▆▃▆▄▂▄▂▃▅▆
(TaskRunner pid=67763) wandb:               timing_s/update_actor ▆▄▃▅▇█▅▄▂▆▇▃▃▄▅▆▃▃▅▆▆▂▂▃▂▄▃▂▃▁▃▃▂▄▁▄▃▂▃▄
(TaskRunner pid=67763) wandb:                      training/epoch ▁▁▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▆▆▇▇▇▇▇▇▇██
(TaskRunner pid=67763) wandb:                training/global_step ▁▁▁▁▁▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇▇████
(TaskRunner pid=67763) wandb: val-core/openai/gsm8k/reward/mean@1 ▁▅▇▇█▇████████████████
(TaskRunner pid=67763) wandb: 
(TaskRunner pid=67763) wandb: Run summary:
(TaskRunner pid=67763) wandb:                       actor/entropy 0.05091
(TaskRunner pid=67763) wandb:                     actor/grad_norm 0.44454
(TaskRunner pid=67763) wandb:                       actor/kl_coef 0.001
(TaskRunner pid=67763) wandb:                       actor/kl_loss 0.00329
(TaskRunner pid=67763) wandb:                            actor/lr 0.0
(TaskRunner pid=67763) wandb:                   actor/pg_clipfrac 0.00083
(TaskRunner pid=67763) wandb:             actor/pg_clipfrac_lower 0
(TaskRunner pid=67763) wandb:                       actor/pg_loss -0.00012
(TaskRunner pid=67763) wandb:                        actor/ppo_kl 0.00016
(TaskRunner pid=67763) wandb:               critic/advantages/max 1.78885
(TaskRunner pid=67763) wandb:              critic/advantages/mean -0.01817
(TaskRunner pid=67763) wandb:               critic/advantages/min -1.78885
(TaskRunner pid=67763) wandb:                  critic/returns/max 1.78885
(TaskRunner pid=67763) wandb:                 critic/returns/mean -0.01817
(TaskRunner pid=67763) wandb:                  critic/returns/min -1.78885
(TaskRunner pid=67763) wandb:                  critic/rewards/max 1
(TaskRunner pid=67763) wandb:                 critic/rewards/mean 0.89863
(TaskRunner pid=67763) wandb:                  critic/rewards/min 0
(TaskRunner pid=67763) wandb:                    critic/score/max 1
(TaskRunner pid=67763) wandb:                   critic/score/mean 0.89863
(TaskRunner pid=67763) wandb:                    critic/score/min 0
(TaskRunner pid=67763) wandb:          global_seqlen/balanced_max 185611
(TaskRunner pid=67763) wandb:          global_seqlen/balanced_min 185610
(TaskRunner pid=67763) wandb:                   global_seqlen/max 192222
(TaskRunner pid=67763) wandb:                  global_seqlen/mean 185610.375
(TaskRunner pid=67763) wandb:                   global_seqlen/min 179276
(TaskRunner pid=67763) wandb:           global_seqlen/minmax_diff 12946
(TaskRunner pid=67763) wandb:             perf/cpu_memory_used_gb 103.69083
(TaskRunner pid=67763) wandb:        perf/max_memory_allocated_gb 45.7422
(TaskRunner pid=67763) wandb:         perf/max_memory_reserved_gb 52.69922
(TaskRunner pid=67763) wandb:                      perf/mfu/actor 0.0654
(TaskRunner pid=67763) wandb:                     perf/throughput 2092.46198
(TaskRunner pid=67763) wandb:                  perf/time_per_step 88.7043
(TaskRunner pid=67763) wandb:               perf/total_num_tokens 1484883
(TaskRunner pid=67763) wandb:            prompt_length/clip_ratio 0
(TaskRunner pid=67763) wandb:                   prompt_length/max 189
(TaskRunner pid=67763) wandb:                  prompt_length/mean 106.77441
(TaskRunner pid=67763) wandb:                   prompt_length/min 69
(TaskRunner pid=67763) wandb:          response_length/clip_ratio 0.00293
(TaskRunner pid=67763) wandb:                 response_length/max 1024
(TaskRunner pid=67763) wandb:                response_length/mean 183.24179
(TaskRunner pid=67763) wandb:                 response_length/min 54
(TaskRunner pid=67763) wandb:             timing_per_token_ms/adv 9e-05
(TaskRunner pid=67763) wandb:             timing_per_token_ms/gen 0.02313
(TaskRunner pid=67763) wandb:             timing_per_token_ms/ref 0.00876
(TaskRunner pid=67763) wandb:    timing_per_token_ms/update_actor 0.01841
(TaskRunner pid=67763) wandb:                        timing_s/adv 0.13354
(TaskRunner pid=67763) wandb:                        timing_s/gen 21.70104
(TaskRunner pid=67763) wandb:         timing_s/generate_sequences 18.6025
(TaskRunner pid=67763) wandb:               timing_s/old_log_prob 7.86395
(TaskRunner pid=67763) wandb:                        timing_s/ref 13.00837
(TaskRunner pid=67763) wandb:                    timing_s/reshard 0.30626
(TaskRunner pid=67763) wandb:                     timing_s/reward 0.87269
(TaskRunner pid=67763) wandb:            timing_s/save_checkpoint 3.01086
(TaskRunner pid=67763) wandb:              timing_s/start_profile 0.00014
(TaskRunner pid=67763) wandb:                       timing_s/step 88.7043
(TaskRunner pid=67763) wandb:               timing_s/stop_profile 7e-05
(TaskRunner pid=67763) wandb:                    timing_s/testing 14.64922
(TaskRunner pid=67763) wandb:               timing_s/update_actor 27.33919
(TaskRunner pid=67763) wandb:                      training/epoch 14
(TaskRunner pid=67763) wandb:                training/global_step 105
(TaskRunner pid=67763) wandb: val-core/openai/gsm8k/reward/mean@1 0.64291
(TaskRunner pid=67763) wandb: 
(TaskRunner pid=67763) wandb: 🚀 View run llama3_1b_function_rm at: https://wandb.ai/haoyugao-google/verl_grpo_example_gsm8k/runs/w73j5mdv
(TaskRunner pid=67763) wandb: ⭐️ View project at: https://wandb.ai/haoyugao-google/verl_grpo_example_gsm8k
(TaskRunner pid=67763) wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
(TaskRunner pid=67763) wandb: Find logs at: ./wandb/run-20250807_063228-w73j5mdv/logs
(WorkerDict pid=79586) INFO:2025-08-07 08:43:28,409:[Rank 2] Saved model to /workspace/verl_space/verl/examples/grpo_trainer/checkpoints/verl_grpo_example_gsm8k/llama3_1b_function_rm/global_step_105/actor/model_world_size_8_rank_2.pt [repeated 7x across cluster]
(WorkerDict pid=79586) INFO:2025-08-07 08:43:29,816:[Rank 2] Saved optim to /workspace/verl_space/verl/examples/grpo_trainer/checkpoints/verl_grpo_example_gsm8k/llama3_1b_function_rm/global_step_105/actor/optim_world_size_8_rank_2.pt [repeated 7x across cluster]
(WorkerDict pid=79586) INFO:2025-08-07 08:43:29,817:[Rank 2] Saved extra_state to /workspace/verl_space/verl/examples/grpo_trainer/checkpoints/verl_grpo_example_gsm8k/llama3_1b_function_rm/global_step_105/actor/extra_state_world_size_8_rank_2.pt [repeated 7x across cluster]
```

