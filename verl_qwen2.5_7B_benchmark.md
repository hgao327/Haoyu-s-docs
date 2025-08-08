# veRL Benchmark Report: GRPO + Qwen2.5-7B on GSM8K

## Experimental Configuration

| Item | Configuration |
|------|---------------|
| **Environment** | GPU H100 × 8 |
| **Framework** | veRL |
| **Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Model** | Qwen2.5-7B Function RM |
| **Task** | GSM8K |
| **Training Steps** | 105 global steps |
| **Training Time** | Approximately 2h25min |
| **Wandb** | [6icl7pbz]() |

---

## Training Results Analysis

### 1. Convergence Metrics

| Metric | Value | Conclusion |
|--------|-------|------------|
| `actor/lr` | 0.0 | Learning rate decayed to 0, training ended |
| `actor/pg_loss` | -0.00265 | Policy updates are minimal, indicates convergence |
| `actor/pg_clipfrac` | 0.00015 | Almost no updates were clipped, stable policy |
| `actor/entropy` | 0.06955 | Low entropy, policy is deterministic |
| `actor/kl_loss` | 0.01276 | Small KL divergence, minimal deviation from reference policy |

**Conclusion**: Model has converged, policy is stable and learning has mostly stopped.

### 2. Performance Metrics

| Metric | Value | Conclusion |
|--------|-------|------------|
| `val-core/openai/gsm8k/reward/mean@1` | 0.92722 | |
| `critic/rewards/mean` | 0.9793 | exceptional generation quality |
| `critic/advantages/mean` | -0.00064 | Advantage is near 0, close to reference policy |
| `critic/returns/mean` | -0.00064 | Return is stable, no drastic change |

**Conclusion**: Training achieved outstanding model performance with 92.7% validation accuracy.

### 3. Efficiency Metrics

| Metric | Value |
|--------|-------|
| `perf/mfu/actor` | 0.34695 |
| `perf/throughput` | 1,811.23 tokens/s |
| `perf/time_per_step` | 114.85s |
| `timing_s/generate_sequences` | 26.73s |
| `timing_s/update_actor` | 29.14s |

### 4. Input/Output Length

| Metric | Value |
|--------|-------|
| `prompt_length/mean` | 92.98 |
| `response_length/mean` | 232.04 |
| `response_length/clip_ratio` | 0.0002 |

---

## Metric Definitions

### Convergence Metrics
- **`actor/lr`**: Current learning rate of the policy model
- **`actor/pg_loss`**: Policy gradient loss; near-zero means little update
- **`actor/pg_clipfrac`**: Fraction of updates that were clipped
- **`actor/entropy`**: Entropy of the policy distribution; low = deterministic output
- **`actor/kl_loss`**: KL divergence from the reference policy

### Performance Metrics
- **`val-core/openai/gsm8k/reward/mean@1`**: Average reward on the validation set
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
```("Final validation metrics: {'val-core/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=132393)  '0.9272175890826384}')
Training Progress: 100%|██████████| 105/105 [2:25:28<00:00, 83.13s/it]
(TaskRunner pid=132393) wandb:                                                                                
(TaskRunner pid=132393) wandb: 
(TaskRunner pid=132393) wandb: Run history:
(TaskRunner pid=132393) wandb:                       actor/entropy █▆▆▄▄▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                     actor/grad_norm █▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                       actor/kl_coef ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                       actor/kl_loss ▁▂▄▄▅▅▅▆▆▆▆▆▆▆▆▇▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇█▇▇▇▇▇█▇█
(TaskRunner pid=132393) wandb:                            actor/lr ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                   actor/pg_clipfrac █▁▁▁▁▁▁▁▂▁▁▂▂▁▂▁▁▂▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:             actor/pg_clipfrac_lower █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                       actor/pg_loss █▃▁▃▄▃▂▁▂▃▃▂▂▃▃▂▃▃▃▂▃▂▃▃▄▃▂▃▂▃▂▂▂▂▂▃▃▂▂▂
(TaskRunner pid=132393) wandb:                        actor/ppo_kl █▂▃▂▂▂▃▃▂▂▁▃▃▂▂▃▃▄▅▃▃▂▄▃▂▂▃▄▄▃▄▃▃▃▂▃▃▃▂▃
(TaskRunner pid=132393) wandb:               critic/advantages/max ████████████████████▁███████████████▁███
(TaskRunner pid=132393) wandb:              critic/advantages/mean ▄▁▇█████████████████████████████████████
(TaskRunner pid=132393) wandb:               critic/advantages/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                  critic/returns/max ███████████████████████▁████████▁███████
(TaskRunner pid=132393) wandb:                 critic/returns/mean ▁▃▇▇████████████████████████████████████
(TaskRunner pid=132393) wandb:                  critic/returns/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                  critic/rewards/max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                 critic/rewards/mean ▁▅▆▇████████████████████████████████████
(TaskRunner pid=132393) wandb:                  critic/rewards/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                    critic/score/max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                   critic/score/mean ▁▄▇█████████████████████████████████████
(TaskRunner pid=132393) wandb:                    critic/score/min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:          global_seqlen/balanced_max █▄▂▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▃▃▃▃▄▄▄▄▃▄▃▃▃▃▃▃▃▃▃▃
(TaskRunner pid=132393) wandb:          global_seqlen/balanced_min █▆▄▁▁▂▂▂▂▃▃▃▃▃▃▃▄▃▃▄▃▃▃▃▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃
(TaskRunner pid=132393) wandb:                   global_seqlen/max █▄▂▁▂▂▂▂▃▃▃▃▃▃▃▃▄▃▃▄▃▄▄▃▄▄▄▄▄▃▄▃▄▃▃▃▃▃▄▃
(TaskRunner pid=132393) wandb:                  global_seqlen/mean █▆▂▁▁▂▂▂▃▃▃▃▄▄▄▃▃▃▃▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▄▃
(TaskRunner pid=132393) wandb:                   global_seqlen/min █▃▁▁▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▃▃▃▃▃▃▄▃▃▃▃▂▂▃▃▂▂▂▂▂
(TaskRunner pid=132393) wandb:           global_seqlen/minmax_diff ▄▆▂▁▃▃▃▃▃▄▃▄▄▄▃▄▄▆▂▃▆▇▃▅▄▅▃▄▄▃▃▄▁▄█▄▂▆▅▃
(TaskRunner pid=132393) wandb:             perf/cpu_memory_used_gb ▃▂▁▂▂▂▃▂▄▄▄▄▄▅▄▄▅▅▅▄▅▅▅▅█▅▆▆▅▅▅▆▅▆▆▆▆▆▆▆
(TaskRunner pid=132393) wandb:        perf/max_memory_allocated_gb ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:         perf/max_memory_reserved_gb ▁▅▅▅▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇█████████████████
(TaskRunner pid=132393) wandb:                      perf/mfu/actor ▃▁▂▇▇▆▆▇▇█▆█▆██▇█▇▇▇▇▇▇▆█▇▇▇▇▇▇▇▇█▇▇▇▆▆█
(TaskRunner pid=132393) wandb:                     perf/throughput ▆▆▃▆▇▇▇▆▇█▆▃▇▁▆▄▆▇▄▆▇▇▇▇▇▄▇▆▄▇▆▇▇▃▇▃▇▆▇▆
(TaskRunner pid=132393) wandb:                  perf/time_per_step ▅▁▁▂▇▆▁▂▅▂▂▂▂▃▂▃▂▂▅▂▂▃▆▃▂▂▃▃▅▂█▂▂▆▂▂▅▂▂▃
(TaskRunner pid=132393) wandb:               perf/total_num_tokens ▃▂▁▂▃▃▃▃▄▄▅▅▅▅▅▇▇███▇▇▇▇▇▇▇█▇▇▅▆▇▆▆▆▆▅▆▆
(TaskRunner pid=132393) wandb:            prompt_length/clip_ratio ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                   prompt_length/max ▃▆▄▄█▆▆▃▃▆█▄▆▆█▄▆▃▂▆▃▆▆█▂▅█▄▂▄▅▄▆▃█▂▆▆▅▁
(TaskRunner pid=132393) wandb:                  prompt_length/mean ▇▄▄▄▆▅▃▆▅▃▄▇▇▁▇▆▄▅▆▆▆▆▄▅▄▅█▆▆▃▂▇▇▆▆▂▄▅▅█
(TaskRunner pid=132393) wandb:                   prompt_length/min ▆▆▅▇▇▆▁▅▆▅█▅▇▇▆▇▁▅▆▅▅▆▇▅▅▇▅▆▅▇▅▆▆▇▇▆▆▆▆▆
(TaskRunner pid=132393) wandb:          response_length/clip_ratio █▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
(TaskRunner pid=132393) wandb:                 response_length/max █████▁▂▇▄▂▂██▆▆▄▇▂▅███▃▃████▂█▂▂████▁██▂
(TaskRunner pid=132393) wandb:                response_length/mean █▅▁▁▁▂▂▂▃▃▃▃▄▄▅▅▅▅▅▅▄▄▄▄▅▅▅▅▄▄▄▄▄▄▄▄▃▄▄▄
(TaskRunner pid=132393) wandb:                 response_length/min ▁▁▁▁▅▆▆▅▆▅▇▆▆█▇▇█▆▆█▇▇▆▇▇▆▇▆▇█▆▆▇▆▆█▇▇▇▇
(TaskRunner pid=132393) wandb:             timing_per_token_ms/adv ▁▆▁▂▂▂▁▂█▁▂▂▇▁▁▁▁▂▁▁▁▂▁▁▂▂▁▂▁▂▁▂▁▇▂▂▂▂▂▂
(TaskRunner pid=132393) wandb:             timing_per_token_ms/gen ▅▅▃▅█▅▅▁▆▆▆▃▅▆▂▄▇▄▃▄▄▅▅█▃▆▄▇▆▅▆▄▆▅▆█▅▅▇▄
(TaskRunner pid=132393) wandb:             timing_per_token_ms/ref ▄▆█▇▆▅▅▆▅▅▅▅▅▄▄▃▃▃▄▃▃▁▃▄▄▄▃▃▄▂▃▃▃▄▄▃▃▅▃▅
(TaskRunner pid=132393) wandb:    timing_per_token_ms/update_actor ▆█▅▄▂▂▃▂▂▃▃▂▂▂▁▂▃▁▂▂▂▂▂▃▂▂▂▁▂▁▂▁▂▂▂▂▂▃▃▂
(TaskRunner pid=132393) wandb:                        timing_s/adv ▁▁▂▁▂█▁▁▁▂▁▁▁▁▁▂▁▂▁▁▁▂▁▁▂▁▁▁▁▁▁▂▂▁▁▂▂▂▁▂
(TaskRunner pid=132393) wandb:                        timing_s/gen ▅▄▁▁▃▄▄▄▅▃▄▄▆▄▅▅▆▇▂▇▃▆▃▄▄█▄▃▆▃▅▄█▃▆▆▅▄▅▄
(TaskRunner pid=132393) wandb:         timing_s/generate_sequences █▆▄▁▁▁▂▁▂▂▁▃▂▅▃▃▂▅▃▃▄▃▄▃▃▂▄▃▃▃▃▃▅▄▃▄▃▃▃▄
(TaskRunner pid=132393) wandb:               timing_s/old_log_prob █▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▃▂▂▂▃▃▂▂▃▂▃▃▃▂▂▂▂▂▂▂▂▂▂▂
(TaskRunner pid=132393) wandb:                        timing_s/ref █▂▁▁▁▂▂▂▃▃▃▃▃▄▃▄▃▃▄▄▄▄▄▄▄▄▄▄▄▄▃▄▄▃▃▅▄▃▄▄
(TaskRunner pid=132393) wandb:                    timing_s/reshard ▅▃▃▃▃▃▅▃▆▅▃▁▄▄▄▄▅▅▇▄▇▁▆▁▆█▇▇▇▅▅▄▆▄▅▆▆▆▆▆
(TaskRunner pid=132393) wandb:                     timing_s/reward █▆▄▃▁▂▂▃▂▃▃▂▄▃▃▄▅▃▃▄▃▄▄▄▄▃▄▅▄▄▃▂▄▃▃▃▂▄▄▃
(TaskRunner pid=132393) wandb:            timing_s/save_checkpoint ▄▄▁█▅▆
(TaskRunner pid=132393) wandb:              timing_s/start_profile ▁▁▁▁▁▇▁▁▁▁▁▁▇▂▁▁▁▁█▁▁█▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▂▂▁
(TaskRunner pid=132393) wandb:                       timing_s/step ▂▄▁▁▁▄▁▁▁▁▅▁▂▂▅▂▂█▃▅▃▂▂▂▂▃▂▂▄▂▂▂▇▂▂▅▂▂▂▇
(TaskRunner pid=132393) wandb:               timing_s/stop_profile ▃▂█▁▁▃▃▆▂▃▂▁▄▂▅▁▂▁▂▂▂▂▂▃▂▁▁▁▅▁▁▂▄▂▂▂▅▁▇▁
(TaskRunner pid=132393) wandb:                    timing_s/testing ▁▇▅▁▆▇██▃▁▃▄▂▂▁▁▆▆▃▆▂
(TaskRunner pid=132393) wandb:               timing_s/update_actor █▅▂▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▃▄▄▄▄▄▄▄▃▄▃▃▃▃▃▃▃▃▃▄▃
(TaskRunner pid=132393) wandb:                      training/epoch ▁▁▁▁▁▂▂▃▃▃▃▃▃▃▃▄▅▅▅▅▅▅▅▅▅▆▇▇▇▇▇▇▇▇▇▇████
(TaskRunner pid=132393) wandb:                training/global_step ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇▇██
(TaskRunner pid=132393) wandb: val-core/openai/gsm8k/reward/mean@1 ▁█████████████████████
(TaskRunner pid=132393) wandb: 
(TaskRunner pid=132393) wandb: Run summary:
(TaskRunner pid=132393) wandb:                       actor/entropy 0.06955
(TaskRunner pid=132393) wandb:                     actor/grad_norm 0.07587
(TaskRunner pid=132393) wandb:                       actor/kl_coef 0.001
(TaskRunner pid=132393) wandb:                       actor/kl_loss 0.01276
(TaskRunner pid=132393) wandb:                            actor/lr 0.0
(TaskRunner pid=132393) wandb:                   actor/pg_clipfrac 0.00015
(TaskRunner pid=132393) wandb:             actor/pg_clipfrac_lower 0
(TaskRunner pid=132393) wandb:                       actor/pg_loss -0.00265
(TaskRunner pid=132393) wandb:                        actor/ppo_kl 7e-05
(TaskRunner pid=132393) wandb:               critic/advantages/max 1.78885
(TaskRunner pid=132393) wandb:              critic/advantages/mean -0.00064
(TaskRunner pid=132393) wandb:               critic/advantages/min -1.78885
(TaskRunner pid=132393) wandb:                  critic/returns/max 1.78885
(TaskRunner pid=132393) wandb:                 critic/returns/mean -0.00064
(TaskRunner pid=132393) wandb:                  critic/returns/min -1.78885
(TaskRunner pid=132393) wandb:                  critic/rewards/max 1
(TaskRunner pid=132393) wandb:                 critic/rewards/mean 0.9793
(TaskRunner pid=132393) wandb:                  critic/rewards/min 0
(TaskRunner pid=132393) wandb:                    critic/score/max 1
(TaskRunner pid=132393) wandb:                   critic/score/mean 0.9793
(TaskRunner pid=132393) wandb:                    critic/score/min 0
(TaskRunner pid=132393) wandb:          global_seqlen/balanced_max 208012
(TaskRunner pid=132393) wandb:          global_seqlen/balanced_min 208011
(TaskRunner pid=132393) wandb:                   global_seqlen/max 212647
(TaskRunner pid=132393) wandb:                  global_seqlen/mean 208011.875
(TaskRunner pid=132393) wandb:                   global_seqlen/min 202019
(TaskRunner pid=132393) wandb:           global_seqlen/minmax_diff 10628
(TaskRunner pid=132393) wandb:             perf/cpu_memory_used_gb 87.73482
(TaskRunner pid=132393) wandb:        perf/max_memory_allocated_gb 77.04508
(TaskRunner pid=132393) wandb:         perf/max_memory_reserved_gb 109.53125
(TaskRunner pid=132393) wandb:                      perf/mfu/actor 0.34695
(TaskRunner pid=132393) wandb:                     perf/throughput 1811.23348
(TaskRunner pid=132393) wandb:                  perf/time_per_step 114.84542
(TaskRunner pid=132393) wandb:               perf/total_num_tokens 1664095
(TaskRunner pid=132393) wandb:            prompt_length/clip_ratio 0
(TaskRunner pid=132393) wandb:                   prompt_length/max 173
(TaskRunner pid=132393) wandb:                  prompt_length/mean 92.97754
(TaskRunner pid=132393) wandb:                   prompt_length/min 55
(TaskRunner pid=132393) wandb:          response_length/clip_ratio 0.0002
(TaskRunner pid=132393) wandb:                 response_length/max 1024
(TaskRunner pid=132393) wandb:                response_length/mean 232.04102
(TaskRunner pid=132393) wandb:                 response_length/min 65
(TaskRunner pid=132393) wandb:             timing_per_token_ms/adv 7e-05
(TaskRunner pid=132393) wandb:             timing_per_token_ms/gen 0.0278
(TaskRunner pid=132393) wandb:             timing_per_token_ms/ref 0.00525
(TaskRunner pid=132393) wandb:    timing_per_token_ms/update_actor 0.01751
(TaskRunner pid=132393) wandb:                        timing_s/adv 0.12448
(TaskRunner pid=132393) wandb:                        timing_s/gen 33.03033
(TaskRunner pid=132393) wandb:         timing_s/generate_sequences 26.73221
(TaskRunner pid=132393) wandb:               timing_s/old_log_prob 8.25264
(TaskRunner pid=132393) wandb:                        timing_s/ref 8.74394
(TaskRunner pid=132393) wandb:                    timing_s/reshard 1.2155
(TaskRunner pid=132393) wandb:                     timing_s/reward 0.84276
(TaskRunner pid=132393) wandb:            timing_s/save_checkpoint 17.9317
(TaskRunner pid=132393) wandb:              timing_s/start_profile 0.00011
(TaskRunner pid=132393) wandb:                       timing_s/step 114.84542
(TaskRunner pid=132393) wandb:               timing_s/stop_profile 7e-05
(TaskRunner pid=132393) wandb:                    timing_s/testing 16.59371
(TaskRunner pid=132393) wandb:               timing_s/update_actor 29.13985
(TaskRunner pid=132393) wandb:                      training/epoch 14
(TaskRunner pid=132393) wandb:                training/global_step 105
(TaskRunner pid=132393) wandb: val-core/openai/gsm8k/reward/mean@1 0.92722 /workspace/verl_space/verl/examples/grpo_trainer/checkpoints/verl_grpo_example_gsm8k/qwen2.5_7b_function_rm/global_step_105/actor/extra_state_world_size_8_rank_0.pt [repeated 7x across cluster]
