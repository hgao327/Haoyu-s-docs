num generation = 4; training-batchsize = 1: ok

num generation = 4; training-batchsize = 2:  Not work. Training step oom

num generation = 2; training-batchsize = 2: ok

num generation = 2; training-batchsize = 4:  Not work. Training step oom



Experiment1

> NUM_GENERATIONS = 4 
>
> BATCH_SIZE = 1(training)
>
> NUM_BATCHES = 200

1.

```
rollout_micro_batch_size=2,
ref_logps_micro_batch_size=2,
old_logps_micro_batch_size=2,
```

```
200/200 [08:28<00:00, 1.23s/step, train_loss=-.02, train_perplexity=0.98, train_steps_per_sec=15.6, train_kl=0.007]
```

2.

```
rollout_micro_batch_size=4,
ref_logps_micro_batch_size=4,
old_logps_micro_batch_size=4,
```

```
200/200 [06:00<00:00, 1.30s/step, train_loss=-.02, train_perplexity=0.98, train_steps_per_sec=16.1, train_kl=0.007
```

3.

```
rollout_micro_batch_size=8,
ref_logps_micro_batch_size=8,
old_logps_micro_batch_size=8,
```

```
200/200 [05:50<00:00, 1.21s/step, train_loss=-.02, train_perplexity=0.98, train_steps_per_sec=18.1, train_kl=0.007]
```



Experiment2

> NUM_GENERATIONS = 4 
>
> BATCH_SIZE = 1(training)
>
> NUM_BATCHES = 200
>
> grad_accu_steps = 8

```
rollout_micro_batch_size=1
ref_logps_micro_batch_size=1
old_logps_micro_batch_size=1
```

```
[06:09, 8.23s/step, train_loss=-.001, train_perplexity=0.999, train_steps_per_sec=13.4, train_kl=0.012]
```



```
rollout_micro_batch_size=2
ref_logps_micro_batch_size=2
old_logps_micro_batch_size=2
```

```
[04:08, 6.66s/step, train_loss=-.004, train_perplexity=0.996, train_steps_per_sec=15.2, train_kl=0.015]
```



```
rollout_micro_batch_size=2
ref_logps_micro_batch_size=2
old_logps_micro_batch_size=2
```

```
OOM
```



```
rollout_micro_batch_size=4
ref_logps_micro_batch_size=2
old_logps_micro_batch_size=2
```

```
[04:06, 6.21s/step, train_loss=-.022, train_perplexity=0.979, train_steps_per_sec=12.3, train_kl=0.005]
```



Experiment3

> NUM_GENERATIONS = 42
>
> BATCH_SIZE = 2(training)
>
> NUM_BATCHES = 200
>
> grad_accu_steps = 8

```
rollout_micro_batch_size=2
ref_logps_micro_batch_size=2
old_logps_micro_batch_size=2
```

```
[05:59, 9.19s/step, train_loss=-.014, train_perplexity=0.986, train_steps_per_sec=13.2, train_kl=0.004]
```



```
rollout_micro_batch_size=4
ref_logps_micro_batch_size=4
old_logps_micro_batch_size=4
```

```
[04:17, 6.70s/step, train_loss=-.003, train_perplexity=0.997, train_steps_per_sec=13.2, train_kl=0.007]
```



```
rollout_micro_batch_size=8
ref_logps_micro_batch_size=8
old_logps_micro_batch_size=8
```

```
OOM
```



```
rollout_micro_batch_size=8
ref_logps_micro_batch_size=4
old_logps_micro_batch_size=4
```

```
[04:03, 5.92s/step, train_loss=-.01, train_perplexity=0.99, train_steps_per_sec=13.9, train_kl=0.003]
```

