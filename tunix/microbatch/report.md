TPU 4: mesh = (1, 4)

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
> NUM_BATCHES = 800
>
> grad_accu_steps = 8

```
rollout_micro_batch_size=1
ref_logps_micro_batch_size=1
old_logps_micro_batch_size=1
```

```
100/100 [23:51<00:00, 8.66s/step, train_loss=-.005, train_perplexity=0.996, train_steps_per_sec=13.3, train_kl=0.011]
```



```
rollout_micro_batch_size=2
ref_logps_micro_batch_size=2
old_logps_micro_batch_size=2
```

```
100/100 [17:44<00:00, 6.72s/step, train_loss=-.004, train_perplexity=0.996, train_steps_per_sec=13.4, train_kl=0.013]
```



```
rollout_micro_batch_size=4
ref_logps_micro_batch_size=4
old_logps_micro_batch_size=4
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

> NUM_GENERATIONS = 2
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









TPU 8 : mesh = (2, 4)



NUM_GENERATIONS = 4

BATCH_SIZE = 1(training)

NUM_BATCHES = 800

grad_accu_steps = 8



111

100/100 [1:25:34<00:00, 32.36s/step, train_loss=-.005, train_perplexity=0.995, train_steps_per_sec=13.1, train_kl=0.011]



222

100/100 [52:38<00:00, 20.18s/step, train_loss=-.003, train_perplexity=0.997, train_steps_per_sec=12.3, train_kl=0.013]



444

100/100 [31:54<00:00, 13.18s/step, train_loss=0, train_perplexity=1, train_steps_per_sec=12.5, train_kl=0.012]



888

100/100 [20:52<00:00, 8.94s/step, train_loss=-.003, train_perplexity=0.997, train_steps_per_sec=8.55, train_kl=0.014]





NUM_GENERATIONS = 4

BATCH_SIZE = 2(training)

NUM_BATCHES = 800

grad_accu_steps = 8



222

100/100 [1:45:21<00:00, 44.62s/step, train_loss=-.001, train_perplexity=0.999, train_steps_per_sec=12, train_kl=0.013]

