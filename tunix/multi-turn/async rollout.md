 • Rollout Orchestrator
Spawns and supervises many TrajectoryCollectEngine instances, distributes work, and passes shared configs/endpoints. Think “concurrency + lifecycle manager.”
 • Trajectory Collect Engine
One independent agent↔env loop: build prompts → call inference → parse actions → env.step() → accumulate rewards → emit a full trajectory at episode end.
 • Agent
Holds policy state and conversation history; turns model outputs into actions and logs transitions back into its trajectory buffer.
 • Environment (Env)
The task container. reset() samples a new problem/state; step(action) executes (optionally via tools), returns observation, reward, and done flags.
 • Tools
Optional external abilities (Python runner, retrieval, etc.) invoked by the env/agent within the loop.
 • Inference layer
 • vLLM Server: separate process with request queue + continuous batching; multiple engines share it.
 • vLLM Engine (in-proc): embedded vLLM, fewer hops; good for single host.
 • Vanilla Engine: plain HF generate; simplest, no token-level batching.
 • Trajectory collect loop_i
The asynchronous sampling loop owned by each engine that pushes completed trajectories to the queues.
 • QueueManager (GroupQueueManager)
Data hub on the training side. It maintains one queue per group_id (e.g., task/difficulty). Inside each group queue it keeps buckets, where one bucket = one episode_id (same prompt).
 • Put: when an engine finishes a trajectory, it goes to queue[group_id].bucket[episode_id].
 • Group size (K): when a bucket accumulates K candidates (e.g., GRPO’s K responses for the same prompt), the bucket becomes ready and is popped as a group.
 • get_batch: waits across all group queues (first-completed), pops oldest episode_id first per queue, and concatenates groups until batch_size is met—ensuring fairness across groups.
 • Backpressure (bounded vs pipe): in async mode a BoundedGroupQueue caps the number of unfinished buckets (max_group_num) to prevent memory blow-up; in sync mode a PipeGroupQueue uses a small “window” (max_traj_per_env) to keep in-flight episodes bounded.
 • Training group data
The ready groups pulled from queues (possibly multiple groups per batch) fed to PPO/GRPO training.
 • Model (weights)
Shared parameters for inference and training; training periodically updates the serving endpoint (or the server pulls).

End-to-end flow: Orchestrator runs many engines → each engine’s loop talks to the inference layer and env → completed trajectories go to QueueManager → buckets aggregate K per prompt → queues yield groups → trainer assembles batches and updates the model.