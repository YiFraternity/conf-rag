def configure_vllm_runtime_env(env):
    env.setdefault("CUDA_VISIBLE_DEVICES", "4,5")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
