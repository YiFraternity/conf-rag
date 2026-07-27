import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from generators import build_vllm_init_kwargs


def test_build_vllm_init_kwargs_uses_explicit_vllm_overrides():
    kwargs = build_vllm_init_kwargs(
        "/models/Qwen2.5-7B-Instruct",
        {
            "max_model_len": 2048,
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.72,
            "max_num_seqs": 32,
            "enforce_eager": True,
        },
    )

    assert kwargs == {
        "model": "/models/Qwen2.5-7B-Instruct",
        "max_model_len": 2048,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.72,
        "max_num_seqs": 32,
        "enforce_eager": True,
    }
