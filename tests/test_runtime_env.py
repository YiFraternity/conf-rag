import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from runtime_env import configure_vllm_runtime_env


def test_preserves_existing_cuda_visible_devices():
    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_TARGET_DEVICE": "rocm",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    }

    configure_vllm_runtime_env(env)

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["VLLM_TARGET_DEVICE"] == "rocm"
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_sets_defaults_when_variables_are_missing():
    env = {}

    configure_vllm_runtime_env(env)

    assert env["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert env["VLLM_TARGET_DEVICE"] == "cuda"
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
