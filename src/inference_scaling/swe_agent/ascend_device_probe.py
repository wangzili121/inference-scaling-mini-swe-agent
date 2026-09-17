"""Exercise Ascend device allocation and an optional HCCL collective."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta


def _device_count() -> int:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible.strip():
        return len([item for item in visible.split(",") if item.strip()])
    return 0


def probe_single_device() -> dict[str, object]:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    value = torch.tensor([7.0], device="npu:0")
    observed = float(value.cpu().item())
    if observed != 7.0:
        raise RuntimeError(f"NPU tensor round trip returned {observed}")
    return {
        "probe": "single_device",
        "status": "ok",
        "logical_device": 0,
        "visible_device_count": _device_count(),
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
    }


def probe_distributed() -> dict[str, object] | None:
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl",
        timeout=timedelta(seconds=120),
    )
    value = torch.tensor([float(rank + 1)], device=f"npu:{local_rank}")
    dist.all_reduce(value)
    observed = float(value.cpu().item())
    expected = world_size * (world_size + 1) / 2
    if observed != expected:
        raise RuntimeError(
            f"HCCL all_reduce returned {observed}; expected {expected}"
        )
    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return None
    return {
        "probe": "hccl_all_reduce",
        "status": "ok",
        "world_size": world_size,
        "visible_device_count": _device_count(),
        "sum": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="run one HCCL all-reduce under torchrun",
    )
    args = parser.parse_args()
    result = probe_distributed() if args.distributed else probe_single_device()
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
