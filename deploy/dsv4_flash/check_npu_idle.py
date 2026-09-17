#!/usr/bin/env python3
"""Fail closed unless every selected Ascend NPU is healthy and idle."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


_IDLE = re.compile(r"No running processes found in NPU\s+(\d+)")
_DEVICE = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|\s*(OK)\s+\|")
_MEMORY = re.compile(r"(\d+)\s*/\s*(\d+)\s*\|")


def parse_npu_smi(text: str) -> tuple[set[int], dict[int, str], dict[int, int]]:
    idle = {int(value) for value in _IDLE.findall(text)}
    health: dict[int, str] = {}
    hbm: dict[int, int] = {}
    current: int | None = None
    for line in text.splitlines():
        device = _DEVICE.match(line)
        if device:
            current = int(device.group(1))
            health[current] = device.group(2)
            continue
        if current is not None and "Bus-Id" not in line:
            matches = _MEMORY.findall(line)
            if matches:
                hbm[current] = int(matches[-1][0])
                current = None
    return idle, health, hbm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--max-idle-hbm-mb", type=int, default=8192)
    args = parser.parse_args()
    devices = tuple(int(value) for value in args.devices.split(","))
    executable = shutil.which("npu-smi") or "/usr/local/bin/npu-smi"
    completed = subprocess.run(
        [executable, "info"], text=True, capture_output=True, check=True
    )
    text = completed.stdout + completed.stderr
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(text, encoding="utf-8")
    idle, health, hbm = parse_npu_smi(text)
    failures: list[str] = []
    for device in devices:
        if health.get(device) != "OK":
            failures.append(f"NPU {device} health is not OK or was not parsed")
        if device not in idle:
            failures.append(f"NPU {device} has a running process or was not parsed")
        if hbm.get(device, args.max_idle_hbm_mb + 1) > args.max_idle_hbm_mb:
            failures.append(
                f"NPU {device} HBM usage {hbm.get(device)} MB exceeds idle limit "
                f"{args.max_idle_hbm_mb} MB"
            )
    result = {
        "checked_at": time.time(),
        "devices": list(devices),
        "idle_devices": sorted(idle),
        "health": health,
        "hbm_used_mb": hbm,
        "max_idle_hbm_mb": args.max_idle_hbm_mb,
        "ok": not failures,
        "failures": failures,
    }
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("selected NPUs are not safely idle")


if __name__ == "__main__":
    main()
