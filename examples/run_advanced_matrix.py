"""Materialize or execute the predefined seed-by-ablation experiment matrix."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/advanced_experiment.json"))
    parser.add_argument("--matrix", type=Path, default=Path("configs/advanced_matrix.json"))
    parser.add_argument("--data-root", type=Path, default=Path("input/advanced"))
    parser.add_argument("--output-root", type=Path, default=Path("output/advanced_matrix"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run training sequentially. Without this flag, only configs and commands are produced.",
    )
    return parser.parse_args()


def deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def main() -> None:
    args = parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    config_directory = args.output_root / "resolved_configs"
    config_directory.mkdir(parents=True, exist_ok=True)
    commands = []

    for seed in matrix["seeds"]:
        for experiment in matrix["experiments"]:
            resolved = copy.deepcopy(base)
            overrides = {key: value for key, value in experiment.items() if key != "name"}
            deep_update(resolved, overrides)
            resolved["seed"] = int(seed)
            data_directory = args.data_root / f"seed_{seed}"
            output_directory = args.output_root / f"seed_{seed}" / experiment["name"]
            resolved["data"]["directory"] = str(data_directory)
            resolved["output_directory"] = str(output_directory)
            config_path = config_directory / f"seed_{seed}_{experiment['name']}.json"
            config_path.write_text(
                json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
            )
            command = [
                sys.executable,
                "-m",
                "examples.train_advanced",
                "--config",
                str(config_path),
            ]
            commands.append(" ".join(command))
            if args.execute:
                subprocess.run(command, check=True)

    command_path = args.output_root / "commands.txt"
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Prepared {len(commands)} experiments. Commands: {command_path}")
    if not args.execute:
        print("Dry run only. Pass --execute on the GPU server to start training.")


if __name__ == "__main__":
    main()
