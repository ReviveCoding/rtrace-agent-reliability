from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import generate_benchmark, validate_benchmark
from .incidents import replay_incidents
from .pipeline import run_all, run_multiseed
from .utils import prepare_output_dir, write_json


def _add_common_options(parser: argparse.ArgumentParser, include_config: bool = True) -> None:
    parser.add_argument("--output", required=True, help="Directory for generated artifacts")
    parser.add_argument("--seed", type=int, default=None, help="Overrides benchmark.seed in config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory instead of failing closed",
    )
    if include_config:
        parser.add_argument(
            "--config",
            default=None,
            help="YAML pipeline configuration; defaults to configs/default.yaml",
        )


def _execute(args: argparse.Namespace) -> None:
    if args.command == "run-all":
        print(
            run_all(
                args.output,
                seed=args.seed,
                overwrite=args.overwrite,
                config_path=args.config,
            )["release"]
        )
        return
    if args.command == "validate-data":
        config, _ = load_config(args.config)
        seed = int(config["benchmark"]["seed"] if args.seed is None else args.seed)
        benchmark_sizes = {
            key: value for key, value in config["benchmark"].items() if key != "seed"
        }
        result = validate_benchmark(generate_benchmark(seed, benchmark_sizes))
        output = prepare_output_dir(Path(args.output), overwrite=args.overwrite)
        write_json(output / "data_quality.json", result)
        write_json(output / "effective_config.json", config)
        print(result)
        return
    if args.command == "replay-incidents":
        seed = 17 if args.seed is None else args.seed
        print(replay_incidents(args.output, seed=seed, overwrite=args.overwrite))
        return

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    print(
        run_multiseed(
            args.output,
            seeds=seeds,
            overwrite=args.overwrite,
            config_path=args.config,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rtrace", description="R-TRACE local agent evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run_all_parser = sub.add_parser("run-all", help="Run the complete local benchmark pipeline")
    _add_common_options(run_all_parser)

    validation_parser = sub.add_parser("validate-data", help="Validate a benchmark configuration")
    _add_common_options(validation_parser)

    replay_parser = sub.add_parser("replay-incidents", help="Replay safety and recovery incidents")
    _add_common_options(replay_parser, include_config=False)

    multi = sub.add_parser("run-multiseed", help="Run or resume independent seed evaluations")
    _add_common_options(multi)
    multi.add_argument("--seeds", default="11,17,23,29,31")

    args = parser.parse_args(argv)
    try:
        _execute(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"rtrace: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
