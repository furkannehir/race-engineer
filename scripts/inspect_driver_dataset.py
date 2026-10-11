"""Validate and inspect the driver-meaning review draft without running or training models."""

import argparse
import io
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from race_engineer.evaluation.driver_dataset import DriverMeaningDataset, dataset_summary
from race_engineer.evaluation.driver_training import coverage_audit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "fixtures/driver-meaning/review-expanded.v2.json"


def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--language", choices=("en", "tr"), help="Filter displayed examples.")
    parser.add_argument("--action", help="Filter by one conversational action.")
    parser.add_argument("--focus", help="Filter by one requested information focus.")
    parser.add_argument("--tag", help="Filter by a coverage tag.")
    parser.add_argument("--group", help="Filter by a leakage group.")
    parser.add_argument("--offset", type=int, default=0, help="Skip matching examples for review.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows listed (default 100).")
    parser.add_argument("--example", help="Print one full example for annotation review.")
    parser.add_argument("--list", action="store_true", help="Display IDs, texts and actions.")
    parser.add_argument("--coverage", action="store_true", help="Include detailed coverage counts.")
    parser.add_argument(
        "--pilot-split-check",
        action="store_true",
        help="Read-only group-held-out action coverage audit; does not train.",
    )
    parser.add_argument("--split-seed", type=int, default=17)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        parser.error("--offset must be nonnegative and --limit must be positive")
    try:
        dataset = DriverMeaningDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    except ValidationError as error:
        # Do not echo private messages from alternate local datasets in validation errors.
        print(f"Invalid dataset: {error.error_count()} validation error(s).", file=sys.stderr)
        for issue in error.errors(include_input=False, include_url=False):
            print(f"  {issue['loc']}: {issue['type']}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Could not read dataset ({type(error).__name__}).", file=sys.stderr)
        return 2
    examples = [
        example
        for example in dataset.examples
        if (args.language is None or example.language == args.language)
        and (args.action is None or args.action in example.annotation.actions)
        and (
            args.focus is None
            or any(request.focus == args.focus for request in example.annotation.requests)
        )
        and (args.tag is None or args.tag in example.coverage_tags)
        and (args.group is None or args.group == example.leakage_group)
    ]
    if args.example:
        selected = next(
            (example for example in examples if example.example_id == args.example),
            None,
        )
        if selected is None:
            print("Example not found in the selected language.", file=sys.stderr)
            return 1
        print(selected.model_dump_json(indent=2))
    else:
        summary = dataset_summary(dataset)
        if args.pilot_split_check:
            try:
                summary["pilot_split_check"] = coverage_audit(dataset, args.split_seed)
            except ValueError:
                print(
                    "Cannot construct an exploratory grouped split with this coverage.",
                    file=sys.stderr,
                )
                return 2
        if not args.coverage:
            for key in ("actions", "request_focuses", "time_scopes", "coverage_tags"):
                summary.pop(key)
        print(json.dumps(summary, indent=2))
        if args.list:
            page = examples[args.offset : args.offset + args.limit]
            print(
                f"Showing {len(page)} of {len(examples)} matching examples; "
                f"offset={args.offset}, limit={args.limit}."
            )
            for example in page:
                actions = ", ".join(example.annotation.actions)
                print(f"{example.example_id} [{example.language}] {actions}: {example.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
