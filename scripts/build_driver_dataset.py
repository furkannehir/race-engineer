"""Mechanically materialize the authored bilingual catalog; no models, training, or approval."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from race_engineer.evaluation.driver_dataset import DriverMeaningDataset, dataset_summary

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "fixtures/driver-meaning/review-draft.v1.json"
CATALOG = ROOT / "fixtures/driver-meaning/expansion-catalog.v1.json"
OUTPUT = ROOT / "fixtures/driver-meaning/review-expanded.v1.json"


def build_dataset(seed: dict[str, Any], catalog: dict[str, Any]) -> DriverMeaningDataset:
    if catalog.get("schema_version") != "driver-meaning-expansion-catalog.v1":
        raise ValueError("unsupported expansion catalog")
    if seed.get("revision") != catalog["seed_revision"]:
        raise ValueError("seed revision differs from the authored catalog")
    # JSON roundtrip copies; the original reviewed-in-conversation seed is not mutated.
    examples = json.loads(json.dumps(seed["examples"]))
    seen_families: set[str] = set()
    for family in catalog["families"]:
        family_id = family["id"]
        if family_id in seen_families:
            raise ValueError("duplicate authored family ID")
        seen_families.add(family_id)
        for variant_index, variant in enumerate(family["variants"], 1):
            for language in ("en", "tr"):
                text = variant[language]
                annotation = json.loads(
                    json.dumps(
                        {
                            **catalog["annotation_defaults"],
                            **family["annotation"],
                        }
                    )
                )
                annotation["requests"] = [
                    {**catalog["request_defaults"], **request} for request in annotation["requests"]
                ]
                for request in annotation["requests"]:
                    # Compound spans are authored per variant; a single request uses the
                    # complete utterance as provenance, not keyword-based classification.
                    spans = variant.get("spans", {}).get(language)
                    request["source_span"] = (
                        spans[request["request_id"]] if spans is not None else text
                    )
                if len(annotation["requests"]) > 1 and "spans" not in variant:
                    raise ValueError("compound variants require explicitly authored spans")
                row = {
                    "example_id": f"ex-{family_id}-{variant_index:02d}-{language}",
                    "scenario_id": f"expanded-{family_id}",
                    "leakage_group": family["leakage_group"],
                    "language": language,
                    "speaker": "driver",
                    "text": text,
                    "recent_dialogue": family.get("dialogue", {}).get(language, []),
                    "origin": "assistant_authored",
                    "prior_exposure": family["prior_exposure"],
                    "split": "unassigned",
                    "review": {"status": "draft"},
                    "annotation": annotation,
                    "coverage_tags": family["coverage_tags"],
                    "input_kind": family.get("input_kind", "typed"),
                }
                if row["input_kind"] == "synthetic_asr":
                    row["intended_text"] = variant["intended"][language]
                examples.append(row)
    dataset = DriverMeaningDataset.model_validate(
        {
            **seed,
            "dataset_id": catalog["dataset_id"],
            "revision": catalog["revision"],
            "examples": examples,
        }
    )
    expected = catalog.get("examples_per_language", 400)
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("invalid examples_per_language")
    if dict(Counter(row.language for row in dataset.examples)) != {"en": expected, "tr": expected}:
        raise ValueError(f"expansion must contain exactly {expected} examples in each language")
    if any(row.review.status != "draft" or row.split != "unassigned" for row in dataset.examples):
        raise ValueError("expansion must not imply review approval or split assignment")
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edition",
        choices=("v1", "v2"),
        default="v1",
        help="v1 preserves the original pilot; v2 adds coverage scenarios.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Create the expanded fixture if absent.")
    mode.add_argument(
        "--check", action="store_true", help="Check the materialized fixture matches."
    )
    args = parser.parse_args()
    seed_path, catalog_path, output_path = SEED, CATALOG, OUTPUT
    if args.edition == "v2":
        seed_path = ROOT / "fixtures/driver-meaning/review-expanded.v1.json"
        catalog_path = ROOT / "fixtures/driver-meaning/coverage-catalog.v2.json"
        output_path = ROOT / "fixtures/driver-meaning/review-expanded.v2.json"
    try:
        dataset = build_dataset(
            json.loads(seed_path.read_text(encoding="utf-8")),
            json.loads(catalog_path.read_text(encoding="utf-8")),
        )
        serialized = dataset.model_dump_json(indent=2) + "\n"
        if args.check and output_path.read_text(encoding="utf-8") != serialized:
            raise ValueError("expanded fixture differs from its authored sources")
        if args.write:
            if output_path.exists():
                if output_path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("refusing to overwrite an existing or edited dataset")
            else:
                with output_path.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(serialized)
        print(json.dumps(dataset_summary(dataset), indent=2))
        print("Authored drafts only: no training, review approval, or split assignment.")
        return 0
    except ValidationError as error:
        print(f"Dataset build failed: {error.error_count()} validation error(s).")
        for issue in error.errors(include_input=False, include_url=False):
            print(f"  {issue['loc']}: {issue['type']}")
        return 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Dataset build failed ({type(error).__name__}): {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
