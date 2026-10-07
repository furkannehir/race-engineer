"""Explicit, reproducible evaluation of local Context Engineer candidates."""

from datetime import UTC, datetime
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.core.interfaces import ContextQueryPlanner
from race_engineer.evaluation.intelligence import (
    IntelligenceCandidateMetadata,
    IntelligenceEvaluationDataset,
    RuntimeProfile,
    load_intelligence_evaluation_dataset,
    validate_intelligence_split_families,
)
from race_engineer.evaluation.intelligence_runner import (
    evaluate_intelligence_datasets,
)
from race_engineer.evaluation.system import (
    files_fingerprint,
    git_dirty,
    git_revision,
    machine_metadata,
)
from race_engineer.intelligence.context_planner import PLANNER_ID, QwenContextQueryPlanner
from race_engineer.intelligence.local_model import LocalJsonModel, ModelRequestMetrics
from race_engineer.intelligence.semantic_planner import (
    CatalogSemanticPlanner,
    LocalSemanticWorker,
)

_QWEN_CANDIDATE = PLANNER_ID
_MINILM_CANDIDATE = "minilm-nli-v1"
_LAYA_CANDIDATE = "laya-multilingual-v1"
_CANDIDATES = {_QWEN_CANDIDATE, _MINILM_CANDIDATE, _LAYA_CANDIDATE}


def _repository_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("evaluation dataset must be a file inside the repository")
    return resolved


def _input_files(
    root: Path,
    dataset_paths: tuple[Path, ...],
    datasets: tuple[IntelligenceEvaluationDataset, ...],
) -> tuple[Path, ...]:
    files = list(dataset_paths)
    for dataset in datasets:
        fixture = (root / dataset.fixture).resolve()
        if not fixture.is_relative_to(root) or not fixture.is_dir():
            raise ValueError("evaluation fixture must be a directory inside the repository")
        files.extend(path for path in fixture.rglob("*") if path.is_file())
    return tuple(files)


async def run_intelligence_evaluation(
    root: Path,
    config_path: Path,
    dataset_paths: tuple[Path, ...],
    *,
    candidate_id: str = _QWEN_CANDIDATE,
    runtime_profile: RuntimeProfile = "portable",
) -> dict[str, object]:
    """Evaluate one candidate without starting or downloading a model runtime."""

    resolved_root = root.resolve()
    if not dataset_paths:
        raise ValueError("at least one intelligence evaluation dataset is required")
    if candidate_id not in _CANDIDATES:
        raise ValueError("intelligence_candidate_not_implemented")
    if runtime_profile != "portable":
        raise ValueError("enhanced_profile_not_implemented")

    paths = tuple(_repository_file(resolved_root, path) for path in dataset_paths)
    datasets = tuple(load_intelligence_evaluation_dataset(path) for path in paths)
    validate_intelligence_split_families(datasets)
    config = load_config(config_path)
    semantic_planner: CatalogSemanticPlanner | None = None
    request_metrics: list[ModelRequestMetrics] = []
    planner: ContextQueryPlanner
    if candidate_id == _QWEN_CANDIDATE:
        model = LocalJsonModel(config.conversation, on_metrics=request_metrics.append)
        planner = QwenContextQueryPlanner(config.conversation, model=model)
        candidate = IntelligenceCandidateMetadata(
            candidate_id=candidate_id,
            runtime_profile=runtime_profile,
            candidate_kind="generative",
            adapter=config.conversation.adapter,
            model=config.conversation.model,
            model_revision=config.conversation.runtime.model_revision,
        )
    else:
        python_path = Path("data/stt-prototype/runtime/Scripts/python.exe")
        if candidate_id == _MINILM_CANDIDATE:
            backend = "minilm"
            model_path = Path("data/dialogue-prototype/minilm")
            package_path = None
            model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
            model_revision = "0a71e92a985b6e1ad1828cf67ce9c459639c1dca"
            adapter = "transformers-nli-worker"
        else:
            backend = "laya"
            model_path = Path("data/dialogue-prototype/laya")
            package_path = Path("data/dialogue-prototype/packages")
            model_name = "convaiinnovations/laya-multilingual"
            model_revision = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
            adapter = "laya-0.3.20-worker"
        worker = LocalSemanticWorker(
            resolved_root,
            python_path=python_path,
            backend=backend,
            model_path=model_path,
            package_path=package_path,
            model_revision=model_revision,
            threads=config.conversation.runtime.threads,
        )
        semantic_planner = CatalogSemanticPlanner(worker, planner_id=candidate_id)
        planner = semantic_planner
        candidate = IntelligenceCandidateMetadata(
            candidate_id=candidate_id,
            runtime_profile=runtime_profile,
            candidate_kind="classifier",
            adapter=adapter,
            model=model_name,
            model_revision=model_revision,
        )

    def collect_metrics(result: dict[str, object]) -> None:
        result["model_requests"] = list(request_metrics)
        request_metrics.clear()

    try:
        results, summary = await evaluate_intelligence_datasets(
            resolved_root,
            config.policy.context,
            planner,
            candidate,
            datasets,
            progress=collect_metrics,
        )
    finally:
        if semantic_planner is not None:
            await semantic_planner.aclose()
    inputs = _input_files(resolved_root, paths, datasets)
    candidate_runtime = (
        semantic_planner.runtime_metadata
        if semantic_planner is not None
        else {
            "configured_mode": config.conversation.runtime.mode,
            "runtime_revision": config.conversation.runtime.runtime_revision,
            "quantization": config.conversation.runtime.quantization,
            "threads": config.conversation.runtime.threads,
            "context_size": config.conversation.runtime.context_size,
            "gpu_layers": config.conversation.runtime.gpu_layers,
        }
    )
    return {
        "schema_version": "intelligence-eval-report.v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_scope": "context_planner_and_deterministic_evidence_execution",
        "candidate": candidate.model_dump(mode="json"),
        "candidate_runtime": candidate_runtime,
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "revision": dataset.revision,
                "split": dataset.split,
                "turns": dataset.turn_count,
                "language_counts": dataset.language_counts(),
            }
            for dataset in datasets
        ],
        "reproducibility": {
            "input_fingerprint_sha256": files_fingerprint(resolved_root, inputs),
            "implementation_fingerprint_sha256": files_fingerprint(
                resolved_root, tuple(sorted((resolved_root / "src/race_engineer").rglob("*.py")))
            ),
            "git_revision": git_revision(resolved_root),
            "git_dirty": git_dirty(resolved_root),
            "machine": machine_metadata(),
        },
        "privacy": {
            "local_only": True,
            "report_contains_transcripts": False,
            "report_contains_model_replies": False,
        },
        "summary": summary,
        "results": results,
    }
