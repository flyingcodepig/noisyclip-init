"""End-to-end read-only data audit and manifest generation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from noisyclip.config.schema import ProjectConfig
from noisyclip.data.catalog import ClassCatalog, scan_class_catalog, write_class_mapping
from noisyclip.data.image_io import inspect_image
from noisyclip.data.leakage import LeakageReport, assert_no_leakage, check_manifest_leakage
from noisyclip.data.manifests import data_digest, make_sample_id, manifest_digest, write_manifest
from noisyclip.data.records import SampleRecord
from noisyclip.data.split import stratified_train_val_split


class AuditConfigError(ValueError):
    """Raised when a config is syntactically valid but unusable for data audit."""


@dataclass(frozen=True, slots=True)
class DataAuditResult:
    """Paths and digests produced by a successful data audit.

    Args:
        class_mapping_path: Written `class_to_idx.json` path.
        train_manifest_path: Written train manifest path.
        val_manifest_path: Written validation manifest path.
        test_manifest_path: Written test manifest path.
        summary_path: Written data summary JSON path.
        leakage_report_path: Written leakage report JSON path.
        class_mapping_digest: Stable digest of the class mapping.
        train_manifest_digest: Stable digest of train manifest rows.
        val_manifest_digest: Stable digest of validation manifest rows.
        test_manifest_digest: Stable digest of test manifest rows.
        data_digest: Stable digest over mapping, manifests, and file hashes.
    """

    class_mapping_path: Path
    train_manifest_path: Path
    val_manifest_path: Path
    test_manifest_path: Path
    summary_path: Path
    leakage_report_path: Path
    class_mapping_digest: str
    train_manifest_digest: str
    val_manifest_digest: str
    test_manifest_digest: str
    data_digest: str


def run_data_audit(config: ProjectConfig) -> DataAuditResult:
    """Generate class mapping, fixed manifests, data summary, and leakage report.

    Args:
        config: Strict resolved project config. `paths.train_root`,
            `paths.test_root`, and `paths.run_root` must be concrete paths, not
            unresolved `${oc.env:...}` placeholders.

    Returns:
        `DataAuditResult` with all artifact paths and digests.

    Raises:
        AuditConfigError: If required paths are unresolved or malformed.
        CatalogError: If training class directories are invalid.
        ImageAuditError: If an image is unreadable under `fail_audit`.
        SplitError: If deterministic stratified splitting is impossible.
        ManifestError: If generated records violate manifest constraints.
        LeakageError: If test data intersects train/validation records.
        OSError: If source roots cannot be read or artifacts cannot be written.
    """

    train_root = _concrete_path(config.paths.train_root, "paths.train_root")
    test_root = _concrete_path(config.paths.test_root, "paths.test_root")
    run_root = _concrete_path(config.paths.run_root, "paths.run_root")
    if not train_root.is_dir():
        raise AuditConfigError(f"paths.train_root is not a directory: {train_root}")
    if not test_root.is_dir():
        raise AuditConfigError(f"paths.test_root is not a directory: {test_root}")

    output_dir = run_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = scan_class_catalog(
        train_root,
        class_id_regex=config.data.class_id_regex,
        expected_num_classes=config.data.expected_num_classes,
    )
    class_mapping_path = write_class_mapping(
        catalog, _path_or_default(config.paths.class_mapping, output_dir / "class_to_idx.json")
    )

    labeled_records = _audit_labeled_records(train_root, catalog, config)
    test_records = _audit_test_records(test_root, config)
    train_records, val_records = stratified_train_val_split(
        labeled_records,
        seed=config.data.split_seed,
        val_fraction=config.data.val_fraction,
    )

    leakage_report = check_manifest_leakage(
        train_records,
        val_records,
        test_records,
        train_root=train_root,
        test_root=test_root,
    )
    leakage_report_path = output_dir / "leakage_report.json"
    _write_json(leakage_report_path, _leakage_report_payload(leakage_report))
    assert_no_leakage(leakage_report)

    train_manifest_path = write_manifest(
        train_records,
        _path_or_default(config.paths.train_manifest, output_dir / "train_manifest.json"),
    )
    val_manifest_path = write_manifest(
        val_records,
        _path_or_default(config.paths.val_manifest, output_dir / "val_manifest.json"),
    )
    test_manifest_path = write_manifest(
        test_records,
        _path_or_default(config.paths.test_manifest, output_dir / "test_manifest.json"),
    )

    combined_digest = data_digest(
        train_records + val_records + test_records,
        dict(catalog.class_to_idx),
    )
    summary_path = output_dir / "data_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "class_mapping_digest": catalog.digest,
            "train_manifest_digest": manifest_digest(train_records),
            "val_manifest_digest": manifest_digest(val_records),
            "test_manifest_digest": manifest_digest(test_records),
            "data_digest": combined_digest,
            "num_classes": len(catalog.class_to_idx),
            "num_train": len(train_records),
            "num_val": len(val_records),
            "num_test": len(test_records),
            "hash_files": config.data.hash_files,
        },
    )
    _write_json(
        output_dir / "manifest_digest.json",
        {
            "schema_version": 1,
            "class_mapping_digest": catalog.digest,
            "train_manifest_digest": manifest_digest(train_records),
            "val_manifest_digest": manifest_digest(val_records),
            "test_manifest_digest": manifest_digest(test_records),
            "data_digest": combined_digest,
        },
    )

    return DataAuditResult(
        class_mapping_path=class_mapping_path,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        test_manifest_path=test_manifest_path,
        summary_path=summary_path,
        leakage_report_path=leakage_report_path,
        class_mapping_digest=catalog.digest,
        train_manifest_digest=manifest_digest(train_records),
        val_manifest_digest=manifest_digest(val_records),
        test_manifest_digest=manifest_digest(test_records),
        data_digest=combined_digest,
    )


def _audit_labeled_records(
    train_root: Path,
    catalog: ClassCatalog,
    config: ProjectConfig,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for class_id in sorted(catalog.class_to_idx):
        class_dir = train_root / class_id
        for image_path in _iter_files(class_dir):
            relative_path = image_path.relative_to(train_root).as_posix()
            info = inspect_image(
                image_path,
                relative_path=relative_path,
                hash_file=config.data.hash_files,
                allow_truncated_images=config.data.allow_truncated_images,
                unreadable_policy=config.data.unreadable_policy,
            )
            if not info.readable and config.data.unreadable_policy == "skip_with_record":
                continue
            records.append(
                SampleRecord(
                    sample_id=make_sample_id(relative_path),
                    relative_path=relative_path,
                    split="train",
                    class_id=class_id,
                    target=catalog.class_to_idx[class_id],
                    file_sha256=info.file_sha256,
                    width=info.width,
                    height=info.height,
                    readable=info.readable,
                )
            )
    return records


def _audit_test_records(test_root: Path, config: ProjectConfig) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for image_path in _iter_files(test_root):
        relative_path = image_path.relative_to(test_root).as_posix()
        info = inspect_image(
            image_path,
            relative_path=relative_path,
            hash_file=config.data.hash_files,
            allow_truncated_images=config.data.allow_truncated_images,
            unreadable_policy=config.data.unreadable_policy,
        )
        if not info.readable and config.data.unreadable_policy == "skip_with_record":
            continue
        records.append(
            SampleRecord(
                sample_id=make_sample_id(relative_path),
                relative_path=relative_path,
                split="test",
                class_id=None,
                target=None,
                file_sha256=info.file_sha256,
                width=info.width,
                height=info.height,
                readable=info.readable,
            )
        )
    return records


def _iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _concrete_path(raw: str, field_name: str) -> Path:
    if raw.startswith("${"):
        raise AuditConfigError(f"{field_name} is unresolved: {raw}")
    return Path(raw).expanduser().resolve()


def _path_or_default(raw: str | None, default: Path) -> Path:
    if raw is None:
        return default
    if raw.startswith("${"):
        raise AuditConfigError(f"configured output path is unresolved: {raw}")
    return Path(raw).expanduser().resolve()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _leakage_report_payload(report: LeakageReport) -> dict[str, object]:
    payload = asdict(report)
    payload["ok"] = report.ok
    return payload
