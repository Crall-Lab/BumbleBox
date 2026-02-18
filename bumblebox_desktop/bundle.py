from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SUPPORTED_SCHEMA_VERSIONS = {"bbx.run_bundle.v1"}


@dataclass
class BundleArtifact:
    kind: str
    required: bool
    source_path: str
    bundle_relpath: str
    size_bytes: int
    sha256: str


@dataclass
class BundleManifest:
    schema_version: str
    bundle_name: str
    session_name: str
    preferred_tracking_csv: Optional[str]
    artifact_count: int
    missing_expected_count: int
    artifacts: list[BundleArtifact]
    raw: dict[str, Any]


@dataclass
class BundleValidationResult:
    bundle_dir: str
    manifest_path: str
    error_count: int
    warning_count: int
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return self.error_count == 0


@dataclass
class ResolvedBundle:
    bundle_dir: Path
    manifest_path: Path
    extracted_tempdir: Optional[Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _find_manifest_in_tree(root: Path) -> Path:
    direct = root / "bundle_manifest.json"
    if direct.exists() and direct.is_file():
        return direct

    candidates = sorted(root.rglob("bundle_manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No bundle_manifest.json found under: {root}")
    if len(candidates) > 1:
        raise ValueError(
            f"Found multiple bundle_manifest.json files under {root}. "
            "Provide a single bundle directory or zip."
        )
    return candidates[0]


def resolve_bundle_input(bundle_input: str | Path) -> ResolvedBundle:
    bundle_input = Path(bundle_input).expanduser().resolve()
    if not bundle_input.exists():
        raise FileNotFoundError(f"Bundle path not found: {bundle_input}")

    if bundle_input.is_dir():
        manifest_path = _find_manifest_in_tree(bundle_input)
        return ResolvedBundle(
            bundle_dir=manifest_path.parent.resolve(),
            manifest_path=manifest_path.resolve(),
            extracted_tempdir=None,
        )

    if bundle_input.is_file() and bundle_input.suffix.lower() == ".zip":
        temp_root = Path(tempfile.mkdtemp(prefix="bbx_bundle_extract_"))
        with zipfile.ZipFile(bundle_input, "r") as zf:
            zf.extractall(temp_root)
        manifest_path = _find_manifest_in_tree(temp_root)
        return ResolvedBundle(
            bundle_dir=manifest_path.parent.resolve(),
            manifest_path=manifest_path.resolve(),
            extracted_tempdir=temp_root,
        )

    raise ValueError(f"Unsupported bundle input: {bundle_input}. Expected folder or .zip")


def cleanup_resolved_bundle(resolved: ResolvedBundle) -> None:
    if resolved.extracted_tempdir and resolved.extracted_tempdir.exists():
        shutil.rmtree(resolved.extracted_tempdir, ignore_errors=True)


def load_manifest(manifest_path: str | Path) -> BundleManifest:
    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")

    artifacts_raw = payload.get("artifacts", [])
    if not isinstance(artifacts_raw, list):
        raise ValueError("Manifest field 'artifacts' must be a list.")

    artifacts = []
    for item in artifacts_raw:
        if not isinstance(item, dict):
            continue
        artifacts.append(
            BundleArtifact(
                kind=str(item.get("kind", "")),
                required=bool(item.get("required", False)),
                source_path=str(item.get("source_path", "")),
                bundle_relpath=str(item.get("bundle_relpath", "")),
                size_bytes=int(item.get("size_bytes", 0)),
                sha256=str(item.get("sha256", "")),
            )
        )

    return BundleManifest(
        schema_version=str(payload.get("schema_version", "")),
        bundle_name=str(payload.get("bundle_name", "")),
        session_name=str(payload.get("session_name", "")),
        preferred_tracking_csv=(
            str(payload.get("preferred_tracking_csv"))
            if payload.get("preferred_tracking_csv") is not None
            else None
        ),
        artifact_count=int(payload.get("artifact_count", len(artifacts))),
        missing_expected_count=int(payload.get("missing_expected_count", 0)),
        artifacts=artifacts,
        raw=payload,
    )


def validate_manifest(
    manifest: BundleManifest,
    bundle_dir: str | Path,
    verify_hashes: bool = True,
) -> BundleValidationResult:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"Unsupported schema_version '{manifest.schema_version}'. "
            f"Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    if manifest.artifact_count != len(manifest.artifacts):
        warnings.append(
            f"artifact_count={manifest.artifact_count} but {len(manifest.artifacts)} artifact entries found."
        )

    for artifact in manifest.artifacts:
        if not artifact.bundle_relpath:
            errors.append(f"Artifact '{artifact.kind}' has empty bundle_relpath.")
            continue
        candidate = (bundle_dir / artifact.bundle_relpath).resolve()
        if not candidate.exists() or not candidate.is_file():
            message = f"Artifact missing: kind={artifact.kind}, path={artifact.bundle_relpath}"
            if artifact.required:
                errors.append(message)
            else:
                warnings.append(message)
            continue

        actual_size = candidate.stat().st_size
        if artifact.size_bytes > 0 and actual_size != artifact.size_bytes:
            warnings.append(
                f"Artifact size mismatch for {artifact.bundle_relpath}: "
                f"manifest={artifact.size_bytes}, actual={actual_size}"
            )

        if verify_hashes and artifact.sha256:
            actual_hash = _sha256(candidate)
            if actual_hash.lower() != artifact.sha256.lower():
                message = (
                    f"Artifact hash mismatch for {artifact.bundle_relpath}: "
                    f"manifest={artifact.sha256}, actual={actual_hash}"
                )
                if artifact.required:
                    errors.append(message)
                else:
                    warnings.append(message)

    return BundleValidationResult(
        bundle_dir=str(bundle_dir),
        manifest_path=str((bundle_dir / "bundle_manifest.json").resolve()),
        error_count=len(errors),
        warning_count=len(warnings),
        errors=errors,
        warnings=warnings,
    )


def preferred_tracking_csv_path(manifest: BundleManifest, bundle_dir: str | Path) -> Optional[Path]:
    bundle_dir = Path(bundle_dir).expanduser().resolve()

    if manifest.preferred_tracking_csv:
        candidate = (bundle_dir / manifest.preferred_tracking_csv).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

    kind_priority = ("tracking_cleaned_csv", "tracking_raw_csv")
    by_kind = {artifact.kind: artifact for artifact in manifest.artifacts}
    for kind in kind_priority:
        artifact = by_kind.get(kind)
        if not artifact:
            continue
        candidate = (bundle_dir / artifact.bundle_relpath).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def preferred_video_mp4_path(manifest: BundleManifest, bundle_dir: str | Path) -> Optional[Path]:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    for artifact in manifest.artifacts:
        if artifact.kind != "video_mp4":
            continue
        candidate = (bundle_dir / artifact.bundle_relpath).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def format_validation_result(result: BundleValidationResult) -> str:
    lines = [
        "Bundle Validation",
        "-----------------",
        f"Bundle directory: {result.bundle_dir}",
        f"Manifest: {result.manifest_path}",
        f"Errors: {result.error_count}",
        f"Warnings: {result.warning_count}",
    ]
    if result.errors:
        lines.append("")
        lines.append("Error details:")
        for item in result.errors:
            lines.append(f"- {item}")
    if result.warnings:
        lines.append("")
        lines.append("Warning details:")
        for item in result.warnings:
            lines.append(f"- {item}")
    return "\n".join(lines)
