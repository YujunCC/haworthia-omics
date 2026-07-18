"""Export, validate, and merge user model packages without replacing images."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


MAX_PACKAGE_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 160 * 1024 * 1024
EXPECTED_ARCHITECTURE = "TemperamentOmicsNet"
EXPECTED_EMBEDDING_DIMENSION = 128
EXPECTED_ATTENTION_HEADS = 4
EXPECTED_PROTOTYPE_BYTES = EXPECTED_EMBEDDING_DIMENSION * 4
REQUIRED_FILES = {
    "MODEL_MANIFEST.json",
    "PACKAGE_MANIFEST.json",
    "model_base.pth",
    "haworthia_omics.db",
}
OPTIONAL_FILES = {"MODEL_LICENSE.txt", "MODEL_PACKAGE_NOTICE.txt"}
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


class ModelPackageError(ValueError):
    """Raised when a model package fails an integrity or format check."""


@dataclass(frozen=True)
class InspectedModelPackage:
    package_sha256: str
    manifest: dict
    model_bytes: bytes
    catalog_bytes: bytes
    catalog_counts: dict


@dataclass(frozen=True)
class ExportedModelPackage:
    filename: str
    payload: bytes
    package_sha256: str
    model_sha256: str
    catalog_counts: dict


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_package_members(payload: bytes) -> dict[str, bytes]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ModelPackageError("模型包不是有效的 ZIP 文件。") from exc

    with archive:
        members: dict[str, zipfile.ZipInfo] = {}
        total_size = 0
        allowed = REQUIRED_FILES | OPTIONAL_FILES
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts or len(path.parts) > 2:
                raise ModelPackageError("模型包包含不安全的文件路径。")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ModelPackageError("模型包不得包含符号链接。")
            if info.flag_bits & 0x1:
                raise ModelPackageError("模型包不得使用加密 ZIP 条目。")
            name = path.name
            if name not in allowed:
                raise ModelPackageError(f"模型包包含不允许的文件：{name}")
            if name in members:
                raise ModelPackageError(f"模型包包含重复文件：{name}")
            total_size += info.file_size
            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise ModelPackageError("模型包解压后体积超过安全上限。")
            if info.compress_size and info.file_size / info.compress_size > 200:
                raise ModelPackageError(f"模型包文件压缩比异常：{name}")
            members[name] = info

        missing = REQUIRED_FILES - members.keys()
        if missing:
            raise ModelPackageError("模型包缺少文件：" + "、".join(sorted(missing)))
        try:
            return {name: archive.read(info) for name, info in members.items()}
        except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
            raise ModelPackageError("模型包解压失败。") from exc


def _validate_manifest(manifest_bytes: bytes, model_bytes: bytes) -> dict:
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackageError("MODEL_MANIFEST.json 无法解析。") from exc

    expected = {
        "architecture": EXPECTED_ARCHITECTURE,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "attention_heads": EXPECTED_ATTENTION_HEADS,
        "filename": "model_base.pth",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ModelPackageError(f"模型清单字段不兼容：{key}")
    if manifest.get("size_bytes") != len(model_bytes):
        raise ModelPackageError("模型文件大小与清单不一致。")
    if manifest.get("sha256") != sha256_bytes(model_bytes):
        raise ModelPackageError("模型权重 SHA-256 与清单不一致。")

    catalog = manifest.get("prototype_catalog")
    if not isinstance(catalog, dict) or catalog.get("filename") != "haworthia_omics.db":
        raise ModelPackageError("模型清单缺少兼容的原型目录说明。")
    return manifest


def _validate_package_manifest(manifest_bytes: bytes) -> dict:
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackageError("PACKAGE_MANIFEST.json 无法解析。") from exc
    if manifest.get("package_type") != "haworthia_model_package":
        raise ModelPackageError("模型包类型不兼容。")
    if manifest.get("images_included") is not False:
        raise ModelPackageError("模型包不得包含图片。")
    if manifest.get("training_checkpoint_included") is not False:
        raise ModelPackageError("模型包不得包含训练断点。")
    return manifest


def _inspect_catalog(catalog_bytes: bytes, manifest: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="haworthia_model_catalog_") as temporary:
        path = Path(temporary) / "catalog.db"
        path.write_bytes(catalog_bytes)
        try:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise ModelPackageError("原型目录不是有效的 SQLite 数据库。") from exc
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise ModelPackageError("原型目录完整性检查失败。")
            objects = conn.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
            allowed_objects = {
                ("table", "taxonomy"),
                ("table", "images"),
                ("table", "prototypes"),
                ("table", "prototype_clusters"),
            }
            if set(objects) != allowed_objects:
                raise ModelPackageError("原型目录包含未知表、视图或触发器。")

            counts = {
                "taxa": conn.execute("SELECT COUNT(*) FROM taxonomy").fetchone()[0],
                "images": conn.execute("SELECT COUNT(*) FROM images").fetchone()[0],
                "mean_prototypes": conn.execute(
                    "SELECT COUNT(*) FROM prototypes"
                ).fetchone()[0],
                "sub_prototypes": conn.execute(
                    "SELECT COUNT(*) FROM prototype_clusters"
                ).fetchone()[0],
            }
            expected = manifest["prototype_catalog"]
            for key, value in counts.items():
                if int(expected.get(key, -1)) != value:
                    raise ModelPackageError(f"原型目录计数与清单不一致：{key}")
            if counts["images"] != 0:
                raise ModelPackageError("模型包不得包含图片记录。")
            if counts["taxa"] == 0 or counts["mean_prototypes"] == 0:
                raise ModelPackageError("模型包的类群或原型目录为空。")

            duplicate_taxa = conn.execute(
                "SELECT species, variant FROM taxonomy "
                "GROUP BY species, variant HAVING COUNT(*) > 1 LIMIT 1"
            ).fetchone()
            if duplicate_taxa:
                raise ModelPackageError("原型目录包含重复类群。")
            invalid_taxa = conn.execute(
                "SELECT id FROM taxonomy WHERE id <= 0 OR species IS NULL OR variant IS NULL "
                "OR length(species) > 256 OR length(variant) > 256 LIMIT 1"
            ).fetchone()
            if invalid_taxa:
                raise ModelPackageError("原型目录包含无效类群字段。")
            orphan_mean = conn.execute(
                "SELECT tax_id FROM prototypes WHERE tax_id NOT IN "
                "(SELECT id FROM taxonomy) LIMIT 1"
            ).fetchone()
            orphan_cluster = conn.execute(
                "SELECT tax_id FROM prototype_clusters WHERE tax_id NOT IN "
                "(SELECT id FROM taxonomy) LIMIT 1"
            ).fetchone()
            if orphan_mean or orphan_cluster:
                raise ModelPackageError("原型目录包含无法映射的类群 ID。")
            invalid_mean = conn.execute(
                "SELECT tax_id FROM prototypes WHERE length(feature_blob) != ? LIMIT 1",
                (EXPECTED_PROTOTYPE_BYTES,),
            ).fetchone()
            invalid_cluster = conn.execute(
                "SELECT tax_id FROM prototype_clusters "
                "WHERE length(feature_blob) != ? OR cluster_index < 0 OR sample_count < 0 LIMIT 1",
                (EXPECTED_PROTOTYPE_BYTES,),
            ).fetchone()
            if invalid_mean or invalid_cluster:
                raise ModelPackageError("原型目录包含维度或计数异常的数据。")
            return counts
        except sqlite3.Error as exc:
            raise ModelPackageError("原型目录结构不兼容。") from exc
        finally:
            conn.close()


def inspect_model_package(payload: bytes, expected_sha256: str) -> InspectedModelPackage:
    expected_sha256 = expected_sha256.strip().lower()
    if not SHA256_RE.fullmatch(expected_sha256):
        raise ModelPackageError("请输入导出模型包时生成的 64 位 SHA-256。")
    if not payload or len(payload) > MAX_PACKAGE_BYTES:
        raise ModelPackageError("模型包为空或超过 128 MB 安全上限。")
    package_sha256 = sha256_bytes(payload)
    if package_sha256 != expected_sha256:
        raise ModelPackageError("整包 SHA-256 不匹配；模型包可能不完整或已被修改。")

    members = _read_package_members(payload)
    _validate_package_manifest(members["PACKAGE_MANIFEST.json"])
    manifest = _validate_manifest(members["MODEL_MANIFEST.json"], members["model_base.pth"])
    counts = _inspect_catalog(members["haworthia_omics.db"], manifest)
    return InspectedModelPackage(
        package_sha256=package_sha256,
        manifest=manifest,
        model_bytes=members["model_base.pth"],
        catalog_bytes=members["haworthia_omics.db"],
        catalog_counts=counts,
    )


def _create_sanitized_catalog(source_db_path: str | Path, target_path: Path) -> dict:
    source = sqlite3.connect(Path(source_db_path).expanduser())
    target = sqlite3.connect(target_path)
    try:
        target.executescript(
            """
            CREATE TABLE taxonomy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT,
                variant TEXT
            );
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tax_id INTEGER,
                orig_path TEXT,
                seg_path TEXT
            );
            CREATE TABLE prototypes (
                tax_id INTEGER PRIMARY KEY,
                feature_blob BLOB
            );
            CREATE TABLE prototype_clusters (
                tax_id INTEGER,
                cluster_index INTEGER,
                feature_blob BLOB,
                sample_count INTEGER,
                PRIMARY KEY (tax_id, cluster_index)
            );
            """
        )
        taxonomy = source.execute(
            "SELECT id, species, variant FROM taxonomy ORDER BY id"
        ).fetchall()
        prototypes = source.execute(
            "SELECT tax_id, feature_blob FROM prototypes ORDER BY tax_id"
        ).fetchall()
        clusters = source.execute(
            "SELECT tax_id, cluster_index, feature_blob, sample_count "
            "FROM prototype_clusters ORDER BY tax_id, cluster_index"
        ).fetchall()
        target.executemany(
            "INSERT INTO taxonomy (id, species, variant) VALUES (?, ?, ?)", taxonomy
        )
        target.executemany(
            "INSERT INTO prototypes (tax_id, feature_blob) VALUES (?, ?)", prototypes
        )
        target.executemany(
            "INSERT INTO prototype_clusters "
            "(tax_id, cluster_index, feature_blob, sample_count) VALUES (?, ?, ?, ?)",
            clusters,
        )
        target.commit()
        return {
            "taxa": len(taxonomy),
            "mean_prototypes": len(prototypes),
            "sub_prototypes": len(clusters),
            "images": 0,
        }
    finally:
        target.close()
        source.close()


def build_model_package(
    model_path: str | Path,
    source_db_path: str | Path,
) -> ExportedModelPackage:
    """Create a portable user model package with no image or checkpoint data."""
    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
        raise ModelPackageError("当前模型文件不存在，无法导出。")
    model_bytes = model_path.read_bytes()
    if not model_bytes or len(model_bytes) > MAX_PACKAGE_BYTES:
        raise ModelPackageError("当前模型为空或超过模型包安全上限。")

    timestamp = datetime.now(timezone.utc)
    version = timestamp.strftime("user-%Y%m%dT%H%M%SZ")
    package_root = f"haworthia-model-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    with tempfile.TemporaryDirectory(prefix="haworthia_model_export_") as temporary:
        staging = Path(temporary) / package_root
        staging.mkdir()
        catalog_path = staging / "haworthia_omics.db"
        counts = _create_sanitized_catalog(source_db_path, catalog_path)
        if counts["mean_prototypes"] == 0:
            raise ModelPackageError("当前数据库没有数值原型，请先完成训练或原型重建。")

        model_sha256 = sha256_bytes(model_bytes)
        manifest = {
            "schema_version": 1,
            "model_version": version,
            "filename": "model_base.pth",
            "sha256": model_sha256,
            "size_bytes": len(model_bytes),
            "architecture": EXPECTED_ARCHITECTURE,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "attention_heads": EXPECTED_ATTENTION_HEADS,
            "created_at": timestamp.isoformat(),
            "prototype_catalog": {
                "filename": "haworthia_omics.db",
                **counts,
                "contents": "Taxonomy labels and numeric prototypes only; no image records or paths.",
            },
            "distribution": {
                "provided_by_application_author": False,
                "license_status": "USER_SUPPLIED_OR_UNSPECIFIED",
            },
        }
        package_manifest = {
            "format_version": 1,
            "package_type": "haworthia_model_package",
            "created_by": "Haworthia OMICS model export interface",
            "model_included": True,
            "prototype_catalog": counts,
            "images_included": False,
            "training_checkpoint_included": False,
            "rights_responsibility": (
                "The exporting and importing users are responsible for model, training-data, "
                "and redistribution rights. The application author does not provide or endorse "
                "this user-generated model package."
            ),
        }
        notice = (
            "This model package was generated by a user-operated Haworthia OMICS instance.\n"
            "It is not a pretrained model supplied or endorsed by the application author.\n"
            "The exporter and importer are responsible for all model and training-data rights.\n"
        )
        (staging / "model_base.pth").write_bytes(model_bytes)
        (staging / "MODEL_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "MODEL_PACKAGE_NOTICE.txt").write_text(notice, encoding="utf-8")

        buffer = io.BytesIO()
        with zipfile.ZipFile(
            buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(staging.iterdir()):
                archive.write(path, path.relative_to(staging.parent))
        payload = buffer.getvalue()
    if len(payload) > MAX_PACKAGE_BYTES:
        raise ModelPackageError("导出的模型包超过 128 MB 安全上限。")
    return ExportedModelPackage(
        filename=f"{package_root}.zip",
        payload=payload,
        package_sha256=sha256_bytes(payload),
        model_sha256=model_sha256,
        catalog_counts=counts,
    )


def merge_catalog(catalog_bytes: bytes, target_db_path: str | Path) -> dict:
    """Merge validated numeric prototypes while preserving local images and extra taxa."""
    target_db_path = Path(target_db_path).expanduser()
    with tempfile.TemporaryDirectory(prefix="haworthia_catalog_merge_") as temporary:
        source_path = Path(temporary) / "catalog.db"
        source_path.write_bytes(catalog_bytes)
        source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
        target = sqlite3.connect(target_db_path)
        try:
            taxa = source.execute(
                "SELECT id, species, variant FROM taxonomy ORDER BY id"
            ).fetchall()
            means = source.execute(
                "SELECT tax_id, feature_blob FROM prototypes ORDER BY tax_id"
            ).fetchall()
            clusters = source.execute(
                "SELECT tax_id, cluster_index, feature_blob, sample_count "
                "FROM prototype_clusters ORDER BY tax_id, cluster_index"
            ).fetchall()

            target.execute("BEGIN IMMEDIATE")
            local_rows = target.execute(
                "SELECT id, species, variant FROM taxonomy ORDER BY id"
            ).fetchall()
            local_ids = {(species, variant): tax_id for tax_id, species, variant in local_rows}
            package_to_local: dict[int, int] = {}
            created = 0
            for package_id, species, variant in taxa:
                key = (species, variant)
                local_id = local_ids.get(key)
                if local_id is None:
                    cursor = target.execute(
                        "INSERT INTO taxonomy (species, variant) VALUES (?, ?)", key
                    )
                    local_id = int(cursor.lastrowid)
                    local_ids[key] = local_id
                    created += 1
                package_to_local[package_id] = local_id

            imported_local_ids = sorted(package_to_local.values())
            target.executemany(
                "DELETE FROM prototype_clusters WHERE tax_id = ?",
                [(tax_id,) for tax_id in imported_local_ids],
            )
            target.executemany(
                "DELETE FROM prototypes WHERE tax_id = ?",
                [(tax_id,) for tax_id in imported_local_ids],
            )
            target.executemany(
                "INSERT INTO prototypes (tax_id, feature_blob) VALUES (?, ?)",
                [(package_to_local[tax_id], blob) for tax_id, blob in means],
            )
            target.executemany(
                "INSERT INTO prototype_clusters "
                "(tax_id, cluster_index, feature_blob, sample_count) VALUES (?, ?, ?, ?)",
                [
                    (package_to_local[tax_id], cluster_index, blob, sample_count)
                    for tax_id, cluster_index, blob, sample_count in clusters
                ],
            )
            target.commit()
            return {
                "taxa_merged": len(taxa),
                "taxa_created": created,
                "mean_prototypes": len(means),
                "sub_prototypes": len(clusters),
            }
        except Exception:
            target.rollback()
            raise
        finally:
            target.close()
            source.close()


def write_bytes_atomic(path: str | Path, payload: bytes) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.importing")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
