import io
import secrets
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
import os

import matplotlib
matplotlib.use('Agg') # 必须在引入 pyplot 之前强制切换后端
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _font_path in ("C:/Windows/Fonts/NotoSansSC-VF.ttf", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(_font_path):
        font_manager.fontManager.addfont(_font_path)
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from fastapi.responses import Response
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
try:
    import networkx as nx
except ImportError:  # Keep the rest of the inference service usable without the graph extra.
    nx = None

import base64
from collections import defaultdict

from database import (
    insert_taxonomy, get_image_count_by_tax, insert_image_record,
    get_taxonomy_records, delete_taxonomy_cascade, get_all_image_records,
    get_all_cluster_prototypes, get_image_record, get_image_records_by_tax,
    delete_image_record, get_database_overview, DB_PATH, IMG_DIR
)

from database import init_db, get_all_prototypes
from models import TemperamentOmicsNet
from dataset import val_transforms
from engine import diagnose_attention, rebuild_prototypes, run_training_loop
from model_package import (
    MAX_PACKAGE_BYTES,
    ModelPackageError,
    build_model_package,
    inspect_model_package,
    merge_catalog,
    write_bytes_atomic,
)
from segmentation import segmentation_engine

HAS_SEGMENTATION_MODELS = segmentation_engine.available

app = FastAPI(title="Haworthia OMICS Inference & Training Engine")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = str(Path(os.getenv("HAWORTHIA_MODEL_PATH", "model_base.pth")).expanduser())
CHKPT_BASE_PATH = str(
    Path(os.getenv("HAWORTHIA_CHECKPOINT_PATH", "checkpoint_base.pth")).expanduser()
)
MODEL_IMPORT_TOKEN = secrets.token_urlsafe(32)

gpu_lock = threading.Lock()
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "status": "idle",
    "message": ""
}
maintenance_state = {
    "is_running": False,
    "status": "idle",
    "message": "",
}
segmentation_state = {
    "is_running": False,
    "status": "idle",
    "message": "",
    "current": 0,
    "total": 0,
    "processed": 0,
    "recovered": 0,
    "failed": 0,
    "backup_path": "",
}

model = None
model_weights_loaded = False


def remove_background(image, mode="adaptive", sensitivity=70, return_metrics=False):
    if not HAS_SEGMENTATION_MODELS:
        raise RuntimeError(
            "缺少 ONNX Runtime 或分割模型 isnet-general-use.onnx、u2net.onnx。"
        )
    result, metrics = segmentation_engine.segment(image, mode, sensitivity)
    return (result, metrics) if return_metrics else result


def black_composite(image):
    rgba = image.convert("RGBA")
    output = Image.new("RGB", rgba.size, (0, 0, 0))
    output.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    return output


def checker_composite(image):
    rgba = image.convert("RGBA")
    width, height = rgba.size
    yy, xx = np.indices((height, width))
    checker = ((xx // 16 + yy // 16) % 2) * 35 + 180
    background = np.stack([checker, checker, checker], axis=-1).astype(np.uint8)
    output = Image.fromarray(background, mode="RGB")
    output.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    return output


def encode_preview(image, checker=False, size=220):
    preview = checker_composite(image) if checker else image.convert("RGB")
    preview = ImageOps.contain(preview, (size, size))
    canvas = Image.new("RGB", (size, size), (28, 28, 28))
    offset = ((size - preview.width) // 2, (size - preview.height) // 2)
    canvas.paste(preview, offset)
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=86)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.on_event("startup")
def startup_event():
    init_db()
    global model, model_weights_loaded
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CHKPT_BASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    model = TemperamentOmicsNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model_weights_loaded = True
    else:
        model_weights_loaded = False
    model.eval()


def require_trained_model():
    if model is None or not model_weights_loaded:
        raise HTTPException(
            status_code=409,
            detail="尚无已完成训练的模型。请先导入数据并完成训练，或配置 HAWORTHIA_MODEL_PATH。",
        )


def require_segmentation_models():
    if not HAS_SEGMENTATION_MODELS:
        raise HTTPException(
            status_code=503,
            detail=(
                "分割功能不可用。请安装 onnxruntime，并在 U2NET_HOME 中放置 "
                "isnet-general-use.onnx 和 u2net.onnx。"
            ),
        )


def _validate_model_state(model_bytes):
    try:
        state = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelPackageError("模型权重无法用安全模式读取。") from exc
    if not isinstance(state, dict) or not state:
        raise ModelPackageError("模型权重不是兼容的 state_dict。")

    expected = model.state_dict()
    if set(state) != set(expected):
        raise ModelPackageError("模型权重的参数名称与当前程序不兼容。")
    for name, expected_tensor in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ModelPackageError(f"模型参数不是张量：{name}")
        if value.shape != expected_tensor.shape or value.dtype != expected_tensor.dtype:
            raise ModelPackageError(f"模型参数形状或类型不兼容：{name}")
    return state


def _backup_database(source_path, target_path):
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def _restore_database(backup_path, target_path):
    source = sqlite3.connect(backup_path)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


class TrainRequest(BaseModel):
    epochs: int = Field(default=300, ge=1, le=1000)
    alpha: float = Field(default=0.05, ge=0.0, le=0.5)
    p_classes: int = Field(default=16, ge=2, le=128)
    k_instances: int = Field(default=4, ge=2, le=16)
    resume: bool = False
    max_lambda_orth: float = Field(default=0.02, ge=0.0, le=0.2)
    lambda_tv: float = Field(default=0.05, ge=0.0, le=0.5)
    lambda_entropy: float = Field(default=0.05, ge=0.0, le=0.5)
    target_attention_entropy: float = Field(default=0.55, ge=0.0, le=1.0)
    prototypes_per_taxon: int = Field(default=3, ge=1, le=6)
    quality_aware: bool = True


class PrototypeRequest(BaseModel):
    prototypes_per_taxon: int = Field(default=3, ge=1, le=6)


class SnapshotRequest(BaseModel):
    include_images: bool = False


class ResegmentRequest(BaseModel):
    mode: Literal["strict", "adaptive", "lenient"] = "adaptive"
    sensitivity: int = Field(default=70, ge=0, le=100)
    scope: Literal["suspicious", "all"] = "suspicious"
    backup_existing: bool = True


class SegmentationRestoreRequest(BaseModel):
    backup_path: str


class IndividualResegmentRequest(BaseModel):
    mode: Literal["strict", "adaptive", "lenient"] = "adaptive"
    sensitivity: int = Field(default=70, ge=0, le=100)


def background_training_task(req: TrainRequest):
    global training_state, model_weights_loaded
    if not gpu_lock.acquire(blocking=False):
        training_state.update({
            "is_training": False, "status": "failed", "message": "GPU 已被锁定"
        })
        return

    try:
        training_state.update({
            "total_epochs": req.epochs,
            "current_epoch": 0,
            "status": "running",
            "message": "流形度量收敛中..."
        })

        # 将新参数透传至计算引擎
        completed = run_training_loop(
            model=model,
            device=DEVICE,
            state_bus=training_state,
            target_epochs=req.epochs,
            alpha_target=req.alpha,
            p_classes=req.p_classes,
            k_instances=req.k_instances,
            resume=req.resume,
            model_path=MODEL_PATH,
            chkpt_path=CHKPT_BASE_PATH,
            max_lambda_orth=req.max_lambda_orth,
            lambda_tv=req.lambda_tv,
            lambda_entropy=req.lambda_entropy,
            target_attention_entropy=req.target_attention_entropy,
            prototypes_per_taxon=req.prototypes_per_taxon,
            quality_aware=req.quality_aware,
        )
        if completed:
            model_weights_loaded = True
        training_state["status"] = "completed" if completed else "stopped"
    except Exception as e:
        training_state.update({"status": "failed", "message": str(e)})
    finally:
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        training_state["is_training"] = False
        gpu_lock.release()

@app.post("/api/train/start")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=400, detail="训练流正在运行。")
    training_state.update({"is_training": True, "status": "queued", "message": "训练任务排队中..."})
    background_tasks.add_task(background_training_task, req)
    return {"message": "GPU 训练指令已下达"}


@app.get("/api/train/status")
def get_status():
    return {
        **training_state,
        "model_weights_loaded": bool(model_weights_loaded),
        "checkpoint_available": Path(CHKPT_BASE_PATH).is_file(),
    }


def _model_file_info(path):
    file_path = Path(path)
    if not file_path.exists():
        return {"exists": False, "name": file_path.name}
    stat = file_path.stat()
    return {
        "exists": True,
        "name": file_path.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        ),
    }


@app.get("/api/overview")
def get_overview():
    """Expose a read-only health summary for the dashboard."""
    overview = get_database_overview()
    overview["model"] = {
        "device": str(DEVICE),
        "initialized": model is not None,
        "loaded": bool(model_weights_loaded),
        "base": _model_file_info(MODEL_PATH),
        "checkpoint": _model_file_info(CHKPT_BASE_PATH),
        "prototype_coverage": (
            round(overview["prototype_count"] / overview["taxonomy_count"] * 100, 1)
            if overview["taxonomy_count"] else 0.0
        ),
    }
    overview["training"] = dict(training_state)
    overview["maintenance"] = dict(maintenance_state)
    overview["segmentation"] = {
        **dict(segmentation_state),
        "available": bool(HAS_SEGMENTATION_MODELS),
        "required_models": ["isnet-general-use.onnx", "u2net.onnx"],
    }
    overview["model_import"] = {
        "format": "Haworthia OMICS 模型包 ZIP + 整包 SHA-256",
        "max_size_mb": MAX_PACKAGE_BYTES // (1024 * 1024),
        "session_token": MODEL_IMPORT_TOKEN,
    }
    return overview


@app.post("/api/train/stop")
def stop_training():
    if training_state["is_training"]:
        training_state["is_training"] = False
        return {"message": "已发送安全中断信号 (Graceful Stop)。等待当前 Epoch 结束。"}
    return {"message": "无运行中的训练任务。"}


# ==========================================
# 替换路由 1: 开放集推理 (加入背景剥离与 Base64 回传)
# ==========================================
@app.post("/api/inference/predict")
async def predict_phenotype(
        file: UploadFile = File(...),
        rejection_threshold: float = Form(0.55),
        segmentation_mode: str = Form("adaptive"),
        segmentation_sensitivity: int = Form(70),
):
    if training_state["is_training"] or segmentation_state["is_running"]:
        raise HTTPException(status_code=503, detail="GPU 独占中，系统正在重构流形空间。")
    require_trained_model()
    require_segmentation_models()

    if segmentation_mode not in segmentation_engine.MODES:
        raise HTTPException(status_code=422, detail="未知分割模式。")
    segmentation_sensitivity = min(max(segmentation_sensitivity, 0), 100)

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    segmented = remove_background(img, segmentation_mode, segmentation_sensitivity)

    display_image = black_composite(segmented)
    buffered = io.BytesIO()
    display_image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    tensor_in, foreground_mask = val_transforms.apply_with_mask(segmented)
    tensor_in = tensor_in.unsqueeze(0).to(DEVICE)
    foreground_mask = foreground_mask.unsqueeze(0).to(DEVICE)

    with gpu_lock:
        with torch.no_grad():
            feat = model(tensor_in, fg_mask=foreground_mask).cpu()

    rows = get_all_cluster_prototypes()
    if not rows:
        raise HTTPException(status_code=404, detail="知识库拓扑为空。")

    best_by_taxon = {}
    for tax_id, species, variant, cluster_index, blob, sample_count in rows:
        proto = torch.from_numpy(np.frombuffer(blob, dtype=np.float32).copy()).unsqueeze(0)
        sim = F.cosine_similarity(feat, proto).item()
        previous = best_by_taxon.get(tax_id)
        if previous is None or sim > previous["confidence"]:
            best_by_taxon[tax_id] = {
                "label": f"{species} - {variant}",
                "confidence": float(sim),
                "prototype": int(cluster_index) + 1,
            }

    sim_list = list(best_by_taxon.values())
    sim_list.sort(key=lambda x: x["confidence"], reverse=True)
    top_similarity = sim_list[0]["confidence"]
    return {
        "predictions": sim_list[:5],
        "segmented_image_base64": img_b64,
        "is_unknown": bool(top_similarity < rejection_threshold),
        "top_similarity": float(top_similarity),
        "rejection_threshold": float(rejection_threshold),
    }


# ==========================================
# 扩展路由 1: 数据流导入与物理回溯删除
# ==========================================
@app.post("/api/taxonomy/upload")
async def upload_image(
        species: str = Form(...),
        variant: str = Form(...),
        segmentation_mode: str = Form("adaptive"),
        segmentation_sensitivity: int = Form(70),
        file: UploadFile = File(...)
):
    if training_state["is_training"] or segmentation_state["is_running"]:
        raise HTTPException(status_code=409, detail="训练或重分割任务正在运行。")
    require_segmentation_models()
    if segmentation_mode not in segmentation_engine.MODES:
        raise HTTPException(status_code=422, detail="未知分割模式。")
    segmentation_sensitivity = min(max(segmentation_sensitivity, 0), 100)
    tax_id, is_new = insert_taxonomy(species, variant)

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((512, 512))

    image_token = uuid.uuid4().hex[:12]
    op = os.path.join(IMG_DIR, f"{tax_id}_{image_token}_orig.jpg")
    sp = os.path.join(IMG_DIR, f"{tax_id}_{image_token}_seg.png")
    img.save(op)

    segmented, metrics = remove_background(
        img, segmentation_mode, segmentation_sensitivity, return_metrics=True
    )
    segmented.save(sp)

    insert_image_record(tax_id, op, sp)
    return {
        "message": "入库成功",
        "tax_id": tax_id,
        "is_new_taxon": is_new,
        "segmentation": metrics,
    }


@app.post("/api/taxonomy/segment-preview")
async def preview_segmentation(
        file: UploadFile = File(...),
        segmentation_mode: str = Form("adaptive"),
        segmentation_sensitivity: int = Form(70),
):
    if training_state["is_training"] or segmentation_state["is_running"]:
        raise HTTPException(status_code=409, detail="训练或重分割任务正在运行。")
    require_segmentation_models()
    if segmentation_mode not in segmentation_engine.MODES:
        raise HTTPException(status_code=422, detail="未知分割模式。")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((512, 512))
    segmented, metrics = remove_background(
        image,
        segmentation_mode,
        min(max(segmentation_sensitivity, 0), 100),
        return_metrics=True,
    )

    buffer = io.BytesIO()
    checker_composite(segmented).save(buffer, format="JPEG", quality=92)
    return {
        "preview_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "metrics": metrics,
    }


@app.get("/api/taxonomy/records")
def list_records():
    records = get_taxonomy_records()
    prototype_tax_ids = {row[0] for row in get_all_cluster_prototypes()}
    results = []
    for r in records:
        count = get_image_count_by_tax(r[0])
        results.append({
            "id": r[0], "species": r[1], "variant": r[2],
            "image_count": count,
            "has_prototype": r[0] in prototype_tax_ids,
        })
    return results


@app.get("/api/taxonomy/{tax_id}/images")
def list_taxonomy_images(tax_id: int, sensitivity: int = 70):
    sensitivity = min(max(sensitivity, 0), 100)
    results = []
    for image_id, record_tax_id, orig_path, seg_path in get_image_records_by_tax(tax_id):
        if not os.path.exists(orig_path) or not os.path.exists(seg_path):
            continue
        with Image.open(orig_path) as original, Image.open(seg_path) as segmented:
            rgba = segmented.convert("RGBA")
            metrics = segmentation_engine.analyze_alpha(
                rgba.getchannel("A"), sensitivity
            )
            results.append({
                "id": image_id,
                "tax_id": record_tax_id,
                "original_preview_base64": encode_preview(original),
                "segmented_preview_base64": encode_preview(rgba, checker=True),
                "metrics": metrics,
            })
    return results


@app.post("/api/taxonomy/images/{image_id}/resegment")
def resegment_single_image(image_id: int, req: IndividualResegmentRequest):
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="已有训练或维护任务正在运行。")
    require_segmentation_models()
    record = get_image_record(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="图像记录不存在。")
    _, tax_id, orig_path, seg_path = record
    if not os.path.exists(orig_path) or not os.path.exists(seg_path):
        raise HTTPException(status_code=404, detail="原图或分割图文件缺失。")
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="计算资源已被锁定。")

    temporary_path = Path(seg_path).with_suffix(".single.new.png")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = Path("backups") / "individual_segments"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{timestamp}_{image_id}_{Path(seg_path).name}"
        shutil.copy2(seg_path, backup_path)

        image = Image.open(orig_path).convert("RGB")
        segmented, metrics = remove_background(
            image, req.mode, req.sensitivity, return_metrics=True
        )
        segmented.save(temporary_path, format="PNG")
        os.replace(temporary_path, seg_path)
        return {
            "message": "单图重分割完成。",
            "image_id": image_id,
            "tax_id": tax_id,
            "backup_path": str(backup_path.resolve()),
            "segmented_preview_base64": encode_preview(segmented, checker=True),
            "metrics": metrics,
        }
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        gpu_lock.release()


@app.delete("/api/taxonomy/images/{image_id}")
def delete_single_image(image_id: int):
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="已有训练或维护任务正在运行。")
    record = get_image_record(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="图像记录不存在。")
    _, tax_id, orig_path, seg_path = record

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_dir = Path("backups") / "deleted_images" / f"{timestamp}_{image_id}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in (orig_path, seg_path):
        if os.path.exists(path):
            shutil.copy2(path, archive_dir / Path(path).name)

    if not delete_image_record(image_id):
        raise HTTPException(status_code=500, detail="数据库图像记录删除失败。")
    for path in (orig_path, seg_path):
        if os.path.exists(path):
            os.remove(path)
    return {
        "message": "图像已从数据集删除并归档。",
        "remaining": get_image_count_by_tax(tax_id),
        "archive_path": str(archive_dir.resolve()),
    }


@app.delete("/api/taxonomy/delete/{tax_id}")
def delete_record(tax_id: int):
    if training_state["is_training"] or segmentation_state["is_running"]:
        raise HTTPException(status_code=503, detail="计算引擎训练中，禁止回溯数据库。")
    delete_taxonomy_cascade(tax_id)
    return {"message": f"成功回溯清除类群节点 (ID: {tax_id}) 及其物理影像"}


@app.get("/api/evolution/dendrogram")
def plot_dendrogram():
    rows = get_all_prototypes()
    if len(rows) < 3:
        raise HTTPException(status_code=400, detail="节点不足。")
    labels = [f"{r[1].replace('Haworthia', '').strip()} - {r[2].strip()}" for r in rows]
    feats = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    sim_matrix = np.dot(feats, feats.T)
    dist_matrix = np.clip(1.0 - sim_matrix, 0, 2)
    dist_array = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]

    Z = linkage(dist_array, method='average')
    fig, ax = plt.subplots(figsize=(8, max(4.0, len(labels) * 0.25)))
    dendrogram(Z, labels=labels, ax=ax, orientation='right', leaf_font_size=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


def _visual_evidence_metrics(orig_path, seg_path, sensitivity=70):
    """Compute image-derived evidence without assigning botanical meaning."""
    with Image.open(seg_path) as segmented:
        if "A" in segmented.getbands():
            alpha_image = segmented.getchannel("A").copy()
        else:
            alpha_image = Image.new("L", segmented.size, 255)
    metrics = segmentation_engine.analyze_alpha(alpha_image, sensitivity)
    alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    foreground = alpha > 0.10
    height, width = alpha.shape
    if foreground.any():
        ys, xs = np.where(foreground)
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        bbox_area_ratio = float(bbox_width * bbox_height / alpha.size)
        bbox_aspect_ratio = float(bbox_width / max(bbox_height, 1))
        weighted = np.clip(alpha, 0.0, 1.0)
        total_weight = float(weighted.sum())
        center_x = float((weighted * np.arange(width)[None, :]).sum() / max(total_weight, 1e-6) / width)
        center_y = float((weighted * np.arange(height)[:, None]).sum() / max(total_weight, 1e-6) / height)
        center_offset = float(np.hypot(center_x - 0.5, center_y - 0.5))
    else:
        bbox_area_ratio = 0.0
        bbox_aspect_ratio = 0.0
        center_offset = 1.0

    with Image.open(orig_path) as original:
        rgb = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.shape[:2] != alpha.shape:
        rgb_image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        rgb = np.asarray(rgb_image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    pixels = rgb[alpha > 0.50]
    if pixels.size:
        brightness = float(pixels.mean())
        saturation = float((pixels.max(axis=1) - pixels.min(axis=1)).mean())
    else:
        brightness = 0.0
        saturation = 0.0
    metrics.update({
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "center_offset": center_offset,
        "masked_brightness": brightness,
        "masked_saturation": saturation,
    })
    return metrics


def _preview_pair(orig_path, seg_path):
    with Image.open(orig_path) as original:
        original_preview = encode_preview(original.convert("RGB"), size=240)
    with Image.open(seg_path) as segmented:
        segmented_preview = encode_preview(segmented.convert("RGBA"), checker=True, size=240)
    return original_preview, segmented_preview


def _extract_record_features(records):
    """Extract current-model embeddings for a small, user-selected taxon."""
    tensors = []
    masks = []
    for _, _, orig_path, seg_path in records:
        with Image.open(orig_path) as original, Image.open(seg_path) as segmented:
            if "A" in segmented.getbands():
                alpha = segmented.getchannel("A").copy()
            else:
                alpha = Image.new("L", segmented.size, 255)
            tensor, mask = val_transforms.apply_with_mask(original.convert("RGB"), alpha)
        tensors.append(tensor)
        masks.append(mask)

    if not tensors:
        return torch.empty((0, 128))
    images = torch.stack(tensors).to(DEVICE)
    foreground = torch.stack(masks).to(DEVICE)
    model.eval()
    with torch.no_grad():
        features = model(images, fg_mask=foreground).cpu()
    return features


def _taxon_lookup():
    return {
        row[0]: {
            "id": row[0],
            "species": row[1],
            "variant": row[2],
        }
        for row in get_taxonomy_records()
    }


@app.get("/api/evidence/report/{tax_id}")
def evidence_report(tax_id: int, sensitivity: int = 70):
    """Return evidence-based visual traits and real multi-prototype examples."""
    if training_state["is_training"] or segmentation_state["is_running"]:
        raise HTTPException(status_code=503, detail="训练或重分割正在占用计算资源。")
    require_trained_model()
    taxon = _taxon_lookup().get(tax_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="类群记录不存在。")
    records = get_image_records_by_tax(tax_id)
    records = [r for r in records if os.path.exists(r[2]) and os.path.exists(r[3])]
    if not records:
        raise HTTPException(status_code=404, detail="该类群没有可用图像。")

    sensitivity = min(max(int(sensitivity), 0), 100)
    visual_rows = []
    for image_id, _, orig_path, seg_path in records:
        metrics = _visual_evidence_metrics(orig_path, seg_path, sensitivity)
        visual_rows.append({
            "image_id": image_id,
            "orig_path": orig_path,
            "seg_path": seg_path,
            "metrics": metrics,
        })

    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="GPU 正在执行其他任务。")
    try:
        features = _extract_record_features(records)
        cluster_rows = [row for row in get_all_cluster_prototypes() if row[0] == tax_id]
        if not cluster_rows:
            cluster_rows = [row for row in get_all_prototypes() if row[0] == tax_id]
        cluster_rows = sorted(cluster_rows, key=lambda row: row[3] if len(row) > 4 else 0)
        if cluster_rows and len(features):
            prototypes = torch.stack([
                torch.from_numpy(np.frombuffer(row[4] if len(row) > 5 else row[3], dtype=np.float32).copy())
                for row in cluster_rows
            ])
            similarities = features @ prototypes.T
            best_cluster = similarities.argmax(dim=1).numpy()
            best_similarity = similarities.max(dim=1).values.numpy()
        else:
            similarities = None
            best_cluster = np.zeros(len(records), dtype=np.int64)
            best_similarity = np.zeros(len(records), dtype=np.float32)

        for index, row in enumerate(visual_rows):
            row["best_prototype"] = int(best_cluster[index]) + 1
            row["model_similarity"] = float(best_similarity[index])
    finally:
        gpu_lock.release()

    numeric_keys = [
        "foreground_ratio", "bbox_area_ratio", "bbox_aspect_ratio", "center_offset",
        "masked_brightness", "masked_saturation", "quality_score", "translucent_fraction",
    ]
    aggregate = {
        key: {
            "median": float(np.median([row["metrics"][key] for row in visual_rows])),
            "p10": float(np.percentile([row["metrics"][key] for row in visual_rows], 10)),
            "p90": float(np.percentile([row["metrics"][key] for row in visual_rows], 90)),
        }
        for key in numeric_keys
    }
    suspicious_count = sum(row["metrics"].get("suspicious", False) for row in visual_rows)
    observations = [
        {
            "label": "主体覆盖",
            "text": f"前景占图像面积中位数 {aggregate['foreground_ratio']['median'] * 100:.1f}%，样本范围覆盖第10至第90百分位。",
            "basis": "分割掩膜",
        },
        {
            "label": "轮廓框比例",
            "text": f"前景外接框面积占比中位数 {aggregate['bbox_area_ratio']['median'] * 100:.1f}%，长宽比中位数 {aggregate['bbox_aspect_ratio']['median']:.2f}。",
            "basis": "分割轮廓",
        },
        {
            "label": "主体位置",
            "text": f"前景重心距画面中心中位数 {aggregate['center_offset']['median']:.3f}；数值越小表示主体越居中。",
            "basis": "Alpha 加权重心",
        },
        {
            "label": "颜色统计",
            "text": f"前景亮度中位数 {aggregate['masked_brightness']['median']:.3f}，颜色离散度代理值 {aggregate['masked_saturation']['median']:.3f}。",
            "basis": "原图前景区域",
        },
        {
            "label": "分割可靠性",
            "text": f"{suspicious_count} / {len(visual_rows)} 张图像仍被自动质量指标标记为疑似问题。",
            "basis": "分割质量指标",
        },
    ]

    board = []
    if cluster_rows:
        for cluster_index, cluster in enumerate(cluster_rows):
            candidates = [i for i, assignment in enumerate(best_cluster) if assignment == cluster_index]
            if not candidates and len(visual_rows):
                candidates = [int(np.argmax(similarities[:, cluster_index].numpy()))] if similarities is not None else [0]
            if not candidates:
                continue
            representative = max(candidates, key=lambda i: visual_rows[i]["model_similarity"])
            source = visual_rows[representative]
            original_preview, segmented_preview = _preview_pair(source["orig_path"], source["seg_path"])
            board.append({
                "prototype": cluster_index + 1,
                "stored_sample_count": int(cluster[5]) if len(cluster) > 5 else 0,
                "assigned_image_count": len(candidates),
                "similarity": source["model_similarity"],
                "image_id": source["image_id"],
                "original_preview_base64": original_preview,
                "segmented_preview_base64": segmented_preview,
                "status": "model_prototype",
            })
    else:
        # A newly imported taxon may not have a prototype until the next training run.
        # Still provide real representative images without pretending they are model clusters.
        fallback_sources = sorted(
            visual_rows,
            key=lambda row: row["metrics"].get("quality_score", 0.0),
            reverse=True,
        )[:3]
        for source in fallback_sources:
            original_preview, segmented_preview = _preview_pair(source["orig_path"], source["seg_path"])
            board.append({
                "prototype": None,
                "stored_sample_count": 0,
                "assigned_image_count": 1,
                "similarity": None,
                "image_id": source["image_id"],
                "original_preview_base64": original_preview,
                "segmented_preview_base64": segmented_preview,
                "status": "real_representative_no_model_prototype",
            })

    evidence_images = []
    for source in sorted(visual_rows, key=lambda row: row["model_similarity"], reverse=True)[:6]:
        original_preview, segmented_preview = _preview_pair(source["orig_path"], source["seg_path"])
        evidence_images.append({
            "image_id": source["image_id"],
            "best_prototype": source["best_prototype"],
            "similarity": source["model_similarity"],
            "metrics": source["metrics"],
            "original_preview_base64": original_preview,
            "segmented_preview_base64": segmented_preview,
        })

    return {
        "taxon": taxon,
        "image_count": len(visual_rows),
        "interpretation_scope": "这些文字是图像和分割掩膜的可观测证据，不是植物学定论。",
        "prototype_status": "available" if cluster_rows else "not_available",
        "aggregate": aggregate,
        "observations": observations,
        "prototype_board": board,
        "evidence_images": evidence_images,
        "mean_model_similarity": float(np.mean(best_similarity)) if len(best_similarity) else 0.0,
        "model_similarity_p10": float(np.percentile(best_similarity, 10)) if len(best_similarity) else 0.0,
    }


_COMPARISON_TRAITS = {
    "foreground_ratio": ("前景面积占比", "percent"),
    "bbox_area_ratio": ("前景外接框面积占比", "percent"),
    "bbox_aspect_ratio": ("前景外接框长宽比", "number"),
    "center_offset": ("主体重心偏移", "number"),
    "masked_brightness": ("前景亮度", "number"),
    "masked_saturation": ("前景颜色离散度", "number"),
    "quality_score": ("分割质量分", "percent"),
    "translucent_fraction": ("半透明前景占比", "percent"),
}


def _bootstrap_median_difference(values_a, values_b, seed, iterations=600):
    """Estimate a deterministic percentile interval for median(A) - median(B)."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if not len(values_a) or not len(values_b):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample_a = rng.choice(values_a, size=len(values_a), replace=True)
        sample_b = rng.choice(values_b, size=len(values_b), replace=True)
        differences[index] = np.median(sample_a) - np.median(sample_b)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def _cliffs_delta(values_a, values_b):
    """Return Cliff's delta without constructing an N by M comparison matrix."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.sort(np.asarray(values_b, dtype=np.float64))
    if not len(values_a) or not len(values_b):
        return 0.0
    greater = sum(np.searchsorted(values_b, value, side="left") for value in values_a)
    smaller = sum(
        len(values_b) - np.searchsorted(values_b, value, side="right")
        for value in values_a
    )
    return float((greater - smaller) / (len(values_a) * len(values_b)))


def _effect_magnitude(delta):
    absolute = abs(delta)
    if absolute < 0.147:
        return "可忽略"
    if absolute < 0.33:
        return "小"
    if absolute < 0.474:
        return "中等"
    return "大"


def _normalized_prototypes(tax_id):
    rows = [row for row in get_all_cluster_prototypes() if row[0] == tax_id]
    if not rows:
        return [], np.empty((0, 128), dtype=np.float32)
    rows = sorted(rows, key=lambda row: row[3])
    vectors = np.stack([
        np.frombuffer(row[4], dtype=np.float32).copy()
        for row in rows
    ])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    return rows, vectors


def _comparison_image_payload(row, own_similarity, other_similarity, margin):
    original_preview, segmented_preview = _preview_pair(row["orig_path"], row["seg_path"])
    return {
        "image_id": row["image_id"],
        "own_similarity": float(own_similarity),
        "other_similarity": float(other_similarity),
        "margin": float(margin),
        "quality_score": float(row["metrics"].get("quality_score", 0.0)),
        "suspicious": bool(row["metrics"].get("suspicious", False)),
        "original_preview_base64": original_preview,
        "segmented_preview_base64": segmented_preview,
    }


@app.get("/api/evidence/compare")
def compare_taxa(tax_id_a: int, tax_id_b: int, sensitivity: int = 70):
    """Compare two taxa using current-gallery evidence and the current model."""
    if tax_id_a == tax_id_b:
        raise HTTPException(status_code=422, detail="请选择两个不同的类群。")
    if training_state["is_training"] or segmentation_state["is_running"]:
        raise HTTPException(status_code=503, detail="训练或重分割正在占用计算资源。")
    require_trained_model()

    taxon_lookup = _taxon_lookup()
    taxon_a = taxon_lookup.get(tax_id_a)
    taxon_b = taxon_lookup.get(tax_id_b)
    if taxon_a is None or taxon_b is None:
        raise HTTPException(status_code=404, detail="至少有一个类群记录不存在。")

    records_a = [
        row for row in get_image_records_by_tax(tax_id_a)
        if os.path.exists(row[2]) and os.path.exists(row[3])
    ]
    records_b = [
        row for row in get_image_records_by_tax(tax_id_b)
        if os.path.exists(row[2]) and os.path.exists(row[3])
    ]
    if not records_a or not records_b:
        raise HTTPException(status_code=404, detail="至少有一个类群没有可用图像。")

    sensitivity = min(max(int(sensitivity), 0), 100)

    def visual_rows(records):
        rows = []
        for image_id, _, orig_path, seg_path in records:
            rows.append({
                "image_id": image_id,
                "orig_path": orig_path,
                "seg_path": seg_path,
                "metrics": _visual_evidence_metrics(orig_path, seg_path, sensitivity),
            })
        return rows

    visual_a = visual_rows(records_a)
    visual_b = visual_rows(records_b)

    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="GPU 正在执行其他任务。")
    try:
        features_a = F.normalize(_extract_record_features(records_a), p=2, dim=1).numpy()
        features_b = F.normalize(_extract_record_features(records_b), p=2, dim=1).numpy()
    finally:
        gpu_lock.release()

    center_a = features_a.mean(axis=0)
    center_b = features_b.mean(axis=0)
    center_a /= max(float(np.linalg.norm(center_a)), 1e-8)
    center_b /= max(float(np.linalg.norm(center_b)), 1e-8)
    centroid_similarity = float(np.clip(center_a @ center_b, -1.0, 1.0))

    own_a = features_a @ center_a
    other_a = features_a @ center_b
    own_b = features_b @ center_b
    other_b = features_b @ center_a
    margins_a = own_a - other_a
    margins_b = own_b - other_b
    separation_a = float(np.mean(margins_a > 0.0))
    separation_b = float(np.mean(margins_b > 0.0))

    embedding = {
        "centroid_cosine_similarity": centroid_similarity,
        "a": {
            "sample_count": len(features_a),
            "within_class_compactness": float(np.mean(own_a)),
            "gallery_separation_rate": separation_a,
            "mean_margin": float(np.mean(margins_a)),
            "median_margin": float(np.median(margins_a)),
        },
        "b": {
            "sample_count": len(features_b),
            "within_class_compactness": float(np.mean(own_b)),
            "gallery_separation_rate": separation_b,
            "mean_margin": float(np.mean(margins_b)),
            "median_margin": float(np.median(margins_b)),
        },
        "scope": "分离率和边界差值来自参与当前图库比较的图像，不是留出测试集准确率。",
    }

    trait_rows = []
    for trait_index, (key, (label, display)) in enumerate(_COMPARISON_TRAITS.items()):
        values_a = np.asarray([row["metrics"][key] for row in visual_a], dtype=np.float64)
        values_b = np.asarray([row["metrics"][key] for row in visual_b], dtype=np.float64)
        median_a = float(np.median(values_a))
        median_b = float(np.median(values_b))
        difference = median_a - median_b
        ci_low, ci_high = _bootstrap_median_difference(
            values_a,
            values_b,
            seed=tax_id_a * 100003 + tax_id_b * 1009 + trait_index,
        )
        delta = _cliffs_delta(values_a, values_b)
        trait_rows.append({
            "key": key,
            "label": label,
            "display": display,
            "a_median": median_a,
            "a_p10": float(np.percentile(values_a, 10)),
            "a_p90": float(np.percentile(values_a, 90)),
            "b_median": median_b,
            "b_p10": float(np.percentile(values_b, 10)),
            "b_p90": float(np.percentile(values_b, 90)),
            "median_difference_a_minus_b": difference,
            "difference_ci95_low": ci_low,
            "difference_ci95_high": ci_high,
            "cliffs_delta": delta,
            "effect_magnitude": _effect_magnitude(delta),
        })

    prototype_rows_a, prototypes_a = _normalized_prototypes(tax_id_a)
    prototype_rows_b, prototypes_b = _normalized_prototypes(tax_id_b)
    prototype_evidence = {
        "status": "available" if len(prototypes_a) and len(prototypes_b) else "incomplete",
        "a_count": len(prototypes_a),
        "b_count": len(prototypes_b),
        "a_sample_counts": [int(row[5]) for row in prototype_rows_a],
        "b_sample_counts": [int(row[5]) for row in prototype_rows_b],
        "matrix": [],
        "strongest_match": None,
        "least_shared_a": None,
        "least_shared_b": None,
    }
    if len(prototypes_a) and len(prototypes_b):
        prototype_matrix = np.clip(prototypes_a @ prototypes_b.T, -1.0, 1.0)
        strongest_a, strongest_b = np.unravel_index(
            int(np.argmax(prototype_matrix)), prototype_matrix.shape
        )
        a_cross_max = prototype_matrix.max(axis=1)
        b_cross_max = prototype_matrix.max(axis=0)
        least_shared_a = int(np.argmin(a_cross_max))
        least_shared_b = int(np.argmin(b_cross_max))
        prototype_evidence.update({
            "matrix": prototype_matrix.tolist(),
            "strongest_match": {
                "a_prototype": int(strongest_a) + 1,
                "b_prototype": int(strongest_b) + 1,
                "similarity": float(prototype_matrix[strongest_a, strongest_b]),
            },
            "least_shared_a": {
                "prototype": least_shared_a + 1,
                "best_cross_similarity": float(a_cross_max[least_shared_a]),
            },
            "least_shared_b": {
                "prototype": least_shared_b + 1,
                "best_cross_similarity": float(b_cross_max[least_shared_b]),
            },
        })

    representative_a = [
        _comparison_image_payload(visual_a[index], own_a[index], other_a[index], margins_a[index])
        for index in np.argsort(-own_a)[:3]
    ]
    representative_b = [
        _comparison_image_payload(visual_b[index], own_b[index], other_b[index], margins_b[index])
        for index in np.argsort(-own_b)[:3]
    ]
    boundary_a_index = int(np.argmin(margins_a))
    boundary_b_index = int(np.argmin(margins_b))

    average_separation = (separation_a + separation_b) / 2.0
    minimum_median_margin = min(float(np.median(margins_a)), float(np.median(margins_b)))
    if average_separation >= 0.85 and minimum_median_margin > 0.03:
        embedding_text = "当前图库中的两组嵌入大多可由各自类群中心分开。"
    elif average_separation >= 0.65 and minimum_median_margin > 0.0:
        embedding_text = "当前图库中的两组嵌入存在可辨方向，但仍有明显交叠或边界样本。"
    else:
        embedding_text = "当前图库中的两组嵌入交叠较强，单凭当前模型不宜作稳定区分。"

    observable_traits = [
        row for row in trait_rows
        if row["key"] not in {"quality_score", "translucent_fraction"}
    ]
    strongest_traits = sorted(
        observable_traits, key=lambda row: abs(row["cliffs_delta"]), reverse=True
    )[:3]
    meaningful_traits = [row for row in strongest_traits if abs(row["cliffs_delta"]) >= 0.147]
    if meaningful_traits:
        trait_fragments = [
            f"{row['label']}（{'A 较高' if row['median_difference_a_minus_b'] > 0 else 'B 较高'}，"
            f"效应量{row['effect_magnitude']}）"
            for row in meaningful_traits
        ]
        trait_text = "当前图库中相对更明显的图像差异为：" + "、".join(trait_fragments) + "。"
    else:
        trait_text = "当前图库的基础轮廓、位置和颜色统计中，没有出现稳定的大幅差异。"

    summary = [
        {
            "title": "模型嵌入",
            "text": f"{embedding_text} 两个图库中心的余弦相似度为 {centroid_similarity:.4f}。",
        },
        {
            "title": "可观测图像证据",
            "text": trait_text + " 这些差异也可能受拍摄、栽培状态和分割质量影响。",
        },
    ]
    if prototype_evidence["status"] == "available":
        strongest = prototype_evidence["strongest_match"]
        summary.append({
            "title": "子原型对应",
            "text": (
                f"最接近的是 A 子原型 {strongest['a_prototype']} 与 B 子原型 "
                f"{strongest['b_prototype']}，余弦相似度 {strongest['similarity']:.4f}；"
                "其余配对可用于观察两个类群内部模式是否一一对应。"
            ),
        })
    else:
        summary.append({
            "title": "子原型对应",
            "text": "至少一个类群尚无当前模型子原型，因此不生成子原型配对结论。",
        })

    suspicious_a = sum(row["metrics"].get("suspicious", False) for row in visual_a)
    suspicious_b = sum(row["metrics"].get("suspicious", False) for row in visual_b)
    warnings = []
    if len(visual_a) < 5 or len(visual_b) < 5:
        warnings.append("至少一个类群少于 5 张可用图片，区间和效应量可能不稳定。")
    if suspicious_a / len(visual_a) > 0.25 or suspicious_b / len(visual_b) > 0.25:
        warnings.append("至少一个类群有超过四分之一的图像被分割质量规则标记，形态统计需谨慎解读。")
    if prototype_evidence["status"] != "available":
        warnings.append("子原型证据不完整；完成包含这两个类群的训练后才能比较其内部原型模式。")
    if average_separation < 0.65:
        warnings.append("图库内分离率偏低，边界样本不应被表述为可靠的物种诊断特征。")

    return {
        "taxon_a": taxon_a,
        "taxon_b": taxon_b,
        "scope": "报告只比较当前图库、当前分割结果和当前模型嵌入，不代表遗传亲缘、演化方向或植物学诊断。",
        "sensitivity": sensitivity,
        "embedding": embedding,
        "summary": summary,
        "traits": trait_rows,
        "sub_prototypes": prototype_evidence,
        "representative_images": {
            "a": representative_a,
            "b": representative_b,
        },
        "boundary_images": {
            "a": _comparison_image_payload(
                visual_a[boundary_a_index], own_a[boundary_a_index],
                other_a[boundary_a_index], margins_a[boundary_a_index],
            ),
            "b": _comparison_image_payload(
                visual_b[boundary_b_index], own_b[boundary_b_index],
                other_b[boundary_b_index], margins_b[boundary_b_index],
            ),
        },
        "segmentation_quality": {
            "a_suspicious": suspicious_a,
            "b_suspicious": suspicious_b,
            "a_total": len(visual_a),
            "b_total": len(visual_b),
        },
        "warnings": warnings,
    }


@app.get("/api/evolution/network")
def plot_phenotype_network(k: int = 2):
    """Build a phenotype similarity network, not a claim of gene flow."""
    if nx is None:
        raise HTTPException(status_code=503, detail="网络分析依赖 networkx，当前环境未安装。")
    k = min(max(int(k), 1), 5)
    cluster_rows = get_all_cluster_prototypes()
    if not cluster_rows:
        raise HTTPException(status_code=400, detail="尚未建立可用于网络分析的原型。")
    grouped = defaultdict(list)
    for row in cluster_rows:
        grouped[row[0]].append(np.frombuffer(row[4], dtype=np.float32).copy())
    taxon_meta = _taxon_lookup()
    distribution = get_database_overview().get("distribution", [])
    image_counts = {row["id"]: row["image_count"] for row in distribution}
    tax_ids = sorted(tax_id for tax_id in grouped if tax_id in taxon_meta)
    if len(tax_ids) < 3:
        raise HTTPException(status_code=400, detail="网络节点不足 3 个。")

    features = []
    for tax_id in tax_ids:
        center = np.mean(grouped[tax_id], axis=0)
        center /= max(np.linalg.norm(center), 1e-8)
        features.append(center)
    features = np.stack(features)
    similarity = np.clip(features @ features.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -1.0)
    neighbor_sets = []
    edge_pairs = set()
    for index in range(len(tax_ids)):
        nearest = [int(neighbor) for neighbor in np.argsort(-similarity[index])[:k]]
        neighbor_sets.append(set(nearest))
        for neighbor in nearest:
            edge_pairs.add(tuple(sorted((index, int(neighbor)))))

    distance = np.clip(1.0 - np.clip(features @ features.T, -1.0, 1.0), 0.0, 2.0)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=1, max_iter=300)
    coords = mds.fit_transform(distance)
    coord_min = coords.min(axis=0)
    coord_span = np.maximum(coords.max(axis=0) - coord_min, 1e-8)
    normalized_coords = (coords - coord_min) / coord_span
    graph = nx.Graph()
    graph.add_nodes_from(range(len(tax_ids)))
    graph.add_edges_from(edge_pairs)
    species = [taxon_meta[tax_id]["species"] for tax_id in tax_ids]
    unique_species = sorted(set(species))
    species_index = {name: index for index, name in enumerate(unique_species)}
    cross_degree = {
        index: sum(species[index] != species[neighbor] for neighbor in graph.neighbors(index))
        for index in range(len(tax_ids))
    }

    fig, ax = plt.subplots(figsize=(13, 9))
    cmap = plt.get_cmap("gist_ncar")
    color_map = {name: cmap(index / max(len(unique_species), 1)) for index, name in enumerate(unique_species)}
    pos = {index: coords[index] for index in range(len(tax_ids))}
    same_edges = []
    cross_edges = []
    for left, right in graph.edges:
        (same_edges if species[left] == species[right] else cross_edges).append((left, right))
    nx.draw_networkx_edges(graph, pos, edgelist=same_edges, ax=ax, edge_color="#aab2bd", alpha=0.35, width=0.8)
    nx.draw_networkx_edges(graph, pos, edgelist=cross_edges, ax=ax, edge_color="#e76f51", alpha=0.75, width=1.4)
    node_colors = [color_map[name] for name in species]
    node_sizes = [45 + 3 * np.sqrt(max(image_counts.get(tax_id, 1), 1)) for tax_id in tax_ids]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=0.6)
    labels = {index: str(tax_ids[index]) for index in range(len(tax_ids))}
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=5)
    for species_name in unique_species:
        ax.scatter([], [], color=color_map[species_name], s=35, label=species_name.replace("Haworthia ", ""))
    ax.legend(title="物种颜色", fontsize=6, title_fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    ax.set_title(f"表型相似度网络 · 每节点保留 {k} 条近邻边\n红边表示跨物种表型相似，不代表已证实的基因流")
    ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    nodes = []
    for index, tax_id in enumerate(tax_ids):
        meta = taxon_meta[tax_id]
        nodes.append({
            "tax_id": tax_id,
            "label": f"{meta['species']} - {meta['variant']}",
            "short_label": f"{meta['species'].replace('Haworthia ', 'H. ')} · {meta['variant']}",
            "species": meta["species"],
            "variant": meta["variant"],
            "species_index": species_index[meta["species"]],
            "image_count": image_counts.get(tax_id, 0),
            "degree": int(graph.degree(index)),
            "cross_species_degree": int(cross_degree[index]),
            "x": float(normalized_coords[index, 0]),
            "y": float(normalized_coords[index, 1]),
        })

    edges = []
    for left, right in sorted(edge_pairs):
        edges.append({
            "source": tax_ids[left],
            "target": tax_ids[right],
            "similarity": float(similarity[left, right]),
            "cross_species": species[left] != species[right],
            "mutual_neighbor": (
                right in neighbor_sets[left] and left in neighbor_sets[right]
            ),
        })

    relationships = []
    relationship_limit = min(12, len(tax_ids) - 1)
    for source_index, source_tax_id in enumerate(tax_ids):
        for rank, target_index in enumerate(
            np.argsort(-similarity[source_index])[:relationship_limit], start=1
        ):
            target_index = int(target_index)
            target_tax_id = tax_ids[target_index]
            same_species = species[source_index] == species[target_index]
            pair = tuple(sorted((source_index, target_index)))
            relationships.append({
                "source_tax_id": source_tax_id,
                "source_label": nodes[source_index]["label"],
                "target_tax_id": target_tax_id,
                "target_label": nodes[target_index]["label"],
                "target_species": species[target_index],
                "target_variant": taxon_meta[target_tax_id]["variant"],
                "target_image_count": image_counts.get(target_tax_id, 0),
                "rank": rank,
                "similarity": float(similarity[source_index, target_index]),
                "relationship": "同物种" if same_species else "跨物种",
                "same_species": same_species,
                "edge_in_network": pair in edge_pairs,
                "mutual_neighbor": (
                    target_index in neighbor_sets[source_index]
                    and source_index in neighbor_sets[target_index]
                ),
            })

    bridge_rows = []
    for index, tax_id in enumerate(tax_ids):
        cross_similarities = [
            float(similarity[index, neighbor])
            for neighbor in graph.neighbors(index)
            if species[index] != species[neighbor]
        ]
        bridge_rows.append({
            "tax_id": tax_id,
            "label": f"{taxon_meta[tax_id]['species']} - {taxon_meta[tax_id]['variant']}",
            "species": taxon_meta[tax_id]["species"],
            "image_count": image_counts.get(tax_id, 0),
            "degree": int(graph.degree(index)),
            "cross_species_degree": int(cross_degree[index]),
            "strongest_cross_similarity": max(cross_similarities) if cross_similarities else 0.0,
            "strongest_cross_neighbor": next((
                nodes[neighbor]["label"]
                for neighbor in sorted(
                    graph.neighbors(index),
                    key=lambda value: similarity[index, value],
                    reverse=True,
                )
                if species[index] != species[neighbor]
            ), ""),
        })
    bridge_rows.sort(key=lambda row: (row["cross_species_degree"], row["strongest_cross_similarity"]), reverse=True)
    return {
        "image_base64": image_base64,
        "node_count": len(tax_ids),
        "edge_count": len(edge_pairs),
        "k": k,
        "interpretation": "这是基于当前模型原型的表型相似度网络。跨物种连接可用于发现桥接表型，但不能单独证明基因流、亲本关系或演化方向。",
        "nodes": nodes,
        "edges": edges,
        "relationships": relationships,
        "bridges": bridge_rows[:30],
    }


# ==========================================
# 替换路由 2: 气质逆向解码 (以去背图像作为热力图底板)
# ==========================================
@app.post("/api/inference/decode")
async def decode_temperament(
        file: UploadFile = File(...),
        segmentation_mode: str = Form("adaptive"),
        segmentation_sensitivity: int = Form(70),
):
    if training_state["is_training"] or segmentation_state["is_running"]:
        raise HTTPException(status_code=503, detail="引擎繁忙，无法执行自注意力追踪。")
    require_trained_model()
    require_segmentation_models()

    if segmentation_mode not in segmentation_engine.MODES:
        raise HTTPException(status_code=422, detail="未知分割模式。")
    segmentation_sensitivity = min(max(segmentation_sensitivity, 0), 100)

    image_bytes = await file.read()
    img_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    segmented = remove_background(
        img_orig, segmentation_mode, segmentation_sensitivity
    )
    img_tensor_ready = ImageOps.pad(
        black_composite(segmented), (224, 224), color=(0, 0, 0)
    )
    tensor_in, foreground_mask = val_transforms.apply_with_mask(segmented)
    tensor_in = tensor_in.unsqueeze(0).to(DEVICE)
    foreground_mask = foreground_mask.unsqueeze(0).to(DEVICE)

    global model
    model.eval()
    with gpu_lock:
        with torch.no_grad():
            # 正确请求掩膜与门控张量，接收 3 个返回值
            _, masks, gates = model(
                tensor_in, fg_mask=foreground_mask, return_masks=True
            )

    masks = F.interpolate(masks, size=(224, 224), mode='bilinear')[0].cpu().numpy()
    gates = [g.item() for g in gates]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    img_gray = np.array(img_tensor_ready).mean(axis=-1)

    for i in range(4):
        # 此时的 img_gray 已无背景干扰，土壤区域为纯黑 (0)
        axes[i].imshow(img_gray, cmap='gray')
        axes[i].imshow(masks[i], cmap='jet', alpha=0.4)
        axes[i].set_title(f"Part {i + 1}: {gates[i]:.3f}", fontsize=9)
        axes[i].axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


def get_existing_segmentation_metrics(seg_path, sensitivity):
    with Image.open(seg_path) as segmented:
        if "A" not in segmented.getbands():
            alpha = Image.new("L", segmented.size, 255)
        else:
            alpha = segmented.getchannel("A")
        return segmentation_engine.analyze_alpha(alpha, sensitivity)


def background_resegment(req: ResegmentRequest):
    if not gpu_lock.acquire(blocking=False):
        segmentation_state.update({
            "is_running": False, "status": "failed", "message": "计算资源已被锁定。"
        })
        return

    try:
        records = get_all_image_records()
        targets = []
        for record in records:
            _, orig_path, seg_path = record
            if not os.path.exists(orig_path) or not os.path.exists(seg_path):
                continue
            metrics = get_existing_segmentation_metrics(seg_path, req.sensitivity)
            if req.scope == "all" or metrics["suspicious"]:
                targets.append((record, metrics))

        segmentation_state.update({
            "status": "running",
            "current": 0,
            "total": len(targets),
            "processed": 0,
            "recovered": 0,
            "failed": 0,
            "message": f"已筛选 {len(targets)} 张图像，正在准备重分割...",
        })

        if req.backup_existing and targets:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("backups") / f"segmentation_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=False)
            for (_, _, seg_path), _ in targets:
                shutil.copy2(seg_path, backup_dir / Path(seg_path).name)
            segmentation_state["backup_path"] = str(backup_dir.resolve())

        for index, ((image_id, orig_path, seg_path), old_metrics) in enumerate(targets, 1):
            segmentation_state.update({
                "current": index,
                "message": f"正在重分割 {index} / {len(targets)}",
            })
            temporary_path = Path(seg_path).with_suffix(".new.png")
            try:
                image = Image.open(orig_path).convert("RGB")
                segmented, new_metrics = remove_background(
                    image, req.mode, req.sensitivity, return_metrics=True
                )
                segmented.save(temporary_path, format="PNG")
                os.replace(temporary_path, seg_path)
                segmentation_state["processed"] += 1
                if (
                    new_metrics.get("recovery_used", False)
                    and new_metrics.get("quality_score", 0.0)
                    > old_metrics.get("quality_score", 0.0) + 0.02
                ):
                    segmentation_state["recovered"] += 1
            except Exception:
                segmentation_state["failed"] += 1
                if temporary_path.exists():
                    temporary_path.unlink()

        segmentation_state.update({
            "status": "completed",
            "message": (
                f"重分割完成：处理 {segmentation_state['processed']} 张，"
                f"宽容恢复 {segmentation_state['recovered']} 张，"
                f"失败 {segmentation_state['failed']} 张。"
            ),
        })
    except Exception as exc:
        segmentation_state.update({"status": "failed", "message": str(exc)})
    finally:
        segmentation_state["is_running"] = False
        gpu_lock.release()


@app.post("/api/taxonomy/resegment")
def start_resegment(
        req: ResegmentRequest,
        background_tasks: BackgroundTasks,
):
    if training_state["is_training"] or maintenance_state["is_running"]:
        raise HTTPException(status_code=409, detail="训练或维护任务正在运行。")
    if segmentation_state["is_running"]:
        raise HTTPException(status_code=409, detail="已有重分割任务正在运行。")
    if not HAS_SEGMENTATION_MODELS:
        raise HTTPException(
            status_code=400,
            detail="服务端缺少 onnxruntime 或 IS-Net/U2Net 分割权重。",
        )

    segmentation_state.update({
        "is_running": True,
        "status": "queued",
        "message": "重分割任务排队中...",
        "backup_path": "",
    })
    background_tasks.add_task(background_resegment, req)
    return {"message": "重分割任务已启动。"}


@app.get("/api/taxonomy/resegment/status")
def get_resegment_status():
    return segmentation_state


@app.post("/api/taxonomy/resegment/restore")
def restore_segmentation_backup(req: SegmentationRestoreRequest):
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="已有任务正在运行。")

    backup_root = (Path("backups").resolve())
    source = Path(req.backup_path).resolve()
    if backup_root not in source.parents or not source.name.startswith("segmentation_"):
        raise HTTPException(status_code=400, detail="备份路径不在合法的分割备份目录中。")
    if not source.is_dir():
        raise HTTPException(status_code=404, detail="分割备份目录不存在。")

    restored = 0
    for backup_file in source.glob("*_seg.png"):
        target = Path(IMG_DIR) / backup_file.name
        if target.exists():
            shutil.copy2(backup_file, target)
            restored += 1
    return {"message": f"已从备份恢复 {restored} 张分割图。", "restored": restored}


def background_prototype_rebuild(req: PrototypeRequest):
    if not gpu_lock.acquire(blocking=False):
        maintenance_state.update({
            "is_running": False, "status": "failed", "message": "GPU 已被锁定"
        })
        return

    try:
        maintenance_state.update({
            "is_running": True,
            "status": "running",
            "message": "正在读取现有图像并重建多原型...",
        })
        class_count = rebuild_prototypes(
            model, DEVICE, req.prototypes_per_taxon, maintenance_state
        )
        maintenance_state.update({
            "status": "completed",
            "message": f"已为 {class_count} 个类群重建多原型。",
        })
    except Exception as exc:
        maintenance_state.update({"status": "failed", "message": str(exc)})
    finally:
        maintenance_state["is_running"] = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gpu_lock.release()


@app.post("/api/maintenance/rebuild-prototypes")
def start_prototype_rebuild(req: PrototypeRequest, background_tasks: BackgroundTasks):
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="已有训练或维护任务正在运行。")
    require_trained_model()
    maintenance_state.update({
        "is_running": True, "status": "queued", "message": "多原型任务排队中..."
    })
    background_tasks.add_task(background_prototype_rebuild, req)
    return {"message": "多原型重建任务已启动。"}


@app.post("/api/maintenance/import-model-package")
async def import_model_package(
        file: UploadFile = File(...),
        package_sha256: str = Form(...),
        import_token: str = Form(...),
):
    """Import a user-selected model and merge prototypes without replacing images."""
    global model_weights_loaded
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="已有训练、维护或重分割任务正在运行。")
    if not secrets.compare_digest(import_token, MODEL_IMPORT_TOKEN):
        raise HTTPException(status_code=403, detail="模型导入会话令牌无效，请刷新页面后重试。")
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="只接受 Haworthia OMICS 模型包 ZIP。")

    payload = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > MAX_PACKAGE_BYTES:
            raise HTTPException(status_code=413, detail="模型包超过 128 MB 安全上限。")

    try:
        inspected = inspect_model_package(bytes(payload), package_sha256)
        state = _validate_model_state(inspected.model_bytes)
    except ModelPackageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="计算引擎正忙，请稍后重试。")

    backup_dir = Path("backups") / (
        "model_import_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    database_backup = backup_dir / Path(DB_PATH).name
    model_path = Path(MODEL_PATH).expanduser()
    checkpoint_path = Path(CHKPT_BASE_PATH).expanduser()
    model_backup = backup_dir / model_path.name
    checkpoint_backup = backup_dir / checkpoint_path.name
    original_model_exists = model_path.exists()
    original_checkpoint_exists = checkpoint_path.exists()
    original_loaded = model_weights_loaded
    original_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    database_changed = False
    model_file_changed = False
    checkpoint_archived = False

    try:
        maintenance_state.update({
            "is_running": True,
            "status": "running",
            "message": "正在校验并导入模型包...",
        })
        backup_dir.mkdir(parents=True, exist_ok=False)
        _backup_database(DB_PATH, database_backup)
        if original_model_exists:
            shutil.copy2(model_path, model_backup)
        if original_checkpoint_exists:
            shutil.copy2(checkpoint_path, checkpoint_backup)
        (backup_dir / "PACKAGE_SHA256.txt").write_text(
            inspected.package_sha256 + "\n", encoding="ascii"
        )

        merged = merge_catalog(inspected.catalog_bytes, DB_PATH)
        database_changed = True
        write_bytes_atomic(model_path, inspected.model_bytes)
        model_file_changed = True
        if original_checkpoint_exists:
            checkpoint_path.unlink()
            checkpoint_archived = True

        model.load_state_dict(state, strict=True)
        model.to(DEVICE)
        model.eval()
        model_weights_loaded = True
        maintenance_state.update({
            "is_running": False,
            "status": "completed",
            "message": "模型包导入完成。",
        })
        return {
            "message": "模型、类群标签和数值原型已导入；本地图片与额外类群已保留。",
            "package_sha256": inspected.package_sha256,
            "model_version": inspected.manifest.get("model_version", "unknown"),
            "catalog": merged,
            "backup_path": str(backup_dir.resolve()),
            "checkpoint_archived": original_checkpoint_exists,
        }
    except Exception as exc:
        restore_failed = False
        try:
            if database_changed and database_backup.exists():
                _restore_database(database_backup, DB_PATH)
            if model_file_changed:
                if original_model_exists and model_backup.exists():
                    shutil.copy2(model_backup, model_path)
                elif model_path.exists():
                    model_path.unlink()
            if checkpoint_archived and checkpoint_backup.exists():
                shutil.copy2(checkpoint_backup, checkpoint_path)
            model.load_state_dict(original_state, strict=True)
            model.to(DEVICE)
            model.eval()
            model_weights_loaded = original_loaded
        except Exception:
            restore_failed = True
        finally:
            maintenance_state.update({
                "is_running": False,
                "status": "failed",
                "message": (
                    "模型包导入失败，自动恢复也未完整完成；请使用导入前备份。"
                    if restore_failed
                    else "模型包导入失败，已恢复导入前状态。"
                ),
            })
        raise HTTPException(
            status_code=500,
            detail=(
                f"模型包导入失败，自动恢复未完整完成；请从 {backup_dir.resolve()} 恢复。"
                if restore_failed
                else "模型包导入失败，程序已恢复导入前的数据库和模型。"
            ),
        ) from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gpu_lock.release()


@app.post("/api/maintenance/export-model-package")
def export_model_package(export_token: str = Form(...)):
    """Export the current user model and numeric prototypes without any images."""
    if not secrets.compare_digest(export_token, MODEL_IMPORT_TOKEN):
        raise HTTPException(status_code=403, detail="模型导出会话令牌无效，请刷新页面后重试。")
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="已有训练、维护或重分割任务正在运行。")
    require_trained_model()
    if get_database_overview().get("prototype_count", 0) == 0:
        raise HTTPException(status_code=409, detail="当前没有数值原型，请先完成训练或原型重建。")
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="计算引擎正忙，请稍后重试。")
    try:
        maintenance_state.update({
            "is_running": True,
            "status": "running",
            "message": "正在生成不含图片的模型包...",
        })
        model_path = Path(MODEL_PATH).expanduser()
        if not model_path.is_file():
            raise ModelPackageError("当前模型文件不存在，无法导出。")
        _validate_model_state(model_path.read_bytes())
        exported = build_model_package(model_path, DB_PATH)
        maintenance_state.update({
            "is_running": False,
            "status": "completed",
            "message": "模型包导出完成。",
        })
        return Response(
            content=exported.payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{exported.filename}"',
                "X-Package-Filename": exported.filename,
                "X-Package-SHA256": exported.package_sha256,
                "X-Model-SHA256": exported.model_sha256,
            },
        )
    except ModelPackageError as exc:
        maintenance_state.update({
            "is_running": False,
            "status": "failed",
            "message": str(exc),
        })
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        maintenance_state.update({
            "is_running": False,
            "status": "failed",
            "message": "模型包导出失败。",
        })
        raise HTTPException(status_code=500, detail="模型包导出失败。") from exc
    finally:
        gpu_lock.release()


@app.get("/api/maintenance/status")
def get_maintenance_status():
    return maintenance_state


@app.get("/api/maintenance/attention-diagnostics")
def attention_diagnostics(sample_limit: int = 128):
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="GPU 正在执行其他任务。")
    require_trained_model()
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="GPU 已被锁定。")
    try:
        return diagnose_attention(model, DEVICE, min(max(sample_limit, 16), 512))
    finally:
        gpu_lock.release()


@app.post("/api/maintenance/snapshot")
def create_snapshot(req: SnapshotRequest):
    if (
        training_state["is_training"]
        or maintenance_state["is_running"]
        or segmentation_state["is_running"]
    ):
        raise HTTPException(status_code=409, detail="请等待训练或维护任务结束后再创建快照。")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = Path("backups") / timestamp
    target.mkdir(parents=True, exist_ok=False)

    source_db = sqlite3.connect(DB_PATH)
    backup_db = sqlite3.connect(target / Path(DB_PATH).name)
    try:
        source_db.backup(backup_db)
    finally:
        backup_db.close()
        source_db.close()

    copied_files = [Path(DB_PATH).name]
    for asset in (MODEL_PATH, CHKPT_BASE_PATH):
        source = Path(asset)
        if source.exists():
            shutil.copy2(source, target / source.name)
            copied_files.append(source.name)

    if req.include_images:
        shutil.copytree(IMG_DIR, target / IMG_DIR)
        copied_files.append(IMG_DIR)

    return {
        "message": "完整快照创建成功。" if req.include_images else "模型与数据库快照创建成功。",
        "path": str(target.resolve()),
        "contents": copied_files,
    }
