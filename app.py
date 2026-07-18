import streamlit as st
import requests
import time
import pandas as pd
import base64
import json
import os

# ==========================================
# 核心配置
# ==========================================
API_BASE_URL = os.getenv(
    "HAWORTHIA_API_URL", "http://127.0.0.1:8000/api"
).rstrip("/")
SEGMENTATION_MODES = {
    "自动宽容": "adaptive",
    "双模型宽松": "lenient",
    "IS-Net 严格": "strict",
}


def render_interactive_network(network_data, height=720):
    payload = {
        "nodes": network_data.get("nodes", []),
        "edges": network_data.get("edges", []),
        "relationships": network_data.get("relationships", []),
    }
    encoded = base64.b64encode(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    html = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, "Microsoft YaHei", sans-serif; color: #24292f; background: #fff; }
  .shell { border: 1px solid #d8dee4; border-radius: 6px; overflow: hidden; background: #fff; }
  .toolbar { height: 48px; display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-bottom: 1px solid #e7ebef; background: #f7f8fa; }
  .toolbar input[type="search"] { flex: 1; min-width: 160px; height: 32px; border: 1px solid #c8d0d8; border-radius: 4px; padding: 0 10px; font-size: 13px; }
  .toolbar label { display: flex; align-items: center; gap: 5px; white-space: nowrap; font-size: 12px; }
  .toolbar button { width: 34px; height: 32px; border: 1px solid #c8d0d8; border-radius: 4px; background: #fff; cursor: pointer; font-size: 17px; }
  .workspace { display: grid; grid-template-columns: minmax(0, 1fr) 260px; height: 650px; }
  .canvas-wrap { position: relative; min-width: 0; background: #fbfcfd; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
  canvas.dragging { cursor: grabbing; }
  .tooltip { position: absolute; display: none; pointer-events: none; max-width: 280px; padding: 7px 9px; border-radius: 4px; background: rgba(24, 29, 34, .94); color: white; font-size: 12px; line-height: 1.45; box-shadow: 0 4px 14px rgba(0,0,0,.18); }
  .details { border-left: 1px solid #e7ebef; padding: 14px; overflow: auto; background: #fff; }
  .details h3 { margin: 0 0 6px; font-size: 15px; line-height: 1.35; }
  .meta { color: #66707a; font-size: 12px; margin-bottom: 12px; }
  .neighbor { width: 100%; border: 0; border-top: 1px solid #edf0f2; background: transparent; padding: 8px 0; text-align: left; cursor: pointer; }
  .neighbor strong { display: block; font-size: 12px; font-weight: 600; color: #24292f; }
  .neighbor span { display: block; color: #68737d; font-size: 11px; margin-top: 2px; }
  .legend { position: absolute; left: 12px; bottom: 10px; padding: 6px 9px; border: 1px solid #d8dee4; border-radius: 4px; background: rgba(255,255,255,.9); font-size: 11px; color: #5a646e; }
  .line-key { display: inline-block; width: 20px; border-top: 2px solid #aab2bd; margin: 0 5px 3px 8px; }
  .line-key.cross { border-color: #d9573f; }
  @media (max-width: 720px) {
    .workspace { grid-template-columns: 1fr; height: 700px; grid-template-rows: 500px 200px; }
    .details { border-left: 0; border-top: 1px solid #e7ebef; }
  }
</style>
</head>
<body>
<div class="shell">
  <div class="toolbar">
    <input id="search" type="search" list="taxa" placeholder="搜索物种或变种">
    <datalist id="taxa"></datalist>
    <label><input id="labels" type="checkbox" checked>显示名称</label>
    <button id="reset" title="重置视图" aria-label="重置视图">⌂</button>
  </div>
  <div class="workspace">
    <div class="canvas-wrap" id="canvasWrap">
      <canvas id="network"></canvas>
      <div id="tooltip" class="tooltip"></div>
      <div class="legend"><span class="line-key"></span>同物种 <span class="line-key cross"></span>跨物种</div>
    </div>
    <aside class="details">
      <h3 id="detailName">未选择类群</h3>
      <div id="detailMeta" class="meta">节点信息与相似关系</div>
      <div id="neighborList"></div>
    </aside>
  </div>
</div>
<script>
const raw = "__DATA__";
const bytes = Uint8Array.from(atob(raw), c => c.charCodeAt(0));
const data = JSON.parse(new TextDecoder().decode(bytes));
const canvas = document.getElementById("network");
const wrap = document.getElementById("canvasWrap");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const labelsToggle = document.getElementById("labels");
const search = document.getElementById("search");
const dataList = document.getElementById("taxa");
const nodeById = new Map(data.nodes.map(node => [node.tax_id, node]));
const relationsBySource = new Map();
for (const relation of data.relationships) {
  if (!relationsBySource.has(relation.source_tax_id)) relationsBySource.set(relation.source_tax_id, []);
  relationsBySource.get(relation.source_tax_id).push(relation);
}
for (const node of data.nodes) {
  const option = document.createElement("option");
  option.value = node.label;
  dataList.appendChild(option);
}

let view = { scale: 1, tx: 0, ty: 0 };
let selected = null;
let hovered = null;
let dragging = false;
let moved = false;
let last = { x: 0, y: 0 };

function colorFor(node) {
  const hue = (node.species_index * 137.508) % 360;
  return `hsl(${hue}, 68%, 43%)`;
}

function viewport() {
  const rect = canvas.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
}

function basePoint(node) {
  const { width, height } = viewport();
  const pad = 64;
  return {
    x: pad + node.x * Math.max(1, width - pad * 2),
    y: pad + node.y * Math.max(1, height - pad * 2),
  };
}

function project(node) {
  const { width, height } = viewport();
  const base = basePoint(node);
  return {
    x: (base.x - width / 2) * view.scale + width / 2 + view.tx,
    y: (base.y - height / 2) * view.scale + height / 2 + view.ty,
  };
}

function focusNode(node, minimumScale=1.8) {
  const { width, height } = viewport();
  const base = basePoint(node);
  view.scale = Math.max(view.scale, minimumScale);
  view.tx = -(base.x - width / 2) * view.scale;
  view.ty = -(base.y - height / 2) * view.scale;
}

function radius(node) {
  return 4.2 + Math.min(4.2, Math.sqrt(Math.max(1, node.image_count)) * 0.34);
}

function focusedIds() {
  if (selected === null) return null;
  const ids = new Set([selected]);
  for (const edge of data.edges) {
    if (edge.source === selected) ids.add(edge.target);
    if (edge.target === selected) ids.add(edge.source);
  }
  return ids;
}

function overlaps(rect, occupied) {
  return occupied.some(other => !(
    rect.right < other.left || rect.left > other.right ||
    rect.bottom < other.top || rect.top > other.bottom
  ));
}

function draw() {
  const { width, height } = viewport();
  ctx.clearRect(0, 0, width, height);
  const focus = focusedIds();
  for (const edge of data.edges) {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    if (!source || !target) continue;
    const a = project(source), b = project(target);
    const active = selected === null || edge.source === selected || edge.target === selected;
    ctx.globalAlpha = selected === null ? (edge.cross_species ? .58 : .24) : (active ? .9 : .05);
    ctx.strokeStyle = edge.cross_species ? "#d9573f" : "#9aa4ad";
    ctx.lineWidth = active && selected !== null ? 2.2 : (edge.mutual_neighbor ? 1.35 : .8);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const node of data.nodes) {
    const p = project(node);
    const isSelected = node.tax_id === selected;
    const isHovered = node.tax_id === hovered;
    ctx.globalAlpha = focus && !focus.has(node.tax_id) ? .16 : 1;
    ctx.fillStyle = colorFor(node);
    ctx.strokeStyle = isSelected ? "#111827" : "#ffffff";
    ctx.lineWidth = isSelected ? 3 : 1.2;
    ctx.beginPath(); ctx.arc(p.x, p.y, radius(node) + (isSelected || isHovered ? 2 : 0), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  if (labelsToggle.checked) {
    const occupied = [];
    const candidates = [...data.nodes].sort((a, b) => {
      if (a.tax_id === selected || a.tax_id === hovered) return -1;
      if (b.tax_id === selected || b.tax_id === hovered) return 1;
      return (b.degree + b.cross_species_degree) - (a.degree + a.cross_species_degree);
    });
    ctx.font = `${view.scale > 1.5 ? 12 : 10}px system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    for (const node of candidates) {
      const p = project(node);
      if (p.x < -100 || p.x > width + 100 || p.y < -20 || p.y > height + 20) continue;
      const text = node.short_label;
      const textWidth = ctx.measureText(text).width;
      const rect = { left: p.x + 8, right: p.x + 12 + textWidth, top: p.y - 8, bottom: p.y + 8 };
      const forced = node.tax_id === selected || node.tax_id === hovered;
      if (!forced && overlaps(rect, occupied)) continue;
      if (!forced && view.scale < .85 && node.degree < 3) continue;
      occupied.push(rect);
      ctx.globalAlpha = focus && !focus.has(node.tax_id) ? .12 : .92;
      ctx.fillStyle = "rgba(255,255,255,.86)";
      ctx.fillRect(rect.left - 2, rect.top, textWidth + 6, 16);
      ctx.fillStyle = "#20262d";
      ctx.fillText(text, p.x + 9, p.y);
    }
  }
  ctx.globalAlpha = 1;
}

function hitTest(x, y) {
  let best = null, bestDistance = 15;
  for (const node of data.nodes) {
    const p = project(node);
    const distance = Math.hypot(p.x - x, p.y - y);
    if (distance < bestDistance) { best = node; bestDistance = distance; }
  }
  return best;
}

function showDetails(node) {
  document.getElementById("detailName").textContent = node.label;
  document.getElementById("detailMeta").textContent =
    `ID ${node.tax_id} · ${node.image_count} 张图 · ${node.degree} 条网络边 · ${node.cross_species_degree} 条跨物种边`;
  const list = document.getElementById("neighborList");
  list.replaceChildren();
  for (const relation of relationsBySource.get(node.tax_id) || []) {
    const button = document.createElement("button");
    button.className = "neighbor";
    const name = document.createElement("strong");
    name.textContent = `${relation.rank}. ${relation.target_label}`;
    const meta = document.createElement("span");
    meta.textContent = `${relation.relationship} · 相似度 ${relation.similarity.toFixed(4)}${relation.mutual_neighbor ? " · 互为近邻" : ""}`;
    button.append(name, meta);
    button.addEventListener("click", () => {
      selected = relation.target_tax_id;
      const selectedNode = nodeById.get(selected);
      focusNode(selectedNode);
      showDetails(selectedNode);
      draw();
    });
    list.appendChild(button);
  }
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

canvas.addEventListener("pointerdown", event => {
  dragging = true; moved = false; last = pointerPosition(event);
  canvas.setPointerCapture(event.pointerId); canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", event => {
  const p = pointerPosition(event);
  if (dragging) {
    const dx = p.x - last.x, dy = p.y - last.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    view.tx += dx; view.ty += dy; last = p; draw(); return;
  }
  const node = hitTest(p.x, p.y);
  hovered = node ? node.tax_id : null;
  if (node) {
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(p.x + 12, wrap.clientWidth - 285)}px`;
    tooltip.style.top = `${Math.max(4, p.y - 8)}px`;
    tooltip.textContent = `${node.label} · ${node.image_count} 张图 · ${node.cross_species_degree} 条跨物种边`;
  } else tooltip.style.display = "none";
  draw();
});
canvas.addEventListener("pointerup", event => {
  const p = pointerPosition(event);
  if (!moved) {
    const node = hitTest(p.x, p.y);
    if (node) { selected = node.tax_id; showDetails(node); }
  }
  dragging = false; canvas.classList.remove("dragging"); draw();
});
canvas.addEventListener("pointerleave", () => { hovered = null; tooltip.style.display = "none"; draw(); });
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  const p = pointerPosition(event);
  const oldScale = view.scale;
  const factor = event.deltaY < 0 ? 1.13 : .885;
  view.scale = Math.max(.55, Math.min(5, view.scale * factor));
  view.tx = p.x - (p.x - view.tx) * (view.scale / oldScale);
  view.ty = p.y - (p.y - view.ty) * (view.scale / oldScale);
  draw();
}, { passive: false });
labelsToggle.addEventListener("change", draw);
document.getElementById("reset").addEventListener("click", () => {
  view = { scale: 1, tx: 0, ty: 0 }; selected = null; draw();
});
function selectFromSearch() {
  const query = search.value.trim().toLowerCase();
  if (!query) return;
  const node = data.nodes.find(item => item.label.toLowerCase() === query)
    || data.nodes.find(item => item.label.toLowerCase().includes(query));
  if (node) { selected = node.tax_id; focusNode(node); showDetails(node); draw(); }
}
search.addEventListener("change", selectFromSearch);
search.addEventListener("keydown", event => { if (event.key === "Enter") selectFromSearch(); });

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
new ResizeObserver(resize).observe(wrap);
resize();
</script>
</body>
</html>
""".replace("__DATA__", encoded)
    st.iframe(html, width="stretch", height=height)

st.set_page_config(page_title="Haworthia OMICS Client", layout="wide")
st.sidebar.title("Haworthia OMICS")
st.sidebar.caption("作者：雨筠")
# 修改 app.py，将模式扩展为 5 个功能
mode = st.sidebar.radio("系统功能", [
    "数据库总览",
    "数据导入与清洗",
    "流形度量训练",
    "表型关系发掘",
    "开放集推理",
    "气质逆向解码",
    "系统维护与诊断"
])

if mode == "数据库总览":
    st.header("数据库与模型总览")
    st.caption("这里显示当前图库、分割文件、原型覆盖率和训练引擎状态。只读统计不会修改数据。")

    model_import_feedback = st.session_state.pop("model_import_feedback", None)
    if model_import_feedback:
        st.success(model_import_feedback["message"])
        st.caption(f"导入前备份：{model_import_feedback['backup_path']}")

    try:
        overview_res = requests.get(f"{API_BASE_URL}/overview", timeout=10)
    except requests.exceptions.RequestException as exc:
        st.error(f"无法连接到计算引擎：{exc}")
        st.stop()

    if overview_res.status_code != 200:
        st.error(overview_res.text)
        st.stop()

    overview = overview_res.json()
    metric_row_1 = st.columns(4)
    metric_row_1[0].metric("类群记录", overview.get("taxonomy_count", 0))
    metric_row_1[1].metric("物种数", overview.get("species_count", 0))
    metric_row_1[2].metric("图片数", overview.get("image_count", 0))
    metric_row_1[3].metric("尚无模型原型", overview.get("taxa_without_prototypes", 0))

    metric_row_2 = st.columns(4)
    metric_row_2[0].metric("每类群平均图片", overview.get("image_mean_per_taxon", 0))
    metric_row_2[1].metric("最少图片数", overview.get("image_min_per_taxon", 0))
    metric_row_2[2].metric("最多图片数", overview.get("image_max_per_taxon", 0))
    metric_row_2[3].metric("缺失文件", overview.get("missing_files", 0))

    st.subheader("模型状态")
    model_info = overview.get("model", {})
    model_row = st.columns(4)
    model_row[0].metric("运行设备", model_info.get("device", "未知"))
    model_row[1].metric("模型已加载", "是" if model_info.get("loaded") else "否")
    model_row[2].metric("原型数量", overview.get("prototype_count", 0))
    model_row[3].metric("原型覆盖率", f"{model_info.get('prototype_coverage', 0):.1f}%")

    if not model_info.get("loaded"):
        st.warning("尚无已完成训练的模型。可以先导入数据并训练，推理与模型证据功能将在训练完成后启用。")
    segmentation_info = overview.get("segmentation", {})
    if not segmentation_info.get("available", False):
        st.warning(
            "背景分割功能不可用。请运行 `python scripts/download_segmentation_models.py` "
            "从上游下载并校验 isnet-general-use.onnx 与 u2net.onnx。"
            "这些第三方权重不随软件分发；缺失时图像导入会被阻止。"
        )

    base_info = model_info.get("base", {})
    checkpoint_info = model_info.get("checkpoint", {})
    model_table = pd.DataFrame([
        {
            "文件": "当前模型",
            "状态": "可用" if base_info.get("exists") else "缺失",
            "大小(MB)": base_info.get("size_mb"),
            "更新时间": base_info.get("modified", "-"),
        },
        {
            "文件": "训练断点",
            "状态": "可用" if checkpoint_info.get("exists") else "缺失",
            "大小(MB)": checkpoint_info.get("size_mb"),
            "更新时间": checkpoint_info.get("modified", "-"),
        },
    ])
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    training_info = overview.get("training", {})
    maintenance_info = overview.get("maintenance", {})
    import_info = overview.get("model_import", {})
    with st.expander(
        "导入模型包" if not model_info.get("loaded") else "更换模型包",
        expanded=not model_info.get("loaded"),
    ):
        if "model_package_widget_version" not in st.session_state:
            st.session_state["model_package_widget_version"] = 0
        model_package_file = st.file_uploader(
            "模型包 ZIP",
            type=["zip"],
            key=f"model_package_{st.session_state['model_package_widget_version']}",
        )
        model_package_sha256 = st.text_input(
            "整包 SHA-256",
            max_chars=64,
            placeholder="导出模型包时生成的 64 位校验值",
        ).strip()
        model_rights_confirmed = st.checkbox(
            "我确认有权使用该模型包，并对模型及其训练数据来源负责",
            value=False,
        )
        st.caption(import_info.get("format", "Haworthia OMICS 模型包 ZIP + 整包 SHA-256"))
        st.warning(
            "导入会替换当前模型并覆盖同名类群的数值原型；本地图片和额外类群会保留。"
            "程序会先自动备份，已有训练断点将移入备份。软件作者不提供或审核用户模型。"
        )
        import_disabled = (
            model_package_file is None
            or len(model_package_sha256) != 64
            or not model_rights_confirmed
            or training_info.get("is_training", False)
            or maintenance_info.get("is_running", False)
            or not import_info.get("session_token")
        )
        if st.button(
            "校验并导入模型包",
            use_container_width=True,
            disabled=import_disabled,
        ):
            with st.spinner("正在校验整包、模型架构和原型目录..."):
                try:
                    import_res = requests.post(
                        f"{API_BASE_URL}/maintenance/import-model-package",
                        files={
                            "file": (
                                model_package_file.name,
                                model_package_file.getvalue(),
                                "application/zip",
                            )
                        },
                        data={
                            "package_sha256": model_package_sha256,
                            "import_token": import_info.get("session_token", ""),
                        },
                        timeout=300,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"模型包导入请求失败：{exc}")
                else:
                    if import_res.status_code == 200:
                        imported = import_res.json()
                        st.session_state["model_import_feedback"] = {
                            "message": imported.get("message", "模型包导入完成。"),
                            "backup_path": imported.get("backup_path", "-"),
                        }
                        st.session_state.pop("model_export", None)
                        st.session_state["model_package_widget_version"] += 1
                        st.rerun()
                    else:
                        try:
                            detail = import_res.json().get("detail", "模型包导入失败。")
                        except ValueError:
                            detail = import_res.text or "模型包导入失败。"
                        st.error(detail)

    if model_info.get("loaded"):
        with st.expander("导出当前模型包", expanded=False):
            st.caption("导出当前权重、类群标签和数值原型；不包含图片、图片路径或训练断点。")
            export_confirmed = st.checkbox(
                "我了解模型包的使用与分发权限由导出者负责",
                value=False,
                key="model_export_rights_confirmation",
            )
            export_disabled = (
                not export_confirmed
                or overview.get("prototype_count", 0) == 0
                or training_info.get("is_training", False)
                or maintenance_info.get("is_running", False)
                or not import_info.get("session_token")
            )
            if st.button(
                "生成模型包",
                use_container_width=True,
                disabled=export_disabled,
            ):
                with st.spinner("正在整理模型权重与零图片原型目录..."):
                    try:
                        export_res = requests.post(
                            f"{API_BASE_URL}/maintenance/export-model-package",
                            data={"export_token": import_info.get("session_token", "")},
                            timeout=300,
                        )
                    except requests.exceptions.RequestException as exc:
                        st.error(f"模型包导出请求失败：{exc}")
                    else:
                        if export_res.status_code == 200:
                            filename = export_res.headers.get(
                                "X-Package-Filename", "haworthia-model.zip"
                            )
                            package_sha256 = export_res.headers.get("X-Package-SHA256", "")
                            st.session_state["model_export"] = {
                                "filename": filename,
                                "payload": export_res.content,
                                "sha256": package_sha256,
                            }
                        else:
                            try:
                                detail = export_res.json().get("detail", "模型包导出失败。")
                            except ValueError:
                                detail = export_res.text or "模型包导出失败。"
                            st.error(detail)

            model_export = st.session_state.get("model_export")
            if model_export:
                st.code(model_export["sha256"], language=None)
                download_col_1, download_col_2 = st.columns(2)
                download_col_1.download_button(
                    "下载模型包 ZIP",
                    data=model_export["payload"],
                    file_name=model_export["filename"],
                    mime="application/zip",
                    use_container_width=True,
                )
                download_col_2.download_button(
                    "下载 SHA-256",
                    data=(
                        f"{model_export['sha256']}  {model_export['filename']}\n"
                    ).encode("ascii"),
                    file_name=f"{model_export['filename']}.sha256",
                    mime="text/plain",
                    use_container_width=True,
                )

    state_col_1, state_col_2 = st.columns(2)
    with state_col_1:
        st.info(
            f"训练：{training_info.get('status', 'idle')} · "
            f"Epoch {training_info.get('current_epoch', 0)} / {training_info.get('total_epochs', 0)}"
        )
    with state_col_2:
        st.info(f"维护：{maintenance_info.get('status', 'idle')}")

    distribution = overview.get("distribution", [])
    if distribution:
        st.subheader("类群数据分布")
        distribution_df = pd.DataFrame(distribution).rename(columns={
            "id": "ID", "species": "物种", "variant": "变种/园艺名",
            "image_count": "图片数",
        })
        st.dataframe(
            distribution_df[["ID", "物种", "变种/园艺名", "图片数"]],
            use_container_width=True,
            hide_index=True,
        )

elif mode == "数据导入与清洗":
    st.header("植物影像入库与物理隔离")

    if "upload_widget_version" not in st.session_state:
        st.session_state["upload_widget_version"] = 0
    upload_feedback = st.session_state.pop("upload_feedback", None)
    if upload_feedback:
        st.success(f"成功将 {upload_feedback['successful']} 张表型图录入知识库。")
        if upload_feedback["suspicious"]:
            st.warning(f"其中 {upload_feedback['suspicious']} 张分割质量仍然偏低。")
        if upload_feedback["failed"]:
            st.error(f"有 {upload_feedback['failed']} 张图片入库失败。")

    col1, col2 = st.columns(2)
    with col1:
        species = st.text_input("物种 (Species)", "Haworthia obtusa")
        variant = st.text_input("变种/园艺名 (Variant/Cultivar)", "紫肌玉露")
        uploaded_files = st.file_uploader(
            "批量上传表型影像",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg'],
            key=f"batch_upload_{st.session_state['upload_widget_version']}",
        )
        upload_segmentation_label = st.selectbox(
            "入库分割模式", list(SEGMENTATION_MODES), index=0
        )
        upload_sensitivity = st.slider("入库分割灵敏度", 0, 100, 70)

    with col2:
        if st.button("特征剥离与落盘", use_container_width=True):
            if not uploaded_files:
                st.error("请先选择图片文件。")
            else:
                progress_bar = st.progress(0)
                failed_uploads = 0
                suspicious_uploads = 0
                for idx, f in enumerate(uploaded_files):
                    payload = {
                        "species": species,
                        "variant": variant,
                        "segmentation_mode": SEGMENTATION_MODES[upload_segmentation_label],
                        "segmentation_sensitivity": upload_sensitivity,
                    }
                    files = {"file": (f.name, f.getvalue(), "image/jpeg")}
                    upload_res = requests.post(
                        f"{API_BASE_URL}/taxonomy/upload", data=payload, files=files
                    )
                    if upload_res.status_code != 200:
                        failed_uploads += 1
                    elif upload_res.json().get("segmentation", {}).get("suspicious", False):
                        suspicious_uploads += 1
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                successful_uploads = len(uploaded_files) - failed_uploads
                st.session_state["upload_feedback"] = {
                    "successful": successful_uploads,
                    "suspicious": suspicious_uploads,
                    "failed": failed_uploads,
                }
                st.session_state["upload_widget_version"] += 1
                st.rerun()

    st.divider()
    st.subheader("数据管理与物理回溯 (Data Rollback)")
    try:
        records = requests.get(f"{API_BASE_URL}/taxonomy/records").json()
    except Exception:
        records = []

    if records:
        record_by_label = {
            f"#{r['id']} · {r['species']} - {r['variant']} · {r['image_count']} 张": r
            for r in records
        }
        selected_label = st.selectbox(
            "选择要管理的类群",
            list(record_by_label),
            key="taxon_selector",
        )
        selected_record = record_by_label[selected_label]
        st.caption("选择类群后再打开详情，页面不会一次性展开全部记录。")
        action_col_1, action_col_2 = st.columns(2)
        with action_col_1:
            if st.button("查看图片与分割", key="open_selected_taxon", use_container_width=True):
                st.session_state["selected_taxon"] = selected_record
                st.session_state["taxon_image_page"] = 1
                st.rerun()
        with action_col_2:
            if st.button("删除当前类群", key="delete_selected_taxon", use_container_width=True):
                st.session_state["pending_delete_taxon"] = selected_record["id"]
                st.rerun()
        if st.session_state.get("pending_delete_taxon") == selected_record["id"]:
            st.warning("删除会移除该类群的数据库记录及其本地图片备份。")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("确认删除", key="confirm_selected_taxon_delete"):
                res = requests.delete(f"{API_BASE_URL}/taxonomy/delete/{selected_record['id']}")
                if res.status_code == 200:
                    st.session_state.pop("pending_delete_taxon", None)
                    if st.session_state.get("selected_taxon", {}).get("id") == selected_record["id"]:
                        st.session_state.pop("selected_taxon", None)
                    st.success("已删除该类群及其图片。")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "删除失败。"))
            if cancel_col.button("取消", key="cancel_selected_taxon_delete"):
                st.session_state.pop("pending_delete_taxon", None)
                st.rerun()
    else:
        st.info("库内暂无已存类群拓扑。")

    selected_taxon = st.session_state.get("selected_taxon")
    if selected_taxon:
        st.divider()
        detail_title, detail_close = st.columns([5, 1])
        detail_title.subheader(
            f"{selected_taxon['species']} - {selected_taxon['variant']}"
        )
        if detail_close.button("关闭", use_container_width=True):
            st.session_state.pop("selected_taxon", None)
            st.session_state.pop("pending_delete_image", None)
            st.rerun()

        image_action_feedback = st.session_state.pop("image_action_feedback", None)
        if image_action_feedback:
            st.success(image_action_feedback)

        image_mode_col, image_sensitivity_col = st.columns(2)
        with image_mode_col:
            individual_mode_label = st.selectbox(
                "单图重分割模式",
                list(SEGMENTATION_MODES),
                index=0,
                key=f"individual_mode_{selected_taxon['id']}",
            )
        with image_sensitivity_col:
            individual_sensitivity = st.slider(
                "单图重分割灵敏度",
                0,
                100,
                70,
                key=f"individual_sensitivity_{selected_taxon['id']}",
            )

        image_res = requests.get(
            f"{API_BASE_URL}/taxonomy/{selected_taxon['id']}/images",
            params={"sensitivity": individual_sensitivity},
        )
        taxon_images = image_res.json() if image_res.status_code == 200 else []
        page_size = 8
        total_pages = max(1, (len(taxon_images) + page_size - 1) // page_size)
        current_page = min(st.session_state.get("taxon_image_page", 1), total_pages)
        if total_pages > 1:
            page_widget_key = f"taxon_page_{selected_taxon['id']}"
            if st.session_state.get(page_widget_key, 1) > total_pages:
                st.session_state[page_widget_key] = total_pages
            current_page = st.number_input(
                "图片页码",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key=page_widget_key,
            )
            st.session_state["taxon_image_page"] = current_page

        page_start = (current_page - 1) * page_size
        for image_record in taxon_images[page_start: page_start + page_size]:
            image_id = image_record["id"]
            original_col, segmented_col, image_action_col = st.columns([2, 2, 1.2])
            original_col.image(
                base64.b64decode(image_record["original_preview_base64"]),
                caption=f"原图 #{image_id}",
                use_container_width=True,
            )
            segmented_col.image(
                base64.b64decode(image_record["segmented_preview_base64"]),
                caption="当前分割",
                use_container_width=True,
            )
            metrics = image_record.get("metrics", {})
            image_action_col.metric(
                "前景", f"{metrics.get('foreground_ratio', 0) * 100:.1f}%"
            )
            if metrics.get("suspicious"):
                image_action_col.warning("疑似失败")

            if image_action_col.button(
                "重新分割", key=f"single_resegment_{image_id}", use_container_width=True
            ):
                with st.spinner(f"正在处理图像 #{image_id}..."):
                    single_res = requests.post(
                        f"{API_BASE_URL}/taxonomy/images/{image_id}/resegment",
                        json={
                            "mode": SEGMENTATION_MODES[individual_mode_label],
                            "sensitivity": individual_sensitivity,
                        },
                    )
                if single_res.status_code == 200:
                    st.session_state["image_action_feedback"] = (
                        f"图像 #{image_id} 已重新分割，旧掩码已备份。"
                    )
                    st.rerun()
                else:
                    image_action_col.error(single_res.json().get("detail", "重分割失败。"))

            if image_action_col.button(
                "删除图片", key=f"request_delete_image_{image_id}", use_container_width=True
            ):
                st.session_state["pending_delete_image"] = image_id
                st.rerun()

            if st.session_state.get("pending_delete_image") == image_id:
                confirm_col, cancel_col = image_action_col.columns(2)
                if confirm_col.button("确认", key=f"confirm_delete_image_{image_id}"):
                    delete_res = requests.delete(
                        f"{API_BASE_URL}/taxonomy/images/{image_id}"
                    )
                    if delete_res.status_code == 200:
                        st.session_state.pop("pending_delete_image", None)
                        st.session_state["image_action_feedback"] = (
                            f"图像 #{image_id} 已删除并归档。"
                        )
                        st.rerun()
                    else:
                        image_action_col.error(delete_res.json().get("detail", "删除失败。"))
                if cancel_col.button("取消", key=f"cancel_delete_image_{image_id}"):
                    st.session_state.pop("pending_delete_image", None)
                    st.rerun()

            st.divider()

    st.divider()
    st.subheader("宽容分割预览与重构")
    col_mode, col_scope = st.columns(2)
    with col_mode:
        resegmentation_label = st.selectbox(
            "重分割模式", list(SEGMENTATION_MODES), index=0, key="resegmentation_mode"
        )
        resegmentation_sensitivity = st.slider(
            "重分割灵敏度", 0, 100, 70, key="resegmentation_sensitivity"
        )
    with col_scope:
        scope_label = st.selectbox(
            "处理范围", ["仅修复疑似失败图", "全库重新分割"], index=0
        )
        backup_existing = st.checkbox("自动备份旧分割图", value=True)

    preview_file = st.file_uploader(
        "上传代表图预览分割", type=['png', 'jpg', 'jpeg'], key="segmentation_preview"
    )
    if preview_file and st.button("预览当前分割参数", use_container_width=True):
        with st.spinner("正在生成分割预览..."):
            preview_res = requests.post(
                f"{API_BASE_URL}/taxonomy/segment-preview",
                data={
                    "segmentation_mode": SEGMENTATION_MODES[resegmentation_label],
                    "segmentation_sensitivity": resegmentation_sensitivity,
                },
                files={
                    "file": (
                        preview_file.name,
                        preview_file.getvalue(),
                        "image/jpeg",
                    )
                },
            )
        if preview_res.status_code == 200:
            preview = preview_res.json()
            preview_col_original, preview_col_mask = st.columns(2)
            preview_col_original.image(preview_file, caption="原图", use_container_width=True)
            preview_col_mask.image(
                base64.b64decode(preview["preview_base64"]),
                caption="透明区棋盘预览",
                use_container_width=True,
            )
            metrics = preview.get("metrics", {})
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("前景占比", f"{metrics.get('foreground_ratio', 0) * 100:.2f}%")
            metric_col2.metric(
                "中心前景", f"{metrics.get('center_foreground_ratio', 0) * 100:.2f}%"
            )
            metric_col3.metric("宽容恢复", "是" if metrics.get("recovery_used") else "否")
            if metrics.get("suspicious"):
                st.warning("该参数下分割结果仍被判定为疑似失败。")
        else:
            st.error(preview_res.json().get("detail", "预览失败。"))

    try:
        segmentation_status = requests.get(
            f"{API_BASE_URL}/taxonomy/resegment/status"
        ).json()
    except requests.exceptions.ConnectionError:
        segmentation_status = {"is_running": False, "status": "unavailable"}

    segmentation_running = segmentation_status.get("is_running", False)
    if segmentation_running:
        current = segmentation_status.get("current", 0)
        total = segmentation_status.get("total", 0)
        st.progress(current / max(total, 1))
        st.info(segmentation_status.get("message", "重分割运行中..."))
        time.sleep(2)
        st.rerun()
    elif segmentation_status.get("status") == "completed":
        st.success(segmentation_status.get("message"))
        if segmentation_status.get("backup_path"):
            st.code(segmentation_status["backup_path"])
            confirm_restore = st.checkbox("确认恢复本次重分割前的旧掩码")
            if st.button(
                "恢复旧分割图",
                use_container_width=True,
                disabled=not confirm_restore,
            ):
                restore_res = requests.post(
                    f"{API_BASE_URL}/taxonomy/resegment/restore",
                    json={"backup_path": segmentation_status["backup_path"]},
                )
                if restore_res.status_code == 200:
                    st.success(restore_res.json().get("message"))
                else:
                    st.error(restore_res.json().get("detail", "恢复失败。"))
    elif segmentation_status.get("status") == "failed":
        st.error(segmentation_status.get("message", "重分割失败。"))

    if st.button(
        "启动安全重分割",
        use_container_width=True,
        disabled=segmentation_running,
    ):
        resegmentation_res = requests.post(
            f"{API_BASE_URL}/taxonomy/resegment",
            json={
                "mode": SEGMENTATION_MODES[resegmentation_label],
                "sensitivity": resegmentation_sensitivity,
                "scope": "suspicious" if scope_label == "仅修复疑似失败图" else "all",
                "backup_existing": backup_existing,
            },
        )
        if resegmentation_res.status_code == 200:
            st.info(resegmentation_res.json().get("message"))
            time.sleep(1)
            st.rerun()
        else:
            st.error(resegmentation_res.json().get("detail", "任务启动失败。"))

# ==========================================
# 模块 1: 流形度量训练
# ==========================================
# 在 app.py 的 “流形度量训练” 模块下，替换对应的表单渲染代码
if mode == "流形度量训练":
    st.header("度量空间构建与退火控制")

    try:
        status_res = requests.get(f"{API_BASE_URL}/train/status").json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到计算引擎，请检查后端服务是否启动。")
        st.stop()

    is_running = status_res.get("is_training", False)
    model_weights_loaded = status_res.get("model_weights_loaded", False)
    checkpoint_available = status_res.get("checkpoint_available", False)

    # 构建高阶超参数配置表单
    with st.expander("⚙️ 训练策略与动力学配置 (Advanced Hyperparameters)", expanded=True):
        col_ep, col_st = st.columns(2)
        with col_ep:
            target_epochs = st.number_input("目标训练轮次 (Epochs)", min_value=1, max_value=1000, value=300,
                                            disabled=is_running)
            p_classes = st.number_input("每批次类群数 (P)", min_value=2, max_value=128, value=16, disabled=is_running)
            k_instances = st.number_input("单类群图像张数 (K)", min_value=2, max_value=16, value=4, disabled=is_running)
        with col_st:
            alpha = st.slider(
                "目标微调先验权重 (α)", 0.0, 0.5, 0.05, step=0.01,
                help="最终训练建议使用 0.05；保持 P=16、K=4。",
                disabled=is_running,
            )
            training_start_options = [
                "当前已加载模型增量训练"
                if model_weights_loaded
                else "随机初始化训练"
            ]
            if checkpoint_available:
                training_start_options.append("最近训练断点续训")
            training_start = st.radio(
                "训练起点",
                training_start_options,
                horizontal=True,
                disabled=is_running,
                help=(
                    "当前模型增量训练会使用新的优化器和轮次；只用少量局部类群微调可能"
                    "造成旧类群遗忘。仅新增类群时优先使用系统维护中的多原型重建。"
                ),
            )
            resume_ckpt = training_start == "最近训练断点续训"
            quality_aware = st.checkbox(
                "启用类群内分割质量平衡", value=True, disabled=is_running
            )
            st.markdown(f"**内存前向传播的实际 Batch Size (包含双视图): {p_classes * k_instances * 2}**")

        col_attn, col_proto = st.columns(2)
        with col_attn:
            lambda_orth = st.slider(
                "注意力头互斥权重", 0.0, 0.10, 0.02, step=0.005, disabled=is_running
            )
            lambda_entropy = st.slider(
                "注意力面积约束权重", 0.0, 0.20, 0.05, step=0.01, disabled=is_running
            )
        with col_proto:
            target_attention_entropy = st.slider(
                "目标注意力覆盖率", 0.20, 0.90, 0.55, step=0.05, disabled=is_running
            )
            prototypes_per_taxon = st.number_input(
                "每类群子原型数", min_value=1, max_value=6, value=3, disabled=is_running
            )

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("启动计算引擎", use_container_width=True, disabled=is_running):
            payload = {
                "epochs": target_epochs,
                "alpha": alpha,
                "p_classes": p_classes,
                "k_instances": k_instances,
                "resume": resume_ckpt,
                "max_lambda_orth": lambda_orth,
                "lambda_tv": 0.05,
                "lambda_entropy": lambda_entropy,
                "target_attention_entropy": target_attention_entropy,
                "prototypes_per_taxon": prototypes_per_taxon,
                "quality_aware": quality_aware,
            }
            res = requests.post(f"{API_BASE_URL}/train/start", json=payload)
            if res.status_code == 200:
                st.success("指令已下发，CUDA 引擎启动。")
                time.sleep(1)
                st.rerun()
            else:
                st.error(res.json().get("detail", "启动失败"))

    with col_stop:
        if st.button("安全中断 (Graceful Stop)", use_container_width=True, disabled=not is_running):
            res = requests.post(f"{API_BASE_URL}/train/stop")
            st.warning(res.json().get("message"))
            time.sleep(1)
            st.rerun()

    st.divider()

    # 状态机轮询渲染
    if status_res["status"] == "running":
        current = status_res["current_epoch"]
        total = status_res["total_epochs"]
        loss = status_res["loss"]

        progress = current / max(1, total)
        st.progress(progress)

        st.write(f"**训练状态**: {status_res['message']}")
        st.write(f"**进度**: {current} / {total} Epochs")
        st.write(f"**当前泛化损失 (Loss)**: {loss:.4f}")
        component_col1, component_col2, component_col3 = st.columns(3)
        component_col1.metric(
            "度量损失", f"{status_res.get('metric_loss', loss):.4f}"
        )
        component_col2.metric(
            "当前 α", f"{status_res.get('alpha', 0.0):.4f}"
        )
        component_col3.metric(
            "平均质量权重", f"{status_res.get('mean_quality_weight', 1.0):.3f}"
        )
        quality_summary = status_res.get("quality_summary", {})
        if quality_summary.get("enabled"):
            st.caption(
                f"自动质量平衡：疑似 {quality_summary.get('suspicious_count', 0)} 张，"
                f"平均质量分 {quality_summary.get('mean_score', 0.0):.3f}"
            )

        time.sleep(2)
        st.rerun()

    elif status_res["status"] == "completed":
        st.success("流形收敛完成，度量空间已固化。")

    elif status_res["status"] == "failed":
        st.error(f"引擎异常中断: {status_res.get('message', '未知错误')}")
    elif status_res["status"] == "queued":
        st.info(status_res.get("message", "训练任务排队中..."))
        time.sleep(1)
        st.rerun()
    elif status_res["status"] == "stopped":
        st.warning(status_res.get("message", "训练已安全中断。"))
elif mode == "表型关系发掘":
    st.header("表型关系发掘")
    st.caption("当前模型与图库中的表型证据，不等同于遗传系统发育结论。")

    try:
        taxon_records_res = requests.get(f"{API_BASE_URL}/taxonomy/records", timeout=10)
        taxon_records = taxon_records_res.json() if taxon_records_res.status_code == 200 else []
    except requests.exceptions.RequestException as exc:
        st.error(f"无法读取类群记录：{exc}")
        st.stop()

    if not taxon_records:
        st.info("请先导入类群和图片。")
        st.stop()

    relationship_view = st.radio(
        "查看内容",
        ["关系网络与单类群证据", "双类群比较", "树状聚类"],
        horizontal=True,
        key="phenotype_relationship_view",
    )

    if relationship_view == "树状聚类":
        st.subheader("表型层次聚类树")
        refresh_tree = st.button("生成/刷新树状图", use_container_width=True)
        if refresh_tree or "phenotype_dendrogram" not in st.session_state:
            with st.spinner("正在根据当前模型原型计算层次聚类..."):
                try:
                    dendrogram_res = requests.get(
                        f"{API_BASE_URL}/evolution/dendrogram", timeout=180
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"树状图请求失败：{exc}")
                else:
                    if dendrogram_res.status_code == 200:
                        st.session_state["phenotype_dendrogram"] = dendrogram_res.content
                    else:
                        try:
                            detail = dendrogram_res.json().get("detail", "树状图生成失败。")
                        except ValueError:
                            detail = "树状图生成失败。"
                        st.warning(detail)
        if st.session_state.get("phenotype_dendrogram"):
            st.image(
                st.session_state["phenotype_dendrogram"],
                caption="基于当前模型类群原型的平均连接层次聚类",
                use_container_width=True,
            )
            st.warning("树状分支表示当前模型中的表型距离，不代表遗传亲缘、祖先后代或演化方向。")
        st.stop()

    if relationship_view == "双类群比较":
        comparison_records = sorted(
            taxon_records,
            key=lambda row: (
                not row.get("has_prototype", False),
                row.get("species", "").casefold(),
                row.get("variant", "").casefold(),
            ),
        )
        if len(comparison_records) < 2:
            st.info("至少需要两个有图片的类群才能进行比较。")
            st.stop()

        comparison_by_label = {
            f"#{row['id']} · {row['species']} - {row['variant']} · {row['image_count']} 张"
            f"{' · 已有模型原型' if row.get('has_prototype') else ''}": row
            for row in comparison_records
        }
        comparison_labels = list(comparison_by_label)
        selector_a, selector_b = st.columns(2)
        with selector_a:
            selected_a_label = st.selectbox(
                "类群 A", comparison_labels, index=0, key="comparison_taxon_a"
            )
        with selector_b:
            selected_b_label = st.selectbox(
                "类群 B", comparison_labels, index=1, key="comparison_taxon_b"
            )
        selected_a = comparison_by_label[selected_a_label]
        selected_b = comparison_by_label[selected_b_label]
        comparison_sensitivity = st.slider(
            "分割质量灵敏度", 0, 100, 70, key="comparison_sensitivity"
        )
        comparison_key = (
            f"{selected_a['id']}:{selected_b['id']}:{comparison_sensitivity}"
        )
        if st.session_state.get("taxon_comparison_request_key") != comparison_key:
            st.session_state.pop("taxon_comparison", None)

        same_taxon = selected_a["id"] == selected_b["id"]
        if same_taxon:
            st.warning("请选择两个不同的类群。")
        generate_comparison = st.button(
            "生成双类群比较报告",
            use_container_width=True,
            disabled=same_taxon,
        )
        if generate_comparison:
            with st.spinner("正在提取两组嵌入、统计图像证据并匹配子原型..."):
                try:
                    comparison_res = requests.get(
                        f"{API_BASE_URL}/evidence/compare",
                        params={
                            "tax_id_a": selected_a["id"],
                            "tax_id_b": selected_b["id"],
                            "sensitivity": comparison_sensitivity,
                        },
                        timeout=300,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"比较请求失败：{exc}")
                else:
                    if comparison_res.status_code == 200:
                        st.session_state["taxon_comparison"] = comparison_res.json()
                        st.session_state["taxon_comparison_request_key"] = comparison_key
                    else:
                        try:
                            detail = comparison_res.json().get("detail", "比较报告生成失败。")
                        except ValueError:
                            detail = "比较报告生成失败。"
                        st.error(detail)

        comparison = st.session_state.get("taxon_comparison")
        if comparison:
            taxon_a = comparison.get("taxon_a", {})
            taxon_b = comparison.get("taxon_b", {})
            name_a = f"{taxon_a.get('species', '')} - {taxon_a.get('variant', '')}"
            name_b = f"{taxon_b.get('species', '')} - {taxon_b.get('variant', '')}"
            st.subheader(f"{name_a}  与  {name_b}")
            st.caption(comparison.get("scope", "仅比较当前图库与模型证据。"))

            embedding = comparison.get("embedding", {})
            embedding_a = embedding.get("a", {})
            embedding_b = embedding.get("b", {})
            metric_columns = st.columns(5)
            metric_columns[0].metric(
                "类群中心相似度",
                f"{embedding.get('centroid_cosine_similarity', 0.0):.4f}",
            )
            metric_columns[1].metric(
                "A 图库内分离率",
                f"{embedding_a.get('gallery_separation_rate', 0.0) * 100:.1f}%",
            )
            metric_columns[2].metric(
                "B 图库内分离率",
                f"{embedding_b.get('gallery_separation_rate', 0.0) * 100:.1f}%",
            )
            metric_columns[3].metric(
                "A 类内紧凑度",
                f"{embedding_a.get('within_class_compactness', 0.0):.4f}",
            )
            metric_columns[4].metric(
                "B 类内紧凑度",
                f"{embedding_b.get('within_class_compactness', 0.0):.4f}",
            )
            margin_columns = st.columns(2)
            margin_columns[0].metric(
                "A 中位边界差值", f"{embedding_a.get('median_margin', 0.0):.4f}"
            )
            margin_columns[1].metric(
                "B 中位边界差值", f"{embedding_b.get('median_margin', 0.0):.4f}"
            )
            st.caption(embedding.get("scope", "图库内证据不是留出测试集准确率。"))

            st.markdown("**证据摘要**")
            for item in comparison.get("summary", []):
                st.markdown(f"**{item.get('title', '')}：** {item.get('text', '')}")

            st.markdown("**基础图像特征比较**")

            def format_trait_value(row, value):
                if row.get("display") == "percent":
                    return f"{value * 100:.1f}%"
                return f"{value:.3f}"

            trait_table = []
            for row in comparison.get("traits", []):
                trait_table.append({
                    "证据维度": row["label"],
                    "A 中位数": format_trait_value(row, row["a_median"]),
                    "A 第10-90百分位": (
                        f"{format_trait_value(row, row['a_p10'])} 至 "
                        f"{format_trait_value(row, row['a_p90'])}"
                    ),
                    "B 中位数": format_trait_value(row, row["b_median"]),
                    "B 第10-90百分位": (
                        f"{format_trait_value(row, row['b_p10'])} 至 "
                        f"{format_trait_value(row, row['b_p90'])}"
                    ),
                    "中位数差 A-B": format_trait_value(
                        row, row["median_difference_a_minus_b"]
                    ),
                    "差值95%区间": (
                        f"{format_trait_value(row, row['difference_ci95_low'])} 至 "
                        f"{format_trait_value(row, row['difference_ci95_high'])}"
                    ),
                    "效应量（Cliff's δ）": (
                        f"{row['effect_magnitude']}（{row['cliffs_delta']:.3f}）"
                    ),
                })
            if trait_table:
                st.dataframe(
                    pd.DataFrame(trait_table),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("正差值或正 delta 表示当前图库中 A 的数值总体较高；质量与半透明指标反映分割状态。")

            st.markdown("**子原型交叉匹配**")
            sub_prototypes = comparison.get("sub_prototypes", {})
            if sub_prototypes.get("status") == "available":
                a_counts = sub_prototypes.get("a_sample_counts", [])
                b_counts = sub_prototypes.get("b_sample_counts", [])
                matrix = pd.DataFrame(
                    sub_prototypes.get("matrix", []),
                    index=[
                        f"A 子原型 {index + 1}（{count} 张）"
                        for index, count in enumerate(a_counts)
                    ],
                    columns=[
                        f"B 子原型 {index + 1}（{count} 张）"
                        for index, count in enumerate(b_counts)
                    ],
                )
                st.dataframe(
                    matrix.style.format("{:.4f}"),
                    use_container_width=True,
                )
                strongest = sub_prototypes.get("strongest_match", {})
                least_a = sub_prototypes.get("least_shared_a", {})
                least_b = sub_prototypes.get("least_shared_b", {})
                st.caption(
                    f"最强配对：A 子原型 {strongest.get('a_prototype')} ↔ "
                    f"B 子原型 {strongest.get('b_prototype')}，相似度 "
                    f"{strongest.get('similarity', 0.0):.4f}。"
                    f"相对独特：A 子原型 {least_a.get('prototype')}、"
                    f"B 子原型 {least_b.get('prototype')}。"
                )
            else:
                st.info("至少一个类群尚无当前模型子原型；图像与嵌入比较仍然有效。")

            st.markdown("**各类群代表图**")
            representative_images = comparison.get("representative_images", {})
            for group_key, group_name in (("a", name_a), ("b", name_b)):
                st.markdown(f"**{group_key.upper()} · {group_name}**")
                images = representative_images.get(group_key, [])
                image_columns = st.columns(max(1, len(images)))
                for image_index, evidence_image in enumerate(images):
                    column = image_columns[image_index]
                    column.image(
                        base64.b64decode(evidence_image["original_preview_base64"]),
                        caption=f"原图 #{evidence_image['image_id']}",
                        use_container_width=True,
                    )
                    column.image(
                        base64.b64decode(evidence_image["segmented_preview_base64"]),
                        caption="当前分割",
                        use_container_width=True,
                    )
                    column.caption(
                        f"类内相似度 {evidence_image['own_similarity']:.4f} · "
                        f"边界差值 {evidence_image['margin']:.4f}"
                    )

            st.markdown("**最接近比较边界的真实图片**")
            boundary_images = comparison.get("boundary_images", {})
            boundary_columns = st.columns(2)
            for column, group_key, group_name in zip(
                boundary_columns, ("a", "b"), (name_a, name_b)
            ):
                boundary = boundary_images.get(group_key)
                if not boundary:
                    continue
                column.markdown(f"**{group_key.upper()} · {group_name}**")
                column.image(
                    base64.b64decode(boundary["original_preview_base64"]),
                    caption=f"原图 #{boundary['image_id']}",
                    use_container_width=True,
                )
                column.image(
                    base64.b64decode(boundary["segmented_preview_base64"]),
                    caption="当前分割",
                    use_container_width=True,
                )
                column.caption(
                    f"本类中心 {boundary['own_similarity']:.4f} · "
                    f"对方中心 {boundary['other_similarity']:.4f} · "
                    f"边界差值 {boundary['margin']:.4f}"
                )

            for warning in comparison.get("warnings", []):
                st.warning(warning)
        st.stop()

    st.subheader("表型相似度网络")
    network_col, network_action_col = st.columns([2, 1])
    with network_col:
        network_k = st.slider("每个节点保留的近邻边", 1, 5, 2, key="phenotype_network_k")
    with network_action_col:
        refresh_network = st.button("生成/刷新网络", use_container_width=True)
    if (
        refresh_network
        or "phenotype_network" not in st.session_state
        or not st.session_state.get("phenotype_network", {}).get("nodes")
    ):
        with st.spinner("正在根据原型相似度计算网络布局..."):
            network_res = requests.get(
                f"{API_BASE_URL}/evolution/network",
                params={"k": network_k},
                timeout=180,
            )
        if network_res.status_code == 200:
            st.session_state["phenotype_network"] = network_res.json()
        else:
            st.error(network_res.json().get("detail", "网络生成失败。"))

    network_data = st.session_state.get("phenotype_network")
    if network_data:
        network_metrics = st.columns(3)
        network_metrics[0].metric("原型节点", network_data.get("node_count", 0))
        network_metrics[1].metric("近邻连接", network_data.get("edge_count", 0))
        network_metrics[2].metric(
            "跨物种连接",
            sum(edge.get("cross_species", False) for edge in network_data.get("edges", [])),
        )
        render_interactive_network(network_data)
        st.warning(network_data.get("interpretation", "网络仅表示表型相似度。"))
        if network_data.get("k") != network_k:
            st.info(f"当前图使用每节点 {network_data.get('k')} 条近邻边；点击“生成/刷新网络”应用新设置。")

        with st.expander("静态网络图", expanded=False):
            st.image(
                base64.b64decode(network_data["image_base64"]),
                caption=f"{network_data['node_count']} 个节点，{network_data['edge_count']} 条近邻边",
                use_container_width=True,
            )

        st.markdown("**逐类群相似关系**")
        network_nodes = sorted(network_data.get("nodes", []), key=lambda row: row["label"])
        node_by_label = {node["label"]: node for node in network_nodes}
        relationship_label = st.selectbox(
            "选择类群查看相似对象",
            list(node_by_label),
            key="network_relationship_taxon",
        )
        relationship_node = node_by_label[relationship_label]
        relation_filter_col, relation_count_col = st.columns(2)
        with relation_filter_col:
            relationship_scope = st.radio(
                "关系范围",
                ["全部", "仅跨物种", "仅同物种"],
                horizontal=True,
                key="network_relationship_scope",
            )
        with relation_count_col:
            relationship_count = st.slider(
                "显示相似对象数量", 3, 12, 8, key="network_relationship_count"
            )

        selected_relationships = [
            row for row in network_data.get("relationships", [])
            if row["source_tax_id"] == relationship_node["tax_id"]
        ]
        if relationship_scope == "仅跨物种":
            selected_relationships = [row for row in selected_relationships if not row["same_species"]]
        elif relationship_scope == "仅同物种":
            selected_relationships = [row for row in selected_relationships if row["same_species"]]
        selected_relationships = selected_relationships[:relationship_count]

        relation_summary = st.columns(4)
        relation_summary[0].metric("图片数", relationship_node.get("image_count", 0))
        relation_summary[1].metric("网络连接", relationship_node.get("degree", 0))
        relation_summary[2].metric("跨物种连接", relationship_node.get("cross_species_degree", 0))
        relation_summary[3].metric(
            "最高相似度",
            f"{selected_relationships[0]['similarity']:.4f}" if selected_relationships else "-",
        )

        if selected_relationships:
            selected_df = pd.DataFrame(selected_relationships).rename(columns={
                "rank": "排名", "target_tax_id": "相似类群ID", "target_label": "相似类群",
                "relationship": "关系", "similarity": "余弦相似度",
                "mutual_neighbor": "互为近邻", "edge_in_network": "显示在网络中",
                "target_image_count": "对方图片数",
            })
            st.dataframe(
                selected_df[[
                    "排名", "相似类群ID", "相似类群", "关系", "余弦相似度",
                    "互为近邻", "显示在网络中", "对方图片数",
                ]].style.format({"余弦相似度": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("当前范围内没有相似关系。")

        with st.expander("全部类群相似关系", expanded=False):
            full_relation_scope = st.radio(
                "完整表筛选",
                ["全部关系", "跨物种关系", "互为近邻"],
                horizontal=True,
                key="full_relationship_scope",
            )
            full_relationships = network_data.get("relationships", [])
            if full_relation_scope == "跨物种关系":
                full_relationships = [row for row in full_relationships if not row["same_species"]]
            elif full_relation_scope == "互为近邻":
                full_relationships = [row for row in full_relationships if row["mutual_neighbor"]]
            full_df = pd.DataFrame(full_relationships).rename(columns={
                "source_tax_id": "来源ID", "source_label": "来源类群",
                "rank": "来源内排名", "target_tax_id": "相似类群ID",
                "target_label": "相似类群", "relationship": "关系",
                "similarity": "余弦相似度", "mutual_neighbor": "互为近邻",
                "target_image_count": "对方图片数",
            })
            if not full_df.empty:
                st.dataframe(
                    full_df[[
                        "来源ID", "来源类群", "来源内排名", "相似类群ID", "相似类群",
                        "关系", "余弦相似度", "互为近邻", "对方图片数",
                    ]].style.format({"余弦相似度": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                    height=430,
                )

        bridge_rows = network_data.get("bridges", [])
        if bridge_rows:
            with st.expander("跨物种桥接概览", expanded=False):
                bridge_df = pd.DataFrame(bridge_rows).rename(columns={
                    "tax_id": "ID", "label": "类群", "image_count": "图片数",
                    "degree": "连接数", "cross_species_degree": "跨物种连接数",
                    "strongest_cross_similarity": "最强跨物种相似度",
                    "strongest_cross_neighbor": "最相似的跨物种类群",
                })
                st.dataframe(
                    bridge_df[[
                        "ID", "类群", "最相似的跨物种类群", "最强跨物种相似度",
                        "连接数", "跨物种连接数", "图片数",
                    ]].style.format({"最强跨物种相似度": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

    st.divider()
    st.subheader("证据型气质报告与多原型外观板")
    record_by_label = {
        f"#{r['id']} · {r['species']} - {r['variant']} · {r['image_count']} 张"
        f"{' · 已有模型原型' if r.get('has_prototype') else ''}": r
        for r in taxon_records
    }
    report_default_index = next(
        (index for index, record in enumerate(taxon_records) if record.get("has_prototype")),
        0,
    )
    selected_label = st.selectbox(
        "选择要分析的类群",
        list(record_by_label),
        index=report_default_index,
        key="evidence_taxon_selector",
    )
    selected_taxon_for_report = record_by_label[selected_label]
    if st.session_state.get("evidence_report_tax_id") != selected_taxon_for_report["id"]:
        st.session_state.pop("evidence_report", None)
        st.session_state["evidence_report_tax_id"] = selected_taxon_for_report["id"]
    report_sensitivity = st.slider("报告使用的分割质量灵敏度", 0, 100, 70, key="evidence_sensitivity")
    report_request_key = f"{selected_taxon_for_report['id']}:{report_sensitivity}"
    if st.session_state.get("evidence_report_request_key") != report_request_key:
        st.session_state.pop("evidence_report", None)
    if st.button("生成证据报告", use_container_width=True):
        with st.spinner("正在读取分割证据、提取模型相似度并选择代表图..."):
            report_res = requests.get(
                f"{API_BASE_URL}/evidence/report/{selected_taxon_for_report['id']}",
                params={"sensitivity": report_sensitivity},
                timeout=180,
            )
        if report_res.status_code == 200:
            st.session_state["evidence_report"] = report_res.json()
            st.session_state["evidence_report_tax_id"] = selected_taxon_for_report["id"]
            st.session_state["evidence_report_request_key"] = report_request_key
        else:
            st.error(report_res.json().get("detail", "证据报告生成失败。"))

    report = st.session_state.get("evidence_report")
    if report:
        report_taxon = report.get("taxon", {})
        st.markdown(
            f"**{report_taxon.get('species', '')} - {report_taxon.get('variant', '')}** · "
            f"{report.get('image_count', 0)} 张可用图片"
        )
        if report.get("prototype_status") != "available":
            st.warning("该类群尚未建立当前模型原型，外观板中的图片是真实代表图，不是模型子原型。")
        st.caption(report.get("interpretation_scope", "报告只整理图像证据。"))
        similarity_col_1, similarity_col_2 = st.columns(2)
        similarity_col_1.metric("平均最佳原型相似度", f"{report.get('mean_model_similarity', 0.0):.4f}")
        similarity_col_2.metric("相似度第10百分位", f"{report.get('model_similarity_p10', 0.0):.4f}")

        st.markdown("**可观测证据**")
        observation_df = pd.DataFrame(report.get("observations", []))
        if not observation_df.empty:
            observation_df = observation_df.rename(columns={
                "label": "证据维度", "text": "当前类群图像中的统计描述", "basis": "依据",
            })
            st.dataframe(observation_df, use_container_width=True, hide_index=True)

        st.markdown("**多原型外观板**")
        board = report.get("prototype_board", [])
        board_columns = st.columns(max(1, min(3, len(board))))
        for index, prototype in enumerate(board):
            column = board_columns[index % len(board_columns)]
            status = prototype.get("status")
            title = (
                f"子原型 {prototype['prototype']}"
                if prototype.get("prototype") is not None
                else "真实代表图"
            )
            column.markdown(f"**{title} · 图片 #{prototype['image_id']}**")
            column.image(base64.b64decode(prototype["original_preview_base64"]), use_container_width=True)
            column.image(base64.b64decode(prototype["segmented_preview_base64"]), use_container_width=True)
            if status == "model_prototype":
                column.caption(
                    f"分配样本 {prototype['assigned_image_count']} · "
                    f"储存样本 {prototype['stored_sample_count']} · "
                    f"相似度 {prototype['similarity']:.4f}"
                )
            else:
                column.caption("尚未建立模型原型，仅作为真实图库代表图。")

        st.markdown("**证据样本（按当前模型相似度排序）**")
        evidence_images = report.get("evidence_images", [])
        evidence_columns = st.columns(3)
        for index, evidence in enumerate(evidence_images):
            column = evidence_columns[index % 3]
            column.image(base64.b64decode(evidence["original_preview_base64"]), use_container_width=True)
            column.caption(
                f"图片 #{evidence['image_id']} · 子原型 {evidence['best_prototype']} · "
                f"相似度 {evidence['similarity']:.4f}"
            )

# ==========================================
# 模块 2: 开放集推理
# ==========================================
elif mode == "开放集推理":
    st.header("表型相似度推理")

    uploaded_file = st.file_uploader("上传待验图像", type=["jpg", "png", "jpeg"])
    inference_mode_col, inference_sensitivity_col = st.columns(2)
    with inference_mode_col:
        inference_segmentation_label = st.selectbox(
            "预测分割模式", list(SEGMENTATION_MODES), index=0
        )
        rejection_threshold = st.slider(
            "未知类群拒绝阈值", -1.0, 1.0, 0.55, step=0.01
        )
    with inference_sensitivity_col:
        inference_sensitivity = st.slider("预测分割灵敏度", 0, 100, 70)

    if uploaded_file:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(uploaded_file, caption="原始输入图像", use_container_width=True)

        if st.button("提取特征并比对", use_container_width=True):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}

            with st.spinner("调度 GPU 执行背景剥离与正向传播计算..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/inference/predict",
                        files=files,
                        data={
                            "rejection_threshold": rejection_threshold,
                            "segmentation_mode": SEGMENTATION_MODES[inference_segmentation_label],
                            "segmentation_sensitivity": inference_sensitivity,
                        }
                    )
                except requests.exceptions.ConnectionError:
                    st.error("后端服务未响应。")
                    st.stop()

            if res.status_code == 200:
                data = res.json()
                predictions = data.get("predictions", [])
                seg_b64 = data.get("segmented_image_base64", "")

                if data.get("is_unknown", False):
                    st.warning(
                        f"未知类群：最高相似度 {data.get('top_similarity', 0.0):.4f} "
                        f"低于阈值 {data.get('rejection_threshold', rejection_threshold):.2f}"
                    )
                else:
                    st.success("已通过开放集拒绝阈值。")

                # 渲染网络实际感知的无背景图像
                if seg_b64:
                    with col_img2:
                        seg_bytes = base64.b64decode(seg_b64)
                        st.image(seg_bytes, caption="网络实际感知的纯净特征图", use_container_width=True)

                if not predictions:
                    st.warning("未匹配到结果。")
                else:
                    st.subheader("拓扑匹配结果")
                    df = pd.DataFrame(predictions)
                    df.index = df.index + 1
                    df = df[["label", "confidence", "prototype"]]
                    df.columns = ["类群标签 (Taxon)", "余弦相似度 (Confidence)", "子原型"]
                    st.table(df.style.format({"余弦相似度 (Confidence)": "{:.4f}"}))

            elif res.status_code == 503:
                st.warning("CUDA 核心被训练任务独占，请等待训练释放计算图。")
            elif res.status_code == 404:
                st.error("知识库拓扑为空，请先导入数据并完成基座训练。")
            else:
                st.error(f"推理请求失败: {res.text}")

elif mode == "气质逆向解码":
    st.header("门控权重与注意力热力图")
    df = st.file_uploader("上传解析图像", type=['png', 'jpg', 'jpeg'])
    decode_mode_col, decode_sensitivity_col = st.columns(2)
    with decode_mode_col:
        decode_segmentation_label = st.selectbox(
            "解码分割模式", list(SEGMENTATION_MODES), index=0
        )
    with decode_sensitivity_col:
        decode_sensitivity = st.slider("解码分割灵敏度", 0, 100, 70)
    if df:
        st.image(df, caption="原图", width=200)
        if st.button("逆向追踪弱监督注意力图谱", use_container_width=True):
            files = {"file": (df.name, df.getvalue(), "image/jpeg")}
            with st.spinner("正向截断注意力追踪中..."):
                dec_res = requests.post(
                    f"{API_BASE_URL}/inference/decode",
                    files=files,
                    data={
                        "segmentation_mode": SEGMENTATION_MODES[decode_segmentation_label],
                        "segmentation_sensitivity": decode_sensitivity,
                    },
                )
            if dec_res.status_code == 200:
                st.image(dec_res.content, caption="多区域门控掩膜融合分布图", use_container_width=True)
            else:
                st.error(dec_res.json().get("detail", "跟踪失败"))

elif mode == "系统维护与诊断":
    st.header("系统维护与注意力诊断")

    try:
        maintenance = requests.get(f"{API_BASE_URL}/maintenance/status").json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端引擎。")
        st.stop()

    is_maintaining = maintenance.get("is_running", False)
    if is_maintaining:
        st.info(maintenance.get("message", "维护任务运行中..."))
        time.sleep(2)
        st.rerun()
    elif maintenance.get("status") == "completed":
        st.success(maintenance.get("message", "维护任务已完成。"))
    elif maintenance.get("status") == "failed":
        st.error(maintenance.get("message", "维护任务失败。"))

    st.subheader("数据与模型快照")
    include_images = st.checkbox("包含全部原图和分割图", value=False)
    if st.button("创建时间戳快照", use_container_width=True, disabled=is_maintaining):
        with st.spinner("正在创建快照..."):
            snapshot_res = requests.post(
                f"{API_BASE_URL}/maintenance/snapshot",
                json={"include_images": include_images}
            )
        if snapshot_res.status_code == 200:
            snapshot = snapshot_res.json()
            st.success(snapshot.get("message"))
            st.code(snapshot.get("path", ""))
        else:
            st.error(snapshot_res.json().get("detail", "快照创建失败。"))

    st.divider()
    st.subheader("多原型重建")
    rebuild_count = st.number_input("每类群子原型数", min_value=1, max_value=6, value=3)
    if st.button("使用当前模型重建多原型", use_container_width=True, disabled=is_maintaining):
        rebuild_res = requests.post(
            f"{API_BASE_URL}/maintenance/rebuild-prototypes",
            json={"prototypes_per_taxon": rebuild_count}
        )
        if rebuild_res.status_code == 200:
            st.info(rebuild_res.json().get("message"))
            time.sleep(1)
            st.rerun()
        else:
            st.error(rebuild_res.json().get("detail", "任务启动失败。"))

    st.divider()
    st.subheader("注意力质量诊断")
    diagnostic_samples = st.number_input("诊断抽样数量", min_value=16, max_value=512, value=128, step=16)
    if st.button("运行注意力诊断", use_container_width=True, disabled=is_maintaining):
        with st.spinner("正在统计前景质量与注意力覆盖..."):
            diagnostic_res = requests.get(
                f"{API_BASE_URL}/maintenance/attention-diagnostics",
                params={"sample_limit": diagnostic_samples}
            )
        if diagnostic_res.status_code == 200:
            st.session_state["attention_diagnostics"] = diagnostic_res.json()
        else:
            st.error(diagnostic_res.json().get("detail", "诊断失败。"))

    diagnostics = st.session_state.get("attention_diagnostics")
    if diagnostics:
        col_samples, col_overlap = st.columns(2)
        col_samples.metric("诊断样本", diagnostics.get("sample_count", 0))
        col_overlap.metric("头间平均重叠", f"{diagnostics.get('mean_head_overlap', 0.0):.4f}")
        diagnostic_df = pd.DataFrame(diagnostics.get("heads", []))
        diagnostic_df.columns = ["注意力头", "前景质量", "归一化覆盖率", "有效网格数", "平均门控"]
        st.dataframe(
            diagnostic_df.style.format({
                "前景质量": "{:.4f}",
                "归一化覆盖率": "{:.4f}",
                "有效网格数": "{:.2f}",
                "平均门控": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
