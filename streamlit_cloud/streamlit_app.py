"""Read-only Streamlit Community Cloud demo for Haworthia OMICS."""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
HF_SOURCE = ROOT / "hf_space"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HF_SOURCE))

U2NET_HOME = Path(tempfile.gettempdir()) / "haworthia-u2net"
os.environ["U2NET_HOME"] = str(U2NET_HOME)
os.environ["HAWORTHIA_LOW_MEMORY_SEGMENTATION"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
LOCAL_ASSET_DIRECTORY = ROOT / "hf_private_assets"
if LOCAL_ASSET_DIRECTORY.is_dir():
    os.environ.setdefault("HAWORTHIA_ASSET_DIR", str(LOCAL_ASSET_DIRECTORY))

from download_segmentation_models import ensure_segmentation_models
from inference_core import HaworthiaInferenceService, SEGMENTATION_MODES
from runtime_assets import prepare_private_assets


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LOGGER = logging.getLogger("haworthia_streamlit")


st.set_page_config(
    page_title="Haworthia OMICS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .block-container {max-width: 1380px; padding-top: 1.7rem;}
    .main-notice {border-left: 3px solid #56705e; padding: .2rem 0 .2rem .9rem;
                  color: #46514a; margin-bottom: 1.1rem;}
    [data-testid="stFileUploader"] {min-height: 118px;}
    [data-testid="stExpander"] summary {font-weight: 600;}
    .small-note {color: #626b65; font-size: .9rem;}
</style>
""",
    unsafe_allow_html=True,
)


def _secret(name):
    try:
        value = st.secrets[name]
    except (KeyError, FileNotFoundError):
        value = os.getenv(name, "")
    return str(value).strip()


@st.cache_resource(show_spinner="正在加载表型模型与数值原型……")
def load_service():
    repo_id = _secret("HF_MODEL_REPO_ID")
    token = _secret("HF_TOKEN")
    directory, manifest = prepare_private_assets(repo_id=repo_id, token=token)
    return HaworthiaInferenceService(directory), manifest


@st.cache_resource(show_spinner="首次使用正在下载并校验背景分割模型……")
def prepare_segmenters():
    ensure_segmentation_models(U2NET_HOME)
    return True


def read_upload(uploaded_file):
    if uploaded_file is None:
        raise ValueError("请先上传一张图片。")
    data = uploaded_file.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("图片文件超过 20 MB 限制。")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取该图片，请改用正常的 JPEG、PNG 或 WebP 文件。") from exc


def require_upload_rights(confirmed):
    if not confirmed:
        raise ValueError("请先确认您有权上传并处理该图片。")


def show_processing_error(exc):
    if isinstance(exc, ValueError):
        st.error(str(exc))
        return
    LOGGER.exception("Hosted image processing failed", exc_info=exc)
    st.error("图片处理失败，请稍后重试；若问题持续出现，请通过项目 Issues 告知维护者。")


def render_usage_notice():
    st.warning(
        "内部图库配对五折评估：类群/变种 Top-1 为 80.42%，物种 Top-1 为 99.85%。"
        "这不是独立外部测试，开放集未知类群检出与误拒也尚未校准。模型很可能出错，"
        "鉴定、注意力和表型关系结果只能用于探索与交叉核对。"
    )
    with st.expander("评测口径、常见失败情形与使用边界（使用前请阅读）"):
        st.markdown(
            """
#### 内部评测口径

当前托管模型采用 300 轮训练、分类先验 `α=0.05`、分割质量感知降权和每类群 3 个子原型；
训练图库包含 126 个类群记录、3371 张人工筛选图片。

| 指标 | 内部评估结果 |
|---|---:|
| 类群/变种级 Top-1 | `80.42%` |
| 物种级 Top-1 | `99.85%` |
| 新旧模型共同覆盖的 116 个类群，按当前 3 子原型规则计算的类群级 Top-1 | `87.29%` |

这些结果来自训练图库内的配对五折中心评估。虽然被测图片会从对应折的中心计算中排除，
模型权重仍见过图库图片，因此不能排除训练集记忆、图片来源风格和类群不平衡的影响。
项目目前没有足够的独立外部测试集，也没有经校准的开放集未知类群检出率、误拒率或置信区间。

#### 容易出错的情况

- 背景分割失败，或图片较暗、模糊、植株不完整；
- 幼苗、胁迫状态或栽培状态变化明显；
- 冷门类群样本不足，或类群之间表型高度相近；
- 输入类群不在当前原型库中。开放集拒绝阈值不等于统计学置信度。

#### 使用与数据边界

- Demo 仅用于学术研究、教育和个人学习；请只上传您有权处理的图片。
- 上传图片不会加入训练集或项目数据库，但托管平台可能在计算期间产生短期临时缓存。
- 模型权重和数值原型仅在托管运行时加载，不在公开 GitHub 仓库或 Release 中提供下载。
- 鉴定名称、相似度、注意力热力图、表型网络、近邻表和聚类树，均不得作为植物鉴定、
  分类学修订、亲缘或杂交判断、保护与交易决策或科研结论的决定性依据。

重要结论应结合模式与原始描述、产地和生态信息、多个器官性状、可靠专家复核，以及适用时的
分子系统学或群体遗传学证据。完整说明见
[GitHub README](https://github.com/YujunCC/haworthia-omics#最终模型的内部评测与重要限制)。
"""
        )


def render_prediction(service):
    with st.form("prediction_form", clear_on_submit=False):
        left, right = st.columns([1.15, 1])
        with left:
            uploaded = st.file_uploader(
                "待分析图片", type=["jpg", "jpeg", "png", "webp"], key="predict_upload"
            )
        with right:
            mode = st.selectbox("背景分割方式", list(SEGMENTATION_MODES), key="predict_mode")
            sensitivity = st.slider("分割灵敏度", 0, 100, 70, 1, key="predict_sensitivity")
            threshold = st.slider(
                "未知类群拒绝阈值", -1.0, 1.0, 0.55, 0.01, key="predict_threshold"
            )
        confirmed = st.checkbox(
            "我确认有权上传并处理该图片，且了解图片会经过临时计算。",
            key="predict_consent",
        )
        submitted = st.form_submit_button("提取特征并比对", type="primary")

    if submitted:
        try:
            require_upload_rights(confirmed)
            image = read_upload(uploaded)
            prepare_segmenters()
            with st.spinner("正在分割图片并提取表型特征……"):
                st.session_state["prediction_result"] = service.predict(
                    image, mode, sensitivity, threshold
                )
        except Exception as exc:
            show_processing_error(exc)

    result = st.session_state.get("prediction_result")
    if result:
        segmented, status, table = result
        preview, findings = st.columns([1, 1.35])
        with preview:
            st.image(segmented, caption="模型实际接收的去背景图像", width="stretch")
        with findings:
            st.info(status)
            st.dataframe(table, hide_index=True, width="stretch")


def render_attention(service):
    with st.form("attention_form", clear_on_submit=False):
        left, right = st.columns([1.15, 1])
        with left:
            uploaded = st.file_uploader(
                "待解析图片", type=["jpg", "jpeg", "png", "webp"], key="attention_upload"
            )
        with right:
            mode = st.selectbox(
                "背景分割方式", list(SEGMENTATION_MODES), key="attention_mode"
            )
            sensitivity = st.slider(
                "分割灵敏度", 0, 100, 70, 1, key="attention_sensitivity"
            )
        confirmed = st.checkbox(
            "我确认有权上传并处理该图片，且了解图片会经过临时计算。",
            key="attention_consent",
        )
        submitted = st.form_submit_button("生成注意力热力图", type="primary")

    if submitted:
        try:
            require_upload_rights(confirmed)
            image = read_upload(uploaded)
            prepare_segmenters()
            with st.spinner("正在分割图片并解码四个注意力头……"):
                st.session_state["attention_result"] = service.attention(
                    image, mode, sensitivity
                )
        except Exception as exc:
            show_processing_error(exc)

    result = st.session_state.get("attention_result")
    if result:
        segmented, heatmap, status = result
        st.info(status)
        preview, decoded = st.columns([1, 2.2])
        with preview:
            st.image(segmented, caption="模型输入", width="stretch")
        with decoded:
            st.image(heatmap, caption="四注意力头与门控权重", width="stretch")


def render_relationships(service):
    labels = [service.taxa[tax_id]["label"] for tax_id in service.tax_ids]
    with st.form("relationship_form"):
        first, second, third, fourth = st.columns([1.7, 1, 1, 1])
        with first:
            label = st.selectbox("选择类群", labels)
        with second:
            scope = st.selectbox("关系范围", ["全部", "仅跨物种", "仅同物种"])
        with third:
            count = st.slider("显示数量", 3, 12, 8, 1)
        with fourth:
            k = st.slider("每节点近邻边", 1, 5, 2, 1)
        submitted = st.form_submit_button("生成关系视图", type="primary")

    if submitted:
        with st.spinner("正在计算数值化表型关系……"):
            st.session_state["relationship_result"] = (
                service.network_figure(k),
                service.nearest_table(label, scope, count, k),
            )

    result = st.session_state.get("relationship_result")
    if result:
        figure, table = result
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
        )
        st.caption("红线表示跨物种表型相似；网络不证明基因流、亲本关系或演化方向。")
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.caption("选择类群和显示范围后生成互动网络及近邻关系表。")


def render_tree(service):
    st.caption("树状分支来自当前模型原型的余弦距离，不代表遗传系统发育。")
    if st.button("生成树状聚类", type="primary"):
        with st.spinner("正在生成层次聚类树……"):
            st.session_state["tree_result"] = service.dendrogram_figure()
    figure = st.session_state.get("tree_result")
    if figure:
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
        )


st.title("Haworthia OMICS")
st.markdown(
    '<div class="main-notice">面向瓦苇属植物表型研究的只读模型演示。结果表示当前模型中的表型相似性，'
    "模型很可能出错，不构成权威鉴定、遗传亲缘、杂交或演化关系结论，也不得作为科研或实际决策的决定性依据。</div>",
    unsafe_allow_html=True,
)

try:
    SERVICE, ASSET_MANIFEST = load_service()
except Exception:
    st.error("模型服务初始化失败。请管理员检查私有模型仓库、Streamlit Secrets 和运行日志。")
    st.stop()

st.caption(
    f"已加载 {len(SERVICE.tax_ids)} 个类群的数值原型。模型权重和原型库不在公开 GitHub 仓库中分发。"
)
render_usage_notice()

prediction_tab, attention_tab, relationship_tab, tree_tab = st.tabs(
    ["开放集推理", "注意力热力图", "表型网络与近邻", "树状聚类"]
)
with prediction_tab:
    render_prediction(SERVICE)
with attention_tab:
    render_attention(SERVICE)
with relationship_tab:
    render_relationships(SERVICE)
with tree_tab:
    render_tree(SERVICE)

st.divider()
st.markdown(
    """
<div class="small-note">
本 Demo 仅用于学术研究、教育和个人学习。应用不会将上传图片用于训练，也不会写入项目数据库；
托管平台可能在处理期间产生短期临时缓存。未经授权的模型提取或再分发不受项目维护者认可。<br><br>
<a href="https://github.com/YujunCC/haworthia-omics">GitHub 源代码</a> ·
<a href="https://github.com/YujunCC/haworthia-omics/issues">问题与权利通知</a>
</div>
""",
    unsafe_allow_html=True,
)
