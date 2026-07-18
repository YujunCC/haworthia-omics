# Haworthia OMICS

Haworthia OMICS 是面向瓦苇属植物表型研究的本地开源工具，提供图像导入与背景分割、
质量感知训练、开放集表型匹配、注意力诊断、表型关系网络、双类群证据比较和层次聚类。

本项目是研究工具，不是权威植物学鉴定服务。模型相似度和表型网络不能单独证明遗传亲缘、
杂交、祖先后代关系或演化方向。

## 开源边界

公开仓库和 Release 只提供软件与文档，不提供任何预训练模型、训练数据、数值原型或数据库。
用户可以导入自己的图片从头训练，也可以使用通用模型包接口导入、导出自己有权使用的模型。
软件作者不提供、审核或认可用户模型，也不对模型及其训练数据权利负责。

以下内容不会进入 Git 或公开应用包：

- 原始图片、分割图和 `local_images/`；
- SQLite 数据库；
- `backups/` 中的快照；
- `model_base.pth`、checkpoint 和其他模型权重；
- 模型卡、模型清单和数值原型；
- IS-Net、U2Net 等第三方 ONNX 权重。

数据边界见 [DATA_POLICY.md](DATA_POLICY.md)，模型包格式见
[MODEL_PACKAGE_FORMAT.md](MODEL_PACKAGE_FORMAT.md)。

## 运行环境

- Python 3.12
- Windows 10/11、Linux 或 macOS
- NVIDIA GPU 可显著加速训练；CPU 可运行界面和推理，但训练较慢
- 约 1 GB Python 依赖空间
- 两个可选背景分割权重合计约 355 MB

## Windows 安装

1. 从 [GitHub Releases](https://github.com/Yujun8Q/haworthia-omics/releases) 下载
   `haworthia-omics-*-core.zip` 并解压。
2. 双击 `install_haworthia.bat` 创建 `.venv` 并安装依赖。
3. 安装器会询问是否从 rembg 上游下载背景分割权重。
4. 双击 `start_haworthia.bat` 启动应用。

也可以手动安装：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_segmentation_models.py
python scripts/check_setup.py
python run_haworthia.py
```

Linux/macOS 激活环境使用：

```bash
source .venv/bin/activate
```

需要特定 CUDA 版本时，请先按 PyTorch 官方安装器安装匹配的 `torch` 和 `torchvision`，
再安装其余依赖。

## 背景分割权重

`onnxruntime` 是 Python 依赖；`isnet-general-use.onnx` 和 `u2net.onnx` 是单独的第三方
模型资产，不随本项目分发，也不适用本项目的 Apache-2.0 许可证。下载脚本从 rembg
GitHub Release 获取固定文件并校验 SHA-256：

```powershell
python scripts/download_segmentation_models.py
```

默认安装到 `~/.u2net`。用户应自行核对上游权重许可和适用范围。缺少这些文件时，程序会
阻止依赖背景分割的图片入库，不会静默使用未分割图片。

## 启动与安全

Windows 双击 `start_haworthia.bat`，或运行：

```powershell
python run_haworthia.py
```

界面默认打开 `http://127.0.0.1:8501`。后端包含训练、删除、重分割、模型导入和数据恢复
等写操作，没有公网身份认证。不要绑定到 `0.0.0.0`，也不要直接部署成公共互联网服务。

## 首次使用

1. 打开“数据库总览”，确认依赖和背景分割状态。
2. 在“数据导入与清洗”中导入自己拥有或获准处理的图片。
3. 检查分割预览和疑似失败图片。
4. 在“流形度量训练”中从头训练模型。
5. 训练完成后可运行推理、注意力、表型关系和证据比较。
6. 使用“导出当前模型包”备份或迁移自己的模型与数值原型。

## 模型包导入与导出

“数据库总览”提供两个对等接口：

- `导入模型包`：选择 ZIP、填写整包 SHA-256、确认模型权利责任后导入；
- `导出当前模型包`：生成模型 ZIP，并分别下载 ZIP 与 SHA-256 文件。

导出包包含当前 `state_dict`、类群标签和数值原型，不包含原图、分割图、图片路径或训练
checkpoint。导入会自动备份现有数据库和模型，按 `(species, variant)` 合并类群，保留本地
图片和额外类群。导入者和导出者必须自行确认模型、训练图片和再分发权利。

## 配置

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `HAWORTHIA_API_URL` | `http://127.0.0.1:8000/api` | 前端访问的 API |
| `HAWORTHIA_DB_PATH` | `haworthia_omics.db` | SQLite 数据库路径 |
| `HAWORTHIA_IMAGE_DIR` | `local_images` | 原图与分割图目录 |
| `HAWORTHIA_MODEL_PATH` | `model_base.pth` | 用户本地模型 |
| `HAWORTHIA_CHECKPOINT_PATH` | `checkpoint_base.pth` | 用户训练断点 |
| `U2NET_HOME` | `~/.u2net` | 第三方分割权重目录 |

程序不会自动读取 `.env`；示例见 [.env.example](.env.example)。

## 开发与发布检查

```powershell
python -m py_compile app.py main.py database.py dataset.py engine.py model_package.py models.py segmentation.py
python scripts/check_setup.py
python scripts/audit_release.py
python scripts/build_release_bundle.py --version 0.1.0
```

公开构建器只生成不含模型和数据的 `core` ZIP。发布流程见
[RELEASE_GUIDE.md](RELEASE_GUIDE.md)。

## 许可证与联系

Copyright 2026 雨筠（GitHub: [Yujun8Q](https://github.com/Yujun8Q)）。

软件和项目文档采用 [Apache License 2.0](LICENSE)，署名信息见 [NOTICE](NOTICE)。该许可证
允许使用、修改、商用和再分发，但必须遵守许可证和 NOTICE 要求。它不覆盖用户图片、数据库、
模型权重、导出模型包或第三方分割权重。

问题反馈：[GitHub Issues](https://github.com/Yujun8Q/haworthia-omics/issues)
