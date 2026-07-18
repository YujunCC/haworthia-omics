# Haworthia OMICS

Haworthia OMICS 是面向瓦苇属植物表型研究的本地开源工具，提供图像导入与背景分割、
质量感知训练、开放集表型匹配、注意力诊断、表型关系网络、双类群证据比较和层次聚类。

本项目是研究工具，不是权威植物学鉴定服务。模型相似度和表型网络不能单独证明遗传亲缘、
杂交、祖先后代关系或演化方向。

## 主要功能

Haworthia OMICS 将图片管理、背景分割、模型训练、开放集匹配和表型证据分析整合在一个本地界面中。

### 数据库总览

- 查看类群数、物种数、图片数和各类群图片分布；
- 检查模型、训练断点、数值原型和背景分割权重状态；
- 查看模型原型对现有类群的覆盖情况；
- 导入或导出 Haworthia OMICS 模型包；
- 模型包不包含图库图片、图片路径或训练断点。

### 数据导入与清洗

- 按“物种 + 变种/园艺名”批量导入图片；
- 提供自动宽容、双模型宽松和 IS-Net 严格三种背景分割方式；
- 可调整分割灵敏度，并在入库前预览结果；
- 按类群浏览原图与分割图，逐张重新分割或删除；
- 自动标记前景过少、半透明异常等疑似分割失败图片；
- 支持仅处理疑似失败图片或重新分割整个图库；
- 批量重分割可备份并恢复原有分割结果。

### 流形度量训练

- 从本地图库随机初始化训练，也可继续现有模型或训练断点；
- 使用 P×K 类群平衡采样和双视图数据增强；
- 通过分类先验权重 α 协调物种级区分与类群内表型结构；
- 可启用分割质量感知，使疑似失败图片保留训练价值但降低影响；
- 支持多子原型、注意力头互斥和注意力覆盖约束；
- 实时显示训练进度、度量损失、质量权重并支持安全中断。

默认参数是项目经验配置，不保证适用于所有数据集。正式训练前建议先创建快照。

### 表型关系发掘

- 生成可缩放、拖动、搜索和显示名称的交互式相似度网络；
- 查看每个类群最相似的同物种或跨物种类群；
- 查看互为近邻、跨物种桥接和余弦相似度；
- 生成单类群证据型气质报告和多原型外观板；
- 比较两个类群的中心相似度、类内紧凑度和图库内分离率；
- 比较基础图像统计、子原型对应关系、代表图和边界样本；
- 生成基于当前模型原型的表型层次聚类树。

这些结果描述当前图库和模型中的表型关系，不等同于遗传亲缘、杂交证据或系统发育结论。

### 开放集推理

- 上传图库外图片并选择与入库阶段相同的分割方式；
- 显示模型实际接收的去背景图像；
- 返回最相似类群、余弦相似度和对应子原型；
- 使用可调拒绝阈值，将低相似度输入标记为未知类群。

本功能用于表型检索和研究辅助，不应作为权威植物鉴定结果。

### 气质逆向解码

- 对单张图片生成多注意力头融合热力图；
- 观察模型在当前预测中关注的区域；
- 使用独立的分割模式和灵敏度处理困难图片；
- 辅助发现模型是否依赖背景、轮廓或局部叶片结构。

注意力热力图是模型行为诊断工具，不是植物学性状标注。

### 系统维护与诊断

- 创建数据库、模型及可选图库图片的时间戳快照；
- 使用当前模型重新建立各类群的多子原型；
- 抽样检查注意力覆盖率、头间重叠和门控权重；
- 在训练或大规模重分割前保存可恢复状态。

## 推荐使用流程

1. 在“数据库总览”确认运行环境和分割权重可用。
2. 在“数据导入与清洗”建立类群并检查分割效果。
3. 创建一次不含或包含图片的本地快照。
4. 在“流形度量训练”完成训练。
5. 使用“开放集推理”检查图库外图片的匹配表现。
6. 使用“表型关系发掘”和“气质逆向解码”检查模型证据。
7. 导出模型包和 SHA-256 校验值，用于个人备份或授权迁移。

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

1. 从 [GitHub Releases](https://github.com/YujunCC/haworthia-omics/releases) 下载
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

Copyright 2026 Yujun（GitHub: [YujunCC](https://github.com/YujunCC)）。

软件和项目文档采用 [Apache License 2.0](LICENSE)，署名信息见 [NOTICE](NOTICE)。该许可证
允许使用、修改、商用和再分发，但必须遵守许可证和 NOTICE 要求。它不覆盖用户图片、数据库、
模型权重、导出模型包或第三方分割权重。

本工具旨在服务植物爱好者与基础科研。若您基于本项目开发了商业产品，或发表了重要成果，欢迎通过 GitHub Issues、作者的 GitHub 主页或邮箱联系。
得知本项目得到实际应用，我会非常高兴，谢谢喵qwq

问题反馈：[GitHub Issues](https://github.com/YujunCC/haworthia-omics/issues)
