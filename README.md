# ICL Vault Prediction Project

## 项目简介

本项目面向 ICL 手术后 Vault（拱高）预测任务，目标是构建一个多模态深度学习研究与工程框架，用于后续融合 OCT、UBM、角膜地形图及临床参数等多源信息，支持模型训练、实验管理与结果复现。

当前仓库处于 V2 工程化推进阶段。项目已从早期原型结构逐步迁移到新的 `src/ + scripts/ + data/ + artifacts/` 组织方式，并开始将公开数据实验纳入统一的数据接口、训练入口与评估流程中。

## 项目目标

1. 构建面向 ICL 术后 Vault 预测的多模态深度学习框架。
2. 建立统一的数据 schema、训练入口和实验组织方式。
3. 为后续真实临床数据接入、训练与实验管理提供稳定的工程基础。
4. 在工程结构清晰化的前提下，逐步推进主任务与公开数据辅助任务的训练实现。

## 当前项目状态

### 1. V2 工程骨架

当前已建立 V2 主结构，包括：

- `configs/`
- `data/`
- `docs/`
- `scripts/`
- `src/`
- `artifacts/`
- `tests/`
- `legacy/`

V2 的核心目标是将数据接口、训练入口、评估逻辑与实验产物管理逐步从原型式仓库组织方式迁移到更规范的工程结构中。

### 2. 主任务 Vault 线

主任务目前仍处于 scaffold 阶段，但基础工程链路已经打通，已完成的组件包括：

- `src/icl_vault/data/schema.py`
- `src/icl_vault/data/datasets/vault_dataset.py`
- `src/icl_vault/data/collate.py`
- `scripts/train_vault.py`
- `src/icl_vault/engine/trainer.py`

当前状态说明：

- 已定义主任务样本字段结构
- 已实现 manifest 驱动的 `VaultDataset`
- 已实现 metadata-based 的 batch collate
- 已有 V2 训练入口与 trainer scaffold
- 目前尚未接入真实多模态主模型训练

### 3. MCOA 预训练线

MCOA 公开数据预训练线已经成为当前仓库中第一条真实可训练路径，已完成的组件包括：

- manifest 驱动数据组织
- `src/icl_vault/data/datasets/mcoa_dataset.py`
- `scripts/pretrain_backbone.py`
- `src/icl_vault/engine/evaluator.py`
- `src/icl_vault/engine/checkpoint.py`

当前能力包括：

- 从 manifest 读取真实 MCOA 图像样本
- 显式标签映射，不再依赖目录名推断
- 基于 `torchvision.models.resnet18` 的最小分类训练流程
- 输出基础指标：`train_loss`、`val_loss`、`val_accuracy`
- 在 `artifacts/checkpoints/` 下保存最小 checkpoint

## 小样本 MCOA 初步实验结论

基于小规模 manifest，当前已完成三组对比实验：

1. from scratch + no augmentation
2. ImageNet pretrained + no augmentation
3. ImageNet pretrained + basic augmentation

当前阶段的结论是：

- ImageNet 预训练权重是当前小样本场景下的主要收益来源。
- 在 ImageNet 预训练基础上加入 basic augmentation 后，训练与验证过程进一步稳定。

更详细的实验记录见：

- `docs/experiments/mcoa_pretrain_ablation_log.md`

## 当前目录结构

```text
ICL_Vault_Project/
├─ configs/
├─ data/
│  ├─ manifests/
│  ├─ processed/
│  ├─ raw/
│  ├─ interim/
│  └─ splits/
├─ docs/
│  ├─ experiments/
│  └─ V2_REFACTOR_PLAN.md
├─ scripts/
│  ├─ pretrain_backbone.py
│  ├─ train_vault.py
│  ├─ train_segmentation.py
│  ├─ infer_vault.py
│  └─ infer_segmentation.py
├─ src/
│  └─ icl_vault/
│     ├─ data/
│     ├─ engine/
│     ├─ models/
│     └─ utils/
├─ artifacts/
│  ├─ checkpoints/
│  ├─ figures/
│  ├─ logs/
│  └─ predictions/
├─ legacy/
├─ tests/
├─ requirements.txt
└─ README.md
```

## 主要模块说明

### `src/icl_vault/data/`

负责 V2 数据接口层，包括：

- 数据 schema
- manifest 驱动的数据集定义
- batch collate
- 后续可扩展的数据变换与数据组织逻辑

### `src/icl_vault/engine/`

负责 V2 训练与评估引擎层，包括：

- `trainer.py`：训练流程 scaffold
- `evaluator.py`：当前最小分类评估器
- `checkpoint.py`：最小 checkpoint 保存工具
- `logger.py`：日志工具占位

### `scripts/`

负责各项实验的顶层运行入口。当前主要包括：

- `pretrain_backbone.py`：MCOA 预训练实验入口
- `train_vault.py`：Vault 主任务训练入口 scaffold
- 其余脚本仍处于骨架或占位阶段

### `docs/`

负责记录：

- V2 重构方案
- 实验记录
- 后续可补充的方法设计与实验总结

### `legacy/`

用于归档旧版 V1 / 原型阶段脚本、文档与产物。当前主开发应以 V2 结构为准，不再以旧版根目录脚本作为主要维护入口。

## 当前可运行内容

当前仓库中最明确、最完整的可运行路径是：MCOA 小规模公开数据预训练实验。

示例命令：

```bash
python scripts/pretrain_backbone.py --manifest_path data/manifests/mcoa_manifest_small.csv --epochs 5
```

使用 ImageNet 预训练权重的对比命令：

```bash
python scripts/pretrain_backbone.py --manifest_path data/manifests/mcoa_manifest_small.csv --epochs 5 --use_imagenet_pretrain
```

如果希望复现实验中的保守增强版本，可直接使用当前脚本默认配置。当前训练脚本已在 train transform 中启用基础且保守的数据增强。

## Checkpoint 说明

当前 MCOA 训练线会在 `artifacts/checkpoints/` 下保存：

- `mcoa_latest.pth`
- `mcoa_best.pth`

其中：

- `mcoa_latest.pth`：每个 epoch 结束后更新
- `mcoa_best.pth`：按当前 `val_accuracy` 最优结果更新

## 当前限制

1. 主任务真实临床数据尚未完全到位。
2. Vault 主任务仍以 scaffold 为主，尚未完成真实多模态训练闭环。
3. 当前公开数据实验规模仍偏小。
4. 当前结论主要用于工程验证与初步观察，尚不能替代更大规模实验结论。
5. 训练引擎、评估器与 checkpoint 机制仍处于最小实现阶段。

## 下一步计划

1. 扩展更大规模的 MCOA manifest。
2. 继续比较 scratch / ImageNet pretrained / augmentation 三组设置。
3. 将主任务 Vault 线从 scaffold 推进到真实训练路径。
4. 逐步接入真实临床数据。
5. 继续完善 trainer / evaluator / checkpoint 等 V2 engine 组件。

## Legacy 说明

旧版 V1 / 原型脚本、旧文档与旧产物已逐步归档到 `legacy/`：

- `legacy/scripts_v1/`
- `legacy/models_v1/`
- `legacy/utils_v1/`
- `legacy/docs_v1/`
- `legacy/artifacts_v1/`
- `legacy/notes_v1/`

当前主开发与后续工程推进应以 V2 结构为准，即优先使用：

- `scripts/`
- `src/icl_vault/`
- `data/manifests/`
- `artifacts/`

而不是继续在旧版根目录原型脚本上叠加新逻辑。
