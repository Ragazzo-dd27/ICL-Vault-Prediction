# ICL Vault Prediction Project

## 项目简介

本项目的最终目标是：基于术前多模态眼科数据，预测 ICL 手术后的 vault 连续值。

当前仓库还不是最终主任务训练成品，而是处于 V2 工程推进中的一个过渡阶段：公开数据过渡阶段的第一轮工程建设已基本完成，V2 vault 主线的第一版 data contract 已经实体化，但真实医院主任务数据仍未到位，因此最终 vault 主任务尚未真正接通。

## 当前项目状态

当前阶段可以概括为三点：

- 公开数据过渡阶段的第一轮工程建设已基本完成。
- V2 vault 主线已经从 metadata-only scaffold 推进到第一版可用的数据契约。
- 真实医院数据仍未接入，因此真实 vault 回归训练尚未开始。

换句话说，仓库里已经有几条真实跑通的工程链路，可以验证样本组织、日志、checkpoint、缺失模态表达和最小训练闭环；但这些结果主要用于工程验证，不应被当作最终临床结论。

## 当前已完成的主要工程线

### 1. MCOA eye-level 单模态线

这条线已经完成：

- 单眼多图组织
- eye-level 训练闭环
- 日志与 checkpoint 落盘
- 对应运行说明文档

关键入口：

- `scripts/build_mcoa_eye_manifest.py`
- `scripts/pretrain_mcoa_eye.py`
- `docs/MCOA_EYELEVEL_RUN.md`

### 2. MCOA OCT+ASP 双模态代理线

这条线已经完成：

- OCT + ASP 双模态最小闭环
- 单模态 / 双模态对照 smoke test
- 缺失模态 smoke test
- 日志、checkpoint 与结果摘要

关键入口：

- `scripts/build_mcoa_multimodal_manifest.py`
- `scripts/pretrain_mcoa_multimodal.py`
- `docs/experiments/mcoa_modality_ablation_smoketest.md`

### 3. keratitis_oct 替代结构预训练线

这条线不是最终主任务，而是在 AIDK 阻塞时提供一个现实可用的结构辅助预训练入口。当前已经完成：

- 基于 `keratitis_oct` 的替代结构线
- `Cornea` 二值分割最小闭环
- 日志、checkpoint、运行说明与结果摘要

关键入口：

- `scripts/build_keratitis_manifest.py`
- `scripts/train_keratitis_structure_pretrain.py`
- `docs/KERATITIS_STRUCTURE_RUN.md`
- `docs/experiments/keratitis_structure_smoketest.md`

### 4. V2 vault 主线第一版 data contract

这条线当前已经推进到：

- `VaultDataset` 已从 metadata-only 推进到第一版真实读取能力
- `collate_vault_batch` 已形成第一版 batch contract
- 缺失模态表达已正式化
- 已有 `scripts/smoke_test_vault_data_contract.py`

当前仍未完成的部分：

- 真实医院主任务数据尚未接入
- 因此真实 vault 主任务训练尚未开始
- `scripts/train_vault.py` 仍是占位式训练入口，不代表主任务已跑通

对应说明：

- `docs/experiments/vault_data_contract_smoketest.md`

## 当前哪些链路已经真实跑通

当前可以明确说“已经真实跑通”的链路有：

- MCOA eye-level 单模态训练链路
- MCOA OCT+ASP 双模态代理链路
- MCOA 缺失模态 smoke test
- `keratitis_oct` 替代结构预训练链路
- V2 vault data contract smoke test

## 当前仍然只是 scaffold / placeholder / waiting for real data 的内容

当前仍未真正接通的内容有：

- 真实医院 vault 主任务数据接入
- 真实 vault 回归训练
- 最终多模态主模型训练闭环
- `scripts/train_vault.py` 对应的真实主任务训练逻辑
- `scripts/infer_vault.py` 对应的真实推理逻辑

## 当前仓库结构

```text
ICL_Vault_Project/
├─ artifacts/
│  ├─ checkpoints/
│  ├─ figures/
│  ├─ logs/
│  └─ predictions/
├─ configs/
├─ data/
│  ├─ manifests/
│  ├─ public_datasets/
│  └─ raw/
├─ docs/
│  ├─ MCOA_EYELEVEL_RUN.md
│  ├─ KERATITIS_STRUCTURE_RUN.md
│  ├─ V2_REFACTOR_PLAN.md
│  └─ experiments/
├─ legacy/
├─ scripts/
│  ├─ build_mcoa_eye_manifest.py
│  ├─ build_mcoa_multimodal_manifest.py
│  ├─ build_keratitis_manifest.py
│  ├─ pretrain_mcoa_eye.py
│  ├─ pretrain_mcoa_multimodal.py
│  ├─ train_keratitis_structure_pretrain.py
│  ├─ smoke_test_vault_data_contract.py
│  └─ train_vault.py
├─ src/
│  └─ icl_vault/
│     ├─ data/
│     └─ engine/
├─ tests/
├─ .gitignore
├─ README.md
└─ requirements.txt