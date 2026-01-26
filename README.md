# ICL Vault Prediction Project (基于多模态深度学习的 ICL 术后拱高预测)

## 📌 项目简介
本项目旨在构建一个多模态深度学习模型，通过融合 **AS-OCT（光学相干断层扫描）**、**UBM（超声生物显微镜）** 及 **临床数值参数**，精准预测 ICL 手术后的拱高（Vault），以辅助医生进行晶体尺寸选型。

> **当前状态：** 已完成工程架构搭建、辅助模型训练及主模型全链路模拟验证。
---

## 📂 项目结构全景图 (Project Structure)
*(更新时间: 2026-01-26)*

    ICL_Vault_Project/
    │
    ├── data/                           # 【燃料库】存放数据
    │   └── public_datasets/            # 公开数据集 (用于预训练辅助模型)
    │       └── keratitis_oct/          # Keratitis AS-OCT 数据集 (bmp images + json masks)
    │
    ├── models/                         # 【引擎室】存放神经网络架构
    │   ├── unet.py                     # [✅已验证] 轻量级 U-Net 分割模型 (用于 UBM 自动 ROI 裁剪)
    │   └── multimodal_net.py           # [✅已验证] 多模态回归主模型 (ResNet提取 + Cross-Attention融合)
    │
    ├── utils/                          # 【加工厂】存放数据处理工具
    │   ├── dataset.py                  # [✅已验证] 分割通用加载器 (支持 LabelMe JSON 解析)
    │   └── multimodal_dataset.py       # [✅已验证] 多模态模拟数据生成器
    │
    ├── checkpoints/                    # 【军火库】存放训练好的模型权重
    │   ├── unet_keratitis.pth          # U-Net 权重 (Loss: 0.0629)
    │   └── multimodal_simulated.pth    # 主模型模拟训练权重
    │
    ├── logs/                           # 【档案馆】实验日志
    │   ├── 20260126_01_dataset_check.txt
    │   ├── 20260126_02_unet_training.txt
    │   ├── 20260126_03_multimodal_data_check.txt
    │   ├── 20260126_04_multimodal_sim_train.txt
    │   └── 20260126_05_demo_inference.txt
    │
    ├── train_unet.py                   # [脚本] U-Net 训练
    ├── train_multimodal.py             # [脚本] 主模型全链路模拟训练
    ├── inference.py                    # [脚本] U-Net 分割效果可视化
    ├── demo.py                         # [脚本] 单样本临床预测演示 (System Demo)
    ├── main.py                         # [脚本] 冒烟测试
    └── GIT_CHEAT_SHEET.md              # [文档] Git 操作备忘录

---

## 🧩 核心模块与验证情况

### 1. 辅助模块：UBM 自动裁剪 (Auxiliary Pipeline)
*   **模型：** Lightweight U-Net
*   **验证数据：** Keratitis AS-OCT 公开数据集 (1000+ images)
*   **结果：** 训练 10 Epochs，Loss 收敛至 **0.0629**。
*   **可视化：** `inference.py` 生成的对比图显示预测掩码与 Ground Truth 高度重合。

### 2. 主预测模型：拱高回归 (Main Prediction Pipeline)
*   **架构：** 双流 ResNet18 (OCT/UBM) + MLP (Clinical) -> **Cross-Attention Fusion** -> Regression Head
*   **验证方式：** 使用 `SimulatedMultimodalDataset` 生成符合真实维度的随机数据流。
*   **结果：** 全链路训练无报错，Loss 正常下降，显存占用稳定，验证了多模态融合逻辑的正确性。

---

## 📅 项目进度追踪 (Milestones)

- [x] **环境搭建**: PyTorch + CUDA 环境配置完成。
- [x] **架构复现**: 完成 ResNet + Cross-Attention 多模态融合网络搭建。
- [x] **数据工程**: 攻克 LabelMe JSON 数据解析与清洗难点。
- [x] **辅助模型落地**: **U-Net 训练完成 (Loss 0.06)，具备 ROI 提取能力。**
- [x] **全链路模拟**: **主模型模拟训练通过，工程链路打通。**
- [x] **原型系统**: **完成 `demo.py`，可演示完整的患者数据分析流程。**
- [ ] **实战迁移**: 接入爱尔眼科真实 UBM 数据，微调 U-Net。
- [ ] **真实训练**: 替换模拟数据，启动主模型在真实数据上的训练。

---

## 🚀 快速指令 (Quick Start)

### 1. 演示原型系统 (Demo)
模拟输入一位患者的临床数据，输出预测报告：

    python demo.py

### 2. 查看 U-Net 分割效果 (Visualization)
随机抽取测试集图片，展示 原图 vs 真值 vs 预测：

    python inference.py

### 3. 运行训练脚本

    python train_unet.py       # 训练 U-Net
    python train_multimodal.py # 训练主模型 (模拟数据)

---

## 📝 实验日志
详细的终端输出记录请查看 `logs/` 文件夹，包含从数据清洗到最终演示的所有关键步骤验证。