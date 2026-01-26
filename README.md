# ICL Vault Prediction Project (基于多模态深度学习的 ICL 术后拱高预测)

## 📌 项目简介
本项目旨在构建一个多模态深度学习模型，通过融合 **AS-OCT（光学相干断层扫描）**、**UBM（超声生物显微镜）** 及 **临床数值参数**，精准预测 ICL 手术后的拱高（Vault），以辅助医生进行晶体尺寸选型。

---

## 📂 项目结构全景图 (Project Structure)
*(更新时间: 2026-01-26)*

    ICL_Vault_Project/
    │
    ├── data/                           # 【燃料库】存放数据
    │   └── public_datasets/
    │       └── keratitis_oct/          # [替代数据] Keratitis AS-OCT 公开数据集
    │           ├── images/             # 原始图片 (.bmp)
    │           └── masks/              # 标注文件 (.json)
    │
    ├── models/                         # 【引擎室】存放神经网络架构
    │   ├── unet.py                     # [✅已验证] 轻量级 U-Net 分割模型 (用于 UBM 自动 ROI 裁剪)
    │   └── multimodal_net.py           # [✅已验证] 多模态回归主模型 (ResNet提取 + Cross-Attention融合)
    │
    ├── utils/                          # 【加工厂】存放数据处理工具
    │   └── dataset.py                  # [✅已验证] 通用分割数据集加载器 (支持 LabelMe JSON 解析)
    │
    ├── checkpoints/                    # 【军火库】存放训练好的模型权重
    │   └── unet_keratitis.pth          # [NEW] 基于公开数据训练好的 U-Net 权重 (Loss: 0.0629)
    │
    ├── train_unet.py                   # 【训练场】U-Net 训练脚本 (已跑通)
    └── main.py                         # 【测试台】主模型冒烟测试脚本 (已跑通)

---

## 🧩 核心模块与协同逻辑

### 1. 预处理与辅助模块 (Auxiliary Pipeline)
*   **当前状态：** ✅ **已跑通**
*   **`dataset.py`**: 成功实现了从 LabelMe JSON 到二值 Mask 的自动转换与标准化。
*   **`unet.py`**: 作为项目的“前哨”，负责从大图中自动裁剪出关键区域（ROI）。目前已在 Keratitis 数据集上完成验证训练，**Loss 降至 0.0629**，具备了极强的特征提取能力。

### 2. 主预测模型 (Main Prediction Pipeline)
*   **当前状态：** ✅ **架构已验证（冒烟测试通过）**
*   **`multimodal_net.py`**: 实现了师兄论文中的核心架构：
    *   **Feature Extraction**: 双流 ResNet18 提取 OCT 与 UBM 特征。
    *   **Fusion Strategy**: 引入 **Cross-Attention (交叉注意力机制)**，利用 OCT 高维特征动态检索 UBM 的形态特征。
    *   **Regression Head**: 最终输出连续的 Vault 预测值。

---

## 📅 项目进度追踪 (Milestones)

- [x] **环境搭建**: PyTorch + CUDA 环境配置完成。
- [x] **主模型架构**: 复现 ResNet + Cross-Attention 多模态融合网络。
- [x] **数据工程**: 完成复杂 LabelMe JSON 数据的解析与清洗管道。
- [x] **辅助模型验证**: **完成 U-Net 在公开数据集上的训练 (Epoch 10/10, Loss 0.06)。**
- [x] **可视化**: 编写推断脚本 (`inference.py`)，可视化 U-Net 的分割效果。
- [x] **全链路模拟**: **完成多模态主模型的模拟训练 (Simulated Training)，架构验证通过。**
- [ ] **实战迁移**: 接入爱尔眼科真实 UBM 数据，微调 U-Net 实现睫状沟自动裁剪。
- [ ] **真实训练**: 替换模拟数据，启动主模型在真实数据上的训练。

---

## 🚀 快速开始 (Quick Start)

### 1. 测试主模型架构

    python main.py

### 2. 训练 U-Net 分割模型

    python train_unet.py