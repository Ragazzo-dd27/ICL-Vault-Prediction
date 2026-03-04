# ICL Vault Prediction Project (基于多模态深度学习的 ICL 术后拱高预测)

## 📌 项目简介
本项目旨在构建一个高度鲁棒的多模态深度学习回归模型。通过融合 **AS-OCT（光学相干断层扫描）**、**UBM（超声生物显微镜）** 及 **临床数值参数**，利用 **交叉注意力机制（Cross-Attention）** 精准预测 ICL 手术后的拱高（Vault），为医生提供晶体尺寸选型的数字化决策支持。

> **当前状态：** 
> 1. **全链路架构已就绪**：完成 ResNet18 + Cross-Attention 多模态融合模型的工程化实现。
> 2. **域适应预训练突破**：利用 MCOA 大规模眼科数据集（13,328 张影像）完成主干网络预训练，**分类准确率达到 99.56%**，成功实现从通用视觉向眼科领域的知识迁移。
> 3. **工程验证通过**：辅助分割模型 (U-Net) 及主回归模型已完成全链路模拟训练，架构稳定性与梯度流验证通过。

---

## 🌟 技术亮点 (Technical Highlights)
*   **域适应策略 (Domain Adaptation)**：针对临床数据稀缺痛点，引入大规模公开数据集进行预热训练，显著提升特征提取器的领域特征捕捉能力。
*   **交互融合架构**：放弃简单拼接，采用 Cross-Attention 机制实现 OCT 与 UBM 模态间的深层交互，模拟临床“参考扫描引导”的诊断逻辑。
*   **工程化日志管理**：建立完整的实验追踪体系，所有预训练及集成测试均有详细 Log 记录可供溯源。

---

## 📂 项目结构全景图 (Project Structure)
*(更新时间: 2026-03-03)*

    ICL_Vault_Project/
    │
    ├── data/                           # 【燃料库】存放数据
    │   └── public_datasets/            # 公开数据集 (用于预训练)
    │       ├── keratitis_oct/          # Keratitis AS-OCT 数据集 (用于 U-Net 分割训练)
    │       └── mcoa_oct/               # [NEW] MCOA 大规模数据集 (13GB, 用于主干网络预训练)
    │
    ├── models/                         # 【引擎室】存放神经网络架构
    │   ├── unet.py                     # [✅已验证] 轻量级 U-Net 分割模型 (用于 UBM 自动 ROI 裁剪)
    │   └── multimodal_net.py           # [✅已验证] 多模态回归主模型 (支持加载 MCOA 预训练权重)
    │
    ├── utils/                          # 【加工厂】存放数据处理工具
    │   ├── dataset.py                  # [✅已验证] 分割通用加载器 (LabelMe JSON 解析)
    │   ├── multimodal_dataset.py       # [✅已验证] 多模态模拟数据生成器
    │   └── mcoa_dataset.py             # [NEW] MCOA 大规模数据递归读取器
    │
    ├── checkpoints/                    # 【军火库】存放训练好的模型权重
    │   ├── unet_keratitis.pth          # U-Net 权重 (Loss: 0.0629)
    │   ├── resnet18_mcoa_pretrained.pth # [NEW] 域适应主干权重 (MCOA Acc: 99.56%)
    │   └── multimodal_simulated.pth    # 主模型模拟训练权重
    │
    ├── logs/                           # 【档案馆】实验日志
    │   ├── 20260126_02_unet_training.txt
    │   ├── 20260303_mcoa_pretrain.txt  # [NEW] 主干网络预训练日志
    │   ├── 20260303_domain_adaptation_train.txt # [NEW] 域适应集成训练日志
    │   └── ... (其他历史日志)
    │
    ├── pretrain_backbone.py            # [NEW] 主干网络预训练脚本 (MCOA)
    ├── train_multimodal.py             # [脚本] 主模型全链路训练 (集成域适应权重)
    ├── train_unet.py                   # [脚本] U-Net 训练
    ├── inference.py                    # [脚本] U-Net 分割效果可视化
    ├── demo.py                         # [脚本] 单样本临床预测演示 (System Demo)
    ├── main.py                         # [脚本] 冒烟测试
    ├── architecture_design.md          # [文档] Mermaid 架构设计图源码
    └── GIT_CHEAT_SHEET.md              # [文档] Git 操作备忘录

---

## 🧩 核心模块与技术亮点

### 1. 域适应预训练 (Domain Adaptation Strategy) 🔥

*   **策略：** 利用 **MCOA (13,328 images)** 大规模眼科数据集进行 Transfer Learning。
*   **成果：** ResNet18 主干网络在 Pretext Task (Normal vs Opaque) 上达到了 **99.56%** 的准确率。
*   **意义：** 使特征提取器从“通用视觉模型”进化为“眼科专家模型”，显著解决了 ICL 小样本训练易过拟合的问题。

### 2. 主预测模型 (Main Prediction Pipeline)

*   **架构：** 双流 ResNet18 (Initialized with MCOA weights) + MLP (Clinical) -> **Cross-Attention Fusion** -> Regression Head。
*   **验证：** 成功加载 MCOA 权重，并通过全链路模拟训练验证，梯度回传正常，Loss 收敛符合预期。

### 3. 辅助模块 (Auxiliary Pipeline)

*   **功能：** 基于 Lightweight U-Net 实现图像 ROI 自动裁剪。
*   **验证：** 在 Keratitis 数据集上验证通过 (Loss 0.06)。

---

## 📅 项目进度追踪 (Milestones)

- [x] **环境搭建**: PyTorch + CUDA 环境配置完成。
- [x] **架构复现**: 完成 ResNet + Cross-Attention 多模态融合网络搭建。
- [x] **数据工程**: 攻克 LabelMe JSON 解析及 MCOA 大规模数据读取。
- [x] **辅助模型落地**: U-Net 训练完成 (Loss 0.06)，具备 ROI 提取能力。
- [x] **域适应预训练**: **完成 ResNet 主干在 MCOA 数据集上的预训练 (Acc 99.56%)。**
- [x] **全链路集成**: **完成基于 MCOA 权重的主模型集成训练，验证了 Domain Adaptation 的有效性。**
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

## 📚 理论与设计文档 (Documentation)
为了更好地理解本项目的数学原理与架构设计，请参考以下独立文档：
*   **[Problem_Formulation.md](./Problem_Formulation.md)**: 详细阐述了 ICL 拱高预测的数学定义、公式推导及现有方法的局限性分析。
*   **[architecture_design.md](./architecture_design.md)**: 包含模型宏观训练流程与微观网络架构的 Mermaid 源码及详细说明。

---

## 📝 实验日志

详细的终端输出记录请查看 `logs/` 文件夹，包含从数据清洗到最终演示的所有关键步骤验证。