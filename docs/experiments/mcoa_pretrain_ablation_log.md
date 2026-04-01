# MCOA Pretraining Ablation Log

## 1. 实验背景

当前项目正在从原型式实现逐步迁移到 V2 工程体系。为尽快验证 V2 结构在真实训练场景下的可用性，项目优先选择公开数据集 MCOA 作为 backbone 预训练入口，并围绕以下关键环节完成了初步工程闭环搭建：

- manifest 驱动的数据组织方式；
- V2 风格的数据接口与训练入口；
- 基础 evaluator 与 checkpoint 机制；
- 小规模真实数据上的端到端训练验证。

在此基础上，本轮实验聚焦于 **MCOA 小样本场景下的预训练策略对比**，重点分析以下两个因素对验证性能的影响：

1. 是否使用 ImageNet 预训练权重初始化；
2. 是否加入基础且保守的数据增强策略。

---

## 2. 实验目的

本轮实验的主要目标如下：

1. 验证 V2 工程体系下的 MCOA 预训练路径已具备真实训练能力；
2. 比较随机初始化（from scratch）与 ImageNet 预训练初始化在小样本场景下的差异；
3. 评估在 ImageNet 预训练基础上加入保守 augmentation 后，训练稳定性与验证表现是否进一步改善；
4. 为后续更大规模实验与主任务迁移提供可参考的工程与实验依据。

---

## 3. 实验设置

### 3.1 任务类型
- 二分类预训练实验

### 3.2 数据来源
本次实验的数据来源限定为 MCOA 数据集中的以下两类 AS-OCT 图像：

- `Normal Cornea/AS-OCT`
- `Opaque Cornea/AS-OCT/AS-OCT Original Images`

### 3.3 工程入口与模块
- 训练入口：`scripts/pretrain_backbone.py`
- 数据接口：`src/icl_vault/data/datasets/mcoa_dataset.py`
- 评估模块：`src/icl_vault/engine/evaluator.py`
- checkpoint 模块：`src/icl_vault/engine/checkpoint.py`

### 3.4 模型与优化设置
- 模型：`torchvision.models.resnet18`
- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam`
- 输入尺寸：`224 × 224`
- batch size：`2`
- 训练轮数：`5`
- 学习率：`1e-4`
- 随机种子：`42`

### 3.5 指标
当前阶段主要记录以下指标：

- `train_loss`
- `val_loss`
- `val_accuracy`

说明：本轮实验的重点是验证小样本条件下不同初始化与增强策略的相对表现，因此暂未引入更复杂的评估指标。

---

## 4. 数据划分说明

本次实验使用小规模 manifest 文件：

- manifest 文件：`data/manifests/mcoa_manifest_small.csv`

对应的数据划分如下：

- 训练集（train）：20 张
- 验证集（val）：10 张

类别构成如下：

- train:
  - normal：10
  - opaque：10
- val:
  - normal：5
  - opaque：5

需要说明的是，该划分主要用于：

1. 验证 V2 工程链路的端到端可运行性；
2. 初步观察小样本条件下的训练行为；

并**不用于形成最终统计结论**。

---

## 5. 对比实验设计

本轮共完成三组对比实验：

### E1：from scratch + no augmentation
- 模型初始化：`resnet18(weights=None)`
- 数据增强：不使用，仅保留基础预处理

### E2：ImageNet pretrained + no augmentation
- 模型初始化：使用 torchvision 的 ImageNet 预训练权重
- 数据增强：不使用，仅保留基础预处理

### E3：ImageNet pretrained + basic augmentation
- 模型初始化：使用 torchvision 的 ImageNet 预训练权重
- 数据增强：使用保守增强策略，包括：
  - `RandomHorizontalFlip`
  - 小角度 `RandomRotation`
  - 轻微 `ColorJitter`

其中，验证集变换均保持纯净，仅使用：
- `Resize`
- `ToTensor`

以避免验证阶段引入额外扰动。

---

## 6. 实验结果

### 6.1 结果汇总表

| 实验编号 | ImageNet 预训练 | Augmentation | 训练现象 | 最终 train_loss | 最终 val_loss | 最终 val_accuracy |
|---|---:|---:|---|---:|---:|---:|
| E1 | 否 | 否 | 训练损失快速下降，验证损失持续升高，出现明显过拟合 | 0.0843 | 1.1992 | 0.5000 |
| E2 | 是 | 否 | 验证性能显著改善，验证集表现明显优于随机水平 | 0.3275 | 0.0374 | 1.0000 |
| E3 | 是 | 是 | 在 E2 基础上训练更平稳，验证损失下降更顺滑 | 0.1367 | 0.0583 | 1.0000 |

### 6.2 分实验记录

#### E1：from scratch + no augmentation
关键结果如下：

- Epoch 1: `train_loss=0.6080`, `val_loss=0.7547`, `val_accuracy=0.5000`
- Epoch 2: `train_loss=0.2536`, `val_loss=0.9990`, `val_accuracy=0.5000`
- Epoch 3: `train_loss=0.0744`, `val_loss=1.0056`, `val_accuracy=0.5000`
- Epoch 4: `train_loss=0.1617`, `val_loss=1.1270`, `val_accuracy=0.5000`
- Epoch 5: `train_loss=0.0843`, `val_loss=1.1992`, `val_accuracy=0.5000`

现象概括：
- 训练损失快速下降；
- 验证准确率长期停留在 0.5；
- 验证损失持续上升；
- 典型表现为小样本条件下的明显过拟合。

#### E2：ImageNet pretrained + no augmentation
关键结果如下：

- Epoch 1: `train_loss=0.5582`, `val_loss=0.2627`, `val_accuracy=1.0000`
- Epoch 2: `train_loss=0.4463`, `val_loss=0.2329`, `val_accuracy=1.0000`
- Epoch 3: `train_loss=0.1262`, `val_loss=0.0940`, `val_accuracy=1.0000`
- Epoch 4: `train_loss=0.4799`, `val_loss=0.0337`, `val_accuracy=1.0000`
- Epoch 5: `train_loss=0.3275`, `val_loss=0.0374`, `val_accuracy=1.0000`

现象概括：
- 相比 E1，验证表现显著改善；
- 即使在当前极小样本验证集上，也可观察到非常明显的提升；
- 表明预训练初始化在当前设定下具有决定性作用。

#### E3：ImageNet pretrained + basic augmentation
关键结果如下：

- Epoch 1: `train_loss=0.5476`, `val_loss=0.3078`, `val_accuracy=0.9000`
- Epoch 2: `train_loss=0.5106`, `val_loss=0.1667`, `val_accuracy=1.0000`
- Epoch 3: `train_loss=0.3723`, `val_loss=0.1392`, `val_accuracy=1.0000`
- Epoch 4: `train_loss=0.2678`, `val_loss=0.1133`, `val_accuracy=1.0000`
- Epoch 5: `train_loss=0.1367`, `val_loss=0.0583`, `val_accuracy=1.0000`

现象概括：
- 在 E2 的基础上，训练与验证过程更平滑；
- 第 1 个 epoch 即取得较高验证准确率；
- `val_loss` 呈现更稳定的下降趋势；
- 说明适度 augmentation 在当前设定下主要提供了训练稳定性收益。

---

## 7. 结果分析

### 7.1 from scratch 条件下的主要问题
在当前小样本条件下，训练集规模仅为 20，验证集规模仅为 10。对 ResNet18 这类模型而言，从随机初始化开始学习容易快速记忆训练集，但难以形成有效泛化。E1 中 `train_loss` 明显下降，而 `val_accuracy` 长期维持在 0.5，且 `val_loss` 持续上升，说明模型基本停留在接近随机分类的验证水平，过拟合现象较为明显。

### 7.2 ImageNet 预训练权重的主要收益
E2 相比 E1 的差异非常明显，说明在当前小样本设定下，ImageNet 预训练权重是提升验证性能的主要来源。尽管自然图像与医学图像之间存在领域差异，但预训练模型所提供的较优初始特征表示，仍然对当前二分类任务起到了显著帮助。

### 7.3 基础 augmentation 的作用
E3 与 E2 相比，在最终 `val_accuracy` 上并未进一步突破，但在训练过程的稳定性上表现更好。其主要体现为：
- 训练损失下降更平稳；
- 验证损失下降更连续；
- 初期训练过程更自然，没有出现明显震荡。

因此，在当前实验条件下，augmentation 的主要作用更接近于**稳定训练过程**，而非单纯提升最终点估计精度。

---

## 8. 当前结论

基于当前已完成的小规模真实实验，可以得到如下阶段性结论：

1. **V2 体系下的 MCOA 预训练路径已经完成真实训练验证**，说明当前工程框架具备承载真实实验的能力；
2. **ImageNet 预训练权重是当前小样本场景下的主要收益来源**，对验证性能提升具有决定性作用；
3. **基础 augmentation 带来了进一步的稳定化收益**，主要体现在训练与验证过程更平滑；
4. 当前实验已能支持后续更大规模数据划分下的进一步对比验证。

---

## 9. 局限性

当前结果仍存在以下局限：

1. 样本规模较小，train 仅 20、val 仅 10，统计稳定性有限；
2. 当前验证准确率达到 1.0，可能受到小验证集规模影响，尚不能作为最终性能结论；
3. 当前仅使用基础指标，尚未引入更系统的分析维度；
4. 当前结果主要用于工程验证与初步实验观察，仍需在更大规模 manifest 上重复验证。

---

## 10. 下一步计划

下一阶段拟推进如下工作：

1. 构建更大规模的 manifest（如 medium 规模），复现当前三组对比实验；
2. 在更大样本划分下继续比较：
   - from scratch
   - ImageNet pretrained
   - ImageNet pretrained + basic augmentation
3. 观察当前结论在更大规模数据上的稳定性；
4. 逐步补充更规范的实验管理能力，包括：
   - 更系统的 checkpoint 管理
   - 更统一的 trainer / evaluator 整合
   - 更细化的 augmentation ablation
5. 在预训练线稳定后，进一步考虑将相关工程经验迁移到主任务 Vault 预测模型中。

---

## 11. 可用于月报/组会汇报的精简版总结

本阶段在 V2 工程体系内完成了 MCOA 预训练路径的真实训练验证，并基于小规模 manifest 开展了三组对比实验。实验结果表明：在当前小样本场景下，从头训练容易出现明显过拟合，验证准确率长期维持在 0.5；引入 ImageNet 预训练权重后，验证性能显著改善，说明预训练初始化是当前设定下的主要收益来源；在此基础上进一步加入保守 augmentation 后，训练与验证过程更加平稳，验证损失下降更连续。总体而言，当前结果支持将“ImageNet 预训练 + 基础 augmentation”作为后续更大规模 MCOA 实验的默认起点，但仍需在更大样本划分上继续验证其稳定性与泛化能力。