# 1. 研究问题定义与数学抽象 (Problem Formulation)

## 1.1 研究要解决的问题是什么？
本研究旨在解决**“ICL（可植入式隐形眼镜）手术后晶体拱高（Vault）的精准预测问题”**。
拱高是指人工晶体后表面与人体自身晶状体前表面之间的垂直安全间隙。若预测不准导致选错晶体尺寸，拱高过低（<250μm）会引发白内障，过高（>750μm）会引发青光眼。因此，术前精准预测 Vault 是一项高风险、高价值的临床决策任务。

## 1.2 问题的数学抽象描述
从计算机科学的视角，ICL 拱高预测本质上是一个**多模态连续值回归问题 (Multimodal Continuous Regression)**。

定义患者的术前多模态观测空间为 $\mathcal{X}$，包含以下三个子空间：
1. **高分辨率眼前节结构**：$I_{OCT} \in \mathbb{R}^{H \times W \times C}$ (AS-OCT 图像)
2. **深层睫状沟形态结构**：$I_{UBM} \in \mathbb{R}^{H \times W \times C}$ (UBM 图像)
3. **临床数值特征向量**：$v_{clin} \in \mathbb{R}^{N}$ (包含 ACD, WTW 等 $N$ 个测量参数)

定义真实术后拱高为连续标量 $y \in \mathbb{R}^{+}$。
本研究的目标是寻找一个非线性映射函数 $\mathcal{F}_{\theta}$（由深度神经网络参数化），使得给定术前观测数据 $X = \{I_{OCT}, I_{UBM}, v_{clin}\}$ 时，预测值 $\hat{y}$ 与真实值 $y$ 的期望损失最小化：

$$ \hat{y} = \mathcal{F}_{\theta}(I_{OCT}, I_{UBM}, v_{clin}) $$
$$ \theta^{*} = \arg\min_{\theta} \mathbb{E}_{(X, y) \sim \mathcal{D}}[\mathcal{L}(y, \mathcal{F}_{\theta}(X))] $$

其中，$\mathcal{L}$ 为回归损失函数（如 MSE Loss），$\theta^{*}$ 为模型的最优权重。

---

# 2. 现有方法的局限性分析 (Limitations of Existing Methods)

该问题并非一个全新的问题，但在现有临床实践中仍未得到良好解决。现有主流方法（如传统厂商 Nomogram 表表法、NK/KS 公式）存在以下核心数学与工程缺陷：

### 2.1 缺陷一：欠拟合非线性耦合关系 (Lack of Non-linear Modeling)
传统统计学公式（如 NK 公式）本质上是**多元线性回归 (Multiple Linear Regression)**：
$$ \hat{y} = w_1 \cdot \text{ACD} + w_2 \cdot \text{WTW} + w_3 \cdot \text{STS} + b $$
然而，人眼是一个复杂的生物力学系统，各解剖参数对最终拱高的影响存在高度的**非线性耦合效应**（例如，前房深度的变化对拱高的影响并非固定斜率）。线性模型能力上限极低，对极端尺寸病例极易产生严重预测偏差。

### 2.2 缺陷二：高维形态学信息的丢失 (Loss of Morphological Information)
现有公式仅提取了图像中的少数 1D 离散距离值（如 WTW 白到白距离）。从信息论角度来看，这种强行降维导致了 $I_{OCT}$ 和 $I_{UBM}$ 中大量 2D 拓扑结构信息（如：虹膜根部的弯曲角度、睫状体的不规则隆起）被彻底丢弃。

### 2.3 本研究的必要性与创新解法
针对上述痛点，本研究提出利用双流卷积神经网络（ResNet18）直接从影像中提取形态学特征，打破了人工提取参数的限制；并引入**交叉注意力机制（Cross-Attention）**处理图像与数值的非线性耦合，在数学上提供了一个更高容量（High Capacity）的假设空间，从而实现更精准的预测。