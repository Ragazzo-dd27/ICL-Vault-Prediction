graph LR
    %% 定义样式
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef backbone fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef fusion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:4px;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    %% 1. 输入层 (Input Layer)
    subgraph Input_Layer ["输入层: 多模态数据"]
        direction TB
        A["AS-OCT 图像<br/>(高分辨率结构)"]:::input
        B["UBM 图像<br/>(睫状沟形态)"]:::input
        C["临床数值 x10<br/>(ACD, WTW...)"]:::input
    end

    %% 2. 特征提取层 (Backbone)
    subgraph Feature_Ext ["双流特征提取"]
        D("ResNet18 Backbone<br/>提取 OCT 特征"):::backbone
        E("ResNet18 Backbone<br/>提取 UBM 特征"):::backbone
        F("MLP 编码器<br/>数值嵌入"):::backbone
    end

    %% 连接输入到骨干
    A --> D
    B --> E
    C --> F

    %% 3. 核心融合层 (Fusion)
    subgraph Attention_Fusion ["Cross-Attention 融合模块"]
        direction LR
        D --> |"Query (Q)"| G{"交叉注意力<br/>Cross-Attention"}:::fusion
        E --> |"Key (K) / Value (V)"| G
        
        G --> |"加权特征"| H["多模态特征拼接"]:::fusion
        F --> |"数值特征"| H
    end

    %% 4. 输出层 (Prediction)
    subgraph Prediction ["回归预测"]
        H --> I("全连接层 FC"):::output
        I --> J(("预测拱高<br/>Vault Value")):::output
    end

    %% 添加注释连接
    G -.->|"OCT引导UBM特征提取"| K["解剖结构对齐"]

## 宏观训练流程图 (Two-Stage Training Pipeline)

```mermaid
graph TD
    %% 定义样式
    classDef dataset fill:#e3f2fd,stroke:#0277bd,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef weight fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef final fill:#fce4ec,stroke:#c2185b,stroke-width:3px;

    subgraph Stage1[第一阶段：大规模域适应预训练 (Domain-Adaptive Pre-training)]
        direction LR
        A[(MCOA 公开数据集<br>13,328张 AS-OCT)]:::dataset --> B[ResNet18 骨干网络<br>辅助分类任务训练]:::process
        B --> C{{MCOA 预训练权重<br>Acc: 99.56%}}:::weight
    end

    subgraph Stage2 [第二阶段：多模态联合微调 (Multi-modal Fine-tuning)]
        direction LR
        D[(ICL 临床私有数据<br>OCT + UBM + 数值)]:::dataset --> E[多模态主模型<br>VaultPredictor]:::process
    end

    %% 连接两个阶段
    C ==>|权重初始化<br>大脑移植| E
    
    %% 第二阶段的详细输出
    E --> F([模型评估与可视化<br>Bland-Altman / Loss 曲线]):::final
    F --> G(((最终交付:<br>高精度拱高预测模型))):::final
```    