# ICL 术后拱高预测正式数据需求清单

## 1. 文档目的

本文档用于说明：为了开展正式的 ICL 术后拱高（vault）预测实验，医院后续需要提供哪些数据、字段和映射关系。

本文档的目的包括：

- 明确正式 ICL vault 预测实验所需的数据内容；
- 区分当前已获得的真实导出图片数据和正式训练数据之间的差异；
- 为后续医院数据补充、数据清洗、manifest 构建和论文实验设计提供依据；
- 强调当前两份导出样例数据只能用于工程可行性检查，不能用于正式模型训练或性能评估。

需要特别说明：

- 这不是实验结果文档；
- 这不是正式 cohort-level evaluation 记录；
- 这不是模型训练完成后的总结；
- 它的定位是正式数据接入前的数据需求说明文档。

## 2. 当前已有数据的定位

当前已有的真实导出样例数据主要来自：

- `data/raw/real_export_demo/`

围绕这部分数据，当前工程已经完成：

- 递归扫描真实导出图片；
- 识别 AS-OCT raw、AS-OCT 2DAnalysis、UBM horizontal、UBM vertical；
- 生成初版 real-export manifest；
- 生成 patient-level summary；
- 生成去标识化多模态样例图。

当前相关文件包括：

- `scripts/inspect_real_export.py`
- `data/manifests/real_export_manifest_initial.csv`
- `data/manifests/real_export_summary.csv`
- `artifacts/figures/real_multimodal_example_paper.png`

当前数据的限制需要明确写清：

- 当前只有 2 个 patient；
- 每个 patient 有 58 张图像；
- 每个 patient 包含 26 张 OCT raw、26 张 OCT 2DAnalysis、3 张 UBM horizontal、3 张 UBM vertical；
- 每个 patient 形成 13 条 OCT visit-eye records；
- 这些 visit-eye records 不是独立 patient-level 样本；
- 当前没有 postoperative vault label；
- 当前没有 structured clinical features；
- UBM 当前只是 patient-level 关联，尚未精确匹配到 `eye` / `date` / `visit`；
- 当前 manifest 不能直接用于正式监督学习训练。

因此，当前这两份真实导出样例数据的定位应当是：

- 用于验证工程能否读取和整理真实导出数据；
- 用于支持论文中的 real-export feasibility check 描述；
- 不能替代正式训练数据集；
- 不能替代正式 cohort-level evaluation。

## 3. 正式实验的数据基本单位

正式建模时，推荐将一个样本定义为：

一只眼的一次 ICL 手术记录

也就是由：

- `patient_uid`
- `eye`
- `surgery_date`

共同构成一个 `sample_id`。

这里需要强调：

- 不要把每一张图像当成一个独立样本；
- 同一病人、同一只眼、同一次检查下的多张 OCT / UBM 图像，应作为同一个样本的候选图像或多图像记录；
- 后续训练、验证、测试划分应尽量按 patient 或 eye 层面进行，避免数据泄漏。

## 4. 最低可启动版本必须提供的数据

### 4.1 样本身份与对应关系

| 字段名 | 是否必须 | 说明 |
|---|---|---|
| `patient_uid` | 是 | 必须去标识化，不能使用姓名、身份证号、手机号、住院号等直接身份信息。 |
| `eye` | 是 | 必须统一为 `OD` / `OS`，若原始记录为 `R` / `L`，应先规范再入表。 |
| `surgery_date` | 是 | 用于界定这次 ICL 手术记录，是正式样本定义的核心字段之一。 |
| `sample_id` | 是 | 建议由 `patient_uid + eye + surgery_date` 生成，作为正式训练 manifest 的主键。 |
| `visit_id` | 是 | 用于区分同一只眼在围手术期或随访中的多次检查。 |
| `exam_date` | 是 | 必须明确每张图、每个标签、每条临床特征属于哪个病人、哪只眼、哪次手术相关检查。 |

补充说明：

- `patient_uid` 必须去标识化；
- `eye` 必须统一为 `OD` / `OS` 或 `R` / `L` 后再转换；
- `sample_id` 建议由 `patient_uid + eye + surgery_date` 生成；
- `visit_id` 用于区分同一眼的多次检查；
- 必须知道每张图、每个标签、每条临床特征属于哪个病人、哪只眼、哪次手术。

### 4.2 图像路径与模态信息

| 字段名 | 是否必须 | 说明 |
|---|---|---|
| `preop_as_oct_raw_path` | 是 | 术前 AS-OCT 原始图像路径，建议使用相对路径。 |
| `preop_as_oct_2d_analysis_path` | 否 | 术前 AS-OCT 2DAnalysis 图像路径，若有则建议保留。 |
| `preop_ubm_horizontal_path` | 否 | 术前 UBM horizontal 图像路径，必须能对齐到同一只眼。 |
| `preop_ubm_vertical_path` | 否 | 术前 UBM vertical 图像路径，必须能对齐到同一只眼。 |
| `oct_exam_date` | 是 | 对应 AS-OCT 检查日期。 |
| `ubm_exam_date` | 否 | 对应 UBM 检查日期；若 UBM 缺失或未做，可为空。 |
| `image_quality` | 建议提供 | 可标记为 `usable` / `poor` / `unclear`，用于后续质控。 |
| `modality_available` | 建议提供 | 用于记录该样本具备哪些模态，以及模态是否完整。 |

补充说明：

- 图像路径建议统一使用相对路径；
- OCT 与 UBM 必须能对应到同一只眼；
- 如果 UBM 只能关联到 patient，但不能关联到 `eye` / `date` / `visit`，则不能直接用于正式训练；
- `image_quality` 可标记 `usable` / `poor` / `unclear`；
- `modality_available` 用于记录模态是否完整。

### 4.3 术后 vault 标签

| 字段名 | 是否必须 | 说明 |
|---|---|---|
| `postop_vault` | 是 | 最关键的监督标签，表示术后测得的 vault 值。 |
| `vault_unit` | 是 | 必须明确单位，例如 `μm` 或 `mm`。 |
| `postop_exam_date` | 是 | 必须知道 vault 是哪一天测量的。 |
| `postop_interval` | 是 | 建议记录术后时间间隔，如术后 1 月、术后 3 月。 |
| `vault_measurement_device` | 建议提供 | 记录 vault 的测量设备，便于后续解释不同来源的偏差。 |
| `vault_quality` | 建议提供 | 标记标签质量是否可靠，例如 `usable` / `unclear`。 |

补充说明：

- `postop_vault` 是最关键标签；
- 必须明确单位，例如 `μm` 或 `mm`；
- 必须知道 vault 是术后哪个时间点测量的；
- 建议统一标签时间窗口，例如术后 1 月或术后 3 月；
- 如果不同样本的标签时间点差异较大，需要在论文和实验设计中说明。

### 4.4 结构化临床特征

| 字段名 | 是否必须 | 说明 |
|---|---|---|
| `age` | 建议提供 | 年龄。 |
| `sex` | 建议提供 | 性别。 |
| `spherical_equivalent` | 高优先级 | 屈光等效球镜度数。 |
| `spherical_power` | 建议提供 | 球镜度数。 |
| `cylindrical_power` | 建议提供 | 柱镜度数。 |
| `axial_length` | 高优先级 | 眼轴长度。 |
| `anterior_chamber_depth` | 高优先级 | 前房深度。 |
| `white_to_white` | 高优先级 | 角膜白到白距离。 |
| `central_cornea_thickness` | 建议提供 | 中央角膜厚度。 |
| `keratometry_k1` | 建议提供 | 角膜曲率 K1。 |
| `keratometry_k2` | 建议提供 | 角膜曲率 K2。 |
| `intraocular_pressure` | 建议提供 | 眼压。 |
| `pupil_diameter` | 建议提供 | 瞳孔直径。 |
| `icl_size` | 高优先级 | ICL 尺寸。 |
| `icl_power` | 建议提供 | ICL 屈光度。 |
| `icl_model` | 建议提供 | ICL 型号。 |

补充说明：

- `anterior_chamber_depth`、`white_to_white`、`axial_length`、`spherical_equivalent`、`icl_size`、`postop_vault` 是优先级最高的字段；
- 如果医院无法一次提供所有字段，至少应优先提供这些核心字段；
- clinical features 应与 `patient_uid + eye + surgery_date` 对齐。

## 5. 推荐的正式训练 manifest 格式

推荐的正式训练 manifest 每一行对应一个 eye-level surgical case，即一只眼的一次 ICL 手术记录。

| 字段名 | 含义 |
|---|---|
| `sample_id` | 样本主键，建议由 `patient_uid + eye + surgery_date` 生成。 |
| `patient_uid` | 去标识化患者编号。 |
| `eye` | 眼别，建议统一为 `OD` / `OS`。 |
| `surgery_date` | 对应 ICL 手术日期。 |
| `oct_exam_date` | 术前 AS-OCT 检查日期。 |
| `ubm_exam_date` | 术前 UBM 检查日期。 |
| `postop_exam_date` | 术后 vault 测量日期。 |
| `preop_as_oct_raw_path` | 术前 AS-OCT 原始图像路径。 |
| `preop_as_oct_2d_analysis_path` | 术前 AS-OCT 2DAnalysis 图像路径。 |
| `preop_ubm_horizontal_path` | 术前 UBM horizontal 图像路径。 |
| `preop_ubm_vertical_path` | 术前 UBM vertical 图像路径。 |
| `age` | 年龄。 |
| `sex` | 性别。 |
| `spherical_equivalent` | 屈光等效球镜度数。 |
| `axial_length` | 眼轴长度。 |
| `anterior_chamber_depth` | 前房深度。 |
| `white_to_white` | 白到白距离。 |
| `central_cornea_thickness` | 中央角膜厚度。 |
| `icl_size` | ICL 尺寸。 |
| `icl_power` | ICL 度数。 |
| `postop_vault` | 术后 vault 标签。 |
| `vault_unit` | vault 标签单位。 |
| `split` | 数据划分字段，可由我们后续生成。 |
| `notes` | 记录缺失、质控说明或特殊备注。 |

需要强调：

- `split` 字段可由我们后续生成；
- 所有路径建议使用相对路径；
- 所有患者信息必须去标识化；
- 一个 `sample_id` 对应一只眼的一次 ICL 手术记录。

## 6. 数据量需求建议

### 6.1 最低工程验证规模

建议规模：20–30 只眼。

该规模只能用于：

- pipeline smoke test；
- loss 是否能下降；
- 数据读取、模型 forward、反向传播是否正常。

该规模不能用于正式论文结论。

### 6.2 初步论文实验规模

建议规模：100–200 只眼。

该规模可以初步支持：

- 单模态 vs 多模态；
- pretrained vs scratch；
- concat vs attention；
- 基本误差分析。

### 6.3 更理想的数据规模

建议规模：300 只眼以上。

该规模更适合：

- 稳定 train / val / test 划分；
- 多模态深度学习；
- baseline comparison；
- ablation study；
- error distribution analysis。

## 7. 推荐数据目录结构

建议正式数据目录结构如下：

```text
data/raw/icl_clinical_cohort/
├── images/
│   ├── patient_001/
│   │   ├── OD/
│   │   │   ├── ASOCT/
│   │   │   └── UBM/
│   │   └── OS/
│   │       ├── ASOCT/
│   │       └── UBM/
│   └── ...
└── tables/
    ├── clinical_features.csv
    ├── vault_labels.csv
    └── modality_mapping.csv
```

三张表的作用建议如下：

- `clinical_features.csv`：记录结构化临床特征；
- `vault_labels.csv`：记录术后 vault 标签；
- `modality_mapping.csv`：记录样本与图像路径之间的对应关系。

## 8. 需要特别提醒医院的事项

1. 请尽量提供去标识化后的数据。
2. 不需要姓名、身份证号、手机号、住院号等直接身份信息。
3. 每张图像需要能对应到 `patient_uid`、`eye`、`exam_date`。
4. vault 标签必须是术后测量值，并且需要提供测量时间点。
5. OCT 和 UBM 最好能对应到同一只眼、同一次术前检查。
6. 不要只提供图片，还需要标签表和临床特征表。
7. 如果字段暂时不完整，应明确缺失原因，方便后续处理 missing modality / missing feature 问题。

## 9. 当前两份导出数据与正式数据需求的差距

| 需求项 | 当前是否满足 | 说明 |
|---|---|---|
| AS-OCT 图像 | 部分满足 | 当前已有 AS-OCT raw 与 AS-OCT 2DAnalysis 图像。 |
| UBM 图像 | 部分满足 | 当前已有 UBM horizontal 与 UBM vertical 图像。 |
| `patient_uid` | 基本满足 | 当前脚本可生成匿名 `patient_uid`，但仅用于可行性检查。 |
| `eye` | 部分满足 | OCT 文件名中可解析 `eye`，但 UBM 尚未精确对齐到 `eye`。 |
| `exam_date` | 部分满足 | OCT 可从文件名解析 `exam_date`，UBM 当前未建立精确日期映射。 |
| `surgery_date` | 不满足 | 当前真实导出样例中未提供正式手术日期字段。 |
| `postop_vault` | 不满足 | 当前没有 postoperative vault label。 |
| `clinical_features` | 不满足 | 当前没有 structured clinical features。 |
| UBM 与 `eye/date/visit` 精确匹配 | 不满足 | 当前 UBM 仅为 patient-level 关联。 |
| `train/val/test split` | 不满足 | 当前没有正式训练数据划分字段。 |
| image quality control | 不满足 | 当前没有系统化的图像质控字段。 |

当前主要差距可以概括为：

- 有图像，但缺标签；
- 有 patient-level 关联，但缺 `eye` / `date` / `visit` 精确映射；
- 有初版 manifest，但不是正式训练 manifest；
- 有可行性检查，但没有正式 cohort-level evaluation。

## 10. 与后续论文实验的关系

只有在正式数据到位后，才能开展以下论文实验内容：

- baseline comparison；
- modality combination experiment；
- pretrained vs scratch；
- concat vs cross-attention；
- ablation study；
- prediction vs ground truth scatter plot；
- absolute error histogram；
- residual analysis；
- clinical interpretation of prediction errors。

需要强调：

- 当前 real-export feasibility check 可以写进论文；
- 但它只能作为“真实数据接入可行性”的工程说明；
- 不能替代正式实验结果；
- 不能替代正式 cohort-level 性能评估。

## 11. 下一步建议

1. 与导师和医院确认正式标签表和临床特征表字段。
2. 明确 vault 标签的测量时间点。
3. 建立 image-to-sample mapping。
4. 构建 `formal_vault_manifest.csv`。
5. 在小规模样本上进行 pipeline smoke test。
6. 数据量扩大后开展正式 baseline、ablation 和 error analysis。
7. 将正式 cohort 统计、主结果表和误差分析图补入论文。
