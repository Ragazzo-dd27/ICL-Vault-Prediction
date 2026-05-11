# Real Export Feasibility Check

## 1. Purpose

本文档记录的是一次真实导出数据可行性检查（Real-Data Feasibility Check），而不是正式模型实验。
本次检查的目标是验证当前工程是否能够：

- 递归扫描医院导出的真实 AS-OCT 与 UBM 图像；
- 在文件名信息可用时，对 OCT 基础元数据进行解析；
- 生成一份初版 manifest，供后续数据整理使用；
- 生成 patient-level summary，便于快速核查数据结构；
- 生成可用于论文或汇报插图的去标识化多模态样例图；
- 记录当前真实数据接入链路的工程状态，而不宣称任何训练或评估结果。

需要明确强调：

- 这不是正式模型训练实验；
- 这不是正式 cohort-level evaluation；
- 本文档不包含训练指标、验证指标或测试指标；
- 本文档不能作为模型性能结论的依据。

## 2. Data Location

当前真实导出演示数据位于：

- `data/raw/real_export_demo/`

截至本次检查，该目录下包含 2 个 patient-level folders。
脚本会将患者文件夹映射为匿名 `patient_uid`，例如：

- `patient_001`
- `patient_002`

文档和导出的 CSV 中都不应记录或暴露可直接识别个体身份的信息，例如：

- 真实姓名；
- 身份证号；
- 手机号；
- 住院号；
- 其他个人敏感标识信息。

## 3. Inspection Script

本次检查使用的脚本位于：

- `scripts/inspect_real_export.py`

运行方式：

```bash
python scripts/inspect_real_export.py
```

脚本当前默认提供以下 CLI 参数：

- `--raw_root`
- `--manifest_out`
- `--summary_out`
- `--figure_out`
- `--paper_figure_out`
- `--example_patient`

默认行为是扫描 `data/raw/real_export_demo/`，将 CSV 输出写入 `data/manifests/`，并将去标识化图像输出写入 `artifacts/figures/`。

## 4. What The Script Checks

当前脚本的定位是一个保守的数据结构检查工具。
它不会修改训练主线，也不会把当前数据直接包装成正式监督学习数据集。

当前检查流程包括：

- 递归扫描 `data/raw/real_export_demo/` 下所有 patient 文件夹；
- 发现 `.jpg`、`.jpeg`、`.png`、`.bmp` 图像文件；
- 将图像粗分为以下类型：
  - `oct_raw`
  - `oct_2d_analysis`
  - `ubm_horizontal`
  - `ubm_vertical`
  - `ubm_unknown`
  - `other`
- 在 OCT 文件名可解析时提取：
  - `exam_id`
  - `date`
  - `time`
  - `eye`
  - `is_2d_analysis`
- 在 patient 级别上对 UBM 做保守聚合，并识别 horizontal / vertical / unknown；
- 按 OCT visit-eye level 构建初版 manifest；
- 生成 patient-level summary；
- 生成用于汇报和论文插图的去标识化多模态样例图。

## 5. Current Outputs

当前检查流程会生成以下文件：

- `data/manifests/real_export_manifest_initial.csv`
- `data/manifests/real_export_summary.csv`
- `artifacts/figures/real_multimodal_example_patient001_deidentified.png`
- `artifacts/figures/real_multimodal_example_paper.png`

当前更适合论文插图使用的版本是：

- `artifacts/figures/real_multimodal_example_paper.png`

如果后续运行中 paper 版本不存在，则应以 `artifacts/figures/` 下实际存在的去标识化版本作为替代，而不是使用原始未脱敏导出图。

## 6. Observed Data Status

根据当前 `data/manifests/real_export_summary.csv`，本地可见的真实导出数据状态如下：

- 当前只有 2 个 patient；
- `patient_001` 有 58 张图像；
- `patient_002` 有 58 张图像。

对每个 patient，当前统计一致，均包含：

- 26 张 OCT raw 图像；
- 26 张 OCT 2DAnalysis 图像；
- 3 张 UBM horizontal 图像；
- 3 张 UBM vertical 图像；
- 13 条 OCT visit-eye records。

这里也需要特别说明：

- 13 条 OCT visit-eye records 不是 13 个独立的 patient-level 样本；
- 它们表示的是同一 patient 在不同检查时间和眼别上的 visit-eye 级记录；
- 因此，当前统计不能被误读为存在更多独立患者样本。

从工程角度看，当前 2 个匿名 patient 都具备：

- 可识别的 OCT 内容；
- 可识别的 UBM 内容；
- 可解析的 OCT 文件名结构；
- 构建初版 manifest 的基本条件；
- 生成去标识化多模态样例图的基本条件。

## 7. Manifest Status

当前生成的 manifest 文件名为：

- `data/manifests/real_export_manifest_initial.csv`

这里的 `initial` 是有意保留的。
它明确表示该文件只是用于真实导出可行性检查的初版 manifest，而不是正式训练 manifest。

当前每一行对应一条 OCT visit-eye level record，字段包括：

- `patient_uid`
- `source_patient_folder`
- `date`
- `time`
- `eye`
- `exam_id`
- `oct_raw_paths`
- `oct_2d_analysis_paths`
- `ubm_horizontal_paths`
- `ubm_vertical_paths`
- `ubm_unknown_paths`
- `has_oct_raw`
- `has_oct_2d_analysis`
- `has_ubm_horizontal`
- `has_ubm_vertical`
- `vault_label`
- `clinical_features_status`
- `notes`

当前状态下必须明确说明：

- `vault_label` 为空；
- `clinical_features_status` 为 `missing`；
- UBM 当前只是 patient-level 关联，尚未精确匹配到具体 `eye` / `date` / `visit`；
- `notes` 字段明确写明这是 initial manifest，而不是 formal training manifest。

因此，当前 manifest 不能直接用于正式监督训练。
它更适合作为后续数据整理、映射规则设计和论文数据管线说明的起点。

## 8. De-Identified Figure Status

脚本当前也会生成去标识化多模态样例图。
目前更适合论文使用的输出是：

- `artifacts/figures/real_multimodal_example_paper.png`

图像采用 2x2 布局，包含：

- `AS-OCT Raw`
- `AS-OCT 2DAnalysis`
- `UBM Horizontal`
- `UBM Vertical`

隐私与去标识化要求必须保留：

- 原始 `AS-OCT 2DAnalysis` 图中可能包含患者姓名、检查 ID、检查日期、报告时间等信息；
- 论文和汇报只能使用去标识化后的图片；
- 不应直接使用原始导出图作为公开展示材料。

当前实现中，`AS-OCT 2DAnalysis` 面板会在导出前执行保守遮盖，对顶部报告信息区和底部时间信息区做白色遮挡。
该策略优先保护隐私，而不是追求最大程度保留图像边缘信息。

## 9. What This Check Demonstrates

本次可行性检查支持以下工程层面的结论：

- 当前工程可以通过递归遍历读取本地真实导出目录结构；
- 当前导出数据中可以识别出 OCT raw、OCT 2DAnalysis 和 UBM 图像；
- OCT 文件名可以被解析为一份可用于初步整理的元数据结构；
- 可以自动生成初版 manifest；
- 可以自动生成 patient-level summary；
- 可以生成用于论文或汇报的数据管线示意图所需的去标识化多模态样例图。

## 10. What This Check Does Not Demonstrate

本文档不应被解读为已经证明以下任一事项：

- 已经构建完成可直接训练的正式数据集；
- 已经完成 UBM 与 OCT 的一一精确匹配；
- 已经获得最终任务标签；
- 已经获得结构化临床变量；
- 已经完成模型训练、验证或测试；
- 已经完成正式 cohort-level evaluation；
- 已经完成正式数据集清洗与纳入排除流程。

## 11. Current Limitations

当前可行性检查仍然存在以下重要限制：

- 当前本地真实导出目录中只有 2 个 patient；
- `vault_label` 尚未提供；
- 结构化 clinical features 尚未提供；
- UBM 目前仅做保守聚合，尚未建立精确 eye/date/visit 对齐；
- OCT 与 UBM 的 visit-level、eye-level 对应关系仍需人工确认或补充更强规则；
- 当前没有用于正式机器学习的 `train` / `val` / `test` 划分字段；
- 当前流程更适合被理解为真实数据接入检查，而不是正式数据集构建完成。

## 12. Next Steps Toward A Formal Training Manifest

如果要把当前 initial manifest 升级为正式训练 manifest，至少还需要补充或确认以下内容：

- anonymous `patient_id` / `eye_id` / `visit_id`；
- `eye laterality`；
- `preoperative AS-OCT path`；
- `preoperative UBM path`；
- `structured clinical features`；
- `postoperative vault label`；
- `surgery date`；
- `exam date`；
- `modality alignment information`；
- `train/val/test split`；
- `quality control fields`；
- `inclusion/exclusion status`。

## 13. Summary

当前这次真实导出数据可行性检查说明，仓库已经能够以一种保守且注重隐私保护的方式，对当前演示数据完成扫描、解析、汇总和可视化。
这代表真实数据接入链路向前推进了一步。

但与此同时，当前产物仍应被视为工程准备阶段的中间结果，而不是正式训练资产。
更准确地说，当前状态是：

- 真实数据可以被扫描；
- 初版 manifest 可以生成；
- patient-level summary 可以生成；
- 去标识化样例图可以生成；
- 但当前 manifest 仍不能直接用于正式监督训练。

## 14. Next Steps

下一步如果要推进到正式 cohort 级研究，应至少包括：

- 获取完整 clinical feature table；
- 获取 postoperative vault labels；
- 建立 UBM 与 OCT 的 `eye` / `date` / `visit` 精确映射；
- 构建正式 training manifest；
- 在正式 cohort 上进行 baseline、ablation、multimodal comparison 和 error analysis；
- 在论文中补充 full-cohort statistics、main results table 和 prediction error visualization。
