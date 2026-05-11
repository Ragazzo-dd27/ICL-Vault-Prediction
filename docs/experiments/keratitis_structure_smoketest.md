# Keratitis OCT 数据结构检查记录

## 目的

这份记录对应的是一次 Keratitis OCT 数据结构检查与最小 smoke test 结果整理。
它的主要目的不是推进正式主任务训练，而是在 **AIDK 数据暂时缺失** 的情况下，验证当前工程是否已经具备一条可运行的结构辅助预训练入口，用于后续 AS-OCT encoder 的结构监督准备。

需要明确说明：

- 这条线基于本地可见的 `keratitis_oct` 公共数据；
- 它不是 AIDK 本身；
- 它也不是最终的 vault 主任务训练线；
- 当前更接近“数据结构检查 + 最小可运行验证”，而不是完整分割实验平台。

## 数据位置

当前使用的数据目录为：

- `data/public_datasets/keratitis_oct/`

目录结构为：

- `images/`
- `masks/`

配对方式为：

- `images/<sample_id>.bmp`
- `masks/<sample_id>.json`

其中：

- 图像尺寸为 `768`
- 背景值为 `0`
- `JSON` 来自 LabelMe polygon mask 标注

## 数据结构

当前数据采用图像与 JSON 标注文件按 `sample_id` 对齐的方式组织。
其中：

- `images/` 下保存原始 OCT 图像，文件名格式为 `<sample_id>.bmp`
- `masks/` 下保存 LabelMe 风格的标注文件，文件名格式为 `<sample_id>.json`
- 训练时使用 JSON 中的 polygon 标注动态栅格化生成 mask

当前文档中保留下来的类别统计信息为：

- `Cornea: 768/768`
- `Iris: 1536 polygon`
- `Lesion: 557/768`

从当前数据结构看：

- `Cornea` 标注覆盖最完整；
- `Lesion` 只在部分样本中存在；
- 因此当前最合适的第一步任务是 `Cornea` 二值分割。

## 当前检查结果

当前 manifest 构建脚本为：

- `scripts/build_keratitis_manifest.py`

默认输出文件为：

- `data/manifests/keratitis_structure_manifest.csv`

当前 manifest 包含字段：

- `sample_id`
- `image_path`
- `mask_path`
- `task`
- `split`

当前 split 策略为：

- 先按 `sample_id` 升序排序；
- 再按 `index % 5 == 0` 划为 `val`；
- 其余样本划为 `train`。

根据当前脚本与日志，当前 manifest 统计为：

- `train = 614`
- `val = 154`

当前最小运行命令为：

先生成 manifest：

```powershell
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/build_keratitis_manifest.py
```

再运行最小 smoke test：

```powershell
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/train_keratitis_structure_pretrain.py --manifest_path data/manifests/keratitis_structure_manifest.csv --epochs 1 --batch_size 4 --num_workers 0 --image_size 128
```

## 关键统计信息

根据 `artifacts/logs/keratitis_structure_smoketest_keratitis_structure_manifest_20260413_183934.log`，本次已记录的最小 smoke test 信息包括：

- 运行环境：`cv_mamba_project`
- Python 解释器：`D:\tools\anaconda\envs\cv_mamba_project\python.exe`
- `PyTorch version = 2.5.1`
- `Manifest = data/manifests/keratitis_structure_manifest.csv`
- `Train split = train`
- `Val split = val`
- `Batch size = 4`
- `Num workers = 0`
- `Seed = 42`
- `Epochs = 1`
- `LR = 0.001`
- `Image size = 128`
- `Device = cuda`
- `Train samples = 614`
- `Val samples = 154`
- `Preview train image shape = (4, 3, 128, 128)`
- `Preview train mask shape = (4, 1, 128, 128)`

当前日志中记录到的最小训练结果为：

- `train_loss = 0.1821`
- `val_loss = 0.0721`
- `val_dice = 0.9674`

当前已生成的运行产物包括：

- 日志：
  - `artifacts/logs/keratitis_structure_smoketest_keratitis_structure_manifest_20260413_183934.log`
- checkpoint：
  - `artifacts/checkpoints/keratitis_structure_latest.pth`
  - `artifacts/checkpoints/keratitis_structure_best.pth`

## 结论

基于当前可见脚本、日志和数据结构，可以得到以下保守结论：

- 当前工程已经可以从 `data/public_datasets/keratitis_oct/` 中读取 `images/` 与 `masks/`；
- 可以基于 `images/<sample_id>.bmp` 与 `masks/<sample_id>.json` 自动构建 `data/manifests/keratitis_structure_manifest.csv`；
- 当前数据中的 LabelMe polygon mask 可以被用于训练阶段的 mask 生成；
- 当前最小 smoke test 已经跑通，说明这条结构辅助预训练链路在工程上是可运行的；
- 当前最适合作为第一步任务的是 `Cornea` 二值分割。

同时也要明确：

- 这条线是 `keratitis_oct` 的替代结构线，不是 AIDK；
- 当前结果只说明“最小结构线已可运行”，不代表已经完成正式研究结论；
- 当前文档更适合被理解为结构检查与过渡验证记录。

## 后续建议

建议下一步按以下方向推进：

- 继续保留这条 `keratitis_oct` 结构线，作为 AIDK 缺失阶段的辅助入口；
- 在文档和代码中持续明确它不是 AIDK 主线，避免后续误用；
- 在保持 `Cornea` 二值分割稳定运行的前提下，再评估是否扩展更多结构或病灶任务；
- 如果未来 AIDK 数据到位，这条线更适合作为补充参考线，而不是继续充当替代主线；
- 如需复现实验，优先检查：
  - `scripts/build_keratitis_manifest.py`
  - `scripts/train_keratitis_structure_pretrain.py`
  - `artifacts/logs/keratitis_structure_smoketest_keratitis_structure_manifest_20260413_183934.log`
  - `artifacts/checkpoints/keratitis_structure_latest.pth`
  - `artifacts/checkpoints/keratitis_structure_best.pth`
