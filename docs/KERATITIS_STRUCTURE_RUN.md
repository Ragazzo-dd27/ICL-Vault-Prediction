# Keratitis OCT 替代结构预训练线运行说明

## 链路定位

这条线是 **AIDK 数据缺失情况下的替代结构辅助预训练线**。  
它基于本地真实存在的 `keratitis_oct`，不是 AIDK 本身，也不是最终 vault 主任务。  
它当前的作用是为未来 AS-OCT encoder 提供一个现实可用的结构监督入口。

## 正式运行环境

- 正式环境：`cv_mamba_project`
- 解释器路径：`D:\tools\anaconda\envs\cv_mamba_project\python.exe`
- 不要把默认 `python` 的缺依赖情况当成这条线的正式阻塞结论

## 数据与当前任务落点

- 数据目录：
  - `data/public_datasets/keratitis_oct/images/`
  - `data/public_datasets/keratitis_oct/masks/`
- 配对方式：
  - `images/<sample_id>.bmp`
  - `masks/<sample_id>.json`
- 标注格式：
  - `masks/` 下是 LabelMe 风格 JSON
  - 当前使用 polygon 动态栅格化生成训练 mask
- 当前第一落点任务：
  - `Cornea` 二值分割
- 为什么先选它：
  - 标注最稳
  - 全样本覆盖
  - 最适合作为最小结构辅助预训练入口

## Manifest 生成

- 构建脚本：`scripts/build_keratitis_manifest.py`
- 默认输出：`data/manifests/keratitis_structure_manifest.csv`
- 当前 manifest 主要字段：
  - `sample_id`
  - `image_path`
  - `mask_path`
  - `task`
  - `split`
- 当前 split 策略：
  - 先按 `sample_id` 升序排序
  - 再按 `index % 5 == 0` 划为 `val`
  - 其余划为 `train`

## 最小运行命令

先生成 manifest：

```powershell
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/build_keratitis_manifest.py
```

再跑最小 smoke test：

```powershell
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/train_keratitis_structure_pretrain.py --manifest_path data/manifests/keratitis_structure_manifest.csv --epochs 1 --batch_size 4 --num_workers 0 --image_size 128
```

## 运行产物位置

- 日志目录：`artifacts/logs/`
- checkpoint 目录：`artifacts/checkpoints/`

当前已验证通过的一次 smoke test 产物是：

- 日志：`artifacts/logs/keratitis_structure_smoketest_keratitis_structure_manifest_20260413_183934.log`
- checkpoint：
  - `artifacts/checkpoints/keratitis_structure_latest.pth`
  - `artifacts/checkpoints/keratitis_structure_best.pth`

## 日志里会记录什么

当前日志至少会记录：

- 解释器路径
- 实际执行命令
- manifest 路径
- 关键超参数
  - `epochs`
  - `batch_size`
  - `num_workers`
  - `lr`
  - `image_size`
  - `seed`
- train / val 结果
  - `train_loss`
  - `val_loss`
  - `val_dice`
- checkpoint 路径

## 当前已知限制

- 这条线是 `keratitis_oct` 的替代结构线，不是 AIDK
- 当前第一步只做 `Cornea` 二值分割
- 当前目标是最小结构辅助预训练入口，不是完整分割平台
- 当前已经完成最小运行验证，但它仍属于公开数据过渡阶段
- 如果以后 AIDK 数据到位，这条线更适合保留为补充参考线，而不是继续充当替代主线

## 适合什么时候看这份文档

- 重新跑这条结构线前
- 查正式环境和运行命令时
- 查日志和 checkpoint 位置时
- 给别人或另一个 AI 助手交接这条替代结构线时
