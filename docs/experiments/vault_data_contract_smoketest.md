# V2 Vault Data Contract Smoketest

## 背景

本轮目标不是推进完整 vault 回归训练，而是把 V2 vault 主线从 metadata-only scaffold 推进到第一版可用的数据契约状态：

- `VaultDataset` 能在本地文件真实存在时读取模态输入
- `collate_vault_batch` 能产出更接近模型输入的 batch
- 缺失模态用明确 mask 表达
- 为未来医院多模态数据接入保留清晰 metadata

## 当前本地 vault 主任务数据现状

本仓库当前可见的 vault 主任务 manifest 只有：

- `data/manifests/vault_manifest_example.csv`

该 manifest 当前字段为：

- `sample_id`
- `patient_id`
- `eye_side`
- `oct_path`
- `ubm_path`
- `topography_path`
- `vault_label`
- `split`
- `device_oct`
- `device_ubm`
- `has_oct`
- `has_ubm`
- `has_topography`

对这份 manifest 做本地核查后的现实结论：

- manifest 中声明了 OCT / UBM / topography 三类路径字段
- `vault_label` 与 `split` 已具备
- `device_oct` / `device_ubm` 已具备
- 没有更多临床结构化字段，其他扩展列只能进入 `extras`
- manifest 指向的 `data/raw/oct/...`、`data/raw/ubm/...`、`data/raw/topography/...` 当前本地都不存在
- 因此，当前仓库里的真实 vault 主任务样本仍然没有任何本地可读模态文件

这意味着：

- 本轮可以把数据契约实体化
- 但不能诚实地说“vault 主任务本地真实图像已经接通”

## 第一版 sample contract

`VaultDataset.__getitem__` 当前返回的核心结构为：

- 标识字段：`sample_id`、`patient_id`、`eye_side`、`split`
- 标签字段：`vault_label`、`label_available`
- 原始 manifest 元信息：`device_oct`、`device_ubm`、`extras`
- 路径字段：`oct_path`、`ubm_path`、`topography_path`
- 模态声明字段：`has_oct`、`has_ubm`、`has_topography`
- 本地可用性字段：`oct_exists`、`ubm_exists`、`topography_exists`
- 加载结果字段：`oct_loaded`、`ubm_loaded`、`topography_loaded`
- 模态输入字段：`oct_image`、`ubm_image`、`topography_image`

约定如下：

- 如果模态文件真实存在，则尽量读成 `torch.Tensor`
- OCT / UBM 按 RGB 图像读取，默认转为 `float32` 的 `C x H x W`
- topography 支持图像文件和 `.csv/.txt` 数值矩阵；CSV 会转为 `1 x H x W`
- 如果路径不存在或模态缺失，则对应输入字段为 `None`
- 不伪造不存在的主任务模态内容

## 第一版 batch contract

`collate_vault_batch` 当前返回：

- `oct_images`
- `ubm_images`
- `topography_images`
- `vault_labels`
- `label_available`
- `has_oct`
- `has_ubm`
- `has_topography`
- `oct_available`
- `ubm_available`
- `topography_available`
- `meta`
- `tensor_shapes`

批处理规则：

- 对当前 batch 中至少有一个真实张量的模态，执行按 batch 堆叠
- 同一模态内部如果尺寸不一致，会自动 pad 到该 batch 的最大尺寸
- 当前 batch 中完全没有可读张量的模态，直接返回 `None`
- 缺失或不可读样本不会被伪造为“可用”，而是通过 `*_available` 标记为 `False`
- `meta` 中保留 sample id、路径、声明字段、exists/loaded 状态、设备字段、`extras`

## Smoketest 结果

### 1. 真实 vault example manifest

运行：

```bash
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/smoke_test_vault_data_contract.py --manifest_path data/manifests/vault_manifest_example.csv --batch_size 2
```

结果：

- batch 成功构建
- `vault_labels` 成功张量化
- `oct_images=None`
- `ubm_images=None`
- `topography_images=None`
- `oct_available / ubm_available / topography_available` 全为 `False`
- 说明当前真实 vault example manifest 仍然只能提供 metadata + label，不能提供本地真实模态输入

日志产物：

- `artifacts/logs/vault_data_contract_smoketest_vault_manifest_example_20260415_104314.json`

### 2. 合约代码路径 fixture 验证

为了验证新 contract 本身能处理真实输入和缺失模态，新增了一个明确标注为测试夹具的最小 manifest：

- `tests/fixtures/vault_contract/vault_contract_fixture.csv`

它不是医院真实数据，只用于验证代码路径。

运行：

```bash
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/smoke_test_vault_data_contract.py --manifest_path tests/fixtures/vault_contract/vault_contract_fixture.csv --split train --batch_size 2
```

结果：

- batch 成功构建
- `oct_images` shape 为 `2 x 3 x 2 x 2`
- `ubm_images` shape 为 `2 x 3 x 2 x 2`
- `topography_images` shape 为 `2 x 1 x 3 x 3`
- 第二个样本没有 UBM，`ubm_available=[true, false]`
- 说明当前 dataset + collate 已能处理
  - 真实图像读取
  - CSV topography 张量化
  - 缺失模态 mask
  - batch 内尺寸 pad

日志产物：

- `artifacts/logs/vault_data_contract_smoketest_vault_contract_fixture_20260415_104314.json`

## 当前已知限制

- 仓库里的真实 vault manifest 仍没有本地可读 OCT / UBM / topography 文件
- 当前还没有把真实医院数据整理成这版 manifest 并实际跑通
- 还没有接入最终 vault 多模态回归模型
- topography 当前只做了基础 CSV / 图像读取，没有加入设备特异的预处理语义
- batch 目前以最小稳定契约为主，还没有把多图 OCT、临床表格特征、时序/配准逻辑展开

## 结论

这轮推进后，V2 vault 主线已经不再是“只能返回路径和 exists 标志”的纯 metadata scaffold。

更务实地说，当前状态是：

- 数据接口已经具备真实模态读取能力
- batch 契约已经具备模型输入形态
- 缺失模态表达已经正式化
- 但真实 vault 主任务本地数据本身仍未到位，所以主线只推进到了“可承接真实数据”的第一版实体化状态，还不能宣称主任务真实训练已接通
