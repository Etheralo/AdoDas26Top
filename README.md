# adodas-sota

参考 AdoDAS 挑战论文公开策略的独立 PyTorch 训练实现，包含 A1 三项二分类、A2 二十一项序数预测及顺序训练流程。本仓库不包含官方 baseline、数据集、论文 PDF 或预训练权重。

**复现边界：**论文未公开冠军完整代码和训练配方；本项目实现相关方法组件，不能声称精确复现冠军模型或达到冠军成绩。模型输出是研究或筛查测量，不是临床诊断。

## 方法与协议

依据 AdoDAS 论文第 2 页 §2.1、第 3 页 §2.3、第 4 页 Table 2/3。官方资料见 [挑战网站](https://adodas.hai-lab.cn/) 和 [官方代码仓库](https://github.com/Multimodel-signal-lab/AdoDAS2026)。

| 赛道 | 当前实现 | 配置 |
|---|---|---|
| A1：参考 HITSZ_HLT 描述 | 帧级音视频注意力、跨会话 Transformer、会话文本适配、A1 引导条目修正、冻结编码器后调整 A1 头 | `aligned_a1.yaml`、`aligned_a1_stage2.yaml` |
| A2：参考 HEALTH FIRST 描述 | 音视频双 Conformer、跨会话融合、21 个 query-CORAL 输出头、验证集条目专家选择 | `aligned_a2.yaml` |

完整官方 train 用于梯度更新；官方 val 用于早停、检查点选择、校准和专家选择。验证分数属于开发集结果，不能当作独立测试成绩。测试清单不含标签，参考标签仅由独立工具在模型选择冻结后评分。详见 [方法对应](METHOD_ALIGNMENT.md) 和 [训练计划](ROUND2_PLAN.md)。

## 目录

- `models/`：特征适配器、Conformer、跨模态/跨会话编码器、文本分支和预测头。
- `data/`：官方特征读取、时间对齐、掩码、标签与参与者划分。
- `training/`：训练循环、损失、指标、校准与配置。
- `configs/`：当前分赛道配置，以及早期实验配置。
- `tools/`：数据检查、合成自检、GPU 探测、文本提取与离线评分。
- `scripts/`：训练和环境脚本；`train.py`、`infer.py`、`ensemble.py` 为主要入口。

## 环境

```bash
conda env create -f environment.yaml
conda activate adodas_train
# A1 文本提取额外需要：
python -m pip install transformers==4.46.3
python tools/self_test.py
python tools/runtime_checks.py
python tools/check_alignment.py
python tools/check_round2.py
```

环境文件选择 Python 3.10、PyTorch 2.10.0 / cu128；这是工程环境选择，不是主办方强制版本。环境文件需要联网安装依赖；文本工具仅加载本地编码器，训练脚本禁止自动下载模型。合成检查不证明真实数据覆盖、GPU 吞吐或模型效果。

## 私有数据准备

通过赛事授权渠道自行获取数据，放在仓库外。目录示例：

```text
/data/dataset-ai/
  train.csv、val.csv、test_hidden.csv
  features/<split>/<school>/<class>/<pid>/{audio,video}/...
  transcripts/<split>/<school>/<class>/<pid>/<session>/...
  references/   # 如有授权参考标签，仅用于审计和最终离线评分
```

训练和验证清单需要 `anon_school, anon_class, anon_pid, session, y_D, y_A, y_S, d01..d21`。测试清单只保留身份键、session 和 split。A01/B01/B02/B03 是四种会话；文本分支使用 B01/B02/B03。

修改配置的 `paths.feature_root`、`manifest_dir`、`text_root` 和 `output_dir`，或通过 `--set` 覆盖：

```bash
python tools/check_data.py --config configs/aligned_a2.yaml --set \
  paths.feature_root=/data/dataset-ai/features paths.manifest_dir=/data/dataset-ai
```

如需提取文本向量，先在仓库外准备完整本地编码器（配置、权重、tokenizer 和词表）：

```bash
python tools/extract_text_embeddings.py \
  --transcript_root /data/dataset-ai/transcripts \
  --manifest_dir /data/dataset-ai --text_root /data/dataset-ai/text_emb \
  --model /data/models/chinese-roberta-wwm-ext-large
```

## 顺序训练

当前顺序脚本是 **AutoDL 环境入口**：默认数据位于 `/root/autodl-tmp/dataset-ai`，解释器位于 `/root/miniconda3/envs/adodas_train/bin/python`。配置、脚本及预检仍有此布局依赖；换机器时需要检查这些路径，仅设置解释器变量不足以迁移整个流程。

```bash
ADODAS_SEEDS="42 43 44" ADODAS_BATCH_SIZE=36 ADODAS_GRAD_ACCUM=1 \
  bash scripts/run_aligned_training.sh
```

每个种子依次执行 A1 Stage1 → A1 Stage2 → A2，共 9 次训练，再分别集成。脚本先检查资源、数据和文本特征，执行两种模型的 GPU 探测；失败即停止，训练锁防止并行启动。

A1 默认启用文本。完整已有文本向量不需要编码器；否则通过 `ADODAS_TEXT_MODEL` 指定本地编码器，`ADODAS_TEXT_ROOT` 指定向量目录。`ADODAS_A1_USE_TEXT=false` 是明确的 AV-only 变体，不包含公开描述中的文本策略。

建议 32 GB GPU、32 GiB RAM、8 CPU；预检最低要求 16 GiB RAM、4 CPU。30 GiB 显存检查是停止条件，不能保证避免单次分配 OOM，也不保证维持 28–30 GB 占用。以当前 GPU 上的真实探测为准。

## 输出与公开范围

运行结果写入 `outputs/`，包括配置、开发集参与者清单、日志、检查点、预测、校准参数和报告。通过 `python infer.py --run_dir <run>` 重新推理；集成配置的成员必须指向实际运行目录。

公开仓库仅保存源码、配置和方法说明。数据、转写、标签、参与者清单、逐人预测、提交文件、密码、模型缓存和权重不得提交。可以公开去除路径及身份信息的汇总指标，注明划分、种子和选模方法。

`.gitignore` 排除运行缓存和本地历史服务器记录；它不代替内容检查。未指定开源许可证，公开代码不等于授权任意再分发；发布前应确认代码来源及适用许可。
