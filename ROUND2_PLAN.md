# 分赛道顺序训练计划

## 数据与协议

依据 AdoDAS 挑战论文（PDF 不随仓库分发） 第 2 页 §2.1、第 3 页 §2.3 和 §3、第 4 页 Table 2/3。

- 官方 train 4,200 人 / 16,800 sessions；val 600 人 / 2,400 sessions；test 1,200 人 / 4,800 sessions。三组参与者无交集。
- 第二轮使用完整 train 做梯度更新，官方 val 负责早停、检查点选择、校准和 A2 条目专家选择。`split.development_source: official_val`，禁止 `split.final=true`。
- `test_hidden.csv` 只含 ID、split、session。`references/` 中测试标签不进入训练、校准和集成；所有模型选择冻结后，由独立工具离线评分。
- 验证集已用于选择，报告为 `val_used_for_selection`，不能称独立测试结果。历史 `inner_dev` 方案仍可使用，旧结果保留。
- `development.csv` 保存选模参与者；`inner_dev.csv` 保留兼容旧阶段脚本，在 official_val 模式下也记录验证集 ID。`evaluation_protocol.json` 说明实际用途。

## 两条赛道

| 赛道 | 已公开策略对应实现 |
|---|---|
| A1 / HITSZ_HLT | 帧级音视频 attention、跨 session Transformer、session prompt 文本适配、A1 引导条目修正、第二阶段冻结编码器调整 A1 头 |
| A2 / HEALTH FIRST | 音视频双 Conformer、跨 session 融合、21 个 query-CORAL ordinal heads、在 val 上逐条目选择候选模型与解码参数 |

独立 seeds 42/43/44，依次执行 A1 Stage1 → A1 Stage2 → A2，共 9 次训练，最后分别集成。A1 按原始 0.5 阈值 F1 选检查点，导出概率可进行验证集 logit 校准；评分仍固定阈值 0.5。A2 按验证集校准 QWK 选检查点，条目专家默认选择最高验证 QWK（min_gain=0）。

## 工程选择与未公开内容

**不能声称精确复现冠军实现。** 挑战论文没有提供冠军模型代码、层数、损失权重、冻结范围、学习率、batch、seed、特征组合和文本预训练模型。

当前这些均为工程选择：A1 第一阶段 A1/A2 权重 1/0.25；A2 仅条目损失；192 维、Conformer 4 层；Hubert-large/DINOv2-large；训练 1,500 帧随机窗、评估 4,500 帧中心窗；A1 文本冻结 Chinese RoBERTa-large 最后 4 层均值、session prompt 和有位置编码的 Transformer，最多 1,024 tokens。窗口不覆盖全部长 session，是仍存在的复现差异，需资源允许时做完整 session 或多窗口验证。

未加入论文未披露的额外一致性损失和 session 辅助监督。每个阶段从独立新模型或本轮 Stage1 最优检查点开始，不复用历史退化模型。

## 启动前与命令

当前顺序脚本及部分预检依赖 AutoDL 的 `/root/autodl-tmp/dataset-ai` 布局；迁移说明见 [README](README.md)。以下命令在该布局下执行。

资源建议：32 GB GPU、至少 32 GiB 主机内存和 8 核 CPU。脚本最低检查 16 GiB / 4 核；这些是本地运行要求，不是论文指定配置。

```bash
cd /root/autodl-tmp/adodas_sota
PY=/root/miniconda3/envs/adodas_train/bin/python
# 可选数据审计：要求授权参考标签，仅把身份键用于测试清单；不输出转写。
$PY tools/prepare_evaluation.py --output /root/autodl-tmp/dataset-ai/evaluation_audit.json
# 资源和数据预检；无 GPU 时仅 --allow-cpu 可做诊断。
$PY tools/preflight.py --config configs/aligned_a2.yaml
# GPU 分配后：先提取全部 ASR 向量，再对两种架构做真实数据/最大长度显存检查（仅使用本地模型或已有向量）。
bash scripts/run_aligned_training.sh
```

禁止自动下载模型。仅接受已准备的本地模型（`ADODAS_TEXT_MODEL`）或完整已有文本向量；资源是否齐全由目标环境预检确定。空转写保存显式空特征，模型使用 missing-text token；不能悄悄关闭 A1 文本。

默认 batch 36，显存预算 30 GiB。不能用旧架构探测代替当前模型实测，也不保证维持 28–30 GB。若探测超预算，脚本停止；调整 `ADODAS_BATCH_SIZE=18 ADODAS_GRAD_ACCUM=2` 后再探测。显存检查不能阻止单次分配 OOM，推理也纳入最大长度检查。禁止依靠填充无用张量提高显存占用。

## 最终测试评分

完整序列状态 COMPLETE 后，分别对 A1 集成的 A1 提交和 A2 集成的 A2 提交评分：

```bash
$PY tools/score_test.py --task a1 --reference_root /root/autodl-tmp/dataset-ai/references --submission <A1集成目录>/submissions/submission_a1_test_hidden.csv --output <A1集成目录>/test_score_a1.json
$PY tools/score_test.py --task a2 --reference_root /root/autodl-tmp/dataset-ai/references --submission <A2集成目录>/submissions/submission_a2_test_hidden.csv --output <A2集成目录>/test_score_a2.json
```

固定所有选择后仅作评估，不能据测试分数调参。此为本地参考标签评分，不等同官方榜单提交。论文目标：A1 F1 .5921 / AUROC .7652；A2 QWK .2776 / MAE .4807。
