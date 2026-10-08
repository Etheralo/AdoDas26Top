# 方法对应与复现边界

依据 AdoDAS 挑战论文第 4 页 Table 3 / §5；论文 PDF 不随仓库分发。官方来源入口见 [README](README.md)。以下对应的是公开方法描述，不是冠军原版代码验证。

## 当前分赛道实现

| 策略 | A1 | A2 | 实现位置 |
|---|---|---|---|
| 双 Conformer | 不启用 | 音频/视频各 4 层 | `models/conformer.py` |
| 帧级跨模态注意力 | 双向音视频注意力 | 不启用 | `models/cross_modal.py` |
| 文本适配 | 冻结编码器向量、会话 prompt、带位置编码的 Transformer | 不启用 | `models/text_branch.py` |
| 跨会话融合 | Transformer | Transformer | `models/cross_session.py` |
| 条目预测 | Linear CORAL + A1 引导修正 | 21 个 query-CORAL 输出 | `models/heads.py` |
| 分阶段训练 | 联合训练后仅调整 A1 头 | A2 条目损失 | `train.py` |
| 最终组合 | 同赛道 logits 平均与校准 | 验证集逐条目专家选择 | `ensemble.py` |

当前配置为 `aligned_a1.yaml`、`aligned_a1_stage2.yaml` 和 `aligned_a2.yaml`。`stage*.yaml`、`server*.yaml` 和旧顺序脚本属于早期联合模型实验，不能混用其结果来证明当前方案效果。

## 工程选择

论文没有公开层数、宽度、学习率、损失权重、冻结范围、随机种子、具体特征组合和文本编码器。以下属于本项目选择：

- 192 维表示；Chinese HuBERT-large / DINOv2-large 官方特征。
- A1 第一阶段 A1/A2 损失权重 1/0.25；第二阶段仅 A1 头可训练。
- A2 关闭 A1 损失和引导；两条赛道均关闭会话辅助监督、分量表一致性损失和随机模态/会话丢弃。
- 第一阶段学习率 1e-4、warmup 4 epochs；A1 第二阶段 3e-5、warmup 2 epochs。
- A1 文本使用冻结 Chinese RoBERTa-large 最后 4 层均值，512-token 分块提取；模型最多读取 1,024 个 token。
- 训练 1,500 帧随机窗、评估 4,500 帧中心窗，不覆盖所有长会话；完整会话或多窗口评价仍需验证。
- 三个独立种子 42/43/44；A2 在验证集选择候选模型、解码器及条目偏移。

上述实现形式和参数没有公开证据证明与冠军相同。

## 选模与评分

A1 按未校准的固定 0.5 阈值 F1 选检查点，平分时比较 AUROC。A2 按验证集校准 QWK 选模，平分时比较 MAE。官方 val 同时用于选择与校准，因此其分数不能当作独立测试成绩。

`test_hidden.csv` 不含标签；训练和集成不读取测试参考标签。所有选择冻结后，独立 `tools/score_test.py` 才可评分，且不得根据测试成绩继续调参。详见 [训练计划](ROUND2_PLAN.md)。

## 验证范围

`tools/self_test.py`、`runtime_checks.py`、`check_alignment.py` 和 `check_round2.py` 检查指标、掩码、梯度、冻结范围及评估协议。通过这些检查不证明达到论文成绩。GPU 显存、完整数据覆盖、吞吐和模型效果必须在目标环境单独验证；旧架构的测量不能直接套用当前两种架构。
