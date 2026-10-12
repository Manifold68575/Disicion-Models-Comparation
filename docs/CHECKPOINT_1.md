# Checkpoint 1：MASSIVE 三模型质量与成本基线

状态：0 / 5 项完成。Mock 和 TF-IDF fixture 的成功不计入正式 Checkpoint。

- [ ] Jev、Qwen3-0.6B、BERT 完成同一批 300 条验证请求。
- [ ] 使用同一冻结版本、审阅过的等价候选类别说明。
- [ ] 每条请求保留预测、状态、耗时、模型版本与 Token 用量（接口提供时）。
- [ ] 从逐请求日志独立复算 Accuracy 与 Macro-F1。
- [ ] 固定配置、输入哈希、模型版本、训练 artifact 和 Worker 环境后可重新执行。

进入实际实验所需信息：Jev 的环境变量密钥；GPU Worker SSH 地址；GPU/CPU 实例账单及币种；模型 revision；MASSIVE 语言（默认 en-US）；人工审阅的类别说明。

开始顺序：冻结数据和目录 → 接通三个 Adapter → 同一 300 条质量评测 → 用实际计费数据复算成本 → 独立复算与复测 → 验收后开始并发和动态实验。

每个模型记录 run 目录、数据/目录哈希、Worker 元数据、合法成功数、Accuracy、Macro-F1、usage 完整性与实际费用。若 API 未提供 usage 或出现不确定计费，不用估算值冒充账单。
