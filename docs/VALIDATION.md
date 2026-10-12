# 验证记录

本轮在 2026-10-10（Asia/Shanghai）的本机验证，Python 3.12.13。

| 验证项 | 结果 |
| --- | --- |
| pytest | 29 项通过；有一条上游 TestClient 使用 httpx 的弃用提示，不影响本轮测试 |
| Ruff | 检查通过，代码已格式化 |
| Compose | 全部 profiles 配置校验通过；未启动容器或拉取 GPU 镜像 |
| Mock quality | 6 个请求完整记录，全部合法返回；另存 2 条预热日志 |
| Mock load | 2 秒、20 RPS、40 个计划请求完整记录 |
| Mock overload | 1 秒、100 RPS、并发上限 1；100 个请求完整记录，90 个 client_overload，10 个成功；SLO 50ms 下 Goodput 为 0 |
| Mock dynamic | 2 秒内切换两个目录版本，20 个请求完整记录 |
| Classic HTTP | 实际训练 TF-IDF/SVM，通过 localhost FastAPI → HTTPX → Harness 完成 6 条 fixture 请求；合法成功 6/6 |
| 成本工具 | 演示实例价格的离线成本重算、峰值容量/月度 TCO 命令执行通过 |
| 结果汇总 | JSONL / Manifest / Summary / CSV 已生成；预热不混入正式统计 |
| 数据转换 | MASSIVE 原始分区、BANKING77 训练内验证集划分、固定种子的类别分层抽样通过小样本测试 |

本机虚拟环境 `.venv/` 可直接使用。已生成结果位于 `results/`，这些文件和本机 classic 模型默认被 Git 忽略。`requirements-controller.lock.txt` 记录已验证依赖版本。

所有本轮实验均使用人工小型 fixture；`synthetic=true` 同时覆盖 Mock Adapter 与 fixture 数据。Classic 的 HTTP 实测只验证链路，不代表真实数据集精度或服务容量。Manifest 另存 `mock_adapter` 与 `fixture_dataset`，区分模型和数据来源。

真实 Jev API、4090 vLLM、BERT 训练/推理、Docker 镜像构建和真实账单尚未验证。本机合成日志与小型 TF-IDF/SVM fixture 不能作为模型性能结论。
